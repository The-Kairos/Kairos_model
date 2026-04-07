"""
Aggregate metrics from raw responses + optional LLM judge output.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.llm_judge import judge_batch
from scripts.utils import bench_root, load_json, load_yaml, outputs_dir, save_json, utc_now_iso


def recall_at_k(
    ranked: list[str] | None,
    gold_ids: list[str] | None,
    k: int,
) -> float | None:
    if not ranked or not gold_ids:
        return None
    gold_set = {str(x) for x in gold_ids}
    top = ranked[:k]
    hits = sum(1 for x in top if str(x) in gold_set)
    return hits / max(len(gold_set), 1)


def precision_at_k(
    ranked: list[str] | None,
    gold_ids: list[str] | None,
    k: int,
) -> float | None:
    if not ranked or not gold_ids:
        return None
    gold_set = {str(x) for x in gold_ids}
    top = ranked[:k]
    if not top:
        return None
    hits = sum(1 for x in top if str(x) in gold_set)
    return hits / len(top)


def compute_retrieval_metrics(rows: list[dict], k_values: list[int]) -> dict[str, Any]:
    per_k_recall: dict[str, list[float]] = defaultdict(list)
    per_k_prec: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        ranked = row.get("ranked_segment_ids")
        gold = (row.get("gold") or {}).get("relevant_segment_ids")
        if not gold or not ranked:
            continue
        for k in k_values:
            r = recall_at_k(ranked, gold, k)
            p = precision_at_k(ranked, gold, k)
            if r is not None:
                per_k_recall[f"recall@{k}"].append(r)
            if p is not None:
                per_k_prec[f"precision@{k}"].append(p)
    out: dict[str, Any] = {}
    for name, vals in per_k_recall.items():
        out[name] = sum(vals) / len(vals) if vals else None
    for name, vals in per_k_prec.items():
        out[name] = sum(vals) / len(vals) if vals else None
    return out


def accuracy_breakdown(judged: list[dict]) -> dict[str, Any]:
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total = defaultdict(int)
    for row in judged:
        v = row.get("verdict") or "incorrect"
        qt = row.get("question_type") or row.get("type") or "unknown"
        cat = row.get("category") or "unknown"
        by_type[qt][v] += 1
        by_cat[cat][v] += 1
        total[v] += 1
    n = len(judged) or 1
    strict_correct = total.get("correct", 0) / n
    partial_credit = (total.get("correct", 0) + 0.5 * total.get("partial", 0)) / n
    return {
        "counts": dict(total),
        "accuracy_strict": strict_correct,
        "accuracy_with_partial_credit": partial_credit,
        "by_question_type": {k: dict(v) for k, v in by_type.items()},
        "by_video_category": {k: dict(v) for k, v in by_cat.items()},
    }


def latency_stats(rows: list[dict]) -> dict[str, Any]:
    lat = [float(r["latency_sec"]) for r in rows if r.get("latency_sec") is not None and not r.get("error")]
    if not lat:
        return {"count": 0}
    return {
        "count": len(lat),
        "mean_sec": statistics.mean(lat),
        "median_sec": statistics.median(lat),
        "stdev_sec": statistics.stdev(lat) if len(lat) > 1 else 0.0,
    }


def run_evaluate(
    responses_path: Path,
    out_path: Path | None,
    skip_judge: bool,
) -> dict[str, Any]:
    rows = load_json(responses_path)
    if not isinstance(rows, list):
        raise TypeError("responses file must be a JSON list")
    eval_cfg_path = bench_root() / "config" / "evaluation.yaml"
    eval_cfg = load_yaml(eval_cfg_path)
    retrieval_cfg = eval_cfg.get("retrieval") or {}
    k_values = retrieval_cfg.get("k_values") or [1, 3, 5]
    compute_rk = retrieval_cfg.get("compute_recall_at_k", True)

    judged = list(rows)
    if not skip_judge:
        to_judge_idx = [
            i
            for i, r in enumerate(judged)
            if not r.get("error") and (r.get("gold") or {}).get("gold_answer")
        ]
        batch = [judged[i] for i in to_judge_idx]
        scored = judge_batch(batch) if batch else []
        for idx, new_row in zip(to_judge_idx, scored):
            judged[idx] = new_row

    metrics: dict[str, Any] = {
        "evaluated_at": utc_now_iso(),
        "source_responses": str(responses_path),
        "latency": latency_stats(judged),
        "responses_with_error": sum(1 for r in judged if r.get("error")),
    }
    if not skip_judge:
        metrics["qa"] = accuracy_breakdown([r for r in judged if "verdict" in r])
    if compute_rk:
        metrics["retrieval"] = compute_retrieval_metrics(judged, k_values)

    out = {"metrics": metrics, "rows": judged}
    dest = out_path or (outputs_dir() / "evaluated" / (responses_path.stem + "_scored.json"))
    save_json(dest, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score benchmark responses (LLM judge + metrics).")
    parser.add_argument(
        "--responses",
        type=Path,
        help="Path to raw responses JSON (list of rows). Default: latest in outputs/raw/",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path.")
    parser.add_argument("--skip-judge", action="store_true", help="Only compute latency / retrieval, no API calls.")
    args = parser.parse_args()

    resp_path = args.responses
    if resp_path is None:
        raw_root = outputs_dir() / "raw"
        candidates = sorted(raw_root.glob("**/*_responses.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("No responses found under outputs/raw; run run_benchmark.py first or pass --responses.")
            sys.exit(1)
        resp_path = candidates[0]
        print(f"Using responses file: {resp_path}")

    run_evaluate(resp_path, args.out, args.skip_judge)
    print("Wrote evaluated output.")


if __name__ == "__main__":
    main()
