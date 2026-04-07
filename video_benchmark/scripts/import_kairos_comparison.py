"""
Import log_reports/*_comparison.json into video_benchmark data files + Kairos recording for replay.

Extracts the user query string and the selected pipeline branch (flat / kmeans / hdbscan).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils import bench_root, data_dir, outputs_dir, save_json, utc_now_iso


def _norm_q(s: str) -> str:
    return " ".join(s.strip().split()).lower()


def import_comparison(
    comparison_path: Path,
    *,
    strategy: str,
    video_id: str,
    video_file_rel: str,
    out_rel: str,
    category: str = "long",
) -> dict[str, Any]:
    with open(comparison_path, encoding="utf-8") as f:
        doc = json.load(f)
    video_name = doc.get("video", comparison_path.stem)
    qlist = doc.get("queries")
    if not isinstance(qlist, list):
        raise ValueError("comparison JSON missing 'queries' list")

    out_root = data_dir() / out_rel
    out_root.mkdir(parents=True, exist_ok=True)

    queries_out: list[dict] = []
    answers_out: list[dict] = []
    recording: list[dict] = []

    for i, block in enumerate(qlist):
        qtext = block.get("query")
        if not qtext:
            continue
        qid = f"ys_{i + 1:03d}"
        results = block.get("results") or {}
        branch = results.get(strategy)
        if not isinstance(branch, dict):
            raise KeyError(f"Query {qid}: no results.{strategy} in comparison file")

        answer = branch.get("answer") or ""
        total_time = float(branch.get("total_time") or 0.0)
        retr = float(branch.get("retrieval_time") or 0.0)
        gen = float(branch.get("generation_time") or 0.0)
        chunks = branch.get("chunks") or []
        ranked = [f"{strategy}_chunk_{j}" for j in range(len(chunks))] if chunks else []

        queries_out.append(
            {
                "question_id": qid,
                "video_id": video_id,
                "question": qtext,
                "type": "retrieval_qa",
                "difficulty": "medium",
            }
        )
        answers_out.append(
            {
                "question_id": qid,
                "video_id": video_id,
                "gold_answer": answer,
                "acceptable_variants": [],
                "relevant_segment_ids": [],
                "annotation_status": f"copied_from_kairos_{strategy}",
                "annotation_note": "Replace gold_answer with human-verified truth for rigorous accuracy; "
                "current text is Kairos output used as baseline or agreement target.",
            }
        )
        recording.append(
            {
                "video_id": video_id,
                "question": qtext,
                "question_id": qid,
                "response": answer,
                "latency_sec": round(total_time, 4),
                "ranked_segment_ids": ranked,
                "kairos": {
                    "strategy": strategy,
                    "retrieval_time_sec": retr,
                    "generation_time_sec": gen,
                    "total_time_sec": total_time,
                    "chunk_count": len(chunks),
                },
            }
        )

    video_info = [
        {
            "video_id": video_id,
            "file": video_file_rel,
            "category": category,
            "duration_seconds": None,
            "source_video_name": video_name,
            "notes": "Point `file` at your local copy of the episode/clips used for the comparison run.",
        }
    ]

    save_json(out_root / "queries.json", queries_out)
    save_json(out_root / "answers.json", answers_out)
    save_json(out_root / "video_info.json", video_info)
    save_json(out_root / f"kairos_recording_{strategy}.json", recording)

    manifest = {
        "imported_at": utc_now_iso(),
        "source_comparison_json": str(comparison_path.resolve()),
        "video_id": video_id,
        "strategy": strategy,
        "query_count": len(queries_out),
        "output_dir": str(out_root.resolve()),
        "recording_file": f"{out_rel}/kairos_recording_{strategy}.json",
    }
    save_json(out_root / "import_manifest.json", manifest)

    run_id = utc_now_iso().replace(":", "-")
    response_rows = []
    for rec in recording:
        g = next((a for a in answers_out if a["question_id"] == rec["question_id"]), None)
        response_rows.append(
            {
                "system": f"kairos_{strategy}",
                "run_id": run_id,
                "video_id": video_id,
                "question_id": rec["question_id"],
                "question": rec["question"],
                "question_type": "retrieval_qa",
                "category": category,
                "response": rec["response"],
                "latency_sec": rec["latency_sec"],
                "ranked_segment_ids": rec["ranked_segment_ids"],
                "system_metadata": {"kairos": rec["kairos"]},
                "gold": g,
                "error": None,
            }
        )
    raw_dir = outputs_dir() / "raw" / f"kairos_{strategy}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    responses_path = raw_dir / f"{run_id}_imported_responses.json"
    save_json(responses_path, response_rows)

    return {"manifest": manifest, "responses_path": str(responses_path)}


def main() -> None:
    default_input = (
        bench_root().parent
        / "log_reports"
        / "comparison_results"
        / "Young_Sheldon_-_First_Day_of_High_School.mp4_comparison.json"
    )
    parser = argparse.ArgumentParser(description="Import Kairos comparison_results JSON into benchmark dataset.")
    parser.add_argument("--input", type=Path, default=default_input, help="Path to *_comparison.json")
    parser.add_argument("--strategy", choices=("flat", "kmeans", "hdbscan"), default="flat")
    parser.add_argument("--video-id", default="young_sheldon_first_day")
    parser.add_argument(
        "--video-file",
        default="videos/long/Young Sheldon - First Day of High School.mp4",
        help="Path relative to video_benchmark/data/ where you store the video file",
    )
    parser.add_argument("--out-dir", default="datasets/young_sheldon", help="Folder under data/ to write")
    parser.add_argument("--category", default="long")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Input file not found: {args.input}")
        sys.exit(1)

    info = import_comparison(
        args.input,
        strategy=args.strategy,
        video_id=args.video_id,
        video_file_rel=args.video_file,
        out_rel=args.out_dir,
        category=args.category,
    )
    print(json.dumps(info["manifest"], indent=2))
    print(f"Wrote pre-built Kairos responses: {info['responses_path']}")


if __name__ == "__main__":
    main()
