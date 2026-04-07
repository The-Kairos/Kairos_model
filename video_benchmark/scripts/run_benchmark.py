"""
Run all (or selected) systems against the dataset: upload videos once per run, then ask each query.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from scripts.evaluate import run_evaluate
from scripts.utils import (
    bench_root,
    load_dataset_files,
    load_json,
    load_yaml,
    merge_query_ground_truth,
    outputs_dir,
    resolve_dataset_yaml,
    resolve_video_path,
    save_json,
    utc_now_iso,
)
from systems.registry import build_system, load_systems_config


def _file_fingerprint(path: Path) -> str:
    st = path.stat()
    raw = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _load_state(path: Path) -> dict[str, Any]:
    if path.is_file():
        return load_json(path)
    return {"uploads": {}}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    save_json(path, state)


def run_for_system(
    system_name: str,
    merged_queries: list[dict],
    video_catalog: list[dict],
    *,
    fresh_upload: bool,
    run_id: str,
) -> list[dict]:
    sys_cfg = load_systems_config()
    system = build_system(system_name, sys_cfg)
    state_path = outputs_dir() / "raw" / system_name / "upload_state.json"
    state = {} if fresh_upload else _load_state(state_path)

    video_by_id = {v["video_id"]: v for v in video_catalog}
    uploads: dict[str, Any] = state.get("uploads") or {}

    remote_ids: dict[str, str] = {}

    try:
        for vid, meta in video_by_id.items():
            vpath = resolve_video_path(meta)
            fp = _file_fingerprint(vpath)
            entry = uploads.get(vid)
            if not fresh_upload and entry and entry.get("fingerprint") == fp and entry.get("remote_id"):
                remote_ids[vid] = entry["remote_id"]
                continue
            try:
                rid = system.upload_video(vpath, vid)
            except Exception as e:
                raise RuntimeError(f"upload failed for video_id={vid} path={vpath}: {e}") from e
            uploads[vid] = {"fingerprint": fp, "remote_id": rid, "path": str(vpath)}
            remote_ids[vid] = rid
            state = {"uploads": uploads, "updated_at": utc_now_iso()}
            _save_state(state_path, state)

        rows: list[dict] = []
        for q in merged_queries:
            vid = q.get("video_id")
            meta = video_by_id.get(vid) or {}
            remote = remote_ids.get(vid)
            if not remote:
                rows.append(
                    {
                        "system": system_name,
                        "run_id": run_id,
                        "video_id": vid,
                        "question_id": q.get("question_id"),
                        "question": q.get("question"),
                        "question_type": q.get("type"),
                        "category": meta.get("category"),
                        "response": "",
                        "latency_sec": None,
                        "ranked_segment_ids": None,
                        "gold": q.get("gold"),
                        "error": "missing_remote_video_id",
                    }
                )
                continue
            try:
                result = system.ask(remote, q.get("question", ""), vid)
            except Exception as e:
                rows.append(
                    {
                        "system": system_name,
                        "run_id": run_id,
                        "video_id": vid,
                        "question_id": q.get("question_id"),
                        "question": q.get("question"),
                        "question_type": q.get("type"),
                        "category": meta.get("category"),
                        "response": "",
                        "latency_sec": None,
                        "ranked_segment_ids": None,
                        "gold": q.get("gold"),
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(),
                    }
                )
                continue
            rows.append(
                {
                    "system": system_name,
                    "run_id": run_id,
                    "video_id": vid,
                    "question_id": q.get("question_id"),
                    "question": q.get("question"),
                    "question_type": q.get("type"),
                    "category": meta.get("category"),
                    "response": result.text,
                    "latency_sec": round(result.latency_sec, 4),
                    "ranked_segment_ids": result.ranked_segment_ids,
                    "system_metadata": result.metadata,
                    "gold": q.get("gold"),
                    "error": None,
                }
            )
        return rows
    finally:
        system.close()


def main() -> None:
    load_dotenv(bench_root() / ".env")
    load_dotenv(bench_root().parent / ".env")

    parser = argparse.ArgumentParser(description="Video QA benchmark runner.")
    parser.add_argument(
        "--systems",
        default="all",
        help='Comma-separated system keys from config/systems.yaml, or "all" for enabled list.',
    )
    parser.add_argument("--fresh-upload", action="store_true", help="Ignore cached upload_state.json.")
    parser.add_argument("--skip-eval", action="store_true", help="Do not run LLM judge after collecting responses.")
    parser.add_argument("--eval-only-skip-judge", action="store_true", help="If evaluating, only latency/recall metrics.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset YAML under config/ (e.g. dataset.young_sheldon.yaml) or path to a dataset yaml file.",
    )
    args = parser.parse_args()

    ds_cfg = load_yaml(resolve_dataset_yaml(args.dataset))
    queries, answers, videos = load_dataset_files(ds_cfg)
    merged = merge_query_ground_truth(queries, answers)

    sys_cfg = load_systems_config()
    if args.systems.strip().lower() == "all":
        names = list(sys_cfg.get("enabled") or [])
    else:
        names = [x.strip() for x in args.systems.split(",") if x.strip()]

    if not names:
        print("No systems selected. Edit config/systems.yaml enabled list or pass --systems.")
        sys.exit(1)

    run_id = utc_now_iso().replace(":", "-")

    for name in names:
        print(f"=== Running system: {name} ===")
        rows = run_for_system(
            name,
            merged,
            videos,
            fresh_upload=args.fresh_upload,
            run_id=run_id,
        )
        out_dir = outputs_dir() / "raw" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_id}_responses.json"
        save_json(out_path, rows)
        print(f"Wrote {out_path} ({len(rows)} rows)")

        if not args.skip_eval:
            eval_out = outputs_dir() / "evaluated" / f"{name}_{run_id}_scored.json"
            run_evaluate(out_path, eval_out, skip_judge=args.eval_only_skip_judge)
            print(f"Wrote {eval_out}")


if __name__ == "__main__":
    main()
