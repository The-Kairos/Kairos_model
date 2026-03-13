from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

LOCAL_DIR = Path(__file__).resolve().parent
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))

from pyscene_utils import DEFAULT_VIDEO

import pyscene_base
import vit_scene
import clip_scene
import blip_scene
import py_vit
import py_clip
import py_blip
from _view_scene_map import build_contact_sheets

RESULTS_PATH = LOCAL_DIR / "_test_logs.json"
TABLE_PATH = LOCAL_DIR / "_test_results.md"


def _summarize(result: dict) -> dict:
    scenes = result.get("scenes", []) if isinstance(result, dict) else []
    durations = []
    for scene in scenes:
        if "duration_seconds" in scene:
            durations.append(scene["duration_seconds"])
        else:
            start = scene.get("start_seconds")
            end = scene.get("end_seconds")
            if start is not None and end is not None:
                durations.append(max(0.0, float(end) - float(start)))

    count = len(durations)
    avg_len = sum(durations) / count if count else 0.0
    min_len = min(durations) if count else 0.0
    max_len = max(durations) if count else 0.0

    return {
        "scene_count": count,
        "avg_len": avg_len,
        "min_len": min_len,
        "max_len": max_len,
        "elapsed": float(result.get("elapsed_seconds", 0.0)) if isinstance(result, dict) else 0.0,
    }


def run_all(video_path: str | Path = DEFAULT_VIDEO) -> dict:
    tests = [
        ("pyscene_base", pyscene_base.run),
        ("vit_scene", vit_scene.run),
        ("clip_scene", clip_scene.run),
        ("blip_scene", blip_scene.run),
        ("py_vit", py_vit.run),
        ("py_clip", py_clip.run),
        ("py_blip", py_blip.run),
    ]

    results: dict = {"video": str(video_path)}

    for name, fn in tests:
        start = time.perf_counter()
        try:
            result = fn(video_path=video_path)
            results[name] = result
        except Exception as exc:
            results[name] = {
                "video": str(video_path),
                "scenes": [],
                "elapsed_seconds": time.perf_counter() - start,
                "error": str(exc),
            }

    return results


def _iter_video_results(results: dict) -> list[dict]:
    if isinstance(results, dict) and isinstance(results.get("videos"), list):
        return [r for r in results["videos"] if isinstance(r, dict)]
    if isinstance(results, dict):
        return [results]
    return []


def _load_existing_results() -> dict:
    if not RESULTS_PATH.exists():
        return {"videos": []}
    try:
        data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"videos": []}
    if isinstance(data, dict) and isinstance(data.get("videos"), list):
        return {"videos": [r for r in data["videos"] if isinstance(r, dict)]}
    if isinstance(data, dict):
        return {"videos": [data]}
    return {"videos": []}


def _merge_results(existing: list[dict], new: list[dict]) -> list[dict]:
    new_by_video = {}
    for entry in new:
        if isinstance(entry, dict) and entry.get("video"):
            new_by_video[str(entry["video"])] = entry

    merged: list[dict] = []
    seen = set()
    for entry in existing:
        if not isinstance(entry, dict) or not entry.get("video"):
            continue
        video_key = str(entry["video"])
        if video_key in new_by_video:
            merged.append(new_by_video.pop(video_key))
        else:
            merged.append(entry)
        seen.add(video_key)

    for entry in new_by_video.values():
        merged.append(entry)

    return merged


def _write_table(results: dict) -> None:
    lines = []
    for video_result in _iter_video_results(results):
        video_label = Path(str(video_result.get("video", ""))).name or "unknown"
        lines.append(f"## {video_label}")
        lines.append("")
        lines.append("| Test | Scenes | Avg Len (s) | Min Len (s) | Max Len (s) | Elapsed (s) | Status |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for name, result in video_result.items():
            if name == "video":
                continue
            if not isinstance(result, dict):
                lines.append(f"| {name} | 0 | 0 | 0 | 0 | 0 | error |")
                continue
            summary = _summarize(result)
            status = "ok" if "error" not in result else "error"
            lines.append(
                "| {name} | {count} | {avg:.2f} | {min:.2f} | {max:.2f} | {elapsed:.2f} | {status} |".format(
                    name=name,
                    count=summary["scene_count"],
                    avg=summary["avg_len"],
                    min=summary["min_len"],
                    max=summary["max_len"],
                    elapsed=summary["elapsed"],
                    status=status,
                )
            )
        lines.append("")

    TABLE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scene tests on one or more videos.")
    parser.add_argument(
        "--video",
        action="append",
        help="Path to a video (repeatable). Defaults to the sample video.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos = args.video if args.video else [str(DEFAULT_VIDEO)]
    existing = _load_existing_results()
    new_results = [run_all(video) for video in videos]
    results_all = {"videos": _merge_results(existing.get("videos", []), new_results)}

    RESULTS_PATH.write_text(json.dumps(results_all, indent=2), encoding="utf-8")
    _write_table(results_all)
    build_contact_sheets()
    print(f"Saved results: {RESULTS_PATH}")
    print(f"Saved table: {TABLE_PATH}")


if __name__ == "__main__":
    main()
