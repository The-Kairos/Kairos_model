"""Baseline scene splitting using PySceneDetect only (no semantic merge)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

LOCAL_DIR = Path(__file__).resolve().parent
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))

from pyscene_utils import (
    DEFAULT_VIDEO,
    finalize_scene_times,
    save_scene_boundary_frames,
    to_relative,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scene_cutting import get_scene_list
from vit_scene import pyscene_shortest, pyscene_threshold, frame_resolution


def run(
    video_path: str | Path = DEFAULT_VIDEO,
    threshold: float = pyscene_threshold,
    min_scene_sec: float = pyscene_shortest,
) -> dict:
    """Run baseline PySceneDetect scene cuts and return results."""
    video_path = Path(video_path)
    if not video_path.is_absolute():
        video_path = (PROJECT_ROOT / video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    start = time.perf_counter()
    scenes = get_scene_list(
        input_video_path=str(video_path),
        threshold=threshold,
        min_scene_sec=min_scene_sec,
    )

    scenes = finalize_scene_times(scenes)
    scenes = save_scene_boundary_frames(
        video_path=video_path,
        scenes=scenes,
        test_name=Path(__file__).stem,
        new_size=frame_resolution,
    )
    elapsed = time.perf_counter() - start

    result = {
        "video": to_relative(video_path),
        "scenes": scenes,
        "elapsed_seconds": elapsed,
        "threshold": threshold,
    }
    return result


def main() -> None:
    """CLI entry-point for baseline PySceneDetect scene cuts."""
    parser = argparse.ArgumentParser(description="Baseline PySceneDetect scene cuts.")
    parser.add_argument(
        "--video",
        default=str(DEFAULT_VIDEO),
        help="Path to the input video.",
    )
    parser.add_argument("--threshold", type=float, default=pyscene_threshold)
    parser.add_argument("--min-scene-sec", type=float, default=pyscene_shortest)
    args = parser.parse_args()

    result = run(
        video_path=args.video,
        threshold=args.threshold,
        min_scene_sec=args.min_scene_sec,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
