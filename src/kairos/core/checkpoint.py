"""Checkpoint reading, writing, and frame cleanup."""

import json
import subprocess
from pathlib import Path

import imageio_ffmpeg as ffmpeg

from kairos.core.utils import print_prefixed


def clear_frames(scene_list: list) -> list:
    omit_keys = {
        "frames", "yolo_frames",
        "frame_paths", "yolo_frame_paths", "frame_indices", "frame_timestamps",
        "sample_fps", "motion_bullets", "yolo_tracks", "yolo_track_summaries",
    }
    return [
        {k: v for k, v in scene.items() if k not in omit_keys}
        for scene in scene_list
    ]


def read_json(json_path: str | Path) -> dict:
    json_path = Path(json_path)
    if not json_path.exists():
        print_prefixed("(Checkpoint)", f"JSON path does not exist: {json_path}")
        return {}

    print_prefixed("(Checkpoint)", f"Reading JSON from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)
        if isinstance(checkpoint, list):
            return {"scenes": checkpoint}
        return checkpoint


def save_checkpoint(checkpoint: dict | list, path: str | Path) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(checkpoint, list):
        checkpoint = {"scenes": clear_frames(checkpoint)}
    elif isinstance(checkpoint, dict):
        checkpoint["scenes"] = clear_frames(checkpoint["scenes"])
    else:
        raise TypeError("checkpoint must be a dict or list")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=4, ensure_ascii=False)

    return checkpoint


def have_key(scenes: list, key: str) -> bool:
    return bool(scenes) and all(key in s for s in scenes)


def save_clips(video_path: str, scenes: list, output_dir: str) -> list:
    ffmpeg_path = ffmpeg.get_ffmpeg_exe()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    updated_scenes = []

    for scene in scenes:
        start = scene["start_seconds"]
        end = scene["end_seconds"]
        duration = end - start

        scene_index = scene.get("scene_index", len(updated_scenes))
        clip_filename = f"scene_{scene_index:04d}.mp4"
        clip_path = output_dir / clip_filename

        cmd = [
            ffmpeg_path,
            "-y",
            "-i", video_path,
            "-ss", str(start),
            "-t", str(duration),
            "-c", "copy",
            str(clip_path),
        ]

        if clip_path.exists():
            print_prefixed("(save_clips)", f"Skipping existing clip: {clip_filename}", indent=4)
        else:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        scene_new = dict(scene)
        scene_new["clip_path"] = str(clip_path)
        updated_scenes.append(scene_new)

    return updated_scenes
