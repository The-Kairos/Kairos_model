import os
import subprocess
import json
import re
from pathlib import Path
import imageio_ffmpeg as ffmpeg
import pandas as pd

def see_first_scene(df):
    print("Printing first captioned scene:")
    print("{")
    for key in df[0]:
        if key == "frames": continue
        print(f"{key}, {df[0][key]},")
    print("}")

def see_scenes_cuts(df):
    print(f"Found {len(df)} scenes.")
    for s in df:
        print(
            f"Scene {s['scene_index']:03d}: "
            f"{s['start_timecode']} -> {s['end_timecode']} "
            f"({s['duration_seconds']:.2f} sec)"
        )

def save_clips(video_path, scenes, output_dir):
    ffmpeg_path = ffmpeg.get_ffmpeg_exe()   # portable FFmpeg binary
    
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
            str(clip_path)
        ]

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        scene_new = dict(scene)
        scene_new["clip_path"] = str(clip_path)
        updated_scenes.append(scene_new)

    return updated_scenes

def clear_frames(scene_list):
    omit_keys = {
        "frames", "yolo_frames", 
        "frame_paths", "yolo_frame_paths", "frame_indices", "frame_timestamps", 
        "sample_fps", "motion_bullets", "yolo_tracks", "yolo_track_summaries", 
        }
    cleaned = [
        {k: v for k, v in scene.items() if k not in omit_keys}
        for scene in scene_list
    ]
    return cleaned

def read_json(json_path):
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"JSON path does not exist: {json_path}")
        return {}

    print(f"Reading JSON from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        checkpoint= json.load(f)
        if isinstance(checkpoint, list):
            return {"scenes": checkpoint}
        return checkpoint

def save_checkpoint(checkpoint, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(checkpoint, list):
        checkpoint = {"scenes": clear_frames(checkpoint)}

    elif isinstance(checkpoint, dict):
        checkpoint["scenes"] = clear_frames(checkpoint["scenes"])
    else:
        raise TypeError("checkpoint must be a dict or list")
        
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=4)

    return checkpoint


def have_key(scenes, key: str) -> bool:
    return bool(scenes) and all(key in s for s in scenes)

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
def load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")

def apply_gpt_normalization(text: str, filename: str = "gpt_normalizations.json") -> str:
    """
    Normalize text before sending to GPT using word-boundary replacements.
    Mapping is loaded from prompts/gpt_normalizations.json by default.
    """
    path = PROMPTS_DIR / filename
    if not path.exists():
        return text

    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    for src, dst in mapping.items():
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text, flags=re.IGNORECASE)
    return text
