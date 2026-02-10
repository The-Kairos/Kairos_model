import os
import subprocess
import json
import re
from pathlib import Path
import imageio_ffmpeg as ffmpeg
import pandas as pd

SECTION_LINE = "=" * 40


def print_section(title: str) -> None:
    print(SECTION_LINE)
    print(title)
    print(SECTION_LINE)


def print_prefixed(prefix: str, message: str, indent: int = 0) -> None:
    pad = " " * indent
    print(f"{prefix} {pad}{message}")


def format_timecode(seconds: float | None) -> str:
    if seconds is None:
        return "??:??:??.???"
    try:
        ms_total = int(round(float(seconds) * 1000))
    except (TypeError, ValueError):
        return "??:??:??.???"
    sec_total, ms = divmod(ms_total, 1000)
    mins_total, sec = divmod(sec_total, 60)
    hrs, mins = divmod(mins_total, 60)
    return f"{hrs:02d}:{mins:02d}:{sec:02d}.{ms:03d}"

def see_first_scene(df):
    print("Printing first captioned scene:")
    print("{")
    for key in df[0]:
        if key == "frames": continue
        print(f"{key}, {df[0][key]},")
    print("}")

def see_scenes_cuts(df):
    print_prefixed("(PysceneDetect)", f"Found {len(df)} scenes.")
    for idx, s in enumerate(df):
        scene_index = s.get("scene_index", idx)
        scene_label = f"{int(scene_index):03d}" if isinstance(scene_index, (int, float)) else str(scene_index)
        start_tc = s.get("start_timecode") or format_timecode(s.get("start_seconds"))
        end_tc = s.get("end_timecode") or format_timecode(s.get("end_seconds"))
        print_prefixed(
            "(PysceneDetect)",
            f"Scene {scene_label}: {start_tc} -> {end_tc} ({s['duration_seconds']:.2f} sec)",
            indent=4,
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

# =================== LLM UTILS ===================
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

    with open(path, "r", encoding="utf-8-sig") as f:
        mapping = json.load(f)

    for src, dst in mapping.items():
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text, flags=re.IGNORECASE)
    return text

# =================== MAIN CLI UTILS ===================

def load_video_catalog(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "videos" in data:
        data = data["videos"]
    if not isinstance(data, list):
        raise ValueError("Expected _all_videos.json to be a list of video objects.")
    return data


def get_video_length_seconds(entry: dict) -> float | None:
    value = entry.get("video_length")
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return None


def categorize_length(seconds: float) -> str:
    minutes = seconds / 60
    if minutes < 10:
        return "short"
    if minutes < 30:
        return "medium"
    if minutes < 90:
        return "long"
    return "extra"


def make_output_dir(video_path: Path, processed_root: Path | str = "processed") -> str:
    name = video_path.name
    if name.startswith("."):
        name = name.lstrip(".")
    name = name.strip().rstrip(".")
    if not name:
        name = "video"
    return str(Path(processed_root) / name)


def resolve_video_arg(arg: str, blob_index: dict, videos_dir: Path) -> Path | None:
    candidate = Path(arg)
    if candidate.exists():
        return candidate
    candidate = videos_dir / arg
    if candidate.exists():
        return candidate
    entry = blob_index.get(arg)
    if entry and entry.get("blob"):
        candidate = videos_dir / entry["blob"]
        if candidate.exists():
            return candidate
    return None


def select_videos(args, catalog: list[dict], videos_dir: Path) -> list[Path]:
    blob_index = {v.get("blob"): v for v in catalog if isinstance(v, dict) and v.get("blob")}
    selected_paths: list[Path] = []

    if args.video:
        items = args.video if isinstance(args.video, list) else [args.video]
        for item in items:
            path = resolve_video_arg(item, blob_index, videos_dir)
            if path is None:
                print(f"Skip: video not found: {item}")
                continue
            selected_paths.append(path)
        return selected_paths

    filter_value = getattr(args, "filter", None)
    include_unknown = getattr(args, "include_unknown", False)
    include_all = getattr(args, "all", False)

    if not (include_all or filter_value):
        print("Select videos with --video, --all, or --filter.")
        raise SystemExit(2)

    entries = catalog
    if filter_value:
        rank = {"short": 1, "medium": 2, "long": 3, "extra": 4}
        selected_entries = []
        unknown = 0
        for entry in entries:
            length = get_video_length_seconds(entry)
            if length is None:
                if include_unknown:
                    selected_entries.append(entry)
                else:
                    unknown += 1
                continue
            if rank[categorize_length(length)] <= rank[filter_value]:
                selected_entries.append(entry)
        if unknown and not include_unknown:
            print(f"Skipping {unknown} video(s) with unknown length. Use --include-unknown to include.")
        entries = selected_entries

    for entry in entries:
        blob = entry.get("blob")
        if not blob:
            continue
        path = videos_dir / blob
        if not path.exists():
            print(f"Skip: missing file on disk: {blob}")
            continue
        selected_paths.append(path)

    return selected_paths
