"""Video catalog loading and selection."""

from __future__ import annotations

import json
from pathlib import Path


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
