"""Video catalog loading and selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_video_catalog(path: Path) -> list[dict]:
    """Load the video catalog from a JSON file.

    The file may contain either a plain list of video objects or a dict
    with a ``"videos"`` key whose value is the list.

    Args:
        path: Filesystem path to the ``_all_videos.json`` catalog file.

    Returns:
        A list of video-entry dictionaries.  Returns an empty list when
        the file does not exist.

    Raises:
        ValueError: If the JSON content is neither a list nor a dict
            containing a ``"videos"`` key that maps to a list.
    """
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
    """Extract the video length in seconds from a catalog entry.

    Args:
        entry: A single video-entry dictionary that may contain a
            ``"video_length"`` key.

    Returns:
        The video length as a float if the value is a positive number,
        or ``None`` when the key is missing, non-numeric, or non-positive.
    """
    value = entry.get("video_length")
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return None


def categorize_length(seconds: float) -> str:
    """Map a video duration to a human-readable length category.

    Args:
        seconds: Duration of the video in seconds.

    Returns:
        One of ``"short"`` (< 10 min), ``"medium"`` (< 30 min),
        ``"long"`` (< 90 min), or ``"extra"`` (≥ 90 min).
    """
    minutes = seconds / 60
    if minutes < 10:
        return "short"
    if minutes < 30:
        return "medium"
    if minutes < 90:
        return "long"
    return "extra"


def make_output_dir(video_path: Path, processed_root: Path | str = "processed") -> str:
    """Build an output directory path for a given video file.

    Leading dots, trailing dots, and surrounding whitespace are stripped
    from the filename to produce a safe directory name.

    Args:
        video_path: Path to the video file whose name is used as the
            output directory name.
        processed_root: Root directory under which the video-specific
            output directory is created.

    Returns:
        A string representation of the output directory path
        (e.g. ``"data/processed/video_name.mp4"``).
    """
    name = video_path.name
    if name.startswith("."):
        name = name.lstrip(".")
    name = name.strip().rstrip(".")
    if not name:
        name = "video"
    return str(Path(processed_root) / name)


def resolve_video_arg(arg: str, blob_index: dict, videos_dir: Path) -> Path | None:
    """Resolve a user-supplied video argument to an existing file path.

    The function tries, in order:

    1. Treat *arg* as a direct path.
    2. Join *arg* with *videos_dir*.
    3. Look up *arg* in *blob_index* and join the blob name with
       *videos_dir*.

    Args:
        arg: A blob name, relative path, or absolute path provided by
            the user.
        blob_index: Mapping of blob names to their catalog-entry dicts.
        videos_dir: Base directory where video files are stored.

    Returns:
        A :class:`~pathlib.Path` to an existing video file, or ``None``
        if the argument could not be resolved.
    """
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


def select_videos(
    args: argparse.Namespace,
    catalog: list[dict],
    videos_dir: Path,
) -> list[Path]:
    """Select video files from the catalog based on CLI arguments.

    Videos can be selected explicitly via ``--video``, or in bulk via
    ``--all`` or ``--filter``.  When ``--filter`` is used, entries with
    unknown length are skipped unless ``--include-unknown`` is set.

    Args:
        args: Parsed CLI arguments (must expose ``.video``, ``.filter``,
            ``.include_unknown``, and ``.all`` attributes).
        catalog: List of video-entry dicts loaded from the catalog file.
        videos_dir: Base directory where video files are stored.

    Returns:
        A list of :class:`~pathlib.Path` objects pointing to the
        selected video files that exist on disk.

    Raises:
        SystemExit: If neither ``--video``, ``--all``, nor ``--filter``
            is specified.
    """
    blob_index = {
        v.get("blob"): v for v in catalog if isinstance(v, dict) and v.get("blob")
    }
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
            print(
                f"Skipping {unknown} video(s) with unknown length. "
                "Use --include-unknown to include."
            )
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
