"""Checkpoint reading, writing, and frame cleanup.

Provides utilities for persisting and restoring pipeline state as JSON
checkpoint files.  Heavy per-frame data (raw frames, YOLO tracks, etc.)
is stripped before serialisation so that checkpoints remain small and
portable.  The module also supports extracting individual scene clips
from a source video via FFmpeg.
"""

import json
import subprocess
from pathlib import Path
from typing import Any

import imageio_ffmpeg as ffmpeg

from kairos.core.utils import print_prefixed


def clear_frames(scene_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip heavy per-frame keys from every scene dictionary.

    This is used before writing a checkpoint so that large binary or
    array data (frames, YOLO tracks, motion bullets, etc.) is not
    serialised to JSON.

    Args:
        scene_list: A list of scene dictionaries, each potentially
            containing frame-level data under keys such as ``"frames"``,
            ``"yolo_frames"``, ``"frame_paths"``, and others.

    Returns:
        A new list of scene dictionaries with the heavy keys removed.
        The original dictionaries are **not** mutated.
    """
    omit_keys = {
        "frames",
        "yolo_frames",
        "frame_paths",
        "yolo_frame_paths",
        "frame_indices",
        "frame_timestamps",
        "sample_fps",
        "motion_bullets",
        "yolo_tracks",
        "yolo_track_summaries",
    }
    return [
        {k: v for k, v in scene.items() if k not in omit_keys} for scene in scene_list
    ]


def read_json(json_path: str | Path) -> dict[str, Any]:
    """Read a JSON checkpoint file and return its contents as a dictionary.

    If the file does not exist an empty dictionary is returned and a
    prefixed warning is printed.  If the top-level JSON value is a bare
    list it is wrapped as ``{"scenes": <list>}`` for consistency with
    the checkpoint schema used elsewhere in the pipeline.

    Args:
        json_path: Filesystem path (string or :class:`~pathlib.Path`)
            to the JSON checkpoint file.

    Returns:
        The parsed checkpoint dictionary.  Returns an empty ``dict`` if
        the file does not exist.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        print_prefixed("(Checkpoint)", f"JSON path does not exist: {json_path}")
        return {}

    print_prefixed("(Checkpoint)", f"Reading JSON from {json_path}")
    with open(json_path, encoding="utf-8") as f:
        checkpoint = json.load(f)
        if isinstance(checkpoint, list):
            return {"scenes": checkpoint}
        return checkpoint


def save_checkpoint(
    checkpoint: dict[str, Any] | list[dict[str, Any]],
    path: str | Path,
) -> dict[str, Any]:
    """Persist a checkpoint to disk after stripping heavy frame data.

    Parent directories are created automatically if they do not exist.
    The heavy per-frame keys are removed via :func:`clear_frames` before
    the data is written so that the resulting JSON stays small.

    Args:
        checkpoint: Either a list of scene dictionaries or a dictionary
            that contains a ``"scenes"`` key mapping to such a list.
        path: Destination file path for the JSON checkpoint.

    Returns:
        The checkpoint dictionary as it was actually written (i.e. with
        frame data stripped).  If *checkpoint* was a list it is returned
        wrapped as ``{"scenes": [...]}``.

    Raises:
        TypeError: If *checkpoint* is neither a ``dict`` nor a ``list``.
    """
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


def have_key(scenes: list[dict[str, Any]], key: str) -> bool:
    """Check whether **every** scene dictionary contains a given key.

    This is a convenience guard used by the pipeline to decide whether a
    processing step has already been applied.

    Args:
        scenes: List of scene dictionaries to inspect.
        key: The dictionary key to check for.

    Returns:
        ``True`` if *scenes* is non-empty **and** every scene contains
        *key*; ``False`` otherwise.
    """
    return bool(scenes) and all(key in s for s in scenes)


def save_clips(
    video_path: str,
    scenes: list[dict[str, Any]],
    output_dir: str,
) -> list[dict[str, Any]]:
    """Extract per-scene video clips from a source video using FFmpeg.

    Each clip is written to *output_dir* with a filename of the form
    ``scene_NNNN.mp4``.  If a clip file already exists it is **not**
    re-encoded — the existing file is reused instead.  The function
    returns a copy of each scene dictionary with the additional key
    ``"clip_path"`` pointing to the extracted clip.

    Args:
        video_path: Filesystem path to the source video file.
        scenes: List of scene dictionaries.  Each must contain at least
            ``"start_seconds"`` and ``"end_seconds"`` numeric values.
            An optional ``"scene_index"`` is used for the clip filename;
            when absent the positional index is used.
        output_dir: Directory where clip files will be written.  Created
            automatically if it does not exist.

    Returns:
        A new list of scene dictionaries, each augmented with a
        ``"clip_path"`` key whose value is the absolute string path to
        the extracted clip.
    """
    ffmpeg_path = ffmpeg.get_ffmpeg_exe()
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    updated_scenes: list[dict[str, Any]] = []

    for scene in scenes:
        start: float = scene["start_seconds"]
        end: float = scene["end_seconds"]
        duration: float = end - start

        scene_index: int = scene.get("scene_index", len(updated_scenes))
        clip_filename: str = f"scene_{scene_index:04d}.mp4"
        clip_path: Path = output_dir_path / clip_filename

        cmd: list[str] = [
            ffmpeg_path,
            "-y",
            "-i",
            video_path,
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-c",
            "copy",
            str(clip_path),
        ]

        if clip_path.exists():
            print_prefixed(
                "(save_clips)", f"Skipping existing clip: {clip_filename}", indent=4
            )
        else:
            subprocess.run(cmd, capture_output=True)

        scene_new: dict[str, Any] = dict(scene)
        scene_new["clip_path"] = str(clip_path)
        updated_scenes.append(scene_new)

    return updated_scenes
