"""Download test videos from a JSON catalog."""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import requests

from kairos.cli.catalog import categorize_length, get_video_length_seconds

_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE = re.compile(r"\s+")

_CATEGORY_RANK = {"short": 1, "medium": 2, "long": 3, "extra": 4}


def sanitize_filename(name: str) -> str:
    """Clean a raw blob name into a safe filesystem filename."""
    name = unquote(name)
    name = _INVALID_CHARS.sub(" ", name)
    name = _WHITESPACE.sub(" ", name).rstrip(" .")
    return name or "video"


def parse_link_expire(url: str) -> str | None:
    """Extract the ``se`` (expiry) query parameter from a SAS URL."""
    try:
        query = urlparse(url).query
        params = parse_qs(query)
        return params.get("se", [None])[0]
    except Exception:
        return None


def probe_video_metadata(
    path: Path,
) -> tuple[float | None, list[int] | None]:
    """Return ``(duration_seconds, [width, height])`` for a video file."""
    if not path.exists():
        return None, None

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        info = json.loads(result.stdout)
        duration = None
        fmt = info.get("format") or {}
        if "duration" in fmt:
            try:
                duration = float(fmt["duration"])
            except (TypeError, ValueError):
                duration = None

        resolution = None
        streams = info.get("streams") or []
        if streams:
            width = streams[0].get("width")
            height = streams[0].get("height")
            if isinstance(width, (int, float)) and isinstance(height, (int, float)):
                width, height = int(width), int(height)
                if width > 0 and height > 0:
                    resolution = [width, height]

        if duration is not None or resolution is not None:
            return duration, resolution
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ):
        pass

    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None, None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        duration = None
        if fps and frame_count and fps > 0:
            duration = frame_count / fps
        resolution = [width, height] if width > 0 and height > 0 else None
        if duration is not None or resolution is not None:
            return duration, resolution
    except Exception:
        pass

    try:
        from moviepy.video.io.VideoFileClip import (  # type: ignore
            VideoFileClip,
        )

        with VideoFileClip(str(path)) as clip:
            duration = float(clip.duration) if clip.duration else None
            resolution = [int(clip.w), int(clip.h)] if clip.w and clip.h else None
        return duration, resolution
    except Exception:
        return None, None


def normalize_downloads(
    record: object,
) -> tuple[dict, list[dict]]:
    """Normalize legacy log formats into ``(metadata, downloads)``."""
    if isinstance(record, list):
        downloads = []
        for item in record:
            if isinstance(item, dict):
                if "timestamp" not in item and "downloaded_at" in item:
                    item = dict(item)
                    item["timestamp"] = item.pop("downloaded_at")
                downloads.append(item)
        return {}, downloads
    if isinstance(record, dict):
        downloads = record.get("downloads")
        if not isinstance(downloads, list):
            downloads = []
        return record, downloads
    return {}, []


def order_record(record: dict) -> dict:
    """Return *record* with ``video_length`` and ``downloads`` first."""
    ordered: dict = {}
    ordered["video_length"] = record.get("video_length")
    ordered["downloads"] = record.get("downloads", [])
    for key, value in record.items():
        if key not in ("video_length", "downloads"):
            ordered[key] = value
    return ordered


def _bash_escape(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("$", "\\$").replace("`", "\\`")
    return f'"{escaped}"'


def _format_duration(seconds: float | int | None) -> str:
    if not isinstance(seconds, (int, float)) or seconds <= 0:
        return "unknown"
    total = round(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours}h:{minutes}m:{secs}s"


def write_run_cheatsheet(videos: list[dict], path: Path) -> None:
    """Generate a CLI cheatsheet markdown file from a video catalog."""
    blobs = [v.get("blob") for v in videos if isinstance(v, dict) and v.get("blob")]

    lines: list[str] = [
        "# Video Processing CLI Cheatsheet",
        "",
        "---",
        "",
        "## Process ALL Videos",
        "",
        "#### CLI commands to process all videos in [Videos](Videos) folder:",
        "",
        "```bash",
        "python main.py process --all",
        "```",
        "",
        (
            "Running all the videos can take time so you can filter"
            " videos based on length category."
        ),
        "",
        "```bash",
        "python main.py process --filter <duration>",
        "```",
        "",
        "| Filter | Duration     |",
        "| ------ | ------------ |",
        "| short  | < 10 minutes |",
        "| medium | < 30 minutes |",
        "| long   | < 90 minutes |",
        "| extra  | All lengths  |",
        "",
        (
            "Add `--include-unknown` to include videos with unknown"
            " length in the filtered set."
        ),
        "",
        "#### **CLI commands to process video by length:**",
        "",
        "```bash",
        "python main.py process --filter short     (<10 min videos)",
        "python main.py process --filter medium    (<30 min videos)",
        "python main.py process --filter long      (<90 min videos)",
        "python main.py process --filter extra     (all video lengths)",
        "python main.py process --filter long --include-unknown",
        "```",
        "",
        "---",
        "",
        "## Process a Single Video",
        "",
        "```bash",
        'python main.py process --video "<file_name>"',
        "```",
        "",
        "#### CLI commands to process individual videos:",
        "",
        "```bash",
    ]

    for video in videos:
        if not isinstance(video, dict) or not video.get("blob"):
            continue
        safe_blob = _bash_escape(video["blob"])
        duration = _format_duration(video.get("video_length"))
        lines.append(f"python main.py process --video {safe_blob}  # {duration}")

    lines += [
        "```",
        "",
        "### Redo a Processing Step",
        "",
        (
            "By default, re-running a step also re-runs all dependent"
            " steps (transitive)."
        ),
        "",
        (
            "When using `--redo`, downstream steps are cleared and"
            " re-run automatically. Using `--redo-only` will only"
            " clear and rerun that step."
        ),
        "",
        "```bash",
        ('python main.py process --video "<file_name>" --redo <step>  (transitive)'),
        (
            'python main.py process --video "<file_name>"'
            " --redo-only <step>  (non-transitive)"
        ),
        "```",
        "",
        "Example:",
        "",
        "```bash",
        (
            "python main.py process"
            ' --video "Young Sheldon - First Day of High School.mp4"'
            " --redo frame_captions"
        ),
        "```",
        "",
        "#### Available Steps",
        "",
        "* `scenes`",
        "* `frame_captions`",
        "* `yolo`",
        "* `audio_natural`",
        "* `audio_speech`",
        "* `llm`",
        "* `narrative`",
        "* `synopsis`",
        "* `rag`",
        "",
        "#### Detailed Relationships",
        "",
        "```",
        "frame_captions -> llm -> narrative -> synopsis -> rag",
        "yolo           -> llm -> narrative -> synopsis -> rag",
        "audio_natural  -> llm -> narrative -> synopsis -> rag",
        "audio_speech   -> llm -> narrative -> synopsis -> rag",
        "llm            -> narrative -> synopsis -> rag",
        "narrative      -> synopsis -> rag",
        "synopsis       -> rag",
        "```",
        "",
        "---",
        "",
        "## Run RAG (Requires Prior Processing)",
        "",
        "```bash",
        'python main.py rag --video "<file_name>"',
        "```",
        "",
        "#### CLI commands to run RAG chatbots on videos:",
        "",
        "```bash",
    ]

    for blob in blobs:
        safe_blob = _bash_escape(blob)
        lines.append(f"python main.py rag --video {safe_blob}")

    lines += ["```", "", "---"]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_catalog(data_path: Path) -> list[dict]:
    """Load the video catalog JSON, supporting legacy formats."""
    if data_path.exists():
        with open(data_path) as f:
            data = json.load(f)
    else:
        legacy_path = data_path.parent / "videos.json"
        if legacy_path.exists():
            with open(legacy_path) as f:
                data = json.load(f)
        else:
            data = []
    if isinstance(data, dict) and "videos" in data:
        data = data["videos"]
    if not isinstance(data, list):
        raise ValueError("Expected catalog to be a list of video objects.")
    return data


def _load_logs(log_path: Path) -> dict:
    if log_path.exists():
        try:
            with open(log_path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def _get_log_record(name: str, logs: dict) -> dict:
    record = logs.get(name)
    record, downloads = normalize_downloads(record)
    record["downloads"] = downloads
    record = order_record(record)
    logs[name] = record
    return record


def main() -> None:
    """Interactive CLI for downloading test videos."""
    base_dir = Path("data/videos")
    data_path = base_dir / "_all_videos.json"
    log_path = base_dir / "_logs.json"
    cheatsheet_path = base_dir.parent / ".cli_cheatsheet.md"

    data = _load_catalog(data_path)
    logs = _load_logs(log_path)

    def _write_logs() -> None:
        with open(log_path, "w") as f:
            json.dump(logs, f, indent=2)

    def _write_data() -> None:
        with open(data_path, "w") as f:
            json.dump(data, f, indent=2)

    print("")
    print("====== Choose which videos to download: ======")
    print("1) Short: under 10 minutes")
    print("2) Medium: up to 30 minutes")
    print("3) Long: up to 90 minutes")
    print("4) Extra: all video lengths")
    print("5) Cheatsheet only (no downloads)")
    print("==============================================")

    choice = ""
    while choice not in {"1", "2", "3", "4", "5"}:
        choice = input("Option (1-5): ").strip()

    if choice == "5":
        write_run_cheatsheet(data, cheatsheet_path)
        print(f"Cheatsheet : {cheatsheet_path}")
        raise SystemExit(0)

    category_map = {
        "1": "short",
        "2": "medium",
        "3": "long",
        "4": "extra",
    }
    selected_category = category_map[choice]
    selected_rank = _CATEGORY_RANK[selected_category]

    selected_videos: list[dict] = []
    unknown_videos: list[dict] = []
    for video in data:
        length_seconds = get_video_length_seconds(video)
        if length_seconds is None:
            unknown_videos.append(video)
            continue
        cat = categorize_length(length_seconds)
        if _CATEGORY_RANK[cat] <= selected_rank:
            selected_videos.append(video)

    if unknown_videos:
        print("")
        print("===== Unknown videos found. Include them? =====")
        for item in unknown_videos:
            name = item.get("blob") or "(no blob)"
            print(f"  - {name}")
        print("==============================================")
        answer = input("Option (y/n): ").strip().lower()
        if answer.startswith("y"):
            selected_videos.extend(unknown_videos)

    if not selected_videos:
        print("")
        print("No videos match the selected criteria.")
        raise SystemExit(0)

    downloaded_count = 0
    skipped_count = 0
    for video in selected_videos:
        url = video.get("url") or video.get("sas")
        if not url:
            print(">> Skipping: missing url/sas.")
            continue

        raw_name = video.get("blob")
        if not raw_name:
            print(">> Skipping: missing blob.")
            continue
        filename = sanitize_filename(raw_name)
        filepath = base_dir / filename

        if filepath.exists() and filepath.stat().st_size > 0:
            skipped_count += 1
        else:
            print(f">> Downloading: {filename}")
            start = time.perf_counter()
            r = requests.get(url, stream=True)
            r.raise_for_status()

            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            elapsed = time.perf_counter() - start

            entry = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "seconds": round(elapsed, 3),
            }
            record = _get_log_record(filename, logs)
            record["downloads"].append(entry)
            logs[filename] = order_record(record)
            _write_logs()
            downloaded_count += 1

        link_expire = parse_link_expire(url)
        durations, resolution = probe_video_metadata(filepath)
        record = _get_log_record(filename, logs)
        downloads = record["downloads"]
        avg_time = None
        if downloads:
            total = sum(item.get("seconds", 0) for item in downloads)
            avg_time = round(total / len(downloads), 3)

        if link_expire is not None:
            video["link_expire"] = link_expire
        elif "link_expire" not in video:
            video["link_expire"] = None

        if durations is not None:
            video["video_length"] = round(durations, 3)
            record["video_length"] = video["video_length"]
        elif "video_length" not in video:
            video["video_length"] = None
            record.setdefault("video_length", None)

        if resolution is not None:
            video["resolution"] = resolution
        elif "resolution" not in video:
            video["resolution"] = None

        if avg_time is not None:
            video["average_download_time"] = avg_time
        elif "average_download_time" not in video:
            video["average_download_time"] = None

        logs[filename] = order_record(record)
        _write_logs()
        _write_data()

    print("")
    print("=================== Summary ==================")
    print(f"Downloaded : {downloaded_count}")
    print(f"Skipped    : {skipped_count}")
    print(f"Folder     : {base_dir.resolve()}")
    print("==============================================")
    write_run_cheatsheet(data, cheatsheet_path)
    print(f"Cheatsheet : {cheatsheet_path}")


if __name__ == "__main__":
    main()
