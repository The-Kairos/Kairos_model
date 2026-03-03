import json
import re
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import requests

base_dir = Path(__file__).resolve().parent
data_path = base_dir / "_all_videos.json"
out_dir = base_dir
log_path = base_dir / "_logs.json"
cheatsheet_path = base_dir.parent / ".cli_cheatsheet.md"

if data_path.exists():
    with open(data_path) as f:
        data = json.load(f)
else:
    legacy_path = base_dir / "videos.json"
    if legacy_path.exists():
        with open(legacy_path) as f:
            data = json.load(f)
    else:
        data = []

if isinstance(data, dict) and "videos" in data:
    data = data["videos"]
if not isinstance(data, list):
    raise ValueError("Expected _all_videos.json to be a list of video objects.")

invalid_chars = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
whitespace = re.compile(r"\s+")

if log_path.exists():
    try:
        with open(log_path) as f:
            logs = json.load(f)
    except json.JSONDecodeError:
        logs = {}
else:
    logs = {}


def sanitize_filename(name: str) -> str:
    # Decode URL-encoded characters like %20 -> space
    name = unquote(name)
    # Replace Windows-invalid/control characters with spaces
    name = invalid_chars.sub(" ", name)
    # Normalize whitespace
    name = whitespace.sub(" ", name).rstrip(" .")
    return name or "video"


def parse_link_expire(url: str) -> str | None:
    try:
        query = urlparse(url).query
        params = parse_qs(query)
        expires = params.get("se", [None])[0]
        return expires
    except Exception:
        return None


def probe_video_metadata(path: Path) -> tuple[float | None, list[int] | None]:
    if not path.exists():
        return None, None

    # Try ffprobe first (best accuracy)
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
                width = int(width)
                height = int(height)
                if width > 0 and height > 0:
                    resolution = [width, height]

        if duration is not None or resolution is not None:
            return duration, resolution
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
        pass

    # Fallback: OpenCV if available
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

        resolution = None
        if width > 0 and height > 0:
            resolution = [width, height]

        if duration is not None or resolution is not None:
            return duration, resolution
    except Exception:
        pass

    # Fallback: MoviePy if available
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip  # type: ignore

        with VideoFileClip(str(path)) as clip:
            duration = float(clip.duration) if clip.duration else None
            if clip.w and clip.h:
                resolution = [int(clip.w), int(clip.h)]
            else:
                resolution = None
        return duration, resolution
    except Exception:
        return None, None


def write_logs() -> None:
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)


def write_data() -> None:
    with open(data_path, "w") as f:
        json.dump(data, f, indent=2)


def normalize_downloads(record: object) -> tuple[dict, list[dict]]:
    # Supports old formats: list of entries or dict without downloads.
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
    ordered = {}
    ordered["video_length"] = record.get("video_length")
    ordered["downloads"] = record.get("downloads", [])
    for key, value in record.items():
        if key not in ("video_length", "downloads"):
            ordered[key] = value
    return ordered


def get_log_record(name: str) -> dict:
    record = logs.get(name)
    record, downloads = normalize_downloads(record)
    record["downloads"] = downloads
    record = order_record(record)
    logs[name] = record
    return record


def _bash_escape(value: str) -> str:
    # Prefer double-quoted strings so single quotes stay readable in bash.
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("$", "\\$").replace("`", "\\`")
    return f'"{escaped}"'


def write_run_cheatsheet(videos: list[dict], path: Path) -> None:
    blobs = [v.get("blob") for v in videos if isinstance(v, dict) and v.get("blob")]

    def _format_duration(seconds: float | int | None) -> str:
        if not isinstance(seconds, (int, float)) or seconds <= 0:
            return "unknown"
        total = int(round(seconds))
        hours = total // 3600
        minutes = (total % 3600) // 60
        secs = total % 60
        return f"{hours}h:{minutes}m:{secs}s"

    lines: list[str] = []
    lines.append("# Video Processing CLI Cheatsheet")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Process ALL Videos")
    lines.append("")
    lines.append("#### CLI commands to process all videos in [Videos](Videos) folder:")
    lines.append("")
    lines.append("```bash")
    lines.append("python main.py process --all")
    lines.append("```")
    lines.append("")
    lines.append("Running all the videos can take time so you can filter videos based on length category.")
    lines.append("")
    lines.append("```bash")
    lines.append("python main.py process --filter <duration>")
    lines.append("```")
    lines.append("")
    lines.append("| Filter | Duration     |")
    lines.append("| ------ | ------------ |")
    lines.append("| short  | < 10 minutes |")
    lines.append("| medium | < 30 minutes |")
    lines.append("| long   | < 90 minutes |")
    lines.append("| extra  | All lengths  |")
    lines.append("")
    lines.append("Add `--include-unknown` to include videos with unknown length in the filtered set.")
    lines.append("")
    lines.append("#### **CLI commands to process video by length:**")
    lines.append("")
    lines.append("```bash")
    lines.append("python main.py process --filter short     (<10 min videos)")
    lines.append("python main.py process --filter medium    (<30 min videos)")
    lines.append("python main.py process --filter long      (<90 min videos)")
    lines.append("python main.py process --filter extra     (all video lengths)")
    lines.append("python main.py process --filter long --include-unknown")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Process a Single Video")
    lines.append("")
    lines.append("```bash")
    lines.append('python main.py process --video "<file_name>"')
    lines.append("```")
    lines.append("")
    lines.append("#### CLI commands to process individual videos:")
    lines.append("")
    lines.append("```bash")
    for video in videos:
        if not isinstance(video, dict) or not video.get("blob"):
            continue
        safe_blob = _bash_escape(video["blob"])
        duration = _format_duration(video.get("video_length"))
        lines.append(f"python main.py process --video {safe_blob}  # {duration}")

    lines.append("```")
    lines.append("")
    lines.append("### Redo a Processing Step")
    lines.append("")
    lines.append("By default, re-running a step also re-runs all dependent steps (transitive).")
    lines.append("")
    lines.append("When using `--redo`, downstream steps are cleared and re-run automatically. Using `--redo-only` will only clear and rerun that step.")
    lines.append("")
    lines.append("```bash")
    lines.append('python main.py process --video "<file_name>" --redo <step>  (transitive)')
    lines.append('python main.py process --video "<file_name>" --redo-only <step>  (non-transitive)')
    lines.append("```")
    lines.append("")
    lines.append("Example:")
    lines.append("")
    lines.append("```bash")
    lines.append('python main.py process --video "Young Sheldon - First Day of High School.mp4" --redo frame_captions')
    lines.append("```")
    lines.append("")
    lines.append("#### Available Steps")
    lines.append("")
    lines.append("* `scenes`")
    lines.append("* `frame_captions`")
    lines.append("* `yolo`")
    lines.append("* `audio_natural`")
    lines.append("* `audio_speech`")
    lines.append("* `llm`")
    lines.append("* `narrative`")
    lines.append("* `synopsis`")
    lines.append("* `rag`")
    lines.append("")
    lines.append("#### Detailed Relationships")
    lines.append("")
    lines.append("```")
    lines.append("frame_captions -> llm -> narrative -> synopsis -> rag")
    lines.append("yolo           -> llm -> narrative -> synopsis -> rag")
    lines.append("audio_natural  -> llm -> narrative -> synopsis -> rag")
    lines.append("audio_speech   -> llm -> narrative -> synopsis -> rag")
    lines.append("llm            -> narrative -> synopsis -> rag")
    lines.append("narrative      -> synopsis -> rag")
    lines.append("synopsis       -> rag")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Run RAG (Requires Prior Processing)")
    lines.append("")
    lines.append("```bash")
    lines.append('python main.py rag --video "<file_name>"')
    lines.append("```")
    lines.append("")
    lines.append("#### CLI commands to run RAG chatbots on videos:")
    lines.append("")
    lines.append("```bash")
    for blob in blobs:
        safe_blob = _bash_escape(blob)
        lines.append(f"python main.py rag --video {safe_blob}")
    lines.append("```")
    lines.append("")
    lines.append("---")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_video_length_seconds(video: dict) -> float | None:
    value = video.get("video_length")
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


print("")
print("====== Choose which videos to download: ======")
print("1) Short: under 10 minutes")
print("2) Medium: up to 30 minutes")
print("3) Long: up to 90 minutes")
print("4) Extra: all video lengths")
print("5) Cheatsheet only (no downloads)")
print("==============================================")
# there should also be another option here to only create the cheaTsheet without downloading any video. 
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
category_rank = {
    "short": 1,
    "medium": 2,
    "long": 3,
    "extra": 4,
}
selected_rank = category_rank[selected_category]

selected_videos = []
unknown_videos = []
for video in data:
    length_seconds = get_video_length_seconds(video)
    if length_seconds is None:
        unknown_videos.append(video)
        continue
    if category_rank[categorize_length(length_seconds)] <= selected_rank:
        selected_videos.append(video)

include_unknown = False
if unknown_videos:
    print("")
    print("===== Unknown videos found. Include them? =====")
    for item in unknown_videos:
        name = item.get("blob") or "(no blob)"
        print(f"  - {name}")
    print("==============================================")
    answer = input("Option (y/n): ").strip().lower()
    include_unknown = answer.startswith("y")
    if include_unknown:
        selected_videos.extend(unknown_videos)

if not selected_videos:
    print("")
    print("No videos match the selected criteria. Nothing to download.")
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
    filepath = out_dir / filename

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
        record = get_log_record(filename)
        record["downloads"].append(entry)
        logs[filename] = order_record(record)
        write_logs()
        downloaded_count += 1

    link_expire = parse_link_expire(url)
    durations, resolution = probe_video_metadata(filepath)
    avg_time = None
    record = get_log_record(filename)
    downloads = record["downloads"]
    if downloads:
        avg_time = round(
            sum(item.get("seconds", 0) for item in downloads) / len(downloads),
            3,
        )

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
    write_logs()
    write_data()

print("")
print("=================== Summary ==================")
print(f"Downloaded : {downloaded_count}")
print(f"Skipped    : {skipped_count}")
print(f"Folder     : {out_dir.resolve()}")
print("==============================================")
write_run_cheatsheet(data, cheatsheet_path)
print(f"Cheatsheet : {cheatsheet_path}")
