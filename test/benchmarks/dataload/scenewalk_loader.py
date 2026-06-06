"""
SceneWalk dataset loader for Kairos benchmarking.

Loads the SceneWalk dataset (IVLLab/SceneWalk) from HuggingFace,
groups segments by video, and downloads YouTube videos via yt-dlp.

Dataset: https://huggingface.co/datasets/IVLLab/SceneWalk
Paper:   https://arxiv.org/abs/2411.16173
"""
import json
import os
import subprocess
from collections import defaultdict
from importlib.util import find_spec
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def _get_scenewalk_token():
    """Resolve a Hugging Face token for SceneWalk access.

    Preference order:
    1. `HF_TOKEN` from the environment
    2. token saved by `huggingface-cli login`
    3. library default resolution
    """
    env_token = (os.environ.get("HF_TOKEN") or "").strip()
    if env_token:
        return env_token

    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    return True


def load_scenewalk_dataset(split="train", streaming=True):
    from datasets import load_dataset
    return load_dataset(
        "IVLLab/SceneWalk",
        split=split,
        streaming=streaming,
        token=_get_scenewalk_token(),
    )


def _parse_time_str(time_str):
    """Parse timestamp string like '00:01:30' or '01:30' to seconds."""
    if not time_str:
        return 0.0
    parts = time_str.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def _extract_video_id(url):
    """Extract YouTube video ID from URL."""
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be",):
        return parsed.path.lstrip("/")
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        qs = parse_qs(parsed.query)
        return qs.get("v", [None])[0]
    return None


def _extract_caption(conversations):
    """Extract the ground truth caption from the conversations field."""
    if not conversations:
        return ""
    for entry in conversations:
        if entry.get("from") == "gpt":
            return entry.get("value", "").strip()
    return ""


def group_segments_by_video(dataset, max_videos=10, min_segments=5,
                            min_duration_sec=120, max_scan=50000):
    """Stream through SceneWalk and group segments by video ID."""
    video_segments = defaultdict(list)
    scanned = 0

    for row in dataset:
        scanned += 1
        if scanned % 5000 == 0:
            print(f"  [SceneWalk] Scanned {scanned} segments, found {len(video_segments)} videos...")

        video_id = row.get("id", "")
        if not video_id:
            continue

        ts = row.get("time_stamp", {})
        start_sec = _parse_time_str(ts.get("start_time", ""))
        end_sec = _parse_time_str(ts.get("end_time", ""))
        duration = end_sec - start_sec if end_sec > start_sec else ts.get("duration", 0)

        caption = _extract_caption(row.get("conversations", []))
        if not caption:
            continue

        video_segments[video_id].append({
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": duration,
            "caption": caption,
        })

        if len(video_segments) >= max_videos * 5 and scanned > max_scan:
            break

    candidates = []
    for vid_id, segs in video_segments.items():
        segs.sort(key=lambda s: s["start_sec"])
        total_dur = max(s["end_sec"] for s in segs) if segs else 0
        if len(segs) >= min_segments and total_dur >= min_duration_sec:
            candidates.append({
                "video_id": vid_id,
                "url": f"https://www.youtube.com/watch?v={vid_id}",
                "segments": segs,
                "total_duration_sec": total_dur,
                "num_segments": len(segs),
            })

    candidates.sort(key=lambda v: v["total_duration_sec"], reverse=True)
    print(f"  [SceneWalk] Found {len(candidates)} candidate videos (>= {min_segments} segments, >= {min_duration_sec}s)")
    return candidates[:max_videos]


def _download_env():
    """Build a subprocess env with ffmpeg and common user bins on PATH."""
    env = os.environ.copy()
    extra = [
        str(Path.home() / ".deno" / "bin"),
        str(Path.home() / ".local" / "bin"),
    ]
    static_ffmpeg_spec = find_spec("static_ffmpeg")
    if static_ffmpeg_spec and static_ffmpeg_spec.origin:
        static_ffmpeg_bin = Path(static_ffmpeg_spec.origin).parent / "bin" / "linux"
        if static_ffmpeg_bin.exists():
            extra.append(str(static_ffmpeg_bin))
    env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    return env


def _check_av1(video_path):
    """Return True if video_path uses AV1 codec that OpenCV can't decode."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return True
    ret, frame = cap.read()
    cap.release()
    return not ret or frame is None


def _transcode_to_h264(input_path, env, timeout=600):
    """Transcode an AV1 video to H.264 in-place using ffmpeg."""
    input_path = Path(input_path)
    tmp_path = input_path.with_suffix(".h264.mp4")
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(input_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            str(tmp_path),
        ],
        capture_output=True, text=True, timeout=timeout, env=env,
    )
    if result.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 0:
        tmp_path.replace(input_path)
        return True
    if tmp_path.exists():
        tmp_path.unlink()
    print(f"  [WARN] ffmpeg transcode failed: {result.stderr[:200]}")
    return False


def _yt_dlp_command(format_selector, output_path, url):
    command = [
        "yt-dlp",
        "--js-runtimes", "node",
        "--remote-components", "ejs:github",
        "-f", format_selector,
        "--merge-output-format", "mp4",
        "-o", str(output_path),
        "--no-playlist",
        "--socket-timeout", "30",
        url,
    ]
    cookies_file = (os.environ.get("YTDLP_COOKIES_FILE") or "").strip()
    cookies_browser = (os.environ.get("YTDLP_COOKIES_FROM_BROWSER") or "").strip()
    if cookies_file:
        command[1:1] = ["--cookies", cookies_file]
    elif cookies_browser:
        command[1:1] = ["--cookies-from-browser", cookies_browser]
    return command


def download_youtube_video(url, output_path, timeout=300):
    """Download a YouTube video via yt-dlp. Returns True on success."""
    output_path = Path(output_path)
    env = _download_env()

    if output_path.exists() and output_path.stat().st_size > 0:
        if _check_av1(str(output_path)):
            print("  [AV1] Existing file uses AV1 codec, transcoding to H.264...")
            if _transcode_to_h264(output_path, env):
                print("  [AV1] Transcode successful")
                return True
            print("  [AV1] Transcode failed, re-downloading...")
            output_path.unlink()
        else:
            return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fmt_h264 = "bestvideo[vcodec~='^avc'][height<=720]+bestaudio/best[vcodec~='^avc'][height<=720]"
    try:
        result = subprocess.run(
            _yt_dlp_command(fmt_h264, output_path, url),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    if output_path.exists():
        output_path.unlink()

    try:
        result = subprocess.run(
            _yt_dlp_command(
                "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
                output_path,
                url,
            ),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        if result.returncode == 0 and output_path.exists():
            if _check_av1(str(output_path)):
                print("  [AV1] Downloaded AV1, transcoding to H.264...")
                if _transcode_to_h264(output_path, env):
                    print("  [AV1] Transcode successful")
                    return True
                print("  [WARN] Transcode failed")
                output_path.unlink()
                return False
            return True
        print(f"  [WARN] yt-dlp failed: {result.stderr[:200]}")
        if output_path.exists():
            output_path.unlink()
        return False
    except subprocess.TimeoutExpired:
        print(f"  [WARN] yt-dlp timed out for {url}")
        if output_path.exists():
            output_path.unlink()
        return False
    except FileNotFoundError:
        print("  [ERROR] yt-dlp not installed. Run: pip install yt-dlp")
        return False


def save_manifest(videos, manifest_path):
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(videos, f, indent=2, ensure_ascii=False)


def load_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)
