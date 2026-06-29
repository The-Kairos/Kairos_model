"""
QVHighlights dataset loader for Kairos clip retrieval benchmarking.

Downloads QVHighlights annotations (JSONL) from the Moment-DETR GitHub repo,
downloads videos from the UNC tarball or via yt-dlp, and provides query grouping.

Dataset: https://github.com/jayleicn/moment_detr
Paper:   Lei et al., "QVHighlights: Detecting Moments and Highlights in Videos
         via Natural Language Queries", NeurIPS 2021.
"""
import json
import os
import subprocess
import tarfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

ANNOTATIONS_BASE_URL = (
    "https://raw.githubusercontent.com/jayleicn/moment_detr/main/data"
)
VIDEOS_TARBALL_URL = (
    "https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz"
)


def download_qvhighlights_annotations(cache_dir, split="val"):
    """Download QVHighlights annotation JSONL from GitHub.

    For test split, downloads the version with ground truth labels.
    Returns path to the downloaded JSONL file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if split == "test":
        filename = "highlight_test_with_gt.jsonl"
    else:
        filename = f"highlight_{split}_release.jsonl"
    local_path = cache_dir / filename

    if local_path.exists() and local_path.stat().st_size > 0:
        print(f"[QVH] Annotations cached: {local_path}")
        return local_path

    url = f"{ANNOTATIONS_BASE_URL}/{filename}"
    print(f"[QVH] Downloading annotations: {url}")
    try:
        urlretrieve(url, str(local_path))
        print(f"[QVH] Saved to {local_path}")
    except Exception as e:
        print(f"[QVH] Failed to download annotations: {e}")
        if local_path.exists():
            local_path.unlink()
        return None

    return local_path


def load_annotations(jsonl_path):
    """Parse QVHighlights JSONL file.

    Returns list of dicts: {qid, query, vid, relevant_windows, saliency_scores, duration}
    """
    annotations = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            annotations.append({
                "qid": entry.get("qid"),
                "query": entry.get("query", ""),
                "vid": entry.get("vid", ""),
                "relevant_windows": entry.get("relevant_windows", []),
                "relevant_clip_ids": entry.get("relevant_clip_ids", []),
                "saliency_scores": entry.get("saliency_scores", []),
                "duration": entry.get("duration", 0),
            })
    return annotations


def group_queries_by_video(annotations):
    """Group annotations by video ID.

    Returns dict: vid -> list of annotation dicts
    """
    groups = defaultdict(list)
    for ann in annotations:
        groups[ann["vid"]].append(ann)
    return dict(groups)


def _extract_youtube_id(vid):
    """Extract YouTube video ID from QVHighlights vid format.

    QVHighlights vid format: {youtube_id}_{start}_{end}
    Example: bP5MrgWJ1io_60.0_210.0 -> bP5MrgWJ1io
    """
    parts = vid.rsplit("_", 2)
    if len(parts) >= 3:
        try:
            float(parts[-1])
            float(parts[-2])
            return parts[0]
        except ValueError:
            pass
    return vid


def _extract_time_range(vid):
    """Extract start/end time from QVHighlights vid format.

    Returns (start_sec, end_sec) or None if not parseable.
    """
    parts = vid.rsplit("_", 2)
    if len(parts) >= 3:
        try:
            start = float(parts[-2])
            end = float(parts[-1])
            return (start, end)
        except ValueError:
            pass
    return None


def download_videos_tarball(cache_dir, video_dir):
    """Download the QVHighlights video tarball from UNC.

    This is the preferred method — contains all pre-cut video clips.
    Returns True on success.
    """
    cache_dir = Path(cache_dir)
    video_dir = Path(video_dir)
    tarball_path = cache_dir / "qvhilights_videos.tar.gz"

    if video_dir.exists() and any(video_dir.iterdir()):
        existing = list(video_dir.glob("*.mp4"))
        if existing:
            print(f"[QVH] {len(existing)} videos already cached in {video_dir}")
            return True

    video_dir.mkdir(parents=True, exist_ok=True)

    if not tarball_path.exists():
        print(f"[QVH] Downloading video tarball (~2GB)... this may take a while")
        print(f"[QVH] URL: {VIDEOS_TARBALL_URL}")
        try:
            urlretrieve(VIDEOS_TARBALL_URL, str(tarball_path))
            print(f"[QVH] Tarball downloaded: {tarball_path}")
        except Exception as e:
            print(f"[QVH] Tarball download failed: {e}")
            print(f"[QVH] Will fall back to yt-dlp per-video download")
            if tarball_path.exists():
                tarball_path.unlink()
            return False

    print(f"[QVH] Extracting tarball to {video_dir}...")
    try:
        with tarfile.open(str(tarball_path), "r:gz") as tar:
            try:
                tar.extractall(path=str(video_dir), filter="data")
            except TypeError:
                tar.extractall(path=str(video_dir))
        print(f"[QVH] Extraction complete")
        return True
    except Exception as e:
        print(f"[QVH] Extraction failed: {e}")
        return False


def _download_env():
    """Build subprocess env with ffmpeg on PATH."""
    env = os.environ.copy()
    extra = [
        str(Path.home() / ".local" / "bin"),
        str(Path.home() / ".deno" / "bin"),
        "/opt/conda/lib/python3.10/site-packages/static_ffmpeg/bin/linux",
    ]
    env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    return env


def download_video_ytdlp(vid, video_dir, timeout=300):
    """Download a single QVHighlights video via yt-dlp.

    Downloads the full YouTube video and trims to the QVH time range.
    Returns path to the video file, or None on failure.
    """
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    output_path = video_dir / f"{vid}.mp4"
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    youtube_id = _extract_youtube_id(vid)
    time_range = _extract_time_range(vid)
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    env = _download_env()

    cmd = [
        "yt-dlp",
        "--js-runtimes", "node",
        "--remote-components", "ejs:github",
        "-f", "bestvideo[vcodec~='^avc'][height<=720]+bestaudio/best[vcodec~='^avc'][height<=720]",
        "--merge-output-format", "mp4",
        "-o", str(output_path),
        "--no-playlist",
        "--socket-timeout", "30",
    ]

    if time_range:
        start, end = time_range
        cmd.extend(["--download-sections", f"*{start}-{end}"])

    cookies_file = (os.environ.get("YTDLP_COOKIES_FILE") or "").strip()
    cookies_browser = (os.environ.get("YTDLP_COOKIES_FROM_BROWSER") or "").strip()
    if cookies_file:
        cmd[1:1] = ["--cookies", cookies_file]
    elif cookies_browser:
        cmd[1:1] = ["--cookies-from-browser", cookies_browser]

    cmd.append(url)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env,
        )
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return output_path
        print(f"  [WARN] yt-dlp failed for {vid}: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print(f"  [WARN] yt-dlp timed out for {vid}")
    except FileNotFoundError:
        print("  [ERROR] yt-dlp not installed. Run: pip install yt-dlp")

    if output_path.exists():
        output_path.unlink()
    return None


def extract_videos_from_tarball(tarball_path, video_dir, cache_dir, split="val"):
    """Extract split-specific videos from the full QVHighlights tarball.

    Reads the annotation JSONL to determine which video IDs belong to the
    requested split, then streams through the tarball extracting only those
    files. This avoids writing the full ~134 GB of videos to disk.

    Returns the number of videos extracted.
    """
    tarball_path = Path(tarball_path)
    video_dir = Path(video_dir)
    cache_dir = Path(cache_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    ann_path = download_qvhighlights_annotations(cache_dir, split=split)
    if ann_path is None:
        print(f"[QVH] Cannot determine {split} video IDs without annotations")
        return 0

    split_vids = set()
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            split_vids.add(entry.get("vid", ""))
    split_vids.discard("")
    print(f"[QVH] {len(split_vids)} unique video IDs in {split} split")

    already = set()
    for mp4 in video_dir.glob("*.mp4"):
        already.add(mp4.stem)
    needed = split_vids - already
    if not needed:
        print(f"[QVH] All {len(split_vids)} {split} videos already extracted")
        return len(split_vids)
    print(f"[QVH] {len(already)} already cached, {len(needed)} to extract")

    count = 0
    print(f"[QVH] Streaming tarball {tarball_path} ...")
    with tarfile.open(str(tarball_path), "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".mp4"):
                continue
            basename = member.name.split("/")[-1]
            stem = basename.replace(".mp4", "")
            if stem not in needed:
                continue
            member.name = basename
            tar.extract(member, path=str(video_dir))
            count += 1
            if count % 50 == 0:
                print(f"  [QVH] Extracted {count}/{len(needed)} ...")
    print(f"[QVH] Extracted {count} {split} videos to {video_dir}")
    return count + len(already)


extract_val_videos_from_tarball = extract_videos_from_tarball


def find_video_file(vid, video_dir):
    """Locate a QVHighlights video file in the cache directory.

    Handles both tarball extraction (may nest in subdirectories) and direct downloads.
    """
    video_dir = Path(video_dir)

    direct = video_dir / f"{vid}.mp4"
    if direct.exists():
        return direct

    for mp4 in video_dir.rglob(f"{vid}.mp4"):
        return mp4

    return None


def prepare_qvhighlights(cache_dir, video_dir, split="val", max_videos=None):
    """Download annotations + videos, return grouped data ready for benchmarking.

    Returns:
        list of dicts: [{
            "vid": str,
            "video_path": str,
            "queries": [{"qid": int, "query": str, "relevant_windows": [[s,e]]}],
            "duration": float,
        }]
    """
    cache_dir = Path(cache_dir)
    video_dir = Path(video_dir)

    ann_path = download_qvhighlights_annotations(cache_dir, split=split)
    if ann_path is None:
        print("[QVH] Failed to download annotations")
        return []

    annotations = load_annotations(ann_path)
    print(f"[QVH] Loaded {len(annotations)} queries from {split} split")

    grouped = group_queries_by_video(annotations)
    video_ids = sorted(grouped.keys())
    print(f"[QVH] {len(video_ids)} unique videos")

    result = []
    skipped = 0
    for vid in video_ids:
        if max_videos is not None and len(result) >= max_videos:
            break

        video_path = find_video_file(vid, video_dir)
        if video_path is None:
            video_path = download_video_ytdlp(vid, video_dir)
        if video_path is None:
            skipped += 1
            continue

        queries = grouped[vid]
        duration = queries[0].get("duration", 0) if queries else 0

        result.append({
            "vid": vid,
            "video_path": str(video_path),
            "queries": queries,
            "num_queries": len(queries),
            "duration": duration,
        })

    if skipped > 0:
        print(f"[QVH] Skipped {skipped} videos (not available)")
    print(f"[QVH] {len(result)} videos ready for benchmarking")
    return result


def save_manifest(data, manifest_path):
    """Save prepared video data to a manifest JSON."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for entry in data:
        serializable.append({
            "vid": entry["vid"],
            "video_path": entry["video_path"],
            "num_queries": entry["num_queries"],
            "duration": entry["duration"],
        })
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def load_manifest(manifest_path):
    """Load manifest JSON."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)
