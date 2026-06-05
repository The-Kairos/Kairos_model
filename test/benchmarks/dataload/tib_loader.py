"""
TIB dataset loader for Kairos benchmarking.

Loads the TIB dataset (gigant/tib) from HuggingFace and provides utilities
for downloading videos and extracting ground truth abstracts.

Dataset: https://huggingface.co/datasets/gigant/tib
Paper:   https://dl.acm.org/doi/10.1145/3617233.3617238
"""
import json
import urllib.request
import urllib.error
from pathlib import Path
from datasets import load_dataset


def load_tib_dataset(split="train", streaming=True):
    return load_dataset("gigant/tib", split=split, streaming=streaming)


def _estimate_duration_sec(row):
    """Estimate video duration from the last transcript segment's end timestamp."""
    segs = row.get("transcript_segments")
    if isinstance(segs, dict):
        ends = segs.get("end", [])
        if ends:
            return float(ends[-1])
    return 0.0


def filter_usable_entries(dataset, max_entries=None, require_abstract=True,
                          language=None, min_duration_sec=0):
    """
    Filter TIB entries to those that have a video URL and a non-empty abstract.
    Yields dicts with the fields we need.
    """
    count = 0
    for row in dataset:
        if max_entries and count >= max_entries:
            break

        video_url = row.get("video_url") or ""
        abstract = (row.get("abstract") or "").strip()

        if not video_url:
            continue
        if require_abstract and not abstract:
            continue
        if language and row.get("language") != language:
            continue

        duration = _estimate_duration_sec(row)
        if min_duration_sec and duration < min_duration_sec:
            continue

        yield {
            "doi": row.get("doi", ""),
            "title": row.get("title", ""),
            "video_url": video_url,
            "abstract": abstract,
            "transcript": row.get("transcript", ""),
            "transcript_segments": row.get("transcript_segments", []),
            "language": row.get("language", ""),
            "genre": row.get("genre", ""),
            "release_year": row.get("release_year", ""),
            "estimated_duration_sec": duration,
        }
        count += 1


def download_video(video_url, output_path, timeout=120):
    """
    Download a video from a direct URL to a local path.
    Returns True on success, False on failure.
    """
    output_path = Path(output_path)
    if output_path.exists() and output_path.stat().st_size > 0:
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(video_url, str(output_path))
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"  [WARN] Failed to download {video_url}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def make_video_filename(entry):
    """Create a safe filename from a TIB entry."""
    title = entry.get("title", "untitled") or "untitled"
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
    safe = safe.strip()[:80]
    doi = entry.get("doi", "")
    suffix = doi.replace("/", "_").replace(".", "_")[-20:] if doi else ""
    if suffix:
        safe = f"{safe}__{suffix}"
    return f"{safe}.mp4"


def save_manifest(entries, manifest_path):
    """Save a list of TIB entries as a JSON manifest for reproducibility."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def load_manifest(manifest_path):
    """Load a previously saved manifest."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)
