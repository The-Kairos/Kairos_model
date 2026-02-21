"""
HOW TO USE (CLIP DOWNLOAD)

Requirements:
- ffmpeg installed and in PATH
- run from repo root

This downloads a specific clip from an Azure Blob video
using the SAS link in Videos/_all_videos.json.

Example:
python src/trim_clip.py --video "Argentina v France Full Penalty Shoot-out.mp4" --start 424.32 --end 428.96 --out clips/messi_celebration.mp4

Start/end times come from the RAG preview link:
...#t=424.32,428.96

PROBLEM IM TRYNNA FIX: when i run this it trims the clip and downloads but theres no audio for come reason icba anymore...

"""

import json
import subprocess
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CATALOG_PATH = BASE_DIR / "Videos" / "_all_videos.json"


def get_sas_url(video_name: str) -> str:
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    for item in data:
        if item["blob"] == video_name:
            return item["sas"]
    raise SystemExit(f"Video not found in catalog: {video_name}")


def main():
    p = argparse.ArgumentParser(description="Trim a clip from an Azure SAS video using ffmpeg.")
    p.add_argument("--video", required=True, help="Video blob name as in Videos/_all_videos.json")
    p.add_argument("--start", type=float, required=True, help="Start time in seconds")
    p.add_argument("--end", type=float, required=True, help="End time in seconds")
    p.add_argument("--out", required=True, help="Output mp4 path")
    p.add_argument("--reencode", action="store_true", help="Re-encode for more accurate cuts (slower)")
    args = p.parse_args()

    if args.end <= args.start:
        raise SystemExit("Error: --end must be > --start")

    sas_url = get_sas_url(args.video)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use duration instead of absolute end time so ffmpeg cuts correctly when -ss is before -i
    duration = args.end - args.start

    if args.reencode:
        # Accurate cuts (slower): re-encode, timestamps naturally start at 0 in most players
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(args.start),         # FAST seek (important for remote URLs)
            "-t", str(duration),
            "-i", sas_url,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(out_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(args.start),
            "-t", str(duration),
            "-i", sas_url,
            "-c:v", "copy",
            "-c:a", "aac",          # re-encode audio (source is AAC anyway, fast)
            "-b:a", "96k",          # match source bitrate
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            str(out_path),
        ]
    subprocess.run(cmd, check=True)
    print(f"Saved clip -> {out_path}")


if __name__ == "__main__":
    main()