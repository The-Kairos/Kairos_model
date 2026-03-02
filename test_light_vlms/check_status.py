"""
Check which (light VLM, video) pairs have completed successfully and which are missing.

Run from project root:

    python test_light_vlms/check_status.py

It will:
- List all videos under Videos/ (excluding files starting with "_").
- For each VLM (blip2, instructblip, llava_mistral, phi3_vision, siglip),
  check for test_light_vlms/results/<vlm>/<video_stem>/pipeline_results.json.
- Print a summary of OK vs MISSING, and suggested rerun commands.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = PROJECT_ROOT / "Videos"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

VLMS = ["blip2", "instructblip", "llava_mistral", "phi3_vision", "siglip"]


def main():
    videos = [v for v in VIDEOS_DIR.glob("*.mp4") if not v.name.startswith("_")]
    if not videos:
        print("No videos in Videos/. Add .mp4 files to run benchmarks.")
        return

    print(f"Found {len(videos)} video(s) in {VIDEOS_DIR}")

    status = {}  # (vlm -> {video_name -> 'ok'|'missing'})
    missing = {}  # (vlm -> [video_name,...])

    for vlm in VLMS:
        status[vlm] = {}
        missing[vlm] = []
        for v in videos:
            video_name = v.name
            video_stem = v.stem
            result_dir = RESULTS_DIR / vlm / video_stem
            result_json = result_dir / "pipeline_results.json"

            if result_json.exists():
                # Treat existence as success; we could also sanity-check JSON.
                status[vlm][video_name] = "ok"
            else:
                status[vlm][video_name] = "missing"
                missing[vlm].append(video_name)

    # Print detailed grid
    print("\nStatus by VLM and video:")
    for vlm in VLMS:
        print(f"\n=== {vlm} ===")
        for v in videos:
            flag = status[vlm][v.name]
            print(f"  {flag.upper():7}  {v.name}")

    # Summary
    print("\nSummary:")
    for vlm in VLMS:
        total = len(videos)
        miss = len(missing[vlm])
        ok = total - miss
        print(f"  {vlm}: {ok}/{total} OK, {miss} missing")

    # Suggested rerun commands (grouped by VLM)
    print("\nSuggested rerun commands for missing pairs:")
    any_missing = False
    for vlm in VLMS:
        if not missing[vlm]:
            continue
        any_missing = True
        # Build a comma-separated list suitable for --videos "a,b,c"
        videos_arg = ",".join(missing[vlm])
        print(f"  # {vlm}")
        print(
            f'  python test_light_vlms/main_test.py --vlms {vlm} --videos "{videos_arg}"'
        )

    if not any_missing:
        print("  None – all VLMs have results for all videos.")


if __name__ == "__main__":
    main()

