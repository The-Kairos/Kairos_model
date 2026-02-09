#!/usr/bin/env python3
"""
Orchestrator: Run VLMs one at a time on all videos.
Each VLM runs in a separate Python process that exits after completion.
This forces OS-level GPU memory cleanup. yep
"""
import subprocess
import sys
import json
from pathlib import Path
import time

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = PROJECT_ROOT / "Videos"  # Kairos_model/Videos (capital V)
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BASE_RESULTS_DIR = RESULTS_DIR / "base"

VLMS = ["llava", "qwenvl", "internvl"]
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]

def get_videos():
    """Get all video files."""
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(VIDEOS_DIR.glob(f"*{ext}"))
    return sorted(videos)

def process_base_data(video_path):
    """Run base processing (scenes, ASR, AST, YOLO) for a video."""
    video_name = video_path.stem
    base_data_file = BASE_RESULTS_DIR / video_name / "base_data.json"
    
    if base_data_file.exists():
        print(f"  ✓ Base data exists for {video_name}")
        return base_data_file
    
    print(f"  [Base] Processing {video_name}...")
    
    # Run the base processing script
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "process_base.py"),
        str(video_path),
        str(base_data_file)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ✗ Base processing FAILED: {result.stderr}")
        return None
    
    print(f"  ✓ Base data saved to {base_data_file}")
    return base_data_file

def run_vlm_on_video(vlm_name, video_path, base_data_file):
    """Run a single VLM on a single video in an isolated process."""
    video_name = video_path.stem
    output_dir = RESULTS_DIR / vlm_name / video_name
    caption_file = output_dir / "vlm_captions.json"
    
    # Skip if already exists
    if caption_file.exists():
        print(f"    ✓ {vlm_name} captions already exist")
        return True
    
    print(f"    [{vlm_name}] Running on {video_name}...")
    
    # Run in isolated process
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_single_vlm.py"),
        vlm_name,
        str(video_path),
        str(base_data_file),
        str(output_dir)
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time
    
    if result.returncode != 0:
        print(f"    ✗ {vlm_name} FAILED after {duration:.1f}s")
        # Show last 500 chars of stderr for debugging
        if result.stderr:
            error_msg = result.stderr.strip()[-500:]
            print(f"       Error: {error_msg}")
        return False
    
    print(f"    ✓ {vlm_name} completed in {duration:.1f}s")
    
    # Wait 5 seconds for OS to fully release GPU memory
    print(f"    [Cooling] Waiting 5s for GPU memory release...")
    time.sleep(5)
    
    return True

def main():
    print("\n" + "="*80)
    print("VLM BENCHMARK - ISOLATED EXECUTION MODE")
    print("="*80 + "\n")
    
    videos = get_videos()
    
    if not videos:
        print("ERROR: No videos found in", VIDEOS_DIR)
        sys.exit(1)
    
    print(f"Found {len(videos)} videos")
    print(f"Will test {len(VLMS)} VLMs: {', '.join(VLMS)}\n")
    
    # Process each video
    for video_idx, video_path in enumerate(videos, 1):
        print(f"\n{'='*80}")
        print(f"VIDEO {video_idx}/{len(videos)}: {video_path.name}")
        print(f"{'='*80}\n")
        
        # Step 1: Process base data (scenes, audio, YOLO)
        base_data_file = process_base_data(video_path)
        if not base_data_file:
            print(f"  ✗ Skipping video due to base processing failure\n")
            continue
        
        # Step 2: Run each VLM in isolation
        for vlm_name in VLMS:
            success = run_vlm_on_video(vlm_name, video_path, base_data_file)
            if not success:
                print(f"    [Warning] {vlm_name} failed, continuing with next VLM...")
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
