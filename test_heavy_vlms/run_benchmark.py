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

VLMS = ["llava", "phi3v", "instructblip"]
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
    
    try:
        # Add 20-minute timeout for base processing (ASR/YOLO can be slow for long videos)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        
        if result.returncode != 0:
            print(f"  ✗ Base processing FAILED: {result.stderr}")
            if result.stdout:
                print(f"  [Stdout Last 500 chars]: {result.stdout[-500:]}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ Base processing TIMED OUT after 300s")
        return None
    except Exception as e:
        print(f"  ✗ Base processing Error: {e}")
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
    
    # Metrics tracking
    overall_start = time.time()
    metrics = {
        "videos": {},
        "vlms": {vlm: {"total_time": 0, "success": 0, "failed": 0} for vlm in VLMS},
        "overall_start": overall_start
    }
    
    # Process each video
    for video_idx, video_path in enumerate(videos, 1):
        video_name = video_path.stem
        print(f"\n{'='*80}")
        print(f"VIDEO {video_idx}/{len(videos)}: {video_path.name}")
        print(f"{'='*80}\n")
        
        video_start = time.time()
        metrics["videos"][video_name] = {"base_time": 0, "vlms": {}}
        
        # Step 1: Process base data (scenes, audio, YOLO)
        base_start = time.time()
        base_data_file = process_base_data(video_path)
        base_time = time.time() - base_start
        metrics["videos"][video_name]["base_time"] = base_time
        
        if not base_data_file:
            print(f"  ✗ Skipping video due to base processing failure\n")
            continue
        
        # Step 2: Run each VLM in isolation
        for vlm_name in VLMS:
            vlm_start = time.time()
            success = run_vlm_on_video(vlm_name, video_path, base_data_file)
            vlm_time = time.time() - vlm_start
            
            metrics["videos"][video_name]["vlms"][vlm_name] = {
                "time": vlm_time,
                "success": success
            }
            metrics["vlms"][vlm_name]["total_time"] += vlm_time
            if success:
                metrics["vlms"][vlm_name]["success"] += 1
            else:
                metrics["vlms"][vlm_name]["failed"] += 1
            
            if not success:
                print(f"    [Warning] {vlm_name} failed, continuing with next VLM...")
        
        video_total = time.time() - video_start
        metrics["videos"][video_name]["total_time"] = video_total
    
    overall_time = time.time() - overall_start
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"\nTotal Time: {overall_time:.1f}s ({overall_time/60:.1f} minutes)")
    
    print(f"\n{'─'*80}")
    print("PER-VIDEO BREAKDOWN")
    print(f"{'─'*80}")
    for video_name, video_metrics in metrics["videos"].items():
        print(f"\n📹 {video_name}")
        print(f"  Base Processing: {video_metrics['base_time']:.1f}s")
        for vlm_name, vlm_metrics in video_metrics.get("vlms", {}).items():
            status = "✓" if vlm_metrics["success"] else "✗"
            print(f"  {vlm_name:12s}: {vlm_metrics['time']:6.1f}s {status}")
        print(f"  {'─'*40}")
        print(f"  Total:        {video_metrics.get('total_time', 0):6.1f}s")
    
    print(f"\n{'─'*80}")
    print("PER-VLM SUMMARY")
    print(f"{'─'*80}")
    for vlm_name, vlm_metrics in metrics["vlms"].items():
        total_runs = vlm_metrics["success"] + vlm_metrics["failed"]
        avg_time = vlm_metrics["total_time"] / max(total_runs, 1)
        success_rate = (vlm_metrics["success"] / max(total_runs, 1)) * 100
        print(f"\n🤖 {vlm_name.upper()}")
        print(f"  Total Time:     {vlm_metrics['total_time']:.1f}s")
        print(f"  Avg Time/Video: {avg_time:.1f}s")
        print(f"  Success Rate:   {vlm_metrics['success']}/{total_runs} ({success_rate:.0f}%)")
    
    # Save metrics
    metrics["overall_time"] = overall_time
    metrics_file = RESULTS_DIR / "benchmark_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Metrics saved to: {metrics_file}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
