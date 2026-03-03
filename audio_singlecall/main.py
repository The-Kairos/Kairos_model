"""
main.py — Audio Single-Call Pipeline Orchestrator

Runs ONLY the audio processing stages (not BLIP, YOLO, LLM):
  1. Scene Detection (reuses src/scene_cutting.py)
  2. Audio Pre-Scan (audio_detector.py)
  3. Whisper Single-Call (whisper_singlecall.py)
  4. AST Parallelized (ast_processor.py)

Outputs each scene with audio_speech and audio_natural,
plus timing comparison table.

Usage:
  python -m audio_singlecall.main --video "Videos\\Young Sheldon - First Day of High School.mp4"
  python -m audio_singlecall.main --all
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scene_cutting import get_scene_list
from audio_singlecall.audio_detector import scan_audio
from audio_singlecall.whisper_singlecall import extract_speech_singlecall
from audio_singlecall.ast_processor import extract_sounds_optimized


# =========================================================
# Config
# =========================================================
PYSCENE_THRESHOLD = 27
PYSCENE_SHORTEST = 2
ASR_MODEL_SIZE = "small"
ASR_USE_VAD = True
AST_TARGET_SR = 16000
AST_MAX_WORKERS = 4

VIDEOS_DIR = PROJECT_ROOT / "Videos"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# =========================================================
# Pipeline
# =========================================================

def run_pipeline(video_path: str, parallel: bool = False, max_workers: int = 4, debug: bool = True):
    """
    Run the full audio-only pipeline on a single video.
    Returns (scenes, timing_report).
    """
    video_name = Path(video_path).stem
    print(f"\n{'='*70}")
    print(f"  AUDIO PIPELINE (High Parallelism Enabled: {parallel})")
    print(f"  VIDEO: {video_name}")
    print(f"{'='*70}\n")

    total_start = time.time()

    # ----- Step 1: Scene Detection -----
    print("[1/4] Scene Detection...")
    t = time.time()
    scenes = get_scene_list(
        input_video_path=video_path,
        threshold=PYSCENE_THRESHOLD,
        min_scene_sec=PYSCENE_SHORTEST,
    )
    scene_time = time.time() - t
    print(f"       Found {len(scenes)} scenes in {scene_time:.2f}s\n")

    # ----- Step 2: Audio Pre-Scan -----
    print("[2/4] Audio Pre-Scan (RMS + Silero VAD)...")
    scan_result = scan_audio(
        video_path=video_path,
        scenes=scenes,
        target_sr=AST_TARGET_SR,
        debug=debug,
    )
    print(f"       Pre-scan completed in {scan_result['scan_time_sec']:.2f}s\n")

    # ----- Step 3: Whisper Transcription -----
    print("[3/4] Whisper Transcription...")
    scenes, whisper_timing = extract_speech_singlecall(
        scenes=scenes,
        scan_result=scan_result,
        model_size=ASR_MODEL_SIZE,
        use_vad=ASR_USE_VAD,
        parallel=parallel,
        debug=debug,
    )
    print(f"       Whisper completed in {whisper_timing['total_time_sec']:.2f}s\n")

    # ----- Step 4: AST Parallelized -----
    print("[4/4] AST Parallelized Per-Scene...")
    
    # Only mask if not too large to avoid memory issues
    ast_audio = scan_result["audio"]
    sr = scan_result["sr"]
    
    # If video is extremely long (>30 min), we don't copy the whole buffer for masking
    # unless we are in parallel mode (where we need the masked buffer).
    if len(ast_audio) / sr < 1800: # 30 min
        ast_audio = ast_audio.copy()
        for seg in scan_result["speech_regions"]:
            i0 = int(seg["start_sec"] * sr)
            i1 = int(seg["end_sec"] * sr)
            ast_audio[i0:i1] = 0.0
        scan_result["audio_masked"] = ast_audio
    else:
        # For huge videos, we might skip full-track masking or do it lazily
        # In High Parallel mode, we'll just pass the raw audio and hope for the best,
        # or we could mask segments in the worker.
        scan_result["audio_masked"] = ast_audio 

    scenes, ast_timing = extract_sounds_optimized(
        scenes=scenes,
        scan_result=scan_result,
        target_sr=AST_TARGET_SR,
        max_workers=max_workers,
        use_processes=parallel,
        debug=debug,
    )
    print(f"       AST completed in {ast_timing['total_time_sec']:.2f}s\n")

    total_time = time.time() - total_start

    # ----- Timing Report -----
    timing = {
        "video": video_name,
        "video_duration_sec": scan_result["duration_sec"],
        "num_scenes": len(scenes),
        "scene_detection_sec": round(scene_time, 2),
        "audio_prescan_sec": round(scan_result["scan_time_sec"], 2),
        "whisper_sec": round(whisper_timing["total_time_sec"], 2),
        "ast_sec": round(ast_timing["total_time_sec"], 2),
        "total_sec": round(total_time, 2),
        "whisper_method": whisper_timing["method"],
        "ast_method": ast_timing["method"],
        "ast_scenes_processed": ast_timing.get("scenes_processed", 0),
        "ast_scenes_skipped": ast_timing.get("scenes_skipped", 0),
        "whisper_segments": whisper_timing.get("segments_found", 0),
        "scenes_with_speech": whisper_timing.get("scenes_with_speech", 0),
        "thresholds": scan_result["thresholds_used"],
    }

    return scenes, timing


def save_results(video_path: str, scenes: list, timing: dict):
    """Save pipeline results to the results/ directory."""
    video_name = Path(video_path).stem
    output_dir = RESULTS_DIR / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save scenes (audio_speech + audio_natural only)
    scene_output = []
    for scene in scenes:
        scene_output.append({
            "scene_index": scene["scene_index"],
            "start_timecode": scene["start_timecode"],
            "end_timecode": scene["end_timecode"],
            "audio_speech": scene.get("audio_speech", ""),
            "audio_natural": scene.get("audio_natural", "none"),
        })

    scenes_path = output_dir / "audio_results.json"
    with open(scenes_path, "w", encoding="utf-8") as f:
        json.dump(scene_output, f, indent=4, ensure_ascii=False)

    timing_path = output_dir / "timing.json"
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=4)

    return output_dir


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Audio High-Parallel Pipeline")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--all", action="store_true", help="Process all videos in Videos/")
    parser.add_argument("--parallel", action="store_true", help="Use multi-process parallelization")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--quiet", action="store_true", help="Suppress debug output")
    args = parser.parse_args()

    debug = not args.quiet

    if args.all:
        video_files = []
        # Use iterdir to catch all files including dot-prefixed ones
        extensions = (".mp4", ".mkv", ".avi", ".mov")
        for p in VIDEOS_DIR.iterdir():
            if p.suffix.lower() in extensions and not p.name.startswith("_"):
                video_files.append(str(p))
        print(f"Found {len(video_files)} videos to process.")
    elif args.video:
        video_files = [args.video]
    else:
        parser.error("Provide --video or --all")
        return

    all_timings = []
    for video_path in video_files:
        if not os.path.exists(video_path):
            continue

        video_name = Path(video_path).stem
        output_dir = RESULTS_DIR / video_name
        if (output_dir / "audio_results.json").exists():
            print(f"Skipping {video_name} (already processed)")
            continue

        try:
            # Memory Check for very long videos
            import psutil
            available_gb = psutil.virtual_memory().available / (1024**3)
            if available_gb < 2.0:
                print(f"[WARN] Low memory ({available_gb:.1f}GB). Pipeline might fail on long videos.")

            scenes, timing = run_pipeline(
                video_path, 
                parallel=args.parallel, 
                max_workers=args.workers, 
                debug=debug
            )
            save_results(video_path, scenes, timing)
            all_timings.append(timing)

            # Cleanup: Free space if needed in deployment scenarios
            # (In this context, we usually keep them for evaluation, but for massive runs we might clear)
            # if args.cleanup:
            #     import shutil
            #     shutil.rmtree(RESULTS_DIR / Path(video_path).stem / ".clips", ignore_errors=True)

        except Exception as e:
            print(f"[ERROR] {video_path}: {e}")
            continue

    if len(all_timings) > 1:
        print(f"\n{'='*90}")
        print("  FINAL COMPARISON TABLE")
        print(f"{'='*90}")
        print(f"  {'Video':<40} {'Dur':>6} {'Scenes':>6} {'Whisper':>8} {'AST':>8} {'Total':>8}")
        for t in all_timings:
            name = t["video"][:38]
            dur = f"{t['video_duration_sec']/60:.1f}m"
            print(f"  {name:<40} {dur:>6} {t['num_scenes']:>6} {t['whisper_sec']:>7.1f}s {t['ast_sec']:>7.1f}s {t['total_sec']:>7.1f}s")
        print(f"{'='*90}\n")


if __name__ == "__main__":
    import multiprocessing
    # Required for Windows when using ProcessPoolExecutor
    multiprocessing.freeze_support()
    main()
