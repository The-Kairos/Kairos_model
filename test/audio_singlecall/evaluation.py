"""
evaluation.py — Compare Single-Call Pipeline vs Original Pipeline

Loads existing checkpoint.json (original pipeline output) and new
audio_results.json (single-call output), then computes:
  1. Speech detection accuracy (per-scene confusion matrix)
  2. Transcription similarity (word overlap / Levenshtein)
  3. Natural audio accuracy (label comparison)
  4. Timing comparison table

Usage:
  python -m audio_singlecall.evaluation --video "Videos\\Young Sheldon - First Day of High School.mp4"
  python -m audio_singlecall.evaluation --all
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# 1. Load Data
# =========================================================

def load_original_results(video_path: str) -> list | None:
    """Load scenes from the original pipeline's checkpoint.json."""
    video_name = Path(video_path).name
    checkpoint_path = PROJECT_ROOT / "_processed" / video_name / "checkpoint.json"
    if not checkpoint_path.exists():
        return None
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("scenes", None)


def load_new_results(video_path: str) -> list | None:
    """Load scenes from the new pipeline's audio_results.json."""
    video_name = Path(video_path).stem
    results_path = Path(__file__).resolve().parent / "results" / video_name / "audio_results.json"
    if not results_path.exists():
        return None
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_new_timing(video_path: str) -> dict | None:
    """Load timing from the new pipeline."""
    video_name = Path(video_path).stem
    timing_path = Path(__file__).resolve().parent / "results" / video_name / "timing.json"
    if not timing_path.exists():
        return None
    with open(timing_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# 2. Speech Analysis
# =========================================================

def word_set(text: str) -> set:
    """Convert text to set of lowercase words."""
    return set(text.lower().split())


def word_overlap(text1: str, text2: str) -> float:
    """Jaccard similarity between word sets."""
    s1 = word_set(text1)
    s2 = word_set(text2)
    if not s1 and not s2:
        return 1.0  # both empty = perfect match
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def speech_confusion_matrix(original_scenes: list, new_scenes: list) -> dict:
    """
    Compare speech detection per scene.
    Original = ground truth. A scene "has speech" if audio_speech is non-empty.

    Returns: {TP, TN, FP, FN, precision, recall, f1, accuracy}
    """
    tp = tn = fp = fn = 0

    for orig, new in zip(original_scenes, new_scenes):
        orig_has = bool(orig.get("audio_speech", "").strip())
        new_has  = bool(new.get("audio_speech", "").strip())

        if orig_has and new_has:
            tp += 1
        elif not orig_has and not new_has:
            tn += 1
        elif not orig_has and new_has:
            fp += 1
        else:  # orig_has and not new_has
            fn += 1

    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "total_scenes": total,
    }


# =========================================================
# 3. Natural Audio Analysis
# =========================================================

def natural_audio_confusion_matrix(original_scenes: list, new_scenes: list) -> dict:
    """
    Compare AST labels per scene.
    "has audio" = anything other than "none".
    """
    tp = tn = fp = fn = 0

    for orig, new in zip(original_scenes, new_scenes):
        orig_has = orig.get("audio_natural", "none").strip().lower() != "none"
        new_has  = new.get("audio_natural", "none").strip().lower() != "none"

        if orig_has and new_has:
            tp += 1
        elif not orig_has and not new_has:
            tn += 1
        elif not orig_has and new_has:
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "total_scenes": total,
    }


# =========================================================
# 4. Full Evaluation
# =========================================================

def evaluate_video(video_path: str, debug: bool = True) -> dict | None:
    """
    Full evaluation for one video: load both pipeline outputs, compute metrics.
    """
    video_name = Path(video_path).stem

    original = load_original_results(video_path)
    new = load_new_results(video_path)
    new_timing = load_new_timing(video_path)

    if new is None:
        if debug:
            print(f"[EVAL] No new results for {video_name} (run main.py first)")
        return None

    if original is None:
        if debug:
            print(f"[EVAL] No original results for {video_name} (no checkpoint.json). Saving single-call timings only.")
        
        # Save a partial evaluation with just the new timings
        result = {
            "video": video_name,
            "status": "No original checkpoint to compare against.",
            "single_call_timing": new_timing
        }
        eval_dir = Path(__file__).resolve().parent / "results" / video_name
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_path = eval_dir / "evaluation.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
            
        return result

    # Align scene counts (take minimum)
    n = min(len(original), len(new))
    orig_scenes = original[:n]
    new_scenes = new[:n]

    if debug:
        print(f"\n{'='*70}")
        print(f"  EVALUATION: {video_name}")
        print(f"{'='*70}")
        print(f"  Scenes compared: {n}")

    # --- Speech detection confusion matrix ---
    speech_cm = speech_confusion_matrix(orig_scenes, new_scenes)

    if debug:
        print(f"\n  SPEECH DETECTION (has speech vs no speech):")
        print(f"  +-----------------+---------+---------+")
        print(f"  |                 | Orig=Y  | Orig=N  |")
        print(f"  +-----------------+---------+---------+")
        print(f"  | New=Y (speech)  | TP={speech_cm['TP']:>3}  | FP={speech_cm['FP']:>3}  |")
        print(f"  | New=N (silent)  | FN={speech_cm['FN']:>3}  | TN={speech_cm['TN']:>3}  |")
        print(f"  +-----------------+---------+---------+")
        print(f"  Accuracy:  {speech_cm['accuracy']:.2%}")
        print(f"  Precision: {speech_cm['precision']:.2%}")
        print(f"  Recall:    {speech_cm['recall']:.2%}")
        print(f"  F1:        {speech_cm['f1']:.2%}")

    # --- Transcription similarity ---
    overlaps = []
    for orig, new in zip(orig_scenes, new_scenes):
        o_text = orig.get("audio_speech", "")
        n_text = new.get("audio_speech", "")
        if o_text.strip() or n_text.strip():
            overlaps.append(word_overlap(o_text, n_text))

    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0

    if debug:
        print(f"\n  TRANSCRIPTION SIMILARITY:")
        print(f"  Mean word overlap (Jaccard): {mean_overlap:.2%}")
        print(f"  Scenes with text compared:   {len(overlaps)}")

    # --- Natural audio confusion matrix ---
    natural_cm = natural_audio_confusion_matrix(orig_scenes, new_scenes)

    if debug:
        print(f"\n  NATURAL AUDIO DETECTION (has labels vs none):")
        print(f"  +-----------------+---------+---------+")
        print(f"  |                 | Orig=Y  | Orig=N  |")
        print(f"  +-----------------+---------+---------+")
        print(f"  | New=Y (labels)  | TP={natural_cm['TP']:>3}  | FP={natural_cm['FP']:>3}  |")
        print(f"  | New=N (none)    | FN={natural_cm['FN']:>3}  | TN={natural_cm['TN']:>3}  |")
        print(f"  +-----------------+---------+---------+")
        print(f"  Accuracy:  {natural_cm['accuracy']:.2%}")
        print(f"  Precision: {natural_cm['precision']:.2%}")
        print(f"  Recall:    {natural_cm['recall']:.2%}")
        print(f"  F1:        {natural_cm['f1']:.2%}")

    # --- Timing comparison ---
    if new_timing and debug:
        # Try to get original timing from checkpoint
        checkpoint_path = PROJECT_ROOT / "_processed" / Path(video_path).name / "checkpoint.json"
        orig_timing = {}
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    cp = json.load(f)
                steps = cp.get("steps", {})
                orig_timing = {
                    "asr_sec": steps.get("asr_timings", {}).get("wall_time_sec", 0),
                    "ast_sec": steps.get("ast_timings", {}).get("wall_time_sec", 0),
                }
            except Exception:
                pass

        if orig_timing:
            orig_total = orig_timing["asr_sec"] + orig_timing["ast_sec"]
            new_total = new_timing["whisper_sec"] + new_timing["ast_sec"]
            speedup = orig_total / new_total if new_total > 0 else float('inf')

            print(f"\n  TIMING COMPARISON:")
            print(f"  +------------------+--------------+--------------+----------+")
            print(f"  | Stage            | Original     | Single-Call  | Speedup  |")
            print(f"  +------------------+--------------+--------------+----------+")
            print(f"  | Pre-Scan         |       -      | {new_timing['audio_prescan_sec']:>8.1f}s   |    -     |")
            print(f"  | ASR (Whisper)    | {orig_timing['asr_sec']:>8.1f}s   | {new_timing['whisper_sec']:>8.1f}s   | {orig_timing['asr_sec']/max(new_timing['whisper_sec'],0.01):>5.1f}x  |")
            print(f"  | AST (Sounds)     | {orig_timing['ast_sec']:>8.1f}s   | {new_timing['ast_sec']:>8.1f}s   | {orig_timing['ast_sec']/max(new_timing['ast_sec'],0.01):>5.1f}x  |")
            print(f"  +------------------+--------------+--------------+----------+")
            print(f"  | TOTAL (audio)    | {orig_total:>8.1f}s   | {new_total + new_timing['audio_prescan_sec']:>8.1f}s   | {speedup:>5.1f}x  |")
            print(f"  +------------------+--------------+--------------+----------+")

    print(f"\n{'='*70}\n")

    # Save evaluation
    result = {
        "video": video_name,
        "scenes_compared": n,
        "speech_confusion_matrix": speech_cm,
        "transcription_word_overlap_mean": round(mean_overlap, 4),
        "natural_audio_confusion_matrix": natural_cm,
    }

    eval_dir = Path(__file__).resolve().parent / "results" / video_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / "evaluation.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    return result


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Audio Single-Call Pipeline")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--all", action="store_true", help="Evaluate all videos with results")
    args = parser.parse_args()

    VIDEOS_DIR = PROJECT_ROOT / "Videos"

    if args.all:
        results_dir = Path(__file__).resolve().parent / "results"
        if not results_dir.exists():
            print("No results directory. Run main.py first.")
            return
        video_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
        for vdir in video_dirs:
            # Try to find the original video
            for ext in ("mp4", "mkv", "avi", "mov"):
                vpath = VIDEOS_DIR / f"{vdir.name}.{ext}"
                if vpath.exists():
                    evaluate_video(str(vpath))
                    break
    elif args.video:
        evaluate_video(args.video)
    else:
        parser.error("Provide --video or --all")


if __name__ == "__main__":
    main()
