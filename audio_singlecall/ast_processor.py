"""
ast_processor.py — Parallelized Per-Scene AST with Skip Logic

Runs MIT AST (Audio Spectrogram Transformer) per scene, but:
  1. Skips scenes where per-scene RMS is below the dynamic threshold
  2. Parallelizes AST inference across scenes using ThreadPoolExecutor
  3. Skips ALL scenes if scan_result says no background audio
"""

import time
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


# AST model is loaded once at module level

# =========================================================
# Load AST model (once)
# =========================================================
AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_FE = AutoFeatureExtractor.from_pretrained(AST_MODEL_NAME)
AST_MODEL = AutoModelForAudioClassification.from_pretrained(AST_MODEL_NAME)


# =========================================================
# 1. Single Scene AST Inference
# =========================================================

def classify_scene_audio(audio_slice: np.ndarray, sr: int,
                         threshold: float = 0.3, device: str = "cpu") -> str:
    """
    Classify a single scene's audio using AST.
    Removes speech regions first, then classifies environmental sounds.

    Returns a string like "music (conf=0.85), crowd (conf=0.72)" or "none".
    """
    if audio_slice.size == 0:
        return "none"

    # Speech is now masked externally before calling this, or passed in.
    # We'll assume the audio_slice passed here is already masked for speech.
    masked = audio_slice

    # AST inference
    inputs = AST_FE(masked, sampling_rate=sr, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = AST_MODEL(**inputs)
        probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

    labels = [
        AST_MODEL.config.id2label[i]
        for i, p in enumerate(probs)
        if p >= threshold
    ]
    scores = [float(p) for p in probs if p >= threshold]

    if not labels:
        return "none"

    out = [
        f"{labels[i].lower().replace('_', ' ')} (conf={scores[i]:.2f})"
        for i in range(len(labels))
    ]
    return ", ".join(out)


# =========================================================
# 2. Process a Single Scene (for thread pool)
# =========================================================

def _process_one_scene(args):
    """Worker function for ThreadPoolExecutor."""
    idx, audio, sr, t0, t1, scene_rms_dbfs, scene_silence_threshold = args

    # Skip if scene is below silence threshold
    if scene_rms_dbfs < scene_silence_threshold:
        return idx, "none", True  # True = skipped

    # Slice audio
    i0 = max(0, int(t0 * sr))
    i1 = min(len(audio), int(t1 * sr))
    audio_slice = audio[i0:i1]

    if len(audio_slice) == 0:
        return idx, "none", True

    label = classify_scene_audio(audio_slice, sr)
    return idx, label, False  # False = not skipped


# =========================================================
# 3. Main Entry Point (Parallelized)
# =========================================================

def extract_sounds_optimized(scenes: list, scan_result: dict,
                             target_sr: int = 16000,
                             max_workers: int = 4,
                             debug: bool = False) -> tuple:
    """
    Run AST per scene with parallel execution and skip logic.

    Args:
        scenes: list of scene dicts with start_seconds/end_seconds
        scan_result: output from audio_detector.scan_audio()
        target_sr: sample rate
        max_workers: number of parallel workers
        debug: print progress

    Returns:
        (scenes, timing_info) where each scene gains "audio_natural" key
    """
    t_start = time.time()

    if not scan_result["has_any_audio"] or not scan_result["has_background_audio"]:
        # No meaningful audio — fill all scenes with "none"
        if debug:
            reason = "no audio" if not scan_result["has_any_audio"] else "no background audio"
            print(f"[AST] Skipping all {len(scenes)} scenes ({reason})")
        for scene in scenes:
            scene["audio_natural"] = "none"
        elapsed = time.time() - t_start
        return scenes, {
            "method": "all_skipped",
            "total_time_sec": elapsed,
            "scenes_processed": 0,
            "scenes_skipped": len(scenes),
        }

    audio = scan_result.get("audio_masked", scan_result["audio"])
    sr = scan_result["sr"]
    per_scene_rms = scan_result["per_scene_rms"]
    scene_silence_threshold = scan_result["thresholds_used"]["SCENE_SILENCE_DBFS"]

    # Prepare args for parallel execution
    task_args = []
    for idx, scene in enumerate(scenes):
        t0 = float(scene["start_seconds"])
        t1 = float(scene["end_seconds"])
        rms_dbfs = per_scene_rms[idx] if idx < len(per_scene_rms) else -200.0
        task_args.append((idx, audio, sr, t0, t1, rms_dbfs, scene_silence_threshold))

    # Execute in parallel
    results = [None] * len(scenes)
    skipped_count = 0
    processed_count = 0

    if debug:
        pre_skip = sum(1 for a in task_args if a[5] < scene_silence_threshold)
        print(f"[AST] Processing {len(scenes)} scenes ({pre_skip} will be skipped by RMS)")

    # Note: AST model is not thread-safe for GPU; use workers=1 for GPU
    # For CPU inference, parallel is safe
    actual_workers = min(max_workers, len(scenes))

    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        futures = {executor.submit(_process_one_scene, args): args[0] for args in task_args}

        for future in as_completed(futures):
            try:
                idx, label, was_skipped = future.result()
                results[idx] = label
                if was_skipped:
                    skipped_count += 1
                else:
                    processed_count += 1
                    if debug:
                        print(f"[AST] Finished Scene {idx:03d}")
            except Exception as e:
                print(f"[AST] [ERROR] Scene processing failed: {e}")
                # Fallback label
                idx = futures[future]
                results[idx] = "error"
                skipped_count += 1

    # Assign results to scenes
    for idx, scene in enumerate(scenes):
        scene["audio_natural"] = results[idx] if results[idx] is not None else "none"

        if debug:
            t0 = scene.get("start_timecode", f"{scene['start_seconds']:.1f}s")
            t1 = scene.get("end_timecode", f"{scene['end_seconds']:.1f}s")
            label = scene["audio_natural"]
            prefix = "(skip)" if label == "none" and per_scene_rms[idx] < scene_silence_threshold else "(AST)"
            print(f"[AST] {prefix} Scene {idx:03d} [{t0} → {t1}]: {label}")

    elapsed = time.time() - t_start

    if debug:
        print(f"[AST] Done: {processed_count} processed, {skipped_count} skipped, {elapsed:.2f}s total")

    return scenes, {
        "method": "parallel_per_scene",
        "total_time_sec": elapsed,
        "scenes_processed": processed_count,
        "scenes_skipped": skipped_count,
        "max_workers": actual_workers,
    }
