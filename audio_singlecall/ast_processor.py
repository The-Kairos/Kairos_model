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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


# =========================================================
# AST Model Helper (for ProcessPool)
# =========================================================
_AST_MODEL = None
_AST_FE = None

def _get_ast_model():
    global _AST_MODEL, _AST_FE
    if _AST_MODEL is None:
        AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
        _AST_FE = AutoFeatureExtractor.from_pretrained(AST_MODEL_NAME)
        _AST_MODEL = AutoModelForAudioClassification.from_pretrained(AST_MODEL_NAME)
    return _AST_FE, _AST_MODEL


# =========================================================
# 1. Single Scene AST Inference
# =========================================================

def classify_scene_audio(audio_slice: np.ndarray, sr: int,
                         threshold: float = 0.3, device: str = "cpu") -> str:
    """
    Classify a single scene's audio using AST.
    """
    if audio_slice.size == 0:
        return "none"

    fe, model = _get_ast_model()

    # AST inference
    inputs = fe(audio_slice, sampling_rate=sr, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

    labels = [
        model.config.id2label[i]
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
# 2. Process a Single Scene (for pool)
# =========================================================

def _process_one_scene(args):
    """Worker function for Pooled Execution."""
    idx, audio_slice, sr, rms_dbfs, scene_silence_threshold, force_cpu = args

    # Skip if scene is below silence threshold
    if rms_dbfs < scene_silence_threshold:
        return idx, "none", True  # True = skipped

    if audio_slice is None or audio_slice.size == 0:
        return idx, "none", True

    if len(audio_slice) == 0:
        return idx, "none", True

    label = classify_scene_audio(audio_slice, sr, device="cpu" if force_cpu else "cpu") 
    # Actually classify_scene_audio default is cpu. 
    # If we want to allow GPU, we'd need to pass it here.
    return idx, label, False


# =========================================================
# 3. Main Entry Point (Parallelized)
# =========================================================

def extract_sounds_optimized(scenes: list, scan_result: dict,
                             target_sr: int = 16000,
                             max_workers: int = 4,
                             use_processes: bool = False,
                             force_cpu: bool = False,
                             debug: bool = False) -> tuple:
    """
    Run AST per scene with parallel execution and skip logic.

    Args:
        max_workers: number of parallel workers
        use_processes: If True, uses ProcessPoolExecutor (better for CPU)
    """
    t_start = time.time()

    if not scan_result["has_any_audio"] or not scan_result["has_background_audio"]:
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

    task_args = []
    for idx, scene in enumerate(scenes):
        t0 = float(scene["start_seconds"])
        t1 = float(scene["end_seconds"])
        rms_dbfs = per_scene_rms[idx] if idx < len(per_scene_rms) else -200.0
        
        # SLICE BEFORE PARALLEL: This is the OOM fix.
        # Passing the full 'audio' buffer to N workers in ProcessPoolExecutor
        # causes N copies of the full audio to be pickled/unpickled.
        if rms_dbfs >= scene_silence_threshold:
            i0 = max(0, int(t0 * sr))
            i1 = min(len(audio), int(t1 * sr))
            audio_slice = audio[i0:i1].copy() # copy to ensure we don't carry the full buffer's data
        else:
            audio_slice = None

        task_args.append((idx, audio_slice, sr, rms_dbfs, scene_silence_threshold, force_cpu))

    results = [None] * len(scenes)
    skipped_count = 0
    processed_count = 0

    if debug:
        pre_skip = sum(1 for a in task_args if a[3] < scene_silence_threshold)
        mode = "ProcessPool" if use_processes else "ThreadPool"
        print(f"[AST] Using {mode} with {max_workers} workers")
        print(f"[AST] Processing {len(scenes)} scenes ({pre_skip} will be skipped by RMS)")

    # Execute
    ExecutorClass = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    actual_workers = min(max_workers, len(scenes) or 1)

    with ExecutorClass(max_workers=actual_workers) as executor:
        futures = {executor.submit(_process_one_scene, args): args[0] for args in task_args}

        for future in as_completed(futures):
            try:
                idx, label, was_skipped = future.result()
                results[idx] = label
                if was_skipped:
                    skipped_count += 1
                else:
                    processed_count += 1
            except Exception as e:
                print(f"[AST] [ERROR] Scene processing failed: {e}")
                idx = futures[future]
                results[idx] = "error"
                skipped_count += 1

    # Assign results
    for idx, scene in enumerate(scenes):
        scene["audio_natural"] = results[idx] if results[idx] is not None else "none"

    elapsed = time.time() - t_start
    if debug:
        print(f"[AST] Done: {processed_count} processed, {skipped_count} skipped, {elapsed:.2f}s total")

    return scenes, {
        "method": "parallel_processes" if use_processes else "parallel_threads",
        "total_time_sec": elapsed,
        "scenes_processed": processed_count,
        "scenes_skipped": skipped_count,
        "max_workers": actual_workers,
    }
