"""MIT AST (Audio Spectrogram Transformer) parallelized per-scene classification."""

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from kairos.core.utils import print_prefixed

# Lazy-loaded AST model
_AST_MODEL = None
_AST_FE = None


def _get_ast_model():
    global _AST_MODEL, _AST_FE
    if _AST_MODEL is None:
        ast_model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        _AST_FE = AutoFeatureExtractor.from_pretrained(ast_model_name)
        _AST_MODEL = AutoModelForAudioClassification.from_pretrained(ast_model_name)
    return _AST_FE, _AST_MODEL


def classify_scene_audio(
    audio_slice: np.ndarray,
    sr: int,
    threshold: float = 0.3,
    device: str = "cpu",
    fe=None,
    model=None,
) -> str:
    if audio_slice.size == 0:
        return "none"
    if fe is None or model is None:
        fe, model = _get_ast_model()
    inputs = fe(audio_slice, sampling_rate=sr, return_tensors="pt", padding=True).to(
        device
    )
    with torch.no_grad():
        probs = torch.sigmoid(model(**inputs).logits)[0].cpu().numpy()
    labels = [model.config.id2label[i] for i, p in enumerate(probs) if p >= threshold]
    scores = [float(p) for p in probs if p >= threshold]
    if not labels:
        return "none"
    return ", ".join(
        f"{labels[i].lower().replace('_', ' ')} (conf={scores[i]:.2f})"
        for i in range(len(labels))
    )


def _process_one_scene(args):
    idx, audio_slice, sr, rms_dbfs, scene_silence_threshold, force_cpu = args
    if rms_dbfs < scene_silence_threshold:
        return idx, "none", True
    if audio_slice is None or audio_slice.size == 0:
        return idx, "none", True
    device = (
        "cpu"
        if force_cpu
        else ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    )
    label = classify_scene_audio(audio_slice, sr, device=device)
    return idx, label, False


def extract_sounds_optimized(
    scenes: list,
    scan_result: dict,
    target_sr: int = 16000,
    max_workers: int = 4,
    use_processes: bool = False,
    force_cpu: bool = False,
    debug: bool = False,
) -> tuple:
    """Run AST per scene with parallel execution and skip logic."""
    t_start = time.time()

    if not scan_result["has_any_audio"] or not scan_result["has_background_audio"]:
        if debug:
            reason = (
                "no audio"
                if not scan_result["has_any_audio"]
                else "no background audio"
            )
            print_prefixed("(AST)", f"Skipping all {len(scenes)} scenes ({reason})")
        for scene in scenes:
            scene["audio_natural"] = "none"
        return scenes, {
            "method": "all_skipped",
            "total_time_sec": time.time() - t_start,
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
        if rms_dbfs >= scene_silence_threshold:
            i0 = max(0, int(t0 * sr))
            i1 = min(len(audio), int(t1 * sr))
            audio_slice = audio[i0:i1].copy()
        else:
            audio_slice = None
        task_args.append(
            (idx, audio_slice, sr, rms_dbfs, scene_silence_threshold, force_cpu)
        )

    results = [None] * len(scenes)
    skipped_count = processed_count = 0

    if debug:
        pre_skip = sum(1 for a in task_args if a[3] < scene_silence_threshold)
        mode = "ProcessPool" if use_processes else "ThreadPool"
        print_prefixed(
            "(AST)", f"Using {mode} with {max_workers} workers, {pre_skip} pre-skipped"
        )

    executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    with executor_cls(max_workers=min(max_workers, len(scenes) or 1)) as executor:
        futures = {
            executor.submit(_process_one_scene, args): args[0] for args in task_args
        }
        for future in as_completed(futures):
            try:
                idx, label, was_skipped = future.result()
                results[idx] = label
                if was_skipped:
                    skipped_count += 1
                else:
                    processed_count += 1
            except Exception as e:
                print_prefixed("(AST)", f"[ERROR] Scene processing failed: {e}")
                results[futures[future]] = "error"
                skipped_count += 1

    for idx, scene in enumerate(scenes):
        scene["audio_natural"] = results[idx] if results[idx] is not None else "none"

    elapsed = time.time() - t_start
    if debug:
        print_prefixed(
            "(AST)",
            f"Done: {processed_count} processed, "
            f"{skipped_count} skipped, {elapsed:.2f}s",
        )

    return scenes, {
        "method": "parallel_processes" if use_processes else "parallel_threads",
        "total_time_sec": elapsed,
        "scenes_processed": processed_count,
        "scenes_skipped": skipped_count,
        "max_workers": min(max_workers, len(scenes) or 1),
    }
