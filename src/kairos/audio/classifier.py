"""MIT AST (Audio Spectrogram Transformer) parallelized per-scene classification."""

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import torch
from transformers import (
    ASTFeatureExtractor,
    ASTForAudioClassification,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)

from kairos.core.utils import print_prefixed

# Lazy-loaded AST model
_AST_MODEL: ASTForAudioClassification | None = None
_AST_FE: ASTFeatureExtractor | None = None


def _get_ast_model() -> tuple[ASTFeatureExtractor, ASTForAudioClassification]:
    """Lazy-load the AST feature extractor and classification model.

    Loads the ``MIT/ast-finetuned-audioset-10-10-0.4593`` model on first
    call and caches it in module-level globals so subsequent calls return
    the already-initialised objects.

    Returns:
        A tuple of ``(feature_extractor, model)`` ready for inference.
    """
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
    fe: ASTFeatureExtractor | None = None,
    model: ASTForAudioClassification | None = None,
) -> str:
    """Classify an audio segment using the AST model.

    Runs the Audio Spectrogram Transformer on a single audio slice and
    returns a comma-separated string of detected labels whose sigmoid
    probabilities meet or exceed ``threshold``.

    Args:
        audio_slice: 1-D float array of raw audio samples for one scene.
        sr: Sampling rate of ``audio_slice`` in Hz.
        threshold: Minimum sigmoid probability for a label to be
            included in the result. Defaults to ``0.3``.
        device: PyTorch device string (``"cpu"`` or ``"cuda"``).
            Defaults to ``"cpu"``.
        fe: Pre-loaded AST feature extractor. If ``None``, the
            module-level cached extractor is used.
        model: Pre-loaded AST classification model. If ``None``, the
            module-level cached model is used.

    Returns:
        A comma-separated string of ``"label (conf=X.XX)"`` entries, or
        ``"none"`` when no label exceeds the threshold or the input is
        empty.
    """
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


def _process_one_scene(
    args: tuple[int, np.ndarray | None, int, float, float, bool],
) -> tuple[int, str, bool]:
    """Worker function that classifies audio for a single scene.

    Designed to be submitted to a :class:`~concurrent.futures.Executor`.
    Scenes whose RMS level falls below the silence threshold are skipped
    immediately, returning ``"none"`` without running the model.

    Args:
        args: A 6-element tuple containing:

            * **idx** (*int*) – Scene index in the original list.
            * **audio_slice** (*np.ndarray | None*) – Audio samples for
              the scene, or ``None`` if pre-skipped.
            * **sr** (*int*) – Sampling rate in Hz.
            * **rms_dbfs** (*float*) – Pre-computed RMS level of the
              scene in dBFS.
            * **scene_silence_threshold** (*float*) – dBFS threshold
              below which a scene is considered silent.
            * **force_cpu** (*bool*) – If ``True``, force inference on
              CPU even when CUDA is available.

    Returns:
        A 3-tuple of ``(scene_index, label_string, was_skipped)``.
    """
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
    scenes: list[dict],
    scan_result: dict,
    target_sr: int = 16000,
    max_workers: int = 4,
    use_processes: bool = False,
    force_cpu: bool = False,
    debug: bool = False,
) -> tuple[list[dict], dict]:
    """Run AST classification per scene with parallel execution and skip logic.

    Orchestrates the full sound-classification pipeline:

    1. Checks ``scan_result`` to determine whether audio / background
       audio exists; if not, every scene is labelled ``"none"``
       immediately.
    2. Builds per-scene task arguments, pre-skipping silent scenes based
       on the RMS threshold from the pre-scan.
    3. Dispatches ``_process_one_scene`` calls to a thread or process
       pool, collects results, and writes the ``audio_natural`` key on
       each scene dict.

    Args:
        scenes: List of scene dictionaries. Each dict must contain
            ``start_seconds`` and ``end_seconds`` keys. An
            ``audio_natural`` key is added in-place with the
            classification result.
        scan_result: Dictionary returned by
            :func:`kairos.audio.prescan.scan_audio` containing audio
            data, RMS values, and threshold configuration.
        target_sr: Target sampling rate in Hz. Defaults to ``16000``.
        max_workers: Maximum number of parallel workers for the
            executor. Defaults to ``4``.
        use_processes: If ``True``, use a
            :class:`~concurrent.futures.ProcessPoolExecutor`; otherwise
            use a :class:`~concurrent.futures.ThreadPoolExecutor`.
            Defaults to ``False``.
        force_cpu: If ``True``, force AST inference on CPU regardless of
            GPU availability. Defaults to ``False``.
        debug: If ``True``, emit progress messages via
            :func:`~kairos.core.utils.print_prefixed`. Defaults to
            ``False``.

    Returns:
        A 2-tuple of:

        * **scenes** – The input list with ``audio_natural`` populated.
        * **stats** – A dictionary summarising the run (method, timing,
          worker count, scenes processed / skipped).
    """
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

    task_args: list[tuple[int, np.ndarray | None, int, float, float, bool]] = []
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

    results: list[str | None] = [None] * len(scenes)
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
