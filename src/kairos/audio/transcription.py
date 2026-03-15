"""Whisper-based speech transcription.

Provides parallel chunked transcription, single-call transcription,
scene mapping, and an orchestration entry point
(:func:`extract_speech_singlecall`) that ties everything together.

The module supports two back-ends:

* **Azure OpenAI Whisper API** – preferred when credentials are available.
* **Local ``openai-whisper`` model** – used as a fallback or when
  ``use_api=False``.
"""

from __future__ import annotations

import concurrent.futures
import gc
import os
import time
from collections.abc import Callable
from typing import Any

import noisereduce as nr
import numpy as np
import torch
import whisper

from kairos.audio.text_filter import filter_hallucinations
from kairos.audio.whisper_api import transcribe_via_api
from kairos.core.utils import print_prefixed, retry_with_backoff

# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------


def clean_audio(
    audio: np.ndarray,
    sr: int,
    silero_model: Any | None = None,
    get_speech_ts: Callable[..., list[dict[str, int]]] | None = None,
) -> np.ndarray:
    """Denoise an audio waveform with optional VAD-guided enhancement.

    First applies broadband noise reduction to the entire signal.  When a
    Silero VAD model and its ``get_speech_timestamps`` utility are
    provided, a second, lighter noise-reduction pass is applied to each
    detected speech region individually.

    Args:
        audio: 1-D float NumPy array of audio samples.
        sr: Sampling rate in Hz.
        silero_model: Pre-loaded Silero VAD model.  Pass ``None`` to skip
            the VAD-guided enhancement step.
        get_speech_ts: The ``get_speech_timestamps`` callable returned by
            ``torch.hub.load("snakers4/silero-vad", ...)``.  Required
            together with *silero_model* for VAD enhancement.

    Returns:
        A noise-reduced copy of *audio* (same shape and dtype).
    """
    audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.9)
    if silero_model is not None and get_speech_ts is not None:
        audio_t: torch.Tensor = torch.from_numpy(audio).float()
        speech_ts: list[dict[str, int]] = get_speech_ts(
            audio_t, silero_model, sampling_rate=sr
        )
        if len(speech_ts) > 0:
            enhanced: np.ndarray = audio.copy()
            for seg in speech_ts:
                segment: np.ndarray = enhanced[seg["start"] : seg["end"]]
                enhanced[seg["start"] : seg["end"]] = nr.reduce_noise(
                    y=segment, sr=sr, prop_decrease=0.7
                )
            audio = enhanced
    return audio


# ---------------------------------------------------------------------------
# Timestamp-to-scene mapping
# ---------------------------------------------------------------------------


def map_segments_to_scenes(
    whisper_segments: list[dict], scenes: list[dict]
) -> list[str]:
    """Map Whisper segments to scene boundaries and aggregate text.

    For each scene, collects all Whisper segments whose temporal overlap
    with the scene is at least 20 % of the segment duration **or** at
    least 0.5 s in absolute terms.

    Args:
        whisper_segments: List of Whisper segment dicts, each with
            ``"start"``, ``"end"``, and ``"text"`` keys.
        scenes: List of scene dicts, each with ``"start_seconds"`` and
            ``"end_seconds"`` keys (floats).

    Returns:
        A list of concatenated transcription strings, one per scene (same
        order as *scenes*).  Scenes without matching speech yield an
        empty string.
    """
    scene_texts: list[str] = []
    for scene in scenes:
        t0: float = float(scene["start_seconds"])
        t1: float = float(scene["end_seconds"])
        parts: list[str] = []
        for seg in whisper_segments:
            seg_start: float = float(seg["start"])
            seg_end: float = float(seg["end"])
            if seg_end <= t0 or seg_start >= t1:
                continue
            overlap: float = min(t1, seg_end) - max(t0, seg_start)
            seg_duration: float = max(seg_end - seg_start, 1e-6)
            if overlap / seg_duration >= 0.2 or overlap >= 0.5:
                parts.append(seg["text"])
        scene_texts.append(" ".join(parts).strip())
    return scene_texts


# ---------------------------------------------------------------------------
# Parallel chunking helpers
# ---------------------------------------------------------------------------


def _transcribe_via_api_with_retry(
    chunk_audio: np.ndarray,
    sr: int,
    language: str | None,
    client: Any,
    debug: bool,
) -> list[dict]:
    """Call the Whisper API with automatic retry and rate-limit back-off.

    Wraps :func:`~kairos.audio.whisper_api.transcribe_via_api` inside
    :func:`~kairos.core.utils.retry_with_backoff` (up to 3 attempts,
    30 s base delay).

    Args:
        chunk_audio: 1-D float NumPy array of the audio chunk.
        sr: Sampling rate in Hz.
        language: ISO-639-1 language hint (e.g. ``"en"``), or ``None``.
        client: Pre-built Azure OpenAI client forwarded to the API call.
        debug: If ``True``, prints error diagnostics on failure.

    Returns:
        List of segment dicts on success, or an empty list when all
        retries are exhausted.
    """
    try:
        return retry_with_backoff(
            lambda: transcribe_via_api(
                chunk_audio, sr, language=language, client=client
            ),
            max_retries=2,
            base_sec=30.0,
            jitter=False,
        )
    except Exception as e:
        if debug:
            print_prefixed("(Whisper)", f"API Error: {e}")
        return []


def _transcribe_via_local_model(
    chunk_audio: np.ndarray,
    model_size: str,
    force_cpu: bool,
    language: str | None,
) -> list[dict]:
    """Transcribe an audio chunk using a locally loaded Whisper model.

    Loads the specified Whisper model, runs inference, and immediately
    frees GPU / CPU memory.

    Args:
        chunk_audio: 1-D float NumPy array of the audio chunk.
        model_size: Whisper model size string (e.g. ``"tiny"``,
            ``"base"``, ``"small"``, ``"medium"``, ``"large"``).
        force_cpu: If ``True``, forces model loading and inference on
            CPU regardless of GPU availability.
        language: ISO-639-1 language hint, or ``None`` for auto-detect.

    Returns:
        List of segment dicts produced by ``model.transcribe()``.
    """
    device: str | None = "cpu" if force_cpu else None
    model = whisper.load_model(model_size, device=device)
    result: dict = model.transcribe(
        chunk_audio, fp16=False, verbose=None, language=language
    )
    segments: list[dict] = result.get("segments", [])
    del model
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return segments


def _transcribe_chunk_worker(
    args: tuple[
        np.ndarray,
        int,
        str,
        float,
        bool,
        bool,
        bool,
        str | None,
        bool,
        Any,
    ],
) -> list[dict]:
    """Worker function for parallel chunk-based transcription.

    Designed to be submitted to a :class:`~concurrent.futures.Executor`.
    Optionally applies noise reduction, then transcribes via the API
    (with local-model fallback) or directly via a local model.

    Args:
        args: A 10-element tuple containing:

            0. ``chunk_audio`` – 1-D float audio array for this chunk.
            1. ``sr`` – sampling rate in Hz.
            2. ``model_size`` – Whisper model size string.
            3. ``chunk_start_time`` – offset in seconds of this chunk
               within the full audio.
            4. ``use_vad`` – whether to apply noise reduction.
            5. ``force_cpu`` – force CPU inference.
            6. ``debug`` – enable debug output.
            7. ``language`` – language hint or ``None``.
            8. ``use_api`` – prefer the Whisper API.
            9. ``client`` – Azure OpenAI client instance.

    Returns:
        List of segment dicts with ``"start"`` / ``"end"`` timestamps
        adjusted to the full-audio timeline.
    """
    (
        chunk_audio,
        sr,
        model_size,
        chunk_start_time,
        use_vad,
        force_cpu,
        debug,
        language,
        use_api,
        client,
    ) = args
    if use_vad:
        chunk_audio = nr.reduce_noise(y=chunk_audio, sr=sr, prop_decrease=0.9)

    if use_api:
        segments: list[dict] = _transcribe_via_api_with_retry(
            chunk_audio, sr, language, client, debug
        )
        if not segments:
            try:
                segments = _transcribe_via_local_model(
                    chunk_audio, model_size, force_cpu, language
                )
            except Exception:
                return []
    else:
        segments = _transcribe_via_local_model(
            chunk_audio, model_size, force_cpu, language
        )

    for seg in segments:
        seg["start"] += chunk_start_time
        seg["end"] += chunk_start_time
    return segments


def _deduplicate_segments(segments: list[dict]) -> list[dict]:
    """Remove near-duplicate segments from sorted Whisper output.

    Two consecutive segments are considered duplicates when their start
    times are within 0.5 s and their texts match (case-insensitive).
    For segments within 1.0 s whose text is a substring of the other,
    the longer text is kept.

    Args:
        segments: A time-sorted list of Whisper segment dicts.

    Returns:
        A deduplicated copy of the segment list.
    """
    if not segments:
        return []
    deduped: list[dict] = [segments[0]]
    for curr in segments[1:]:
        prev: dict = deduped[-1]
        time_diff: float = abs(curr["start"] - prev["start"])
        text_match: bool = curr["text"].strip().lower() == prev["text"].strip().lower()
        if time_diff < 0.5 and text_match:
            continue
        if time_diff < 1.0 and (
            curr["text"].strip() in prev["text"].strip()
            or prev["text"].strip() in curr["text"].strip()
        ):
            if len(curr["text"]) > len(prev["text"]):
                deduped[-1] = curr
            continue
        deduped.append(curr)
    return deduped


def transcribe_parallel(
    audio: np.ndarray,
    sr: int,
    model_size: str = "medium",
    chunk_size_sec: int = 600,
    overlap_sec: int = 30,
    lang_info: dict | None = None,
    use_vad: bool = True,
    force_cpu: bool = False,
    debug: bool = False,
    use_api: bool = True,
    client: Any | None = None,
) -> dict:
    """Transcribe audio in parallel chunks and merge the results.

    Splits the audio into overlapping chunks, dispatches each to a
    thread (API mode) or process (local-model mode), merges the
    resulting segments, deduplicates, and filters hallucinations.

    Args:
        audio: 1-D float NumPy array of the full audio signal.
        sr: Sampling rate in Hz.
        model_size: Whisper model size (e.g. ``"medium"``).
        chunk_size_sec: Target chunk length in seconds (before overlap).
        overlap_sec: Overlap between consecutive chunks in seconds.
        lang_info: Optional dict with ``"primary_language"`` and
            ``"is_multilingual"`` keys.  Used to force a language hint
            when the audio is not multilingual.
        use_vad: Apply noise reduction to each chunk before transcription.
        force_cpu: Force CPU-only inference.
        debug: Enable diagnostic output.
        use_api: Prefer the Azure Whisper API over a local model.
        client: Pre-built Azure OpenAI client (used when *use_api* is
            ``True``).

    Returns:
        A dict with keys ``"segments"`` (list of segment dicts),
        ``"full_text"`` (concatenated transcription string),
        ``"total_time_sec"`` (wall-clock seconds), and ``"method"``
        (descriptive label).
    """
    duration: float = len(audio) / sr
    t_start: float = time.time()
    force_lang: str | None = None
    if lang_info and not lang_info.get("is_multilingual", False):
        force_lang = lang_info.get("primary_language")
    chunks_args: list[tuple] = []
    start: float = 0
    while start < duration:
        end: float = min(start + chunk_size_sec + overlap_sec, duration)
        chunks_args.append(
            (
                audio[int(start * sr) : int(end * sr)].copy(),
                sr,
                model_size,
                start,
                use_vad,
                force_cpu,
                debug,
                force_lang,
                use_api,
                client,
            )
        )
        if end >= duration:
            break
        start += chunk_size_sec

    max_rec_workers: int = 4 if use_api else 2
    num_workers: int = min(len(chunks_args), os.cpu_count() or 4, max_rec_workers)
    all_segments: list[dict] = []
    Executor = (
        concurrent.futures.ThreadPoolExecutor
        if use_api
        else concurrent.futures.ProcessPoolExecutor
    )
    with Executor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_transcribe_chunk_worker, arg) for arg in chunks_args
        ]
        for future in concurrent.futures.as_completed(futures):
            all_segments.extend(future.result())
            gc.collect()

    all_segments.sort(key=lambda x: x["start"])
    deduped: list[dict] = _deduplicate_segments(all_segments)

    primary_lang: str | None = lang_info.get("primary_language") if lang_info else None
    final_segments: list[dict] = filter_hallucinations(deduped, primary_lang)
    return {
        "segments": final_segments,
        "full_text": " ".join(s["text"] for s in final_segments),
        "total_time_sec": time.time() - t_start,
        "method": f"parallel_{len(chunks_args)}_chunks",
    }


# ---------------------------------------------------------------------------
# Single-call transcription
# ---------------------------------------------------------------------------


def transcribe_full_video(
    audio: np.ndarray,
    sr: int,
    model_size: str = "small",
    use_vad: bool = True,
    force_cpu: bool = False,
    debug: bool = False,
    silero_model: Any | None = None,
    get_speech_ts_fn: Callable[..., list[dict[str, int]]] | None = None,
) -> dict:
    """Transcribe an entire audio track in a single Whisper call.

    Applies aggressive noise reduction (and optionally VAD-guided
    cleaning) before feeding the audio to a locally loaded Whisper
    model.

    Args:
        audio: 1-D float NumPy array of the full audio signal.
        sr: Sampling rate in Hz.
        model_size: Whisper model size string.
        use_vad: Apply VAD-guided cleaning via :func:`clean_audio`.
        force_cpu: Force CPU-only inference.
        debug: Enable diagnostic output (currently unused; reserved).
        silero_model: Pre-loaded Silero VAD model (forwarded to
            :func:`clean_audio`).
        get_speech_ts_fn: Silero ``get_speech_timestamps`` callable
            (forwarded to :func:`clean_audio`).

    Returns:
        A dict with keys ``"result"`` (raw Whisper result dict),
        ``"segments"``, ``"full_text"``, ``"total_time_sec"``, and
        ``"method"`` (always ``"single_call"``).
    """
    t_start: float = time.time()
    device: str | None = "cpu" if force_cpu else None
    model = whisper.load_model(model_size, device=device)
    cleaned: np.ndarray = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.95)
    if use_vad:
        cleaned = clean_audio(cleaned, sr, silero_model, get_speech_ts_fn)
    result: dict = model.transcribe(cleaned, fp16=False, verbose=None)
    t_end: float = time.time()
    return {
        "result": result,
        "segments": result.get("segments", []),
        "full_text": result.get("text", ""),
        "total_time_sec": t_end - t_start,
        "method": "single_call",
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def extract_speech_singlecall(
    scenes: list[dict],
    scan_result: dict,
    model_size: str = "small",
    use_vad: bool = True,
    language: str | None = None,
    parallel: bool = False,
    use_api: bool = True,
    force_cpu: bool = False,
    debug: bool = False,
) -> tuple[list[dict], dict]:
    """Main entry point: scan audio, transcribe, and map speech to scenes.

    Inspects *scan_result* to determine whether speech is present.  If so,
    selects parallel or single-call transcription based on duration and
    the *parallel* flag, runs transcription, and maps the resulting
    segments back to *scenes*.

    Args:
        scenes: List of scene dicts (must contain ``"start_seconds"``
            and ``"end_seconds"`` keys).  Each scene will have an
            ``"audio_speech"`` key added in-place.
        scan_result: Dict returned by the audio scanner, containing at
            minimum ``"has_speech"`` (bool), ``"audio"`` (np.ndarray),
            ``"sr"`` (int), and optionally ``"lang_info"`` (dict).
        model_size: Whisper model size string (e.g. ``"small"``).
        use_vad: Enable VAD-guided noise reduction.
        language: Force a specific ISO-639-1 language code.  When
            ``None`` the language is auto-detected or taken from
            *scan_result*.
        parallel: Force parallel chunked transcription regardless of
            duration.
        use_api: Prefer the Azure OpenAI Whisper API.
        force_cpu: Force CPU-only inference for local models.
        debug: Enable verbose diagnostic printing.

    Returns:
        A 2-tuple ``(scenes, stats)`` where *scenes* is the (mutated)
        input list with ``"audio_speech"`` populated, and *stats* is a
        dict of timing / count metrics.
    """
    t_start: float = time.time()

    if not scan_result["has_speech"]:
        if debug:
            print_prefixed("(Whisper)", "No speech detected. Skipping.")
        for scene in scenes:
            scene["audio_speech"] = ""
        return scenes, {
            "method": "singlecall_skipped",
            "total_time_sec": time.time() - t_start,
            "whisper_time_sec": 0,
            "segments_found": 0,
            "scenes_with_speech": 0,
        }

    audio: np.ndarray = scan_result["audio"]
    sr: int = scan_result["sr"]
    duration: float = len(audio) / sr
    should_parallel: bool = parallel or (duration > 900)

    if should_parallel:
        lang_data: dict | None = (
            {"primary_language": language, "is_multilingual": False}
            if language
            else scan_result.get("lang_info")
        )
        whisper_result: dict = transcribe_parallel(
            audio,
            sr,
            model_size=model_size,
            lang_info=lang_data,
            use_vad=use_vad,
            force_cpu=force_cpu,
            debug=debug,
            use_api=use_api,
        )
    else:
        from kairos.audio.vad import _get_silero_vad

        silero_model, get_ts_fn = _get_silero_vad()
        whisper_result = transcribe_full_video(
            audio,
            sr,
            model_size=model_size,
            use_vad=use_vad,
            force_cpu=force_cpu,
            debug=debug,
            silero_model=silero_model,
            get_speech_ts_fn=get_ts_fn,
        )

    scene_texts: list[str] = map_segments_to_scenes(
        whisper_result["segments"], scenes
    )
    for i, scene in enumerate(scenes):
        scene["audio_speech"] = scene_texts[i]

    if debug:
        scenes_with_speech: int = sum(1 for t in scene_texts if t.strip())
        print_prefixed(
            "(Whisper)", f"Mapped speech to {scenes_with_speech}/{len(scenes)} scenes"
        )

    return scenes, {
        "method": whisper_result.get("method", "unknown"),
        "total_time_sec": time.time() - t_start,
        "whisper_time_sec": whisper_result.get(
            "transcribe_time_sec", time.time() - t_start
        ),
        "segments_found": len(whisper_result["segments"]),
        "scenes_with_speech": sum(1 for t in scene_texts if t.strip()),
    }
