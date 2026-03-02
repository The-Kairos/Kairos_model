"""
whisper_singlecall.py — Single-Call Whisper Transcription with Scene Mapping

Instead of calling model.transcribe() per scene (N calls),
we call it ONCE on the full audio and map segments back to scenes
using timestamp overlap.
"""

import time
import whisper
import numpy as np
import noisereduce as nr
import torch


# =========================================================
# 1. Audio Preprocessing (reuses logic from src/audio_speech.py)
# =========================================================

def clean_audio(audio: np.ndarray, sr: int, silero_model=None, get_speech_ts=None) -> np.ndarray:
    """
    Clean audio: full-track noise reduction + optional soft VAD enhancement.
    This is the same preprocessing as src/audio_speech.py.
    """
    # Full-track noise reduction
    audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.9)

    # Soft VAD enhancement (if model available)
    if silero_model is not None and get_speech_ts is not None:
        audio_t = torch.from_numpy(audio).float()
        speech_ts = get_speech_ts(audio_t, silero_model, sampling_rate=sr)

        if len(speech_ts) > 0:
            enhanced = audio.copy()
            for seg in speech_ts:
                start, end = seg["start"], seg["end"]
                segment = enhanced[start:end]
                enhanced[start:end] = nr.reduce_noise(
                    y=segment, sr=sr, prop_decrease=0.7
                )
            audio = enhanced

    return audio


# =========================================================
# 2. Timestamp-to-Scene Mapping
# =========================================================

def map_segments_to_scenes(whisper_segments: list, scenes: list) -> list:
    """
    Map Whisper segments to scenes using timestamp overlap.

    For each scene [t0, t1], collects all Whisper segments that overlap.
    A segment is assigned to the scene(s) it temporally overlaps with.
    If a segment spans a boundary, the scene that contains the majority
    (>50%) of the segment gets it; the minority scene gets it only if the
    overlap is at least 0.5s (avoids tiny boundary duplicates).

    Returns list of strings (one per scene).
    """
    scene_texts = []

    for scene in scenes:
        t0 = float(scene["start_seconds"])
        t1 = float(scene["end_seconds"])

        parts = []
        for seg in whisper_segments:
            seg_start = float(seg["start"])
            seg_end   = float(seg["end"])

            # No overlap at all
            if seg_end <= t0 or seg_start >= t1:
                continue

            overlap = min(t1, seg_end) - max(t0, seg_start)
            seg_duration = max(seg_end - seg_start, 1e-6)
            overlap_ratio = overlap / seg_duration

            # Include if this scene contains the majority of the segment,
            # OR the segment is very short (entirely within a reasonable margin)
            if overlap_ratio >= 0.5 or overlap >= 0.5:
                parts.append(seg["text"])

        scene_texts.append(" ".join(parts).strip())

    return scene_texts


# =========================================================
# 3. Single-Call Transcription
# =========================================================

def transcribe_full_video(audio: np.ndarray, sr: int, model_size: str = "small",
                          use_vad: bool = True, debug: bool = False,
                          silero_model=None, get_speech_ts_fn=None) -> dict:
    """
    Run Whisper ONCE on the entire video audio.

    Accepts optional pre-loaded Silero model to avoid reloading.
    Returns the full Whisper result dict with 'segments' containing
    per-segment timestamps.
    """
    t_start = time.time()

    if debug:
        print(f"[WhisperSingle] Loading Whisper model: {model_size}")

    model = whisper.load_model(model_size)

    if debug:
        print(f"[WhisperSingle] Cleaning audio ({len(audio)} samples)...")

    # Only load Silero if not already provided
    if use_vad and silero_model is None:
        _sil, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        silero_model = _sil
        get_speech_ts_fn = utils[0]
    elif not use_vad:
        silero_model = None
        get_speech_ts_fn = None

    cleaned = clean_audio(audio, sr, silero_model, get_speech_ts_fn)

    t_clean = time.time()
    if debug:
        print(f"[WhisperSingle] Audio cleaned in {t_clean - t_start:.2f}s")
        print(f"[WhisperSingle] Transcribing full audio ({len(cleaned) / sr:.1f}s)...")

    # Single transcription call — verbose=None suppresses segment-by-segment output
    result = model.transcribe(cleaned, fp16=False, verbose=None)

    t_transcribe = time.time()
    if debug:
        n_segments = len(result.get("segments", []))
        print(f"[WhisperSingle] Transcription complete: {n_segments} segments in {t_transcribe - t_clean:.2f}s")
        print(f"[WhisperSingle] Total time: {t_transcribe - t_start:.2f}s")

    return {
        "result": result,
        "segments": result.get("segments", []),
        "full_text": result.get("text", ""),
        "clean_time_sec": t_clean - t_start,
        "transcribe_time_sec": t_transcribe - t_clean,
        "total_time_sec": t_transcribe - t_start,
    }


# =========================================================
# 4. Main Entry Point
# =========================================================

def extract_speech_singlecall(scenes: list, scan_result: dict,
                              model_size: str = "small", use_vad: bool = True,
                              debug: bool = False) -> tuple:
    """
    Main entry point. Checks scan_result, runs single-call Whisper if
    speech detected, maps segments to scenes.

    Args:
        scenes: list of scene dicts with start_seconds/end_seconds
        scan_result: output from audio_detector.scan_audio()
        model_size: Whisper model size
        use_vad: whether to use VAD for audio cleaning
        debug: print progress

    Returns:
        (scenes, timing_info) where each scene gains "audio_speech" key
    """
    t_start = time.time()

    if not scan_result["has_speech"]:
        # No speech detected — fill all scenes with empty string
        if debug:
            print(f"[WhisperSingle] No speech detected. Skipping transcription for all {len(scenes)} scenes.")
        for scene in scenes:
            scene["audio_speech"] = ""
        elapsed = time.time() - t_start
        return scenes, {
            "method": "singlecall_skipped",
            "total_time_sec": elapsed,
            "whisper_time_sec": 0,
            "segments_found": 0,
            "scenes_with_speech": 0,
        }

    # Run single-call transcription — reuse Silero from the pre-scan
    audio = scan_result["audio"]
    sr = scan_result["sr"]

    from audio_singlecall.audio_detector import _silero_model, _utils as _silero_utils
    _get_ts = _silero_utils[0]

    whisper_result = transcribe_full_video(
        audio, sr, model_size=model_size, use_vad=use_vad, debug=debug,
        silero_model=_silero_model, get_speech_ts_fn=_get_ts,
    )

    # Map segments to scenes
    scene_texts = map_segments_to_scenes(whisper_result["segments"], scenes)

    for i, scene in enumerate(scenes):
        scene["audio_speech"] = scene_texts[i]

    if debug:
        scenes_with_speech = sum(1 for t in scene_texts if t.strip())
        print(f"[WhisperSingle] Mapped speech to {scenes_with_speech}/{len(scenes)} scenes")

    elapsed = time.time() - t_start

    return scenes, {
        "method": "singlecall",
        "total_time_sec": elapsed,
        "whisper_time_sec": whisper_result["transcribe_time_sec"],
        "clean_time_sec": whisper_result["clean_time_sec"],
        "segments_found": len(whisper_result["segments"]),
        "scenes_with_speech": sum(1 for t in scene_texts if t.strip()),
    }
