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
# 3. Parallel Dynamic Chunking (New)
# =========================================================

def _transcribe_chunk_worker(args):
    """
    Worker function for ProcessPoolExecutor.
    Transcribes a single chunk of audio.
    """
    chunk_audio, sr, model_size, chunk_start_time, use_vad, debug = args
    
    # Reload model in each process if using ProcessPoolExecutor
    # (standard Whisper model isn't easily shared across processes without IPC)
    model = whisper.load_model(model_size)
    
    # Optional VAD cleaning per chunk
    if use_vad:
        # We don't want to reload Silero in every worker if possible, 
        # but for simplicity in CPU-parallel mode, we load it or skip 
        # the soft-VAD and just do noise reduction.
        # Let's do just noise reduction here to save memory/load time.
        chunk_audio = nr.reduce_noise(y=chunk_audio, sr=sr, prop_decrease=0.9)
    
    result = model.transcribe(chunk_audio, fp16=False, verbose=None)
    
    # Offset segments by chunk start time
    segments = result.get("segments", [])
    for seg in segments:
        seg["start"] += chunk_start_time
        seg["end"] += chunk_start_time
        
    return segments


def transcribe_parallel(audio: np.ndarray, sr: int, model_size: str = "small",
                       chunk_size_sec: int = 600, overlap_sec: int = 30,
                       use_vad: bool = True, debug: bool = False) -> dict:
    """
    Split audio into chunks and transcribe in parallel using ProcessPoolExecutor.
    """
    import concurrent.futures
    import os
    
    duration = len(audio) / sr
    t_start = time.time()
    
    # 1. Create chunks
    chunks_args = []
    start = 0
    while start < duration:
        end = min(start + chunk_size_sec + overlap_sec, duration)
        chunk_samples = audio[int(start * sr) : int(end * sr)]
        chunks_args.append((chunk_samples, sr, model_size, start, use_vad, debug))
        
        if end >= duration:
            break
        start += chunk_size_sec # Move forward by chunk size, keeping the overlap
        
    if debug:
        print(f"[WhisperParallel] Split {duration:.1f}s audio into {len(chunks_args)} chunks ({chunk_size_sec}s each)")

    # 2. Run in parallel
    num_workers = min(len(chunks_args), os.cpu_count() or 4)
    all_segments = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_transcribe_chunk_worker, arg) for arg in chunks_args]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            segments = future.result()
            all_segments.extend(segments)
            if debug:
                print(f"[WhisperParallel] Chunk {i+1}/{len(chunks_args)} completed")

    # 3. Deduplicate segments in overlaps
    # Rule: Each chunk "owns" a specific time range (chunk_size_sec).
    # We keep a segment if its midpoint falls within the range owned by the chunk it came from.
    # This naturally deduplicates segments in the overlap areas without checking text.
    final_segments = []
    
    # But wait, we didn't track which chunk each segment came from in the worker return.
    # Let's use a simpler unique-by-start-time-and-text approach for now, 
    # or just keep the best fit.
    
    # Improved Deduplication: Sort by start time.
    # If two segments are very close in time and text, they are likely duplicates from the overlap.
    all_segments.sort(key=lambda x: x["start"])
    deduped = []
    if all_segments:
        deduped.append(all_segments[0])
        for i in range(1, len(all_segments)):
            prev = deduped[-1]
            curr = all_segments[i]
            
            # If they start within 0.5s of each other and have similar text, they are duplicates
            time_diff = abs(curr["start"] - prev["start"])
            text_match = (curr["text"].strip().lower() == prev["text"].strip().lower())
            
            if time_diff < 0.5 and text_match:
                continue
                
            # If one is a complete substring of the other and they overlap heavily
            if time_diff < 1.0 and (curr["text"].strip() in prev["text"].strip() or prev["text"].strip() in curr["text"].strip()):
                # Keep the longer one
                if len(curr["text"]) > len(prev["text"]):
                    deduped[-1] = curr
                continue

            deduped.append(curr)
            
    t_end = time.time()
    
    return {
        "segments": deduped,
        "full_text": " ".join([s["text"] for s in deduped]),
        "total_time_sec": t_end - t_start,
        "method": f"parallel_{len(chunks_args)}_chunks"
    }


# =========================================================
# 4. Single-Call Transcription
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
        try:
            _sil, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True
            )
            silero_model = _sil
            get_speech_ts_fn = utils[0]
        except Exception as e:
            if debug: print(f"[WARN] Silero VAD load failed: {e}. Skipping soft-VAD.")
            silero_model = None

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
        "method": "single_call"
    }


# =========================================================
# 5. Main Entry Point
# =========================================================

def extract_speech_singlecall(scenes: list, scan_result: dict,
                               model_size: str = "small", use_vad: bool = True,
                               parallel: bool = False, debug: bool = False) -> tuple:
    """
    Main entry point. Checks scan_result, runs single-call or parallel Whisper.
    
    Args:
        parallel: If True, uses chunked parallel transcription regardless of length.
                 If False, only uses parallel if video > 15 minutes.
    """
    t_start = time.time()

    if not scan_result["has_speech"]:
        if debug:
            print(f"[WhisperSingle] No speech detected. Skipping transcription.")
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

    audio = scan_result["audio"]
    sr = scan_result["sr"]
    duration = len(audio) / sr

    # Decision: Parallel vs Full
    should_parallel = parallel or (duration > 900) # 15 minutes

    if should_parallel:
        if debug: print(f"[WhisperSingle] Using Parallel Dynamic Chunking (Duration: {duration:.1f}s)")
        # In parallel mode, we don't pass the pre-loaded Silero model because 
        # it can't be easily shared across processes. Each process handles its own NR.
        whisper_result = transcribe_parallel(
            audio, sr, model_size=model_size, debug=debug, use_vad=use_vad
        )
    else:
        if debug: print(f"[WhisperSingle] Using Single-Call Transcription (Duration: {duration:.1f}s)")
        from audio_singlecall.audio_detector import _silero_model, _utils as _silero_utils
        _get_ts = _silero_utils[0] if _silero_utils else None

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
        "method": whisper_result.get("method", "unknown"),
        "total_time_sec": elapsed,
        "whisper_time_sec": whisper_result.get("transcribe_time_sec", elapsed),
        "segments_found": len(whisper_result["segments"]),
        "scenes_with_speech": sum(1 for t in scene_texts if t.strip()),
    }
