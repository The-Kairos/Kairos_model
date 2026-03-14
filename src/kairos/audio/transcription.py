"""Whisper-based speech transcription: parallel chunking, scene mapping, and orchestration."""

import concurrent.futures
import gc
import os
import time

import noisereduce as nr
import numpy as np
import torch
import whisper

from kairos.audio.whisper_api import transcribe_via_api
from kairos.audio.text_filter import filter_hallucinations


# Audio preprocessing

def clean_audio(audio: np.ndarray, sr: int, silero_model=None, get_speech_ts=None) -> np.ndarray:
    audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.9)
    if silero_model is not None and get_speech_ts is not None:
        audio_t = torch.from_numpy(audio).float()
        speech_ts = get_speech_ts(audio_t, silero_model, sampling_rate=sr)
        if len(speech_ts) > 0:
            enhanced = audio.copy()
            for seg in speech_ts:
                segment = enhanced[seg["start"]:seg["end"]]
                enhanced[seg["start"]:seg["end"]] = nr.reduce_noise(y=segment, sr=sr, prop_decrease=0.7)
            audio = enhanced
    return audio


# Timestamp-to-scene mapping

def map_segments_to_scenes(whisper_segments: list, scenes: list) -> list:
    scene_texts = []
    for scene in scenes:
        t0, t1 = float(scene["start_seconds"]), float(scene["end_seconds"])
        parts = []
        for seg in whisper_segments:
            seg_start, seg_end = float(seg["start"]), float(seg["end"])
            if seg_end <= t0 or seg_start >= t1:
                continue
            overlap = min(t1, seg_end) - max(t0, seg_start)
            seg_duration = max(seg_end - seg_start, 1e-6)
            if overlap / seg_duration >= 0.2 or overlap >= 0.5:
                parts.append(seg["text"])
        scene_texts.append(" ".join(parts).strip())
    return scene_texts


# Parallel chunking

def _transcribe_via_api_with_retry(chunk_audio, sr, language, client, debug):
    """Try the Whisper API up to 3 times with rate-limit backoff."""
    for attempt in range(3):
        try:
            return transcribe_via_api(chunk_audio, sr, language=language, client=client)
        except Exception as e:
            if ("429" in str(e) or "RateLimitReached" in str(e)) and attempt < 2:
                time.sleep(65)
                continue
            if debug:
                print(f"[WhisperWorker] API Error: {e}")
            return []
    return []


def _transcribe_via_local_model(chunk_audio, model_size, force_cpu, language):
    """Transcribe using a locally loaded Whisper model."""
    device = "cpu" if force_cpu else None
    model = whisper.load_model(model_size, device=device)
    result = model.transcribe(chunk_audio, fp16=False, verbose=None, language=language)
    segments = result.get("segments", [])
    del model
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return segments


def _transcribe_chunk_worker(args):
    (chunk_audio, sr, model_size, chunk_start_time, use_vad, force_cpu, debug, language, use_api, client) = args
    if use_vad:
        chunk_audio = nr.reduce_noise(y=chunk_audio, sr=sr, prop_decrease=0.9)

    if use_api:
        segments = _transcribe_via_api_with_retry(chunk_audio, sr, language, client, debug)
        if not segments:
            try:
                segments = _transcribe_via_local_model(chunk_audio, model_size, force_cpu, language)
            except Exception:
                return []
    else:
        segments = _transcribe_via_local_model(chunk_audio, model_size, force_cpu, language)

    for seg in segments:
        seg["start"] += chunk_start_time
        seg["end"] += chunk_start_time
    return segments


def transcribe_parallel(audio, sr, model_size="medium", chunk_size_sec=600, overlap_sec=30,
                        lang_info=None, use_vad=True, force_cpu=False, debug=False,
                        use_api=True, client=None) -> dict:
    duration = len(audio) / sr
    t_start = time.time()
    force_lang = None
    if lang_info and not lang_info.get("is_multilingual", False):
        force_lang = lang_info.get("primary_language")
    chunks_args = []
    start = 0
    while start < duration:
        end = min(start + chunk_size_sec + overlap_sec, duration)
        chunks_args.append((
            audio[int(start * sr):int(end * sr)].copy(), sr, model_size, start,
            use_vad, force_cpu, debug, force_lang, use_api, client,
        ))
        if end >= duration:
            break
        start += chunk_size_sec

    max_rec_workers = 4 if use_api else 2
    num_workers = min(len(chunks_args), os.cpu_count() or 4, max_rec_workers)
    all_segments = []
    Executor = concurrent.futures.ThreadPoolExecutor if use_api else concurrent.futures.ProcessPoolExecutor
    with Executor(max_workers=num_workers) as executor:
        futures = [executor.submit(_transcribe_chunk_worker, arg) for arg in chunks_args]
        for future in concurrent.futures.as_completed(futures):
            all_segments.extend(future.result())
            gc.collect()

    all_segments.sort(key=lambda x: x["start"])
    deduped = []
    if all_segments:
        deduped.append(all_segments[0])
        for curr in all_segments[1:]:
            prev = deduped[-1]
            time_diff = abs(curr["start"] - prev["start"])
            text_match = curr["text"].strip().lower() == prev["text"].strip().lower()
            if time_diff < 0.5 and text_match:
                continue
            if time_diff < 1.0 and (curr["text"].strip() in prev["text"].strip() or prev["text"].strip() in curr["text"].strip()):
                if len(curr["text"]) > len(prev["text"]):
                    deduped[-1] = curr
                continue
            deduped.append(curr)

    primary_lang = lang_info.get("primary_language") if lang_info else None
    final_segments = filter_hallucinations(deduped, primary_lang)
    return {
        "segments": final_segments,
        "full_text": " ".join(s["text"] for s in final_segments),
        "total_time_sec": time.time() - t_start,
        "method": f"parallel_{len(chunks_args)}_chunks",
    }


# Single-call transcription

def transcribe_full_video(audio, sr, model_size="small", use_vad=True, force_cpu=False,
                          debug=False, silero_model=None, get_speech_ts_fn=None) -> dict:
    t_start = time.time()
    device = "cpu" if force_cpu else None
    model = whisper.load_model(model_size, device=device)
    cleaned = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.95)
    if use_vad:
        cleaned = clean_audio(cleaned, sr, silero_model, get_speech_ts_fn)
    result = model.transcribe(cleaned, fp16=False, verbose=None)
    t_end = time.time()
    return {
        "result": result, "segments": result.get("segments", []),
        "full_text": result.get("text", ""),
        "total_time_sec": t_end - t_start, "method": "single_call",
    }


# Main entry point

def extract_speech_singlecall(
    scenes: list, scan_result: dict, model_size: str = "small",
    use_vad: bool = True, language: str = None, parallel: bool = False,
    use_api: bool = True, force_cpu: bool = False, debug: bool = False,
) -> tuple:
    """Check scan_result and run single-call or parallel Whisper."""
    t_start = time.time()

    if not scan_result["has_speech"]:
        if debug:
            print("[Whisper] No speech detected. Skipping.")
        for scene in scenes:
            scene["audio_speech"] = ""
        return scenes, {
            "method": "singlecall_skipped", "total_time_sec": time.time() - t_start,
            "whisper_time_sec": 0, "segments_found": 0, "scenes_with_speech": 0,
        }

    audio, sr = scan_result["audio"], scan_result["sr"]
    duration = len(audio) / sr
    should_parallel = parallel or (duration > 900)

    if should_parallel:
        lang_data = {"primary_language": language, "is_multilingual": False} if language else scan_result.get("lang_info")
        whisper_result = transcribe_parallel(audio, sr, model_size=model_size, lang_info=lang_data,
                                             use_vad=use_vad, force_cpu=force_cpu, debug=debug, use_api=use_api)
    else:
        from kairos.audio.vad import _get_silero_vad
        silero_model, get_ts_fn = _get_silero_vad()
        whisper_result = transcribe_full_video(audio, sr, model_size=model_size, use_vad=use_vad,
                                               force_cpu=force_cpu, debug=debug,
                                               silero_model=silero_model, get_speech_ts_fn=get_ts_fn)

    scene_texts = map_segments_to_scenes(whisper_result["segments"], scenes)
    for i, scene in enumerate(scenes):
        scene["audio_speech"] = scene_texts[i]

    if debug:
        scenes_with_speech = sum(1 for t in scene_texts if t.strip())
        print(f"[Whisper] Mapped speech to {scenes_with_speech}/{len(scenes)} scenes")

    return scenes, {
        "method": whisper_result.get("method", "unknown"),
        "total_time_sec": time.time() - t_start,
        "whisper_time_sec": whisper_result.get("transcribe_time_sec", time.time() - t_start),
        "segments_found": len(whisper_result["segments"]),
        "scenes_with_speech": sum(1 for t in scene_texts if t.strip()),
    }
