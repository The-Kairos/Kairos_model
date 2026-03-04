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
import gc
import os
import re

import tempfile
import scipy.io.wavfile as wav
from openai import AzureOpenAI
from dotenv import load_dotenv
from pathlib import Path

# Load credentials from .env in project root
load_dotenv(Path(__file__).parent.parent / ".env")


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
# 2b. Azure OpenAI Client (New)
# =========================================================

AZURE_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

if AZURE_KEY and AZURE_ENDPOINT:
    # Handle the case where the user provided the full deployment URL instead of base endpoint
    # AzureOpenAI client expects the base endpoint (https://xxx.openai.azure.com/)
    base_endpoint = AZURE_ENDPOINT.split("/openai")[0]
    
    _azure_client = AzureOpenAI(
        api_key=AZURE_KEY,
        azure_endpoint=base_endpoint,
        api_version=AZURE_API_VERSION
    )
else:
    _azure_client = None

def transcribe_via_api(audio_np: np.ndarray, sr: int, language: str = None) -> list:
    """
    Save chunk to temp wav and call Azure Whisper API.
    Returns list of segments.
    """
    if _azure_client is None:
        raise ValueError("Azure OpenAI credentials not found in environment.")

    # Save to temporary WAV file (Whisper API requirement)
    # Use a high-quality format (16kHz, mono)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        wav.write(tmp.name, sr, (audio_np * 32767).astype(np.int16))
        
        with open(tmp.name, "rb") as audio_file:
            # Azure OpenAI audio transcriptions call
            response = _azure_client.audio.transcriptions.create(
                model=AZURE_DEPLOYMENT,
                file=audio_file,
                language=language,
                response_format="verbose_json"
            )
            
    # The API returns a 'segments' field in verbose_json mode
    segments = getattr(response, "segments", [])
    # Convert to list of dicts if it's a list of objects
    if segments and not isinstance(segments[0], dict):
        segments = [
            {
                "start": float(s.start),
                "end": float(s.end),
                "text": str(s.text),
                "avg_logprob": getattr(s, "avg_logprob", 0),
                "no_speech_prob": getattr(s, "no_speech_prob", 0)
            }
            for s in segments
        ]
    return segments

# =========================================================
# 3. Parallel Dynamic Chunking (New)
# =========================================================

def _transcribe_chunk_worker(args):
    """
    Worker function for ProcessPoolExecutor.
    Transcribes a single chunk of audio (Local or API).
    """
    chunk_audio, sr, model_size, chunk_start_time, use_vad, force_cpu, debug, language, use_api = args
    
    # Optional VAD cleaning per chunk
    if use_vad:
        chunk_audio = nr.reduce_noise(y=chunk_audio, sr=sr, prop_decrease=0.9)
    
    if use_api:
        # Use Azure OpenAI API with automatic local fallback
        segments = []
        api_success = False
        max_retries = 3
        for attempt in range(max_retries):
            try:
                segments = transcribe_via_api(chunk_audio, sr, language=language)
                api_success = True
                break
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RateLimitReached" in error_str:
                    if attempt < max_retries - 1:
                        if debug: print(f"[WhisperWorker] Rate limit hit. Retrying in 65s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(65)
                        continue
                if debug: print(f"[WhisperWorker] API Error: {e}")
                break  # exit retry loop, fall through to local fallback

        if not api_success:
            # Fallback to local Whisper model (cached on disk after first use)
            if debug: print(f"[WhisperWorker] API exhausted. Falling back to local Whisper ({model_size})...")
            try:
                device = "cpu" if force_cpu else None
                model = whisper.load_model(model_size, device=device)
                result = model.transcribe(chunk_audio, fp16=False, verbose=None, language=language)
                segments = result.get("segments", [])
                del model
                gc.collect()
            except Exception as e2:
                if debug: print(f"[WhisperWorker] Local fallback also failed: {e2}")
                return []

    else:
        # Use Local Whisper
        device = "cpu" if force_cpu else None
        model = whisper.load_model(model_size, device=device)
        
        result = model.transcribe(chunk_audio, fp16=False, verbose=None, language=language)
        segments = result.get("segments", [])
        
        # Explicit cleanup
        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    for seg in segments:
        seg["start"] += chunk_start_time
        seg["end"] += chunk_start_time
        
    if not use_api:
        # Explicit cleanup for local model
        try:
            del model
        except:
            pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
    return segments


# =========================================================
# 3b. Hallucination & Noise Filtering (New)
# =========================================================

def clean_repetitive_text(text: str) -> str:
    """Removes back-to-back repetitions of words or phrases, common in Whisper hallucinations."""
    if not text:
        return text
        
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 1. Phrase level deduplication (e.g. "Bye. Bye. Bye.")
    phrases = re.split(r'([.?!,]+)', text)
    cleaned_phrases = []
    last_p = None
    
    i = 0
    while i < len(phrases):
        p = phrases[i]
        punct = phrases[i+1] if i+1 < len(phrases) else ""
        p_norm = p.strip().lower()
        
        if p_norm:
            if p_norm == last_p:
                if punct and cleaned_phrases and not re.search(r'[.?!,]$', cleaned_phrases[-1]):
                     cleaned_phrases[-1] = cleaned_phrases[-1].rstrip() + punct
            else:
                cleaned_phrases.append(p.strip() + punct)
                last_p = p_norm
        i += 2
        
    text = " ".join(cleaned_phrases).strip()
    
    # 2. Word level deduplication (e.g. "Bye bye bye")
    words = text.split()
    if not words: 
        return text
    
    cleaned_words = [words[0]]
    for w in words[1:]:
        w_norm = w.lower().strip(".,!?")
        last_norm = cleaned_words[-1].lower().strip(".,!?")
        
        if w_norm == last_norm and len(w_norm) > 0:
            if re.search(r'[.,!?]$', w):
                punct = re.search(r'[.,!?]+$', w).group()
                cleaned_words[-1] = cleaned_words[-1].rstrip(".,!?") + punct
        else:
            cleaned_words.append(w)
            
    return " ".join(cleaned_words)


def filter_hallucinations(segments: list, primary_lang: str = None) -> list:
    """
    Remove common Whisper hallucinations:
    - Melodic/Noise patterns (e.g. "Ə Ɓ Ɵ ƙ", "Ơ Ơ Ơ")
    - Extremely low logprobs
    - Repeated identical phrases (loops)
    """
    final = []
    seen_texts = set()
    
    # Common "noise characters" when Whisper trips over music
    NOISE_CHARS = set("ƏƁƟƙƒƠƣƜơ")

    for seg in segments:
        text = seg["text"].strip()
        
        # Strip out emojis, musical notes, and symbols Whisper injects during background noise
        import emoji
        text = emoji.replace_emoji(text, replace="")
        # Also strip specifically common symbols missed by some emoji libraries
        text = re.sub(r'[♪♫♥️✿❀❁]+', '', text).strip()
        
        if not text:
            continue
            
        # 1. Filter out music/noise "garbage" characters
        # If > 15% of characters are noise-weirdness, it's a hallucination
        special_count = sum(1 for c in text if c in NOISE_CHARS)
        if len(text) > 0 and special_count / len(text) > 0.15:
            continue

        # 2. Logprob check (Whisper confidence)
        # avg_logprob: Closer to 0 is better. -0.9 is stricter for noisy environments.
        if seg.get("avg_logprob", 0) < -0.9:
            continue
            
        # 3. No speech probability (Silero VAD handles most, but API adds safety)
        if seg.get("no_speech_prob", 0) > 0.6:
             continue

        # 4. Clean intra-segment repetitions ("Bye. Bye. Bye.")
        text = clean_repetitive_text(text)
        text_lower = text.lower().strip(".,!? ")

        if not text_lower:
            continue

        # 5. Filter Loop hallucinations (same sentence over and over across multiple segments)
        if text_lower in seen_texts:
            continue
        
        if len(text_lower) > 2: # Don't track single letters or blank space
            seen_texts.add(text_lower)
            
        seg["text"] = text # Keep the cleaned version
        final.append(seg)

        
    return final


def transcribe_parallel(audio: np.ndarray, sr: int, model_size: str = "medium",
                       chunk_size_sec: int = 600, overlap_sec: int = 30,
                       lang_info: dict = None, 
                       use_vad: bool = True, force_cpu: bool = False, 
                       debug: bool = False, use_api: bool = True) -> dict:
    """
    Split audio into chunks and transcribe in parallel.
    Handles Single-Language Lock vs Multi-Language Flexibility.
    """
    import concurrent.futures
    import os
    
    duration = len(audio) / sr
    t_start = time.time()
    
    # 1. Decide Language Strategy
    # For the Azure API, we ALWAYS pass language=None so Whisper auto-detects the language
    # natively per chunk. Locking to a language (e.g., 'en') forces Whisper to TRANSLATE,
    # which is wrong for bilingual videos like mixed Arabic/English content.
    # The downstream LLM (GPT-4o) handles multilingual text just fine.
    #
    # For local Whisper only, we use the language lock as a lightweight hallucination guard.
    force_lang = None
    if not use_api and lang_info and not lang_info.get("is_multilingual", False):
        force_lang = lang_info.get("primary_language")
        if debug: print(f"[WhisperParallel] (Local mode) Locking language to: {force_lang}")
    elif debug:
        print("[WhisperParallel] Auto-detecting language per chunk (no global lock).")


    # 2. Create chunks
    chunks_args = []
    start = 0
    while start < duration:
        end = min(start + chunk_size_sec + overlap_sec, duration)
        chunk_samples = audio[int(start * sr) : int(end * sr)].copy()
        # Pass the locked language to all workers
        chunks_args.append((chunk_samples, sr, model_size, start, use_vad, force_cpu, debug, force_lang, use_api))
        
        if end >= duration:
            break
        start += chunk_size_sec
        
    if debug:
        print(f"[WhisperParallel] Split {duration:.1f}s audio into {len(chunks_args)} chunks ({chunk_size_sec}s each)")

    # 2. Run in parallel
    # If using API, we can use more workers as it's not bound by local VRAM/CPU
    # But Azure usually has RPM/TPM limits. Let's stick to 4 workers for API.
    max_rec_workers = 4 if use_api else 2
    num_workers = min(len(chunks_args), os.cpu_count() or 4, max_rec_workers)
    all_segments = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_transcribe_chunk_worker, arg) for arg in chunks_args]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            segments = future.result()
            all_segments.extend(segments)
            if debug:
                print(f"[WhisperParallel] Chunk {i+1}/{len(chunks_args)} completed")
            # Clear memory after each future completes
            gc.collect()

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
            
    # 3. Apply Hallucination & Noise Filtering
    primary_lang = lang_info.get("primary_language") if lang_info else None
    final_segments = filter_hallucinations(deduped, primary_lang)
    
    if debug:
        removed = len(deduped) - len(final_segments)
        if removed > 0:
            print(f"[WhisperParallel] Filtered out {removed} hallucinations/low-quality segments")

    t_end = time.time()
    
    return {
        "segments": final_segments,
        "full_text": " ".join([s["text"] for s in final_segments]),
        "total_time_sec": t_end - t_start,
        "method": f"parallel_{len(chunks_args)}_chunks"
    }


# =========================================================
# 4. Single-Call Transcription
# =========================================================

def transcribe_full_video(audio: np.ndarray, sr: int, model_size: str = "small",
                          use_vad: bool = True, force_cpu: bool = False, debug: bool = False,
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

    device = "cpu" if force_cpu else None
    model = whisper.load_model(model_size, device=device)

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

    # Multi-pass noise reduction for ASR clarity
    cleaned = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.95) # High reduction
    
    if use_vad:
        cleaned = clean_audio(cleaned, sr, silero_model, get_speech_ts_fn)

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
                                language: str = None, # Added language override
                                parallel: bool = False, use_api: bool = True,
                                force_cpu: bool = False, debug: bool = False) -> tuple:
    """
    Main entry point. Checks scan_result, runs single-call or parallel Whisper.
    
    Args:
        language: ISO code (e.g. 'en', 'ar'). If None, performs one-time Global Detection.
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
        # If manual language is provided, use it. Otherwise use the smart lang_info from the scan.
        lang_data = {"primary_language": language, "is_multilingual": False} if language else scan_result.get("lang_info")
        
        whisper_result = transcribe_parallel(
            audio, sr, model_size=model_size, 
            lang_info=lang_data, 
            use_vad=use_vad, force_cpu=force_cpu, debug=debug,
            use_api=use_api
        )
    else:
        if debug: print(f"[WhisperSingle] Using Single-Call Transcription (Duration: {duration:.1f}s)")
        from audio_singlecall.audio_detector import _silero_model, _utils as _silero_utils
        _get_ts = _silero_utils[0] if _silero_utils else None

        whisper_result = transcribe_full_video(
            audio, sr, model_size=model_size, 
            use_vad=use_vad, force_cpu=force_cpu, debug=debug,
            silero_model=scan_result.get("silero_model"),
            get_speech_ts_fn=scan_result.get("get_speech_ts")
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
