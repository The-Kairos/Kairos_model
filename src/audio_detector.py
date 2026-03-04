"""
audio_detector.py - 2-stage audio pre-scan with dynamic thresholds.

Scans the full video audio once to determine:
  1. Is there any audible audio? (RMS energy)
  2. Is there speech?            (Silero VAD)
  3. Is there background audio?  (Spectral flatness)
"""

import math
import time
import gc
import numpy as np
import torch
import librosa
import av
import whisper


_silero_model, _utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
)
_get_speech_ts = _utils[0]


# =========================================================
# 1. Dynamic Thresholds
# =========================================================

def get_sensitivity_multiplier(duration_minutes: float) -> float:
    """
    Log2-scaled multiplier: 1.0 (short) -> 1.5 (very long).
    Longer videos -> more sensitive detection.
    """
    dur = max(1.0, duration_minutes)
    multiplier = 1.0 + 0.1 * math.log2(dur)
    return min(multiplier, 1.5)


def get_dynamic_thresholds(duration_minutes: float) -> dict:
    """
    Compute all audio detection thresholds, scaled by video duration.
    """
    m = get_sensitivity_multiplier(duration_minutes)
    return {
        "SILENCE_THRESHOLD_DBFS":       -60.0 * m,
        "SCENE_SILENCE_DBFS":           -50.0 * m,
        "VAD_THRESHOLD":                0.3 / m,
        "MIN_SPEECH_DURATION_MS":       int(250 / m),
        "MIN_SILENCE_DURATION_MS":      300,
        "SPEECH_PAD_MS":                int(50 * m),
        "SPECTRAL_FLATNESS_THRESHOLD":  min(0.85 * m, 0.95),
        "sensitivity_multiplier":       round(m, 4),
    }


# =========================================================
# 2. Audio Extraction
# =========================================================

def load_audio_av(video_path: str, target_sr: int = 16000):
    """
    Extract full audio from video using PyAV. Returns (audio_np, sr).
    """
    try:
        container = av.open(
            video_path,
            options={"fflags": "+genpts", "ignore_editlist": "1"},
        )
        audio_stream = next(
            (s for s in container.streams if s.type == "audio"), None
        )
        if audio_stream is None:
            return np.zeros(1, dtype=np.float32), target_sr

        audio_stream.thread_type = "AUTO"
        samples = []
        for frame in container.decode(audio_stream):
            pcm = frame.to_ndarray()
            pcm = pcm.mean(axis=0)  # stereo -> mono
            samples.append(pcm)

        if not samples:
            return np.zeros(1, dtype=np.float32), target_sr

        audio = np.concatenate(samples).astype(np.float32)
        orig_sr = audio_stream.rate
        container.close()

        if orig_sr != target_sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

        return audio, target_sr

    except Exception:
        return np.zeros(1, dtype=np.float32), target_sr


# =========================================================
# 3. RMS Energy Scan
# =========================================================

def compute_rms_profile(audio: np.ndarray, sr: int, window_sec: float = 1.0):
    """
    Compute RMS energy per window. Returns dict with:
      - rms_values: array of RMS per window
      - rms_dbfs:   array of dBFS per window
      - max_rms_dbfs, mean_rms_dbfs, min_rms_dbfs
    """
    window_samples = int(sr * window_sec)
    n_windows = max(1, len(audio) // window_samples)

    rms_values = np.zeros(n_windows, dtype=np.float32)
    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        chunk = audio[start:end]
        rms_values[i] = np.sqrt(np.mean(chunk ** 2))

    eps = 1e-10
    rms_dbfs = 20.0 * np.log10(rms_values + eps)

    return {
        "rms_values": rms_values,
        "rms_dbfs": rms_dbfs,
        "max_rms_dbfs": float(np.max(rms_dbfs)),
        "mean_rms_dbfs": float(np.mean(rms_dbfs)),
        "min_rms_dbfs": float(np.min(rms_dbfs)),
    }


def compute_per_scene_rms(audio: np.ndarray, sr: int, scenes: list) -> list:
    """
    Compute RMS (dBFS) for each scene using its timestamps.
    Returns list of dBFS values, one per scene.
    """
    eps = 1e-10
    per_scene = []
    for scene in scenes:
        t0 = float(scene["start_seconds"])
        t1 = float(scene["end_seconds"])
        i0 = max(0, int(t0 * sr))
        i1 = min(len(audio), int(t1 * sr))
        chunk = audio[i0:i1]
        if len(chunk) == 0:
            per_scene.append(-200.0)
        else:
            rms = np.sqrt(np.mean(chunk ** 2))
            per_scene.append(float(20.0 * np.log10(rms + eps)))
    return per_scene


# =========================================================
# 4. Silero VAD Speech Detection
# =========================================================

def detect_speech_regions(audio: np.ndarray, sr: int, thresholds: dict) -> list:
    """
    Run Silero VAD on full audio. Returns list of
    {"start_sec": float, "end_sec": float} dicts.
    """
    audio_tensor = torch.from_numpy(audio).float()

    speech_ts = _get_speech_ts(
        audio_tensor,
        _silero_model,
        sampling_rate=sr,
        threshold=thresholds["VAD_THRESHOLD"],
        min_speech_duration_ms=thresholds["MIN_SPEECH_DURATION_MS"],
        min_silence_duration_ms=thresholds["MIN_SILENCE_DURATION_MS"],
        speech_pad_ms=thresholds["SPEECH_PAD_MS"],
    )

    regions = []
    for seg in speech_ts:
        regions.append({
            "start_sec": seg["start"] / sr,
            "end_sec":   seg["end"] / sr,
        })
    return regions


# =========================================================
# 5. Spectral Flatness
# =========================================================

def compute_spectral_flatness_mean(audio: np.ndarray, sr: int) -> float:
    """
    Compute mean spectral flatness (0 -> tonal, 1 -> noise).
    """
    if len(audio) < sr:  # less than 1 second
        return 0.0
    flatness = librosa.feature.spectral_flatness(y=audio)
    return float(np.mean(flatness))


# =========================================================
# 5b. Language Detection (Whisper-powered)
# =========================================================

def detect_languages(audio: np.ndarray, sr: int, speech_regions: list, debug: bool = False) -> dict:
    """
    Sample speech regions and detect languages using Whisper.
    Returns:
        {
            "primary_language": str,
            "detected_languages": dict,
            "is_multilingual": bool
        }
    """
    if not speech_regions:
        return {"primary_language": None, "detected_languages": {}, "is_multilingual": False}

    model = whisper.load_model("tiny", device="cpu")

    sample_indices = np.linspace(0, len(speech_regions) - 1, min(5, len(speech_regions)), dtype=int)
    detected_counts = {}

    for idx in sample_indices:
        region = speech_regions[idx]
        start = int(region["start_sec"] * sr)
        end = min(start + 30 * sr, int(region["end_sec"] * sr))

        if end - start < 1 * sr:
            continue

        chunk = whisper.pad_or_trim(audio[start:end])
        mel = whisper.log_mel_spectrogram(chunk).to("cpu")
        _, probs = model.detect_language(mel)
        lang = max(probs, key=probs.get)
        detected_counts[lang] = detected_counts.get(lang, 0) + 1

    del model
    gc.collect()

    if not detected_counts:
        return {"primary_language": None, "detected_languages": {}, "is_multilingual": False}

    sorted_langs = sorted(detected_counts.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_langs[0][0]

    is_multilingual = False
    if len(sorted_langs) > 1:
        for lang, count in sorted_langs[1:]:
            if count >= 2:
                is_multilingual = True
                break

    if debug:
        print(f"[AudioDetector] Languages detected: {detected_counts}")
        print(f"[AudioDetector] Primary: {primary}, Multilingual: {is_multilingual}")

    return {
        "primary_language": primary,
        "detected_languages": detected_counts,
        "is_multilingual": is_multilingual,
    }


# =========================================================
# 6. Main Entry Point
# =========================================================

def scan_audio(video_path: str, scenes: list, target_sr: int = 16000, debug: bool = False) -> dict:
    """
    Full 2-stage audio pre-scan with dynamic thresholds.
    Returns a decision dict usable by whisper_parallel and ast_parallel.
    """
    t_start = time.time()

    audio, sr = load_audio_av(video_path, target_sr)
    duration_sec = len(audio) / sr
    duration_min = duration_sec / 60.0

    if debug:
        print(f"[AudioDetector] Audio extracted: {duration_sec:.1f}s ({duration_min:.1f} min), {len(audio)} samples")

    thresholds = get_dynamic_thresholds(duration_min)

    if debug:
        print(f"[AudioDetector] Sensitivity multiplier: {thresholds['sensitivity_multiplier']}")
        print(f"[AudioDetector] Silence threshold: {thresholds['SILENCE_THRESHOLD_DBFS']:.1f} dBFS")
        print(f"[AudioDetector] Scene silence:     {thresholds['SCENE_SILENCE_DBFS']:.1f} dBFS")
        print(f"[AudioDetector] VAD threshold:     {thresholds['VAD_THRESHOLD']:.3f}")

    rms = compute_rms_profile(audio, sr)

    if debug:
        print(f"[AudioDetector] RMS max={rms['max_rms_dbfs']:.1f} dBFS, mean={rms['mean_rms_dbfs']:.1f} dBFS")

    has_any_audio = rms["max_rms_dbfs"] > thresholds["SILENCE_THRESHOLD_DBFS"]

    if not has_any_audio:
        elapsed = time.time() - t_start
        if debug:
            print(f"[AudioDetector] DECISION: No audio detected. Skipping all. ({elapsed:.2f}s)")
        return {
            "audio": audio,
            "sr": sr,
            "duration_sec": duration_sec,
            "has_any_audio": False,
            "has_speech": False,
            "has_background_audio": False,
            "speech_regions": [],
            "rms_profile": rms,
            "per_scene_rms": compute_per_scene_rms(audio, sr, scenes),
            "spectral_flatness_mean": 0.0,
            "thresholds_used": thresholds,
            "scan_time_sec": elapsed,
        }

    speech_regions = detect_speech_regions(audio, sr, thresholds)
    has_speech = len(speech_regions) > 0

    flatness_mean = compute_spectral_flatness_mean(audio, sr)
    has_background_audio = flatness_mean <= thresholds["SPECTRAL_FLATNESS_THRESHOLD"]

    lang_info = detect_languages(audio, sr, speech_regions, debug=debug)

    audio_masked = audio.copy()
    for region in speech_regions:
        s_idx = int(region["start_sec"] * sr)
        e_idx = int(region["end_sec"] * sr)
        audio_masked[s_idx:e_idx] = 0.0

    if debug:
        total_speech_sec = sum(r["end_sec"] - r["start_sec"] for r in speech_regions)
        print(f"[AudioDetector] Speech regions: {len(speech_regions)}, total speech: {total_speech_sec:.1f}s")
        print(f"[AudioDetector] Spectral flatness: {flatness_mean:.3f} (threshold: {thresholds['SPECTRAL_FLATNESS_THRESHOLD']:.3f})")
        print(f"[AudioDetector] has_speech={has_speech}, has_background_audio={has_background_audio}")

    per_scene_rms = compute_per_scene_rms(audio, sr, scenes)

    elapsed = time.time() - t_start

    if debug:
        silent_scenes = sum(1 for r in per_scene_rms if r < thresholds["SCENE_SILENCE_DBFS"])
        print(f"[AudioDetector] Scenes with audio: {len(scenes) - silent_scenes}/{len(scenes)}")
        print(f"[AudioDetector] Pre-scan completed in {elapsed:.2f}s")

    return {
        "audio": audio,
        "audio_masked": audio_masked,
        "sr": sr,
        "duration_sec": duration_sec,
        "has_any_audio": True,
        "has_speech": has_speech,
        "has_background_audio": has_background_audio,
        "speech_regions": speech_regions,
        "lang_info": lang_info,
        "rms_profile": rms,
        "per_scene_rms": per_scene_rms,
        "spectral_flatness_mean": flatness_mean,
        "thresholds_used": thresholds,
        "scan_time_sec": elapsed,
    }
