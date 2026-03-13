"""Audio pre-scan with dynamic thresholds: RMS energy, Silero VAD, spectral flatness."""

import gc
import math
import subprocess
import time

import av
import librosa
import numpy as np
import torch
import whisper

# Lazy-loaded Silero VAD
_silero_model = None
_get_speech_ts = None


def _get_silero_vad():
    """Load Silero VAD model on first use, then cache."""
    global _silero_model, _get_speech_ts
    if _silero_model is None:
        _silero_model, _utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        _get_speech_ts = _utils[0]
    return _silero_model, _get_speech_ts


# Dynamic thresholds

def get_sensitivity_multiplier(duration_minutes: float) -> float:
    dur = max(1.0, duration_minutes)
    return min(1.0 + 0.1 * math.log2(dur), 1.5)


def get_dynamic_thresholds(duration_minutes: float) -> dict:
    m = get_sensitivity_multiplier(duration_minutes)
    return {
        "SILENCE_THRESHOLD_DBFS": -60.0 * m,
        "SCENE_SILENCE_DBFS": -50.0 * m,
        "VAD_THRESHOLD": 0.3 / m,
        "MIN_SPEECH_DURATION_MS": int(250 / m),
        "MIN_SILENCE_DURATION_MS": 300,
        "SPEECH_PAD_MS": int(50 * m),
        "SPECTRAL_FLATNESS_THRESHOLD": min(0.85 * m, 0.95),
        "sensitivity_multiplier": round(m, 4),
    }


# Audio extraction

def _load_audio_ffmpeg(video_path: str, target_sr: int = 16000):
    cmd = [
        "ffmpeg", "-v", "error", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(target_sr), "-f", "f32le", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg failed (code {proc.returncode}): {err}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        return np.zeros(1, dtype=np.float32), target_sr
    return audio, target_sr


def load_audio_av(video_path: str, target_sr: int = 16000, debug: bool = False):
    """Extract full audio from video using PyAV, with ffmpeg fallback."""
    try:
        container = av.open(video_path, options={"fflags": "+genpts", "ignore_editlist": "1"})
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is None:
            if debug:
                print("[AudioDetector] PyAV found no audio stream; trying ffmpeg fallback.")
            return _load_audio_ffmpeg(video_path, target_sr)

        audio_stream.thread_type = "AUTO"
        samples = []
        for frame in container.decode(audio_stream):
            pcm = frame.to_ndarray().mean(axis=0)
            samples.append(pcm)

        if not samples:
            if debug:
                print("[AudioDetector] PyAV decoded no audio frames; trying ffmpeg fallback.")
            return _load_audio_ffmpeg(video_path, target_sr)

        audio = np.concatenate(samples).astype(np.float32)
        orig_sr = audio_stream.rate
        container.close()

        if orig_sr != target_sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        return audio, target_sr

    except Exception as e:
        if debug:
            print(f"[AudioDetector] PyAV audio extraction failed: {e!r}")
        try:
            return _load_audio_ffmpeg(video_path, target_sr)
        except Exception:
            return np.zeros(1, dtype=np.float32), target_sr


# RMS energy

def compute_rms_profile(audio: np.ndarray, sr: int, window_sec: float = 1.0) -> dict:
    window_samples = int(sr * window_sec)
    n_windows = max(1, len(audio) // window_samples)
    rms_values = np.zeros(n_windows, dtype=np.float32)
    for i in range(n_windows):
        chunk = audio[i * window_samples: (i + 1) * window_samples]
        rms_values[i] = np.sqrt(np.mean(chunk ** 2))
    eps = 1e-10
    rms_dbfs = 20.0 * np.log10(rms_values + eps)
    return {
        "rms_values": rms_values, "rms_dbfs": rms_dbfs,
        "max_rms_dbfs": float(np.max(rms_dbfs)),
        "mean_rms_dbfs": float(np.mean(rms_dbfs)),
        "min_rms_dbfs": float(np.min(rms_dbfs)),
    }


def compute_per_scene_rms(audio: np.ndarray, sr: int, scenes: list) -> list:
    eps = 1e-10
    per_scene = []
    for scene in scenes:
        i0 = max(0, int(float(scene["start_seconds"]) * sr))
        i1 = min(len(audio), int(float(scene["end_seconds"]) * sr))
        chunk = audio[i0:i1]
        if len(chunk) == 0:
            per_scene.append(-200.0)
        else:
            per_scene.append(float(20.0 * np.log10(np.sqrt(np.mean(chunk ** 2)) + eps)))
    return per_scene


# Silero VAD speech detection

def detect_speech_regions(audio: np.ndarray, sr: int, thresholds: dict) -> list:
    silero_model, get_speech_ts_fn = _get_silero_vad()
    if not audio.flags.writeable:
        audio = audio.copy()
    audio_tensor = torch.from_numpy(audio).float()
    speech_ts = get_speech_ts_fn(
        audio_tensor, silero_model, sampling_rate=sr,
        threshold=thresholds["VAD_THRESHOLD"],
        min_speech_duration_ms=thresholds["MIN_SPEECH_DURATION_MS"],
        min_silence_duration_ms=thresholds["MIN_SILENCE_DURATION_MS"],
        speech_pad_ms=thresholds["SPEECH_PAD_MS"],
    )
    return [{"start_sec": seg["start"] / sr, "end_sec": seg["end"] / sr} for seg in speech_ts]


# Spectral flatness

def _spectral_flatness_numpy(audio: np.ndarray, sr: int) -> float:
    n_fft = min(2048, max(256, 2 ** int(np.floor(np.log2(len(audio))))))
    hop = max(1, n_fft // 4)
    if len(audio) < n_fft:
        return 0.0
    window = np.hanning(n_fft).astype(np.float32)
    eps = 1e-12
    vals = []
    for i in range(0, len(audio) - n_fft + 1, hop):
        power = np.abs(np.fft.rfft(audio[i:i + n_fft] * window)) ** 2 + eps
        vals.append(float(np.exp(np.mean(np.log(power))) / np.mean(power)))
    return float(np.mean(vals)) if vals else 0.0


def compute_spectral_flatness_mean(audio: np.ndarray, sr: int, debug: bool = False) -> float:
    if len(audio) < sr:
        return 0.0
    try:
        return float(np.mean(librosa.feature.spectral_flatness(y=audio)))
    except ModuleNotFoundError:
        return _spectral_flatness_numpy(audio, sr)


# Language detection

def detect_languages(audio: np.ndarray, sr: int, speech_regions: list, debug: bool = False) -> dict:
    if not speech_regions:
        return {"primary_language": None, "detected_languages": {}, "is_multilingual": False}
    model = whisper.load_model("base", device="cpu")
    sample_indices = np.linspace(0, len(speech_regions) - 1, min(10, len(speech_regions)), dtype=int)
    detected_counts = {}
    for idx in sample_indices:
        region = speech_regions[idx]
        start = int(region["start_sec"] * sr)
        end = min(start + 30 * sr, int(region["end_sec"] * sr))
        if end - start < sr:
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
    is_multilingual = any(count >= 2 for _, count in sorted_langs[1:])
    if debug:
        print(f"[AudioDetector] Languages: {detected_counts}, Primary: {primary}, Multilingual: {is_multilingual}")
    return {"primary_language": primary, "detected_languages": detected_counts, "is_multilingual": is_multilingual}


# Main entry point

def scan_audio(video_path: str, scenes: list, target_sr: int = 16000, debug: bool = False) -> dict:
    """Full 2-stage audio pre-scan with dynamic thresholds."""
    t_start = time.time()
    audio, sr = load_audio_av(video_path, target_sr, debug=debug)
    duration_sec = len(audio) / sr
    duration_min = duration_sec / 60.0

    if debug:
        print(f"[AudioDetector] Audio extracted: {duration_sec:.1f}s ({duration_min:.1f} min)")

    thresholds = get_dynamic_thresholds(duration_min)
    rms = compute_rms_profile(audio, sr)
    has_any_audio = rms["max_rms_dbfs"] > thresholds["SILENCE_THRESHOLD_DBFS"]

    if not has_any_audio:
        elapsed = time.time() - t_start
        if debug:
            print(f"[AudioDetector] No audio detected. Skipping all. ({elapsed:.2f}s)")
        return {
            "audio": audio, "sr": sr, "duration_sec": duration_sec,
            "has_any_audio": False, "has_speech": False, "has_background_audio": False,
            "speech_regions": [], "rms_profile": rms,
            "per_scene_rms": compute_per_scene_rms(audio, sr, scenes),
            "spectral_flatness_mean": 0.0, "thresholds_used": thresholds,
            "scan_time_sec": elapsed,
        }

    speech_regions = detect_speech_regions(audio, sr, thresholds)
    has_speech = len(speech_regions) > 0
    flatness_mean = compute_spectral_flatness_mean(audio, sr, debug=debug)
    has_background_audio = flatness_mean <= thresholds["SPECTRAL_FLATNESS_THRESHOLD"]
    lang_info = detect_languages(audio, sr, speech_regions, debug=debug)

    audio_masked = audio.copy()
    for region in speech_regions:
        s_idx = int(region["start_sec"] * sr)
        e_idx = int(region["end_sec"] * sr)
        audio_masked[s_idx:e_idx] = 0.0

    per_scene_rms = compute_per_scene_rms(audio, sr, scenes)
    elapsed = time.time() - t_start

    if debug:
        total_speech_sec = sum(r["end_sec"] - r["start_sec"] for r in speech_regions)
        silent_scenes = sum(1 for r in per_scene_rms if r < thresholds["SCENE_SILENCE_DBFS"])
        print(f"[AudioDetector] Speech: {len(speech_regions)} regions, {total_speech_sec:.1f}s total")
        print(f"[AudioDetector] Spectral flatness: {flatness_mean:.3f}")
        print(f"[AudioDetector] Scenes with audio: {len(scenes) - silent_scenes}/{len(scenes)}")
        print(f"[AudioDetector] Pre-scan completed in {elapsed:.2f}s")

    return {
        "audio": audio, "audio_masked": audio_masked, "sr": sr,
        "duration_sec": duration_sec, "has_any_audio": True,
        "has_speech": has_speech, "has_background_audio": has_background_audio,
        "speech_regions": speech_regions, "lang_info": lang_info,
        "rms_profile": rms, "per_scene_rms": per_scene_rms,
        "spectral_flatness_mean": flatness_mean, "thresholds_used": thresholds,
        "scan_time_sec": elapsed,
    }
