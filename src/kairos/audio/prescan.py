"""Audio pre-scan with dynamic thresholds.

Orchestrates extraction, RMS, VAD, spectral, and language detection.
"""

import math
import time

import numpy as np

from kairos.audio.extraction import load_audio_av
from kairos.audio.language import detect_languages
from kairos.audio.rms import compute_per_scene_rms, compute_rms_profile
from kairos.audio.spectral import compute_spectral_flatness_mean
from kairos.audio.vad import detect_speech_regions
from kairos.core.utils import print_prefixed


def get_sensitivity_multiplier(duration_minutes: float) -> float:
    """Compute a sensitivity multiplier based on video duration.

    Longer videos receive a higher multiplier (up to 1.5) so that
    detection thresholds can be scaled to reduce false positives on
    extended content. The formula is:

        ``min(1.0 + 0.1 * log2(max(1, duration_minutes)), 1.5)``

    Args:
        duration_minutes: Duration of the video in minutes.

    Returns:
        A float multiplier in the range ``[1.0, 1.5]``.
    """
    dur = max(1.0, duration_minutes)
    return min(1.0 + 0.1 * math.log2(dur), 1.5)


def get_dynamic_thresholds(duration_minutes: float) -> dict[str, float | int]:
    """Get dynamically scaled audio analysis thresholds.

    Computes a set of thresholds used throughout the audio pre-scan
    pipeline (silence detection, VAD, spectral flatness, etc.),
    adjusted by the sensitivity multiplier derived from the video's
    duration.

    Args:
        duration_minutes: Duration of the video in minutes.

    Returns:
        A dictionary with the following keys:

        * **SILENCE_THRESHOLD_DBFS** (*float*) – Global silence
          threshold in dBFS.
        * **SCENE_SILENCE_DBFS** (*float*) – Per-scene silence
          threshold in dBFS.
        * **VAD_THRESHOLD** (*float*) – Voice activity detection
          confidence threshold.
        * **MIN_SPEECH_DURATION_MS** (*int*) – Minimum speech segment
          length in milliseconds.
        * **MIN_SILENCE_DURATION_MS** (*int*) – Minimum inter-speech
          silence in milliseconds (fixed at 300).
        * **SPEECH_PAD_MS** (*int*) – Padding added around detected
          speech segments in milliseconds.
        * **SPECTRAL_FLATNESS_THRESHOLD** (*float*) – Maximum spectral
          flatness below which background audio is deemed present.
        * **sensitivity_multiplier** (*float*) – The raw multiplier
          value used to derive the other thresholds.
    """
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


def scan_audio(
    video_path: str,
    scenes: list[dict],
    target_sr: int = 16000,
    debug: bool = False,
) -> dict:
    """Perform a full 2-stage audio pre-scan with dynamic thresholds.

    Stage 1 extracts audio and checks global RMS to detect total
    silence. If audio is present, Stage 2 runs VAD, spectral-flatness
    analysis, and language detection, then masks speech regions and
    computes per-scene RMS values.

    The returned dictionary carries all data needed by downstream
    classifiers (e.g.,
    :func:`kairos.audio.classifier.extract_sounds_optimized`).

    Args:
        video_path: Path to the input video file.
        scenes: List of scene dicts, each containing
            ``"start_seconds"`` and ``"end_seconds"`` keys.
        target_sr: Sampling rate to which extracted audio is resampled.
            Defaults to ``16000``.
        debug: If ``True``, emit timing and diagnostic messages via
            :func:`~kairos.core.utils.print_prefixed`.
            Defaults to ``False``.

    Returns:
        A dictionary containing:

        * **audio** (*np.ndarray*) – Raw mono audio waveform.
        * **audio_masked** (*np.ndarray*) – Audio with speech regions
          zeroed out (only when audio is present).
        * **sr** (*int*) – Sampling rate of the returned audio.
        * **duration_sec** (*float*) – Total audio duration in seconds.
        * **has_any_audio** (*bool*) – Whether the file has non-silent
          audio.
        * **has_speech** (*bool*) – Whether speech was detected.
        * **has_background_audio** (*bool*) – Whether non-speech
          background audio was detected.
        * **speech_regions** (*list[dict]*) – Detected speech segments.
        * **lang_info** (*dict*) – Language detection results (only
          when audio is present).
        * **rms_profile** (*dict*) – Global RMS statistics.
        * **per_scene_rms** (*list[float]*) – Per-scene RMS in dBFS.
        * **spectral_flatness_mean** (*float*) – Mean spectral flatness.
        * **thresholds_used** (*dict*) – The dynamic thresholds applied.
        * **scan_time_sec** (*float*) – Wall-clock time of the scan.
    """
    t_start = time.time()
    audio, sr = load_audio_av(video_path, target_sr, debug=debug)
    duration_sec: float = len(audio) / sr
    duration_min: float = duration_sec / 60.0

    if debug:
        print_prefixed(
            "(AudioDetector)",
            f"Audio extracted: {duration_sec:.1f}s ({duration_min:.1f} min)",
        )

    thresholds = get_dynamic_thresholds(duration_min)
    rms = compute_rms_profile(audio, sr)
    has_any_audio: bool = rms["max_rms_dbfs"] > thresholds["SILENCE_THRESHOLD_DBFS"]

    if not has_any_audio:
        elapsed = time.time() - t_start
        if debug:
            print_prefixed(
                "(AudioDetector)", f"No audio detected. Skipping all. ({elapsed:.2f}s)"
            )
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
    has_speech: bool = len(speech_regions) > 0
    flatness_mean: float = compute_spectral_flatness_mean(audio, sr, debug=debug)
    has_background_audio: bool = (
        flatness_mean <= thresholds["SPECTRAL_FLATNESS_THRESHOLD"]
    )
    lang_info = detect_languages(audio, sr, speech_regions, debug=debug)

    audio_masked: np.ndarray = audio.copy()
    for region in speech_regions:
        s_idx = int(region["start_sec"] * sr)
        e_idx = int(region["end_sec"] * sr)
        audio_masked[s_idx:e_idx] = 0.0

    per_scene_rms: list[float] = compute_per_scene_rms(audio, sr, scenes)
    elapsed = time.time() - t_start

    if debug:
        total_speech_sec = sum(r["end_sec"] - r["start_sec"] for r in speech_regions)
        silent_scenes = sum(
            1 for r in per_scene_rms if r < thresholds["SCENE_SILENCE_DBFS"]
        )
        print_prefixed(
            "(AudioDetector)",
            f"Speech: {len(speech_regions)} regions, {total_speech_sec:.1f}s total",
        )
        print_prefixed("(AudioDetector)", f"Spectral flatness: {flatness_mean:.3f}")
        print_prefixed(
            "(AudioDetector)",
            f"Scenes with audio: {len(scenes) - silent_scenes}/{len(scenes)}",
        )
        print_prefixed("(AudioDetector)", f"Pre-scan completed in {elapsed:.2f}s")

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
