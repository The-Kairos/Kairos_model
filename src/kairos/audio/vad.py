"""Silero VAD speech detection with lazy model loading."""

import numpy as np
import torch

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


def detect_speech_regions(
    audio: np.ndarray,
    sr: int,
    thresholds: dict,
    silero_model=None,
    get_speech_ts_fn=None,
) -> list:
    if silero_model is None or get_speech_ts_fn is None:
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
