"""Silero VAD speech detection with lazy model loading.

Uses the Silero VAD model (loaded once via ``torch.hub``) to identify
speech regions in a raw audio waveform.  The detected regions are
returned as a list of start/end timestamps in seconds.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

_silero_model: Any | None = None
_get_speech_ts: Callable[..., list[dict[str, int]]] | None = None


def _get_silero_vad() -> tuple[Any, Callable[..., list[dict[str, int]]]]:
    """Lazy-load the Silero VAD model and cache it as a module-level singleton.

    On the first invocation the model is downloaded (or loaded from the
    ``torch.hub`` cache) from the ``snakers4/silero-vad`` repository.
    Subsequent calls return the cached objects immediately.

    Returns:
        A 2-tuple of ``(silero_model, get_speech_timestamps_fn)`` where
        *silero_model* is the loaded VAD network and
        *get_speech_timestamps_fn* is the utility function used to obtain
        speech timestamp dicts from an audio tensor.
    """
    global _silero_model, _get_speech_ts
    if _silero_model is None:
        _silero_model, _utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        _get_speech_ts = _utils[0]
    return _silero_model, _get_speech_ts  # type: ignore[return-value]


def detect_speech_regions(
    audio: np.ndarray,
    sr: int,
    thresholds: dict[str, float | int],
    silero_model: Any | None = None,
    get_speech_ts_fn: Callable[..., list[dict[str, int]]] | None = None,
) -> list[dict[str, float]]:
    """Detect speech regions in an audio waveform using Silero VAD.

    Converts *audio* to a ``torch.Tensor`` and passes it to the Silero VAD
    model together with the caller-supplied detection thresholds.  Each
    detected region is returned as a dict with ``start_sec`` and
    ``end_sec`` keys (values in seconds).

    Args:
        audio: 1-D NumPy array of audio samples (mono, float32 expected).
        sr: Sampling rate of *audio* in Hz (typically 16 000).
        thresholds: Dict with the following keys controlling Silero VAD
            behaviour:

            * ``"VAD_THRESHOLD"`` – speech probability threshold.
            * ``"MIN_SPEECH_DURATION_MS"`` – minimum speech duration in ms.
            * ``"MIN_SILENCE_DURATION_MS"`` – minimum silence gap in ms.
            * ``"SPEECH_PAD_MS"`` – padding added around each speech
              region in ms.
        silero_model: Pre-loaded Silero VAD model.  If ``None``, the model
            is loaded via :func:`_get_silero_vad`.
        get_speech_ts_fn: The ``get_speech_timestamps`` utility returned by
            Silero.  If ``None``, it is obtained via :func:`_get_silero_vad`.

    Returns:
        List of dicts, each containing ``"start_sec"`` and ``"end_sec"``
        keys with float values representing the boundaries of detected
        speech regions in seconds.
    """
    if silero_model is None or get_speech_ts_fn is None:
        silero_model, get_speech_ts_fn = _get_silero_vad()
    if not audio.flags.writeable:
        audio = audio.copy()
    audio_tensor: torch.Tensor = torch.from_numpy(audio).float()
    speech_ts: list[dict[str, int]] = get_speech_ts_fn(
        audio_tensor,
        silero_model,
        sampling_rate=sr,
        threshold=thresholds["VAD_THRESHOLD"],
        min_speech_duration_ms=thresholds["MIN_SPEECH_DURATION_MS"],
        min_silence_duration_ms=thresholds["MIN_SILENCE_DURATION_MS"],
        speech_pad_ms=thresholds["SPEECH_PAD_MS"],
    )
    return [
        {"start_sec": seg["start"] / sr, "end_sec": seg["end"] / sr}
        for seg in speech_ts
    ]
