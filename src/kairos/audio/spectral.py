"""Spectral flatness computation for audio signals.

Provides both a pure-NumPy fallback and a librosa-based implementation for
computing the mean spectral flatness of an audio waveform. Spectral flatness
measures how tone-like vs. noise-like a signal is (0 = tonal, 1 = white noise).
"""

from __future__ import annotations

import numpy as np


def _spectral_flatness_numpy(audio: np.ndarray, sr: int) -> float:
    """Compute spectral flatness without librosa using a pure-NumPy STFT.

    The function windows the signal with a Hann window, computes the power
    spectrum via ``numpy.fft.rfft``, and returns the ratio of the geometric
    mean to the arithmetic mean of the power spectrum averaged across all
    frames.

    Args:
        audio: 1-D float array of audio samples.
        sr: Sampling rate in Hz (used only for consistency with the public
            API; not referenced internally).

    Returns:
        Mean spectral flatness across all STFT frames, in the range
        ``[0.0, 1.0]``.  Returns ``0.0`` when the signal is shorter than
        the chosen FFT size or no valid frames can be computed.
    """
    n_fft = min(2048, max(256, 2 ** int(np.floor(np.log2(len(audio))))))
    hop = max(1, n_fft // 4)
    if len(audio) < n_fft:
        return 0.0
    window = np.hanning(n_fft).astype(np.float32)
    eps = 1e-12
    vals: list[float] = []
    for i in range(0, len(audio) - n_fft + 1, hop):
        power = np.abs(np.fft.rfft(audio[i : i + n_fft] * window)) ** 2 + eps
        vals.append(float(np.exp(np.mean(np.log(power))) / np.mean(power)))
    return float(np.mean(vals)) if vals else 0.0


def compute_spectral_flatness_mean(
    audio: np.ndarray, sr: int, debug: bool = False
) -> float:
    """Compute the mean spectral flatness of an audio signal.

    Tries to use ``librosa.feature.spectral_flatness`` for efficiency and
    accuracy.  Falls back to :func:`_spectral_flatness_numpy` when *librosa*
    is not installed.

    Args:
        audio: 1-D float array of audio samples.
        sr: Sampling rate in Hz.
        debug: If ``True``, additional diagnostic information may be printed
            (currently unused; reserved for future use).

    Returns:
        Mean spectral flatness as a float in ``[0.0, 1.0]``.  Returns
        ``0.0`` when the signal is shorter than one second.
    """
    if len(audio) < sr:
        return 0.0
    try:
        import librosa

        return float(np.mean(librosa.feature.spectral_flatness(y=audio)))
    except ModuleNotFoundError:
        return _spectral_flatness_numpy(audio, sr)
