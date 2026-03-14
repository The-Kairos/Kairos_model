"""Spectral flatness computation for audio signals."""

import numpy as np


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
        import librosa
        return float(np.mean(librosa.feature.spectral_flatness(y=audio)))
    except ModuleNotFoundError:
        return _spectral_flatness_numpy(audio, sr)
