"""RMS energy profiling for audio signals."""

from __future__ import annotations

import numpy as np


def compute_rms_profile(
    audio: np.ndarray, sr: int, window_sec: float = 1.0
) -> dict[str, np.ndarray | float]:
    """Compute the RMS energy profile of an audio signal.

    Divides the waveform into fixed-length windows and calculates the
    root-mean-square amplitude of each window, then converts to dBFS.

    Args:
        audio: 1-D float array of audio samples.
        sr: Sampling rate of ``audio`` in Hz.
        window_sec: Duration of each analysis window in seconds.
            Defaults to ``1.0``.

    Returns:
        A dictionary with the following keys:

        * **rms_values** (*np.ndarray*) – Linear RMS amplitude per
          window (``float32``).
        * **rms_dbfs** (*np.ndarray*) – RMS values converted to dBFS.
        * **max_rms_dbfs** (*float*) – Maximum dBFS value across all
          windows.
        * **mean_rms_dbfs** (*float*) – Mean dBFS value across all
          windows.
        * **min_rms_dbfs** (*float*) – Minimum dBFS value across all
          windows.
    """
    window_samples: int = int(sr * window_sec)
    n_windows: int = max(1, len(audio) // window_samples)
    rms_values = np.zeros(n_windows, dtype=np.float32)
    for i in range(n_windows):
        chunk = audio[i * window_samples : (i + 1) * window_samples]
        rms_values[i] = np.sqrt(np.mean(chunk**2))
    eps: float = 1e-10
    rms_dbfs = 20.0 * np.log10(rms_values + eps)
    return {
        "rms_values": rms_values,
        "rms_dbfs": rms_dbfs,
        "max_rms_dbfs": float(np.max(rms_dbfs)),
        "mean_rms_dbfs": float(np.mean(rms_dbfs)),
        "min_rms_dbfs": float(np.min(rms_dbfs)),
    }


def compute_per_scene_rms(
    audio: np.ndarray, sr: int, scenes: list[dict]
) -> list[float]:
    """Compute the RMS energy level in dBFS for each scene.

    For every scene, extracts the corresponding audio slice and
    calculates its RMS amplitude in dBFS. Scenes with zero-length
    audio are assigned ``-200.0`` dBFS (effectively silent).

    Args:
        audio: 1-D float array of the full audio waveform.
        sr: Sampling rate of ``audio`` in Hz.
        scenes: List of scene dicts, each containing
            ``"start_seconds"`` and ``"end_seconds"`` keys (values
            convertible to ``float``).

    Returns:
        A list of floats (one per scene) representing the RMS level in
        dBFS. A value of ``-200.0`` indicates an empty or missing audio
        segment.
    """
    eps: float = 1e-10
    per_scene: list[float] = []
    for scene in scenes:
        i0: int = max(0, int(float(scene["start_seconds"]) * sr))
        i1: int = min(len(audio), int(float(scene["end_seconds"]) * sr))
        chunk = audio[i0:i1]
        if len(chunk) == 0:
            per_scene.append(-200.0)
        else:
            per_scene.append(float(20.0 * np.log10(np.sqrt(np.mean(chunk**2)) + eps)))
    return per_scene
