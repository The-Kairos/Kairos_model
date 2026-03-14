"""RMS energy profiling for audio signals."""

import numpy as np


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
