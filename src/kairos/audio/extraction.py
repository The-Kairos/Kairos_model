"""Audio extraction from video files using PyAV with ffmpeg fallback."""

import subprocess

import av
import librosa
import numpy as np

from kairos.core.utils import print_prefixed


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
                print_prefixed("(AudioDetector)", "PyAV found no audio stream; trying ffmpeg fallback.")
            return _load_audio_ffmpeg(video_path, target_sr)

        audio_stream.thread_type = "AUTO"
        samples = []
        for frame in container.decode(audio_stream):
            pcm = frame.to_ndarray().mean(axis=0)
            samples.append(pcm)

        if not samples:
            if debug:
                print_prefixed("(AudioDetector)", "PyAV decoded no audio frames; trying ffmpeg fallback.")
            return _load_audio_ffmpeg(video_path, target_sr)

        audio = np.concatenate(samples).astype(np.float32)
        orig_sr = audio_stream.rate
        container.close()

        if orig_sr != target_sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        return audio, target_sr

    except Exception as e:
        if debug:
            print_prefixed("(AudioDetector)", f"PyAV audio extraction failed: {e!r}")
        try:
            return _load_audio_ffmpeg(video_path, target_sr)
        except Exception:
            return np.zeros(1, dtype=np.float32), target_sr
