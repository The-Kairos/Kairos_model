"""Audio extraction from video files using PyAV with ffmpeg fallback."""

from __future__ import annotations

import subprocess

import av
import librosa
import numpy as np

from kairos.core.utils import print_prefixed


def _load_audio_ffmpeg(
    video_path: str, target_sr: int = 16000
) -> tuple[np.ndarray, int]:
    """Extract audio from a video file using an ffmpeg subprocess.

    This is a fallback method used when PyAV extraction fails or finds
    no audio stream. It spawns an ``ffmpeg`` process that decodes the
    audio track, down-mixes to mono, resamples to ``target_sr``, and
    pipes raw 32-bit float PCM to stdout.

    Args:
        video_path: Absolute or relative path to the input video file.
        target_sr: Desired output sampling rate in Hz.
            Defaults to ``16000``.

    Returns:
        A tuple of ``(audio, sample_rate)`` where *audio* is a 1-D
        ``np.float32`` array and *sample_rate* equals ``target_sr``.
        If the file contains no decodable audio, a single-sample zero
        array is returned.

    Raises:
        RuntimeError: If the ``ffmpeg`` process exits with a non-zero
            return code.
    """
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-f",
        "f32le",
        "-",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, check=False
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg failed (code {proc.returncode}): {err}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        return np.zeros(1, dtype=np.float32), target_sr
    return audio, target_sr


def load_audio_av(
    video_path: str, target_sr: int = 16000, debug: bool = False
) -> tuple[np.ndarray, int]:
    """Extract full audio from a video file using PyAV, with ffmpeg fallback.

    Opens the video with :mod:`av`, decodes every audio frame, averages
    channels to mono, and resamples to ``target_sr`` via
    :func:`librosa.resample` when the native rate differs. If PyAV
    cannot find or decode an audio stream, the function transparently
    falls back to :func:`_load_audio_ffmpeg`.

    Args:
        video_path: Absolute or relative path to the input video file.
        target_sr: Desired output sampling rate in Hz.
            Defaults to ``16000``.
        debug: If ``True``, emit diagnostic messages through
            :func:`~kairos.core.utils.print_prefixed` when fallback
            paths are taken. Defaults to ``False``.

    Returns:
        A tuple of ``(audio, sample_rate)`` where *audio* is a 1-D
        ``np.float32`` array and *sample_rate* equals ``target_sr``.
        In the worst case (all extraction methods fail), a single-sample
        zero array is returned.
    """
    try:
        container = av.open(
            video_path, options={"fflags": "+genpts", "ignore_editlist": "1"}
        )
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is None:
            if debug:
                print_prefixed(
                    "(AudioDetector)",
                    "PyAV found no audio stream; trying ffmpeg fallback.",
                )
            return _load_audio_ffmpeg(video_path, target_sr)

        audio_stream.thread_type = "AUTO"
        samples: list[np.ndarray] = []
        for frame in container.decode(audio_stream):
            pcm = frame.to_ndarray().mean(axis=0)
            samples.append(pcm)

        if not samples:
            if debug:
                print_prefixed(
                    "(AudioDetector)",
                    "PyAV decoded no audio frames; trying ffmpeg fallback.",
                )
            return _load_audio_ffmpeg(video_path, target_sr)

        audio = np.concatenate(samples).astype(np.float32)
        orig_sr: int = audio_stream.rate
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
