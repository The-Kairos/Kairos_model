"""
audio_utils.py - FFmpeg-based scene audio extraction.

Provides extract_scene_audio_ffmpeg to extract a temporal segment from a video
into a WAV file. Used by vlms_light and vlms_heavy pipelines.
"""

import subprocess


def extract_scene_audio_ffmpeg(
    video_path: str, wav_path: str, start_sec: float, end_sec: float, sr: int = 16000
) -> None:
    """
    Extract a segment of audio from a video file to a WAV file using ffmpeg.

    Args:
        video_path: Path to the source video.
        wav_path: Path to the output WAV file.
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
        sr: Output sample rate (default 16000).
    """
    duration = max(0.001, end_sec - start_sec)
    cmd = [
        "ffmpeg",
        "-y",
        "-v", "error",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration),
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore").strip() or proc.stdout.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg failed (code {proc.returncode}): {err}")
