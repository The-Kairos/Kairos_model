"""
Audio extraction utilities. Extracts scene-level audio from video to WAV for ASR.
"""
import subprocess
from pathlib import Path

def get_ffmpeg_exe():
    try:
        import imageio_ffmpeg as ffmpeg
        return ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"

def extract_scene_audio_ffmpeg(video_path, wav_path, start_sec, end_sec, sample_rate=16000):
    """
    Extract a segment of the video's audio track to a WAV file using ffmpeg.

    Args:
        video_path: Path to the input video file.
        wav_path: Path for the output WAV file.
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
        sample_rate: Output sample rate (default 16000 for ASR).
    """
    duration = end_sec - start_sec
    if duration <= 0:
        raise ValueError(f"Invalid segment: start={start_sec}, end={end_sec}")
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(video_path),
        "-ss", str(start_sec),
        "-t", str(duration),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(wav_path),
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
