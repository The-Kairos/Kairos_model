"""PySceneDetect adapter for Kairos scene segmentation.

This module keeps scene detection functional and small:
- read basic video stats
- run PySceneDetect
- retry with a more sensitive threshold if needed
- fall back to fixed intervals when no cuts are found
"""

#
# In the future, a lot of these functions will be moved. I want this 
# file to be ONLY pyscene model loading 
#

from typing import Any
import cv2
from src.kairos.logging.schemas import scene_schema

DEFAULT_FPS = 30.0
DEFAULT_FALLBACK_INTERVAL_SEC = 20
MIN_RETRY_THRESHOLD = 0.1

def format_timecode(seconds: float | None) -> str:
    """Convert seconds into `HH:MM:SS.mmm` for fallback scene metadata."""
    if seconds is None:
        return "??:??:??.???"
    try:
        ms_total = int(round(float(seconds) * 1000))
    except (TypeError, ValueError):
        return "??:??:??.???"
    sec_total, ms = divmod(ms_total, 1000)
    mins_total, sec = divmod(sec_total, 60)
    hrs, mins = divmod(mins_total, 60)
    return f"{hrs:02d}:{mins:02d}:{sec:02d}.{ms:03d}"

def build_scene_schema(scene_index: int, start_time: Any, end_time: Any) -> dict:
    """Convert a PySceneDetect time range into the standard scene dictionary."""
    return scene_schema(
        scene_index=scene_index,
        start_seconds=start_time.get_seconds(),
        end_seconds=end_time.get_seconds(),
        start_timecode=str(start_time),
        end_timecode=str(end_time),
    )


def normalize_fps(fps: float) -> float:
    """Use the detected FPS when valid, otherwise fall back to a safe default."""
    return fps if fps > 0 else DEFAULT_FPS


def normalize_interval_seconds(fallback_interval_sec: int) -> float:
    """Normalize fallback interval input so fixed splitting always has a valid step."""
    interval = float(fallback_interval_sec)
    return interval if interval > 0 else float(DEFAULT_FALLBACK_INTERVAL_SEC)


def compute_min_scene_len(fps: float, min_scene_sec: int) -> int:
    """Convert the minimum scene duration from seconds into frames."""
    return max(1, int(round(fps * min_scene_sec)))


def compute_duration_seconds(frame_count: float, fps: float, min_scene_sec: int) -> float:
    """Estimate total video duration for fallback segmentation."""
    if frame_count > 0:
        return frame_count / fps
    return max(float(min_scene_sec), 1.0)


def compute_retry_threshold(threshold: float, retry_threshold_factor: float) -> float:
    """Lower the threshold for the retry pass without going below the floor."""
    return max(MIN_RETRY_THRESHOLD, threshold * retry_threshold_factor)


def build_fallback_scenes(duration_seconds: float, fallback_interval_sec: int) -> list[dict]:
    """Split a video duration into fixed-length scene dictionaries."""
    scenes = []
    interval = normalize_interval_seconds(fallback_interval_sec)
    start = 0.0
    while start < duration_seconds:
        end = min(start + interval, duration_seconds)
        if end <= start:
            break
        scenes.append(scene_schema(len(scenes), start, end, format_timecode(start), format_timecode(end)))
        start = end
    return scenes


def read_video_stats(input_video_path: str) -> tuple[float, float]:
    """Read FPS and frame count once so later logic can stay pure."""
    capture = cv2.VideoCapture(input_video_path)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        capture.release()
    return (fps or 0.0, frame_count or 0.0)


def detect_scene_ranges(input_video_path: str, threshold: float, min_scene_len: int, frame_skip: int) -> list:
    """Run PySceneDetect and return its raw `(start_time, end_time)` ranges."""
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    scene_manager.detect_scenes(open_video(input_video_path), frame_skip=frame_skip)
    return scene_manager.get_scene_list()


def format_detected_scenes(scene_ranges: list) -> list[dict]:
    """Format raw PySceneDetect ranges into Kairos scene dictionaries."""
    return [build_scene_schema(idx, start, end) for idx, (start, end) in enumerate(scene_ranges)]


def detect_with_retry(
    input_video_path: str,
    threshold: float,
    min_scene_len: int,
    frame_skip: int,
    retry_threshold_factor: float,
) -> list:
    """Try scene detection twice, making the second pass more sensitive."""
    scene_ranges = detect_scene_ranges(input_video_path, threshold, min_scene_len, frame_skip)
    if scene_ranges:
        return scene_ranges
    retry_threshold = compute_retry_threshold(threshold, retry_threshold_factor)
    return detect_scene_ranges(input_video_path, retry_threshold, min_scene_len, frame_skip)


def detect_scenes(
    input_video_path: str,
    threshold: float = 27,
    min_scene_sec: int = 2,
    frame_skip: int = 3,
    retry_threshold_factor: float = 0.5,
    fallback_interval_sec: int = 20,
) -> list[dict]:
    """Return Kairos scene metadata from PySceneDetect or fixed-interval fallback.

    Input:
    - `input_video_path`: path to the video file
    - tuning values for sensitivity, minimum scene duration, and fallback size

    Output:
    - list of scene dictionaries with scene index, timecodes, timestamps, and
      duration in seconds
    """
    fps, frame_count = read_video_stats(input_video_path)
    normalized_fps = normalize_fps(fps)
    min_scene_len = compute_min_scene_len(normalized_fps, min_scene_sec)
    scene_ranges = detect_with_retry(
        input_video_path,
        threshold,
        min_scene_len,
        frame_skip,
        retry_threshold_factor,
    )
    if scene_ranges:
        return format_detected_scenes(scene_ranges)
    duration_seconds = compute_duration_seconds(frame_count, normalized_fps, min_scene_sec)
    return build_fallback_scenes(duration_seconds, fallback_interval_sec)


get_scene_list = detect_scenes
