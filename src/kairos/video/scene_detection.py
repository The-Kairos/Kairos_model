from typing import Dict, List

import cv2
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

from kairos.core.utils import format_timecode


def get_scene_list(
    input_video_path: str,
    threshold: float = 27,
    min_scene_sec: int = 2,
    frame_skip: int = 3,
    retry_threshold_factor: float = 0.5,
    fallback_interval_sec: int = 20,
) -> List[Dict]:
    """
    Detect scenes in a video using PySceneDetect and return structured metadata.

    Parameters
    ----------
    input_video_path : str
        Path to the input video file.
    threshold : float, optional
        Sensitivity for the ContentDetector. Lower values detect more scene cuts.
        Default is 27.0.
    min_scene_len : int, optional
        Minimum scene length in frames. Default is 15.
    retry_threshold_factor : float, optional
        If no scenes are detected, retry with `threshold * retry_threshold_factor`.
        Default is 0.5 (more sensitive).
    fallback_interval_sec : int, optional
        If still no scenes are detected, split the video into fixed-duration
        segments of this many seconds. Default is 20.

    Returns
    -------
    List[Dict]
        A list of dictionaries, each containing:
        - "scene_index": Index of the detected scene.
        - "start_timecode": Start timecode (HH:MM:SS.mmm).
        - "end_timecode": End timecode (HH:MM:SS.mmm).
        - "start_seconds": Start time in seconds (float).
        - "end_seconds": End time in seconds (float).
        - "duration_seconds": Duration of the scene in seconds.

    Notes
    -----
    This function uses PySceneDetect's ContentDetector to locate abrupt content
    changes. It is suitable for preprocessing steps in segmentation, retrieval,
    summarization, and other video analysis workflows.
    If no scenes are detected on the first pass, a more sensitive retry is
    attempted. If still empty, a fixed-duration fallback segmentation is used.
    """

    def detect_scenes_with_threshold(thresh: float) -> list:
        video = open_video(input_video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(threshold=thresh, min_scene_len=min_scene_len)
        )
        scene_manager.detect_scenes(video, frame_skip=frame_skip)
        return scene_manager.get_scene_list()

    # Read video metadata once
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if not fps or fps <= 0:
        fps = 30.0

    # Getting the min_scene_len based on fps
    min_scene_len = max(1, int(round(fps * min_scene_sec)))

    scene_list = detect_scenes_with_threshold(threshold)
    if not scene_list:
        retry_threshold = max(0.1, threshold * retry_threshold_factor)
        scene_list = detect_scenes_with_threshold(retry_threshold)

    result = []
    if scene_list:
        for idx, (start_time, end_time) in enumerate(scene_list):
            start_sec = start_time.get_seconds()
            end_sec = end_time.get_seconds()
            result.append(
                {
                    "scene_index": idx,
                    "start_timecode": str(start_time),
                    "end_timecode": str(end_time),
                    "start_seconds": start_sec,
                    "end_seconds": end_sec,
                    "duration_seconds": end_sec - start_sec,
                }
            )
        return result

    # Fallback: fixed-duration segmentation when no scenes are found
    if frame_count and frame_count > 0:
        duration_sec = frame_count / fps
    else:
        duration_sec = max(float(min_scene_sec), 1.0)

    if fallback_interval_sec <= 0:
        fallback_interval_sec = 20

    start = 0.0
    idx = 0
    while start < duration_sec:
        end = min(start + float(fallback_interval_sec), duration_sec)
        if end <= start:
            break
        result.append(
            {
                "scene_index": idx,
                "start_timecode": format_timecode(start),
                "end_timecode": format_timecode(end),
                "start_seconds": start,
                "end_seconds": end,
                "duration_seconds": end - start,
            }
        )
        idx += 1
        start = end

    return result
