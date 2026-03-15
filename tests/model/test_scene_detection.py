"""Model tests for PySceneDetect scene detection.

Runs scene detection on a real sample video and validates the output structure,
key presence, temporal coverage, and duration consistency.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.model

EXPECTED_KEYS = {
    "scene_index",
    "start_seconds",
    "end_seconds",
    "duration_seconds",
    "start_timecode",
    "end_timecode",
}


def test_scene_detection_real_video(sample_video_path: Path) -> None:
    """Verify scene detection returns at least one scene from a real video."""
    from kairos.video.scene_detection import get_scene_list

    scenes = get_scene_list(str(sample_video_path), threshold=27, min_scene_sec=2)
    assert isinstance(scenes, list)
    assert len(scenes) >= 1


def test_scene_detection_keys(sample_video_path: Path) -> None:
    """Verify every detected scene dict contains all expected keys."""
    from kairos.video.scene_detection import get_scene_list

    scenes = get_scene_list(str(sample_video_path), threshold=27, min_scene_sec=2)
    for scene in scenes:
        missing = EXPECTED_KEYS - set(scene.keys())
        assert not missing, f"Scene missing keys: {missing}"


def test_scene_detection_coverage(sample_video_path: Path) -> None:
    """Verify scenes span the full video from near-zero to near the end."""
    from kairos.video.scene_detection import get_scene_list

    scenes = get_scene_list(str(sample_video_path), threshold=27, min_scene_sec=2)
    assert scenes[0]["start_seconds"] < 1.0, "First scene should start near 0"
    assert scenes[-1]["end_seconds"] > 200, (
        "Last scene should end near video length (273s)"
    )


def test_scene_detection_durations(sample_video_path: Path) -> None:
    """Verify each scene has a positive duration consistent with start/end times."""
    from kairos.video.scene_detection import get_scene_list

    scenes = get_scene_list(str(sample_video_path), threshold=27, min_scene_sec=2)
    for scene in scenes:
        assert scene["duration_seconds"] > 0
        assert (
            abs(
                scene["end_seconds"]
                - scene["start_seconds"]
                - scene["duration_seconds"]
            )
            < 0.1
        )
