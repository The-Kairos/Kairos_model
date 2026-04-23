"""Unit tests for the Kairos PySceneDetect adapter.

These tests keep scene detection isolated from real video files and the
external PySceneDetect runtime by mocking metadata reads and detected ranges.
"""

# Run: python test/unit/test_pyscenedetect.py

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.kairos.model import pyscenedetect


def make_timecode(seconds: float, label: str):
    """Build a fake timecode object with the methods the adapter expects."""
    timecode = MagicMock()
    timecode.get_seconds.return_value = seconds
    timecode.__str__.return_value = label
    return timecode


def test_format_timecode_handles_bad_input():
    """Invalid time values should fall back to the unknown timecode marker."""
    assert pyscenedetect.format_timecode(None) == "??:??:??.???"
    assert pyscenedetect.format_timecode("bad") == "??:??:??.???"


def test_normalize_helpers_apply_safe_defaults():
    """Helper defaults should stay stable when metadata or input is missing."""
    assert pyscenedetect.normalize_fps(0.0) == 30.0
    assert pyscenedetect.normalize_interval_seconds(0) == 20.0
    assert pyscenedetect.compute_retry_threshold(0.05, 0.5) == 0.1


@patch("src.kairos.model.pyscenedetect.detect_scene_ranges")
@patch("src.kairos.model.pyscenedetect.read_video_stats")
def test_detect_scenes_formats_detected_ranges(mock_stats, mock_detect):
    """Detected scene ranges should be converted into Kairos scene dictionaries."""
    mock_stats.return_value = (24.0, 240.0)
    mock_detect.return_value = [
        (
            make_timecode(0.0, "00:00:00.000"),
            make_timecode(3.5, "00:00:03.500"),
        ),
        (
            make_timecode(3.5, "00:00:03.500"),
            make_timecode(7.0, "00:00:07.000"),
        ),
    ]

    scenes = pyscenedetect.detect_scenes("video.mp4", threshold=30, min_scene_sec=2, frame_skip=4)

    assert len(scenes) == 2
    assert scenes[0]["scene_index"] == 0
    assert scenes[0]["start_seconds"] == 0.0
    assert scenes[0]["end_seconds"] == 3.5
    assert scenes[0]["duration_seconds"] == 3.5
    assert scenes[1]["start_timecode"] == "00:00:03.500"
    mock_detect.assert_called_once_with("video.mp4", 30, 48, 4)


@patch("src.kairos.model.pyscenedetect.detect_scene_ranges")
@patch("src.kairos.model.pyscenedetect.read_video_stats")
def test_detect_scenes_retries_with_more_sensitive_threshold(mock_stats, mock_detect):
    """An empty first pass should trigger one retry with a lower threshold."""
    mock_stats.return_value = (30.0, 300.0)
    mock_detect.side_effect = [
        [],
        [(make_timecode(1.0, "00:00:01.000"), make_timecode(5.0, "00:00:05.000"))],
    ]

    scenes = pyscenedetect.detect_scenes("video.mp4", threshold=20, retry_threshold_factor=0.25)

    assert len(scenes) == 1
    assert mock_detect.call_count == 2
    assert mock_detect.call_args_list[0].args == ("video.mp4", 20, 60, 3)
    assert mock_detect.call_args_list[1].args == ("video.mp4", 5.0, 60, 3)


@patch("src.kairos.model.pyscenedetect.detect_scene_ranges")
@patch("src.kairos.model.pyscenedetect.read_video_stats")
def test_detect_scenes_falls_back_to_fixed_intervals(mock_stats, mock_detect):
    """No detected cuts should produce fixed-size fallback scenes."""
    mock_stats.return_value = (30.0, 300.0)
    mock_detect.side_effect = [[], []]

    scenes = pyscenedetect.detect_scenes("video.mp4", fallback_interval_sec=4)

    assert scenes == [
        {
            "scene_index": 0,
            "start_timecode": "00:00:00.000",
            "end_timecode": "00:00:04.000",
            "start_seconds": 0.0,
            "end_seconds": 4.0,
            "duration_seconds": 4.0,
        },
        {
            "scene_index": 1,
            "start_timecode": "00:00:04.000",
            "end_timecode": "00:00:08.000",
            "start_seconds": 4.0,
            "end_seconds": 8.0,
            "duration_seconds": 4.0,
        },
        {
            "scene_index": 2,
            "start_timecode": "00:00:08.000",
            "end_timecode": "00:00:10.000",
            "start_seconds": 8.0,
            "end_seconds": 10.0,
            "duration_seconds": 2.0,
        },
    ]


@patch("src.kairos.model.pyscenedetect.detect_scene_ranges")
@patch("src.kairos.model.pyscenedetect.read_video_stats")
def test_detect_scenes_uses_default_fps_when_metadata_is_missing(mock_stats, mock_detect):
    """Missing FPS metadata should fall back to the adapter's default FPS."""
    mock_stats.return_value = (0.0, 0.0)
    mock_detect.side_effect = [[], []]

    scenes = pyscenedetect.detect_scenes("video.mp4", min_scene_sec=2, fallback_interval_sec=5)

    assert len(scenes) == 1
    assert scenes[0]["duration_seconds"] == 2.0
    assert mock_detect.call_args_list[0].args == ("video.mp4", 27, 60, 3)


def run_tests():
    """Run the file directly without needing a separate test runner."""
    test_format_timecode_handles_bad_input()
    test_normalize_helpers_apply_safe_defaults()
    test_detect_scenes_formats_detected_ranges()
    test_detect_scenes_retries_with_more_sensitive_threshold()
    test_detect_scenes_falls_back_to_fixed_intervals()
    test_detect_scenes_uses_default_fps_when_metadata_is_missing()


if __name__ == "__main__":
    run_tests()
