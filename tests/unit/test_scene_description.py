"""Tests for kairos.llm.scene_description formatting functions.

Covers bounding-box normalisation, per-frame description formatting
(with legacy dict, empty, and list YOLO formats), and raw description
assembly including audio transcript/sound handling.
"""

from typing import Any

from kairos.llm.scene_description import (
    format_single_description,
    normalize_bbox,
    raw_descriptions,
)


def test_normalize_bbox() -> None:
    """Verify bbox normalisation returns correct centre and area for a square."""
    cx, cy, area = normalize_bbox([0, 0, 10, 10])
    assert cx == 5.0
    assert cy == 5.0
    assert area == 100.0


def test_normalize_bbox_rectangle() -> None:
    """Verify bbox normalisation returns correct centre and area for a rectangle."""
    cx, cy, area = normalize_bbox([10, 20, 30, 60])
    assert cx == 20.0
    assert cy == 40.0
    assert area == 20 * 40


def test_format_single_description_legacy_yolo() -> None:
    """Verify formatting with a legacy per-frame YOLO detection dict."""
    captions: list[str] = ["a woman driving a car", "a road ahead"]
    yolo: dict[str, Any] = {
        "0": [{"label": "person", "confidence": 0.9, "bbox": [10, 10, 100, 200]}],
        "1": [],
    }
    result = format_single_description(captions, yolo)
    assert "Frame 0:" in result
    assert "Frame 1:" in result
    assert 'Caption: "a woman driving a car"' in result
    assert "person" in result
    assert "Objects: none detected." in result


def test_format_single_description_empty_yolo() -> None:
    """Verify formatting when the YOLO dict is empty."""
    captions: list[str] = ["a scene"]
    yolo: dict[str, Any] = {}
    result = format_single_description(captions, yolo)
    assert "Frame 0:" in result
    assert 'Caption: "a scene"' in result


def test_format_single_description_list_yolo() -> None:
    """Verify formatting when YOLO detections are in list format."""
    captions: list[str] = ["a boy walking"]
    yolo: list[dict[str, Any]] = [{"label": "person", "track_id": 1}]
    result = format_single_description(captions, yolo)
    assert "Frame 0:" in result
    assert 'Caption: "a boy walking"' in result


def test_raw_descriptions_with_audio() -> None:
    """Verify raw descriptions include audio transcript and sound labels."""
    # NOTE: raw_descriptions has ASR_key="audio_natural" and AST_key="audio_speech"
    # (confusingly swapped in the source), so "Audio transcript" maps to audio_natural
    # and "Audio sounds" maps to audio_speech.
    scenes: list[dict[str, Any]] = [
        {
            "frame_captions": ["a cat"],
            "yolo_detections": {},
            "audio_natural": "Meowing",
            "audio_speech": "Hello there",
        }
    ]
    result = raw_descriptions(scenes)
    assert len(result) == 1
    assert "Audio transcript: Meowing" in result[0]
    assert "Audio sounds: Hello there" in result[0]


def test_raw_descriptions_no_audio() -> None:
    """Verify raw descriptions omit audio lines when fields are empty."""
    scenes: list[dict[str, Any]] = [
        {
            "frame_captions": ["a dog"],
            "yolo_detections": {},
            "audio_natural": "",
            "audio_speech": "",
        }
    ]
    result = raw_descriptions(scenes)
    assert len(result) == 1
    assert "Audio transcript" not in result[0]
    assert "Audio sounds" not in result[0]


def test_raw_descriptions_multiple_scenes() -> None:
    """Verify raw descriptions returns one entry per scene."""
    scenes: list[dict[str, Any]] = [
        {
            "frame_captions": ["cap1"],
            "yolo_detections": {},
            "audio_natural": "",
            "audio_speech": "",
        },
        {
            "frame_captions": ["cap2"],
            "yolo_detections": {},
            "audio_natural": "",
            "audio_speech": "",
        },
    ]
    result = raw_descriptions(scenes)
    assert len(result) == 2
