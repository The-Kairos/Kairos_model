"""Tests for kairos.llm.scene_description formatting functions."""

from kairos.llm.scene_description import (
    format_single_description,
    normalize_bbox,
    raw_descriptions,
)


def test_normalize_bbox():
    cx, cy, area = normalize_bbox([0, 0, 10, 10])
    assert cx == 5.0
    assert cy == 5.0
    assert area == 100.0


def test_normalize_bbox_rectangle():
    cx, cy, area = normalize_bbox([10, 20, 30, 60])
    assert cx == 20.0
    assert cy == 40.0
    assert area == 20 * 40


def test_format_single_description_legacy_yolo():
    captions = ["a woman driving a car", "a road ahead"]
    yolo = {
        "0": [{"label": "person", "confidence": 0.9, "bbox": [10, 10, 100, 200]}],
        "1": [],
    }
    result = format_single_description(captions, yolo)
    assert "Frame 0:" in result
    assert "Frame 1:" in result
    assert 'Caption: "a woman driving a car"' in result
    assert "person" in result
    assert "Objects: none detected." in result


def test_format_single_description_empty_yolo():
    captions = ["a scene"]
    yolo = {}
    result = format_single_description(captions, yolo)
    assert "Frame 0:" in result
    assert 'Caption: "a scene"' in result


def test_format_single_description_list_yolo():
    captions = ["a boy walking"]
    yolo = [{"label": "person", "track_id": 1}]
    result = format_single_description(captions, yolo)
    assert "Frame 0:" in result
    assert 'Caption: "a boy walking"' in result


def test_raw_descriptions_with_audio():
    # NOTE: raw_descriptions has ASR_key="audio_natural" and AST_key="audio_speech"
    # (confusingly swapped in the source), so "Audio transcript" maps to audio_natural
    # and "Audio sounds" maps to audio_speech.
    scenes = [
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


def test_raw_descriptions_no_audio():
    scenes = [
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


def test_raw_descriptions_multiple_scenes():
    scenes = [
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
