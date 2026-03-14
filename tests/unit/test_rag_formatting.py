"""Tests for kairos.llm.rag formatting functions."""

from kairos.llm.rag import format_paragraph_embedding, format_scene_embedding


def test_format_scene_embedding(sample_scenes):
    result = format_scene_embedding(sample_scenes)
    assert len(result) == 3
    for text in result:
        assert isinstance(text, str)
        assert len(text) > 0


def test_format_scene_embedding_contains_timecodes(sample_scenes):
    result = format_scene_embedding(sample_scenes)
    assert "00:00:00.000" in result[0]
    assert "00:00:05.000" in result[0]


def test_format_scene_embedding_contains_objects(sample_scenes):
    result = format_scene_embedding(sample_scenes)
    assert "person" in result[0]


def test_format_scene_embedding_contains_audio(sample_scenes):
    result = format_scene_embedding(sample_scenes)
    assert "Speech" in result[0]
    assert "Thank you" in result[0]


def test_format_scene_embedding_empty_yolo():
    scenes = [
        {
            "scene_index": 0,
            "start_timecode": "00:00:00.000",
            "end_timecode": "00:00:05.000",
            "frame_captions": ["test"],
            "yolo_detections": {},
            "audio_natural": "",
            "audio_speech": "",
            "llm_scene_description": "A test scene.",
        }
    ]
    result = format_scene_embedding(scenes)
    assert len(result) == 1
    assert "none" in result[0]


def test_format_scene_embedding_list_yolo():
    scenes = [
        {
            "scene_index": 0,
            "start_timecode": "00:00:00.000",
            "end_timecode": "00:00:05.000",
            "frame_captions": ["test"],
            "yolo_detections": [{"label": "car", "track_id": 1}],
            "audio_natural": "",
            "audio_speech": "",
            "llm_scene_description": "Cars driving.",
        }
    ]
    result = format_scene_embedding(scenes)
    assert "car" in result[0]


def test_format_paragraph_embedding_empty():
    result = format_paragraph_embedding([])
    assert result == []


def test_format_paragraph_embedding_none():
    result = format_paragraph_embedding(None)
    assert result == []
