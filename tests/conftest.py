"""Shared pytest fixtures for the Kairos test suite."""

from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def sample_video_path():
    path = FIXTURES_DIR / "sample_video.mp4"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    return path


@pytest.fixture
def sample_frame_path():
    path = FIXTURES_DIR / "sample_frame.jpg"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    return path


@pytest.fixture
def sample_scenes():
    return [
        {
            "scene_index": 0,
            "start_seconds": 0.0,
            "end_seconds": 5.0,
            "duration_seconds": 5.0,
            "start_timecode": "00:00:00.000",
            "end_timecode": "00:00:05.000",
            "frame_captions": ["a woman speaking at a podium"],
            "yolo_detections": {
                "0": [
                    {"label": "person", "confidence": 0.95, "bbox": [100, 50, 300, 400]}
                ]
            },
            "audio_natural": "Speech",
            "audio_speech": "Thank you for this honor.",
            "llm_scene_description": (
                "A woman speaks at a podium during an awards ceremony."
            ),
        },
        {
            "scene_index": 1,
            "start_seconds": 5.0,
            "end_seconds": 12.0,
            "duration_seconds": 7.0,
            "start_timecode": "00:00:05.000",
            "end_timecode": "00:00:12.000",
            "frame_captions": ["a crowd clapping in an auditorium"],
            "yolo_detections": {
                "0": [
                    {"label": "person", "confidence": 0.88, "bbox": [10, 10, 600, 400]}
                ]
            },
            "audio_natural": "Applause",
            "audio_speech": "",
            "llm_scene_description": "The audience applauds enthusiastically.",
        },
        {
            "scene_index": 2,
            "start_seconds": 12.0,
            "end_seconds": 20.0,
            "duration_seconds": 8.0,
            "start_timecode": "00:00:12.000",
            "end_timecode": "00:00:20.000",
            "frame_captions": ["a close-up of a young woman with a pink scarf"],
            "yolo_detections": {},
            "audio_natural": "Speech, Music",
            "audio_speech": "I am proud to be here today.",
            "llm_scene_description": (
                "A close-up shot of the speaker, wearing a pink headscarf."
            ),
        },
    ]


@pytest.fixture
def sample_catalog():
    return [
        {"blob": "video_a.mp4", "video_length": 120.0, "resolution": [1280, 720]},
        {"blob": "video_b.mp4", "video_length": 1500.0, "resolution": [640, 360]},
        {"blob": "video_c.mp4", "video_length": 7200.0, "resolution": [1920, 1080]},
    ]


@pytest.fixture
def sample_checkpoint(sample_scenes):
    return {
        "scenes": sample_scenes,
        "steps": {
            "get_scene_list": {"wall_time_sec": 1.5},
            "caption_frames": {"wall_time_sec": 10.2},
        },
    }
