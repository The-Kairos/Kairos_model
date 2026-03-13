"""Tests for kairos.config.PipelineConfig."""

from kairos.config import PipelineConfig


def test_default_values():
    cfg = PipelineConfig.default()
    assert cfg.pyscene_threshold == 27.0
    assert cfg.frames_per_scene == 3
    assert cfg.frame_resolution == 320
    assert cfg.blip_start_prompt == "a video frame of"
    assert cfg.yolo_conf_thres == 0.8
    assert cfg.asr_model_size == "medium"
    assert cfg.llm_cooldown_sec == 0.0
    assert cfg.rag_top_k_context == 10


def test_fast_preset():
    cfg = PipelineConfig.fast()
    assert cfg.pyscene_threshold == 40
    assert cfg.frames_per_scene == 1
    assert cfg.llm_chunk_len == 500000
    assert cfg.llm_summary_len == 500000
    # unchanged defaults
    assert cfg.frame_resolution == 320
    assert cfg.yolo_conf_thres == 0.8


def test_motion_sensitive_preset():
    cfg = PipelineConfig.motion_sensitive()
    assert cfg.pyscene_threshold == 15
    assert cfg.pyscene_shortest == 0.5
    assert cfg.frames_per_scene == 5
    assert cfg.yolo_action_fps == 8
    # unchanged
    assert cfg.frame_resolution == 320


def test_static_video_preset():
    cfg = PipelineConfig.static_video()
    assert cfg.pyscene_threshold == 3
    assert cfg.frames_per_scene == 1
    assert cfg.yolo_action_fps == 0.5


def test_to_dict():
    cfg = PipelineConfig.default()
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert "pyscene_threshold" in d
    assert "frames_per_scene" in d
    assert "yolo_model_path" in d
    assert "rag_top_k_context" in d
    assert d["pyscene_threshold"] == 27.0
