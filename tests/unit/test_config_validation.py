"""Tests for PipelineConfig __post_init__ validation and new fields."""

import pytest

from kairos.config import PipelineConfig
from kairos.core.exceptions import KairosConfigError


class TestConfigValidation:
    def test_default_creates_valid(self):
        cfg = PipelineConfig.default()
        assert cfg.pyscene_threshold > 0
        assert cfg.llm_max_workers >= 1

    def test_fast_preset_has_more_workers(self):
        cfg = PipelineConfig.fast()
        assert cfg.llm_max_workers == 8

    def test_negative_threshold_raises(self):
        with pytest.raises(KairosConfigError, match="pyscene_threshold"):
            PipelineConfig(pyscene_threshold=-1)

    def test_zero_threshold_raises(self):
        with pytest.raises(KairosConfigError, match="pyscene_threshold"):
            PipelineConfig(pyscene_threshold=0)

    def test_negative_shortest_raises(self):
        with pytest.raises(KairosConfigError, match="pyscene_shortest"):
            PipelineConfig(pyscene_shortest=-0.5)

    def test_zero_frames_per_scene_raises(self):
        with pytest.raises(KairosConfigError, match="frames_per_scene"):
            PipelineConfig(frames_per_scene=0)

    def test_yolo_conf_out_of_range(self):
        with pytest.raises(KairosConfigError, match="yolo_conf_thres"):
            PipelineConfig(yolo_conf_thres=1.5)

    def test_zero_llm_workers_raises(self):
        with pytest.raises(KairosConfigError, match="llm_max_workers"):
            PipelineConfig(llm_max_workers=0)

    def test_negative_cooldown_raises(self):
        with pytest.raises(KairosConfigError, match="llm_cooldown_sec"):
            PipelineConfig(llm_cooldown_sec=-1.0)


class TestConfigNewFields:
    def test_has_llm_max_workers(self):
        cfg = PipelineConfig()
        assert hasattr(cfg, "llm_max_workers")
        assert cfg.llm_max_workers == 4

    def test_has_data_dir(self):
        cfg = PipelineConfig()
        assert cfg.data_dir == "data"

    def test_prompts_dir_auto_resolved(self):
        cfg = PipelineConfig()
        assert cfg.prompts_dir != ""
        assert "prompts" in cfg.prompts_dir

    def test_to_dict_includes_new_fields(self):
        cfg = PipelineConfig()
        d = cfg.to_dict()
        assert "llm_max_workers" in d
        assert "data_dir" in d
        assert "prompts_dir" in d
        assert "logs_dir" in d
