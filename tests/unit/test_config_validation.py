"""Tests for PipelineConfig __post_init__ validation and new fields.

Ensures that invalid configuration values raise ``KairosConfigError`` and
that new fields like ``llm_max_workers``, ``data_dir``, and ``prompts_dir``
are correctly initialised and included in serialization.
"""

import pytest

from kairos.config import PipelineConfig
from kairos.core.exceptions import KairosConfigError


class TestConfigValidation:
    """Tests for invalid-value rejection in PipelineConfig.__post_init__."""

    def test_default_creates_valid(self) -> None:
        """Verify the default preset creates a valid configuration."""
        cfg = PipelineConfig.default()
        assert cfg.pyscene_threshold > 0
        assert cfg.llm_max_workers >= 1

    def test_fast_preset_has_more_workers(self) -> None:
        """Verify the fast preset uses 8 LLM workers."""
        cfg = PipelineConfig.fast()
        assert cfg.llm_max_workers == 8

    def test_negative_threshold_raises(self) -> None:
        """Verify a negative pyscene_threshold raises KairosConfigError."""
        with pytest.raises(KairosConfigError, match="pyscene_threshold"):
            PipelineConfig(pyscene_threshold=-1)

    def test_zero_threshold_raises(self) -> None:
        """Verify a zero pyscene_threshold raises KairosConfigError."""
        with pytest.raises(KairosConfigError, match="pyscene_threshold"):
            PipelineConfig(pyscene_threshold=0)

    def test_negative_shortest_raises(self) -> None:
        """Verify a negative pyscene_shortest raises KairosConfigError."""
        with pytest.raises(KairosConfigError, match="pyscene_shortest"):
            PipelineConfig(pyscene_shortest=-0.5)

    def test_zero_frames_per_scene_raises(self) -> None:
        """Verify zero frames_per_scene raises KairosConfigError."""
        with pytest.raises(KairosConfigError, match="frames_per_scene"):
            PipelineConfig(frames_per_scene=0)

    def test_yolo_conf_out_of_range(self) -> None:
        """Verify yolo_conf_thres > 1.0 raises KairosConfigError."""
        with pytest.raises(KairosConfigError, match="yolo_conf_thres"):
            PipelineConfig(yolo_conf_thres=1.5)

    def test_zero_llm_workers_raises(self) -> None:
        """Verify zero llm_max_workers raises KairosConfigError."""
        with pytest.raises(KairosConfigError, match="llm_max_workers"):
            PipelineConfig(llm_max_workers=0)

    def test_negative_cooldown_raises(self) -> None:
        """Verify a negative llm_cooldown_sec raises KairosConfigError."""
        with pytest.raises(KairosConfigError, match="llm_cooldown_sec"):
            PipelineConfig(llm_cooldown_sec=-1.0)


class TestConfigNewFields:
    """Tests for newly added PipelineConfig fields."""

    def test_has_llm_max_workers(self) -> None:
        """Verify llm_max_workers defaults to 4."""
        cfg = PipelineConfig()
        assert hasattr(cfg, "llm_max_workers")
        assert cfg.llm_max_workers == 4

    def test_has_data_dir(self) -> None:
        """Verify data_dir defaults to 'data'."""
        cfg = PipelineConfig()
        assert cfg.data_dir == "data"

    def test_prompts_dir_auto_resolved(self) -> None:
        """Verify prompts_dir is auto-resolved and contains 'prompts'."""
        cfg = PipelineConfig()
        assert cfg.prompts_dir != ""
        assert "prompts" in cfg.prompts_dir

    def test_to_dict_includes_new_fields(self) -> None:
        """Verify to_dict serialises all new fields."""
        cfg = PipelineConfig()
        d = cfg.to_dict()
        assert "llm_max_workers" in d
        assert "data_dir" in d
        assert "prompts_dir" in d
        assert "logs_dir" in d
