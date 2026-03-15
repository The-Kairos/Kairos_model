"""Tests for kairos.core.redo.

Validates pipeline-stage dependency resolution, redo application
(with and without cascade), stop-after logic, and checkpoint mutation
when specific stages are marked for re-execution.
"""

from typing import Any

from kairos.core.redo import (
    PIPELINE_ORDER,
    apply_redo,
    get_stop_after_step,
    resolve_dependents,
    should_stop_after,
)


def test_resolve_dependents_scenes() -> None:
    """Verify redoing 'scenes' cascades to all downstream stages."""
    result = resolve_dependents(["scenes"])
    assert result == set(PIPELINE_ORDER)


def test_resolve_dependents_synopsis() -> None:
    """Verify redoing 'synopsis' cascades only to synopsis and rag."""
    result = resolve_dependents(["synopsis"])
    assert result == {"synopsis", "rag"}


def test_resolve_dependents_rag() -> None:
    """Verify redoing 'rag' does not cascade further."""
    result = resolve_dependents(["rag"])
    assert result == {"rag"}


def test_resolve_dependents_llm() -> None:
    """Verify redoing 'llm' cascades to narrative, synopsis, and rag."""
    result = resolve_dependents(["llm"])
    assert "llm" in result
    assert "narrative" in result
    assert "synopsis" in result
    assert "rag" in result
    assert "scenes" not in result


def test_apply_redo_clears_llm(sample_scenes: list[dict[str, Any]]) -> None:
    """Verify apply_redo removes LLM descriptions and step timings."""
    checkpoint: dict[str, Any] = {
        "scenes": list(sample_scenes),
        "steps": {"describe_scenes": {"wall_time_sec": 5}},
    }
    result, info = apply_redo(checkpoint, None, ["llm"])
    assert info["changed"] is True
    for scene in result["scenes"]:
        assert "llm_scene_description" not in scene
    assert "describe_scenes" not in result["steps"]


def test_apply_redo_clears_scenes(sample_scenes: list[dict[str, Any]]) -> None:
    """Verify apply_redo with 'scenes' empties the scene list."""
    checkpoint: dict[str, Any] = {"scenes": list(sample_scenes), "steps": {}}
    result, info = apply_redo(checkpoint, None, ["scenes"])
    assert info["changed"] is True
    assert result["scenes"] == []


def test_apply_redo_only_no_cascade(sample_scenes: list[dict[str, Any]]) -> None:
    """Verify redo_only=True prevents cascade to downstream stages."""
    checkpoint: dict[str, Any] = {
        "scenes": list(sample_scenes),
        "steps": {"describe_scenes": {"wall_time_sec": 5}},
    }
    _result, info = apply_redo(checkpoint, None, ["llm"], redo_only=True)
    assert "llm" in info["redo_set"]
    # redo_only=True should NOT cascade to narrative/synopsis/rag
    assert "narrative" not in info["redo_set"]
    assert "synopsis" not in info["redo_set"]


def test_apply_redo_empty_steps() -> None:
    """Verify apply_redo with no stages reports no changes."""
    checkpoint: dict[str, Any] = {"scenes": [], "steps": {}}
    _result, info = apply_redo(checkpoint, None, [])
    assert info["changed"] is False


def test_get_stop_after_step() -> None:
    """Verify get_stop_after_step returns the latest stage from the list."""
    assert get_stop_after_step(["llm", "scenes"]) == "llm"
    assert get_stop_after_step(["rag"]) == "rag"
    assert get_stop_after_step([]) is None
    assert get_stop_after_step(None) is None


def test_should_stop_after() -> None:
    """Verify should_stop_after correctly compares current stage to stop target."""
    assert should_stop_after("llm", "llm") is True
    assert should_stop_after("scenes", "llm") is False
    assert should_stop_after("llm", None) is False
