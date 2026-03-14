"""Tests for kairos.core.redo."""

from kairos.core.redo import (
    PIPELINE_ORDER,
    apply_redo,
    get_stop_after_step,
    resolve_dependents,
    should_stop_after,
)


def test_resolve_dependents_scenes():
    result = resolve_dependents(["scenes"])
    assert result == set(PIPELINE_ORDER)


def test_resolve_dependents_synopsis():
    result = resolve_dependents(["synopsis"])
    assert result == {"synopsis", "rag"}


def test_resolve_dependents_rag():
    result = resolve_dependents(["rag"])
    assert result == {"rag"}


def test_resolve_dependents_llm():
    result = resolve_dependents(["llm"])
    assert "llm" in result
    assert "narrative" in result
    assert "synopsis" in result
    assert "rag" in result
    assert "scenes" not in result


def test_apply_redo_clears_llm(sample_scenes):
    checkpoint = {
        "scenes": list(sample_scenes),
        "steps": {"describe_scenes": {"wall_time_sec": 5}},
    }
    result, info = apply_redo(checkpoint, None, ["llm"])
    assert info["changed"] is True
    for scene in result["scenes"]:
        assert "llm_scene_description" not in scene
    assert "describe_scenes" not in result["steps"]


def test_apply_redo_clears_scenes(sample_scenes):
    checkpoint = {"scenes": list(sample_scenes), "steps": {}}
    result, info = apply_redo(checkpoint, None, ["scenes"])
    assert info["changed"] is True
    assert result["scenes"] == []


def test_apply_redo_only_no_cascade(sample_scenes):
    checkpoint = {
        "scenes": list(sample_scenes),
        "steps": {"describe_scenes": {"wall_time_sec": 5}},
    }
    result, info = apply_redo(checkpoint, None, ["llm"], redo_only=True)
    assert "llm" in info["redo_set"]
    # redo_only=True should NOT cascade to narrative/synopsis/rag
    assert "narrative" not in info["redo_set"]
    assert "synopsis" not in info["redo_set"]


def test_apply_redo_empty_steps():
    checkpoint = {"scenes": [], "steps": {}}
    result, info = apply_redo(checkpoint, None, [])
    assert info["changed"] is False


def test_get_stop_after_step():
    assert get_stop_after_step(["llm", "scenes"]) == "llm"
    assert get_stop_after_step(["rag"]) == "rag"
    assert get_stop_after_step([]) is None
    assert get_stop_after_step(None) is None


def test_should_stop_after():
    assert should_stop_after("llm", "llm") is True
    assert should_stop_after("scenes", "llm") is False
    assert should_stop_after("llm", None) is False
