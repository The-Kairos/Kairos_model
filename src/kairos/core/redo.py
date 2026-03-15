"""Redo logic for selectively re-running pipeline stages.

This module provides utilities to clear previously-computed results from a
pipeline checkpoint so that specific stages (and, optionally, their
downstream dependents) can be re-executed without restarting the entire
pipeline from scratch.

Key data structures:

* :data:`PIPELINE_ORDER` — canonical execution order of all stages.
* :data:`DEPENDENTS` — maps each stage to the stages that depend on it.
* :data:`SCENE_KEYS` / :data:`TOP_LEVEL_KEYS` — checkpoint keys owned by
  each stage, used when clearing data.

Typical usage::

    from kairos.core.redo import apply_redo

    checkpoint, info = apply_redo(
        checkpoint=checkpoint,
        output_dir="output/my_video",
        redo_steps=["llm"],
        redo_only=False,
    )
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

PIPELINE_ORDER: list[str] = [
    "scenes",
    "frame_captions",
    "yolo",
    "audio_natural",
    "audio_speech",
    "llm",
    "narrative",
    "synopsis",
    "rag",
]
"""Canonical execution order of every pipeline stage."""

REDO_CHOICES: list[str] = list(PIPELINE_ORDER)
"""Valid stage names accepted by the ``--redo`` CLI flag."""

DEPENDENTS: dict[str, list[str]] = {
    "scenes": [
        "frame_captions",
        "yolo",
        "audio_natural",
        "audio_speech",
        "llm",
        "narrative",
        "synopsis",
        "rag",
    ],
    "frame_captions": ["llm", "narrative", "synopsis", "rag"],
    "yolo": ["llm", "narrative", "synopsis", "rag"],
    "audio_natural": ["llm", "narrative", "synopsis", "rag"],
    "audio_speech": ["llm", "narrative", "synopsis", "rag"],
    "llm": ["narrative", "synopsis", "rag"],
    "narrative": ["synopsis", "rag"],
    "synopsis": ["rag"],
    "rag": [],
}
"""Mapping from each stage to the stages that transitively depend on it.

When a stage is re-done, all of its dependents must also be cleared so that
stale data does not pollute downstream results.
"""

STEP_LOG_KEYS: dict[str, list[str]] = {
    "scenes": ["get_scene_list", "save_clips"],
    "frame_captions": ["sample_frames", "caption_frames"],
    "yolo": ["sample_fps", "detect_object_yolo"],
    "audio_natural": ["ast_timings"],
    "audio_speech": ["asr_timings"],
    "llm": ["describe_scenes"],
    "narrative": ["summarize_scenes"],
    "synopsis": ["synthesize_synopsis"],
    "rag": ["make_embedding"],
}
"""Step-log keys associated with each pipeline stage.

Used to remove timing / resource entries from
``checkpoint["steps"]`` when a stage is redone.
"""

SCENE_KEYS: dict[str, list[str]] = {
    "frame_captions": ["frame_captions"],
    "yolo": ["yolo_detections"],
    "audio_natural": ["audio_natural"],
    "audio_speech": ["audio_speech"],
    "llm": ["llm_scene_description"],
}
"""Per-scene checkpoint keys owned by each stage.

These keys live inside each element of ``checkpoint["scenes"]`` and are
removed when the owning stage is cleared.
"""

TOP_LEVEL_KEYS: dict[str, list[str]] = {
    "narrative": ["narratives"],
    "synopsis": ["synopsis"],
    "rag": ["rag_embedding"],
}
"""Top-level checkpoint keys owned by each stage.

These keys live directly on the ``checkpoint`` dict (not inside individual
scenes) and are removed when the owning stage is cleared.
"""


def _normalize_steps(steps: Iterable[str] | None) -> list[str]:
    """Normalize and deduplicate an iterable of step name strings.

    Each value is stripped, lower-cased, and non-string / empty values are
    silently discarded.

    Args:
        steps: Raw step names from user input (e.g. CLI arguments).
            May be ``None``, in which case an empty list is returned.

    Returns:
        A cleaned list of step name strings suitable for lookup in
        :data:`PIPELINE_ORDER`.
    """
    cleaned: list[str] = []
    if not steps:
        return cleaned
    for step in steps:
        if not isinstance(step, str):
            continue
        value = step.strip().lower()
        if value:
            cleaned.append(value)
    return cleaned


def resolve_dependents(steps: Iterable[str]) -> set[str]:
    """Compute the full set of stages affected by redoing *steps*.

    Starting from the given *steps*, transitively expands all downstream
    dependents using :data:`DEPENDENTS` until no new stages are added.

    Args:
        steps: Initial stage names to resolve.

    Returns:
        A set containing every stage that must be cleared — the original
        *steps* plus all transitive dependents.
    """
    resolved: set[str] = set()
    stack = list(steps)
    while stack:
        step = stack.pop()
        if step in resolved:
            continue
        resolved.add(step)
        stack.extend(DEPENDENTS.get(step, []))
    return resolved


def _sort_by_pipeline(steps: Iterable[str]) -> list[str]:
    """Sort *steps* according to :data:`PIPELINE_ORDER`.

    Steps not present in :data:`PIPELINE_ORDER` are placed at the end.

    Args:
        steps: Stage names to sort.

    Returns:
        A new list of stage names ordered by their canonical pipeline
        position.
    """
    order = {step: idx for idx, step in enumerate(PIPELINE_ORDER)}
    return sorted(steps, key=lambda step: order.get(step, len(order)))


def get_stop_after_step(redo_steps: Iterable[str] | None) -> str | None:
    """Return the latest pipeline stage present in *redo_steps*.

    This is used to determine the point at which the pipeline should stop
    after completing only the requested redo stages.

    Args:
        redo_steps: Stage names requested for redo.  May be ``None``.

    Returns:
        The stage name that appears latest in :data:`PIPELINE_ORDER`,
        or ``None`` if *redo_steps* is empty or ``None``.
    """
    normalized = _normalize_steps(redo_steps)
    if not normalized:
        return None
    order = {step: idx for idx, step in enumerate(PIPELINE_ORDER)}
    return max(normalized, key=lambda step: order.get(step, -1))


def should_stop_after(current_step: str, stop_after: str | None) -> bool:
    """Check whether the pipeline should stop after *current_step*.

    Args:
        current_step: The pipeline stage that just completed.
        stop_after: The stage name returned by
            :func:`get_stop_after_step`, or ``None`` if no stop is
            requested.

    Returns:
        ``True`` if execution should halt after *current_step*;
        ``False`` otherwise.
    """
    return bool(stop_after) and current_step == stop_after


def _clear_scene_key(scenes: list[dict[str, object]], key: str) -> bool:
    """Remove *key* from every scene dict in *scenes*.

    Args:
        scenes: List of mutable scene dictionaries from the checkpoint.
        key: The key to remove from each scene dict.

    Returns:
        ``True`` if at least one scene contained *key* (and it was
        removed); ``False`` if no scene was modified.
    """
    changed = False
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        if key in scene:
            scene.pop(key, None)
            changed = True
    return changed


def apply_redo(
    checkpoint: dict[str, object],
    output_dir: str | Path | None,
    redo_steps: Iterable[str] | None,
    redo_only: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    """Clear checkpoint data for the requested stages so they will re-run.

    Depending on *redo_only*, either only the explicitly listed stages are
    cleared, or all transitive dependents are cleared as well.

    The following data is removed for each affected stage:

    * Per-scene keys listed in :data:`SCENE_KEYS`.
    * Top-level keys listed in :data:`TOP_LEVEL_KEYS`.
    * Step-log entries listed in :data:`STEP_LOG_KEYS`.
    * The ``rag_embedding.json`` file on disk (when ``"rag"`` is affected).
    * The entire ``checkpoint["scenes"]`` list (when ``"scenes"`` is
      affected).

    Args:
        checkpoint: Mutable pipeline checkpoint dictionary.  Modified
            in-place by removing keys associated with the targeted stages.
        output_dir: Pipeline output directory.  Used to locate and delete
            the ``rag_embedding.json`` file when the RAG stage is
            affected.  May be ``None`` if RAG cleanup is not needed.
        redo_steps: Stage names to redo (e.g. ``["llm", "synopsis"]``).
            ``None`` or empty means *no redo*.
        redo_only: When ``True``, **only** the stages explicitly listed
            in *redo_steps* are cleared.  When ``False`` (default),
            downstream dependents are also included via
            :func:`resolve_dependents`.

    Returns:
        A ``(checkpoint, info)`` tuple where *info* is a dict with:

        * ``"redo_steps"`` — the normalised list of requested stages.
        * ``"redo_set"`` — the full set of stages that were cleared
          (sorted by pipeline order).
        * ``"changed"`` — ``True`` if any checkpoint data was actually
          modified.
    """
    normalized = _normalize_steps(redo_steps)
    if not normalized:
        return checkpoint, {
            "redo_steps": [],
            "redo_set": [],
            "changed": False,
        }

    redo_set = set(normalized) if redo_only else resolve_dependents(normalized)

    changed = False

    if "scenes" in redo_set:
        checkpoint["scenes"] = []
        changed = True

    scenes = checkpoint.get("scenes")
    if isinstance(scenes, list) and scenes:
        for step in redo_set:
            for key in SCENE_KEYS.get(step, []):
                changed |= _clear_scene_key(scenes, key)

    for step in redo_set:
        for key in TOP_LEVEL_KEYS.get(step, []):
            if key in checkpoint:
                checkpoint.pop(key, None)
                changed = True

    steps_log = checkpoint.get("steps")
    if isinstance(steps_log, dict):
        for step in redo_set:
            for key in STEP_LOG_KEYS.get(step, []):
                if key in steps_log:
                    steps_log.pop(key, None)
                    changed = True

    if "rag" in redo_set and output_dir:
        rag_path = Path(output_dir) / "rag_embedding.json"
        if rag_path.exists():
            rag_path.unlink()
            changed = True

    return checkpoint, {
        "redo_steps": normalized,
        "redo_set": _sort_by_pipeline(redo_set),
        "changed": changed,
    }
