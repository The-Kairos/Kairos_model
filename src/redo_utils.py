from __future__ import annotations

from pathlib import Path
from typing import Iterable

PIPELINE_ORDER = [
    "scenes",
    "frame_captions",
    "yolo",
    "audio_natural",
    "audio_speech",
    "llm",
    "kg_extract",
    "narrative",
    "synopsis",
    "rag",
]

REDO_CHOICES = list(PIPELINE_ORDER)

DEPENDENTS = {
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
    "llm": ["kg_extract", "narrative", "synopsis", "rag"],
    "kg_extract": ["narrative", "synopsis", "rag"],
    "narrative": ["synopsis", "rag"],
    "synopsis": ["rag"],
    "rag": [],
}

STEP_LOG_KEYS = {
    "scenes": ["get_scene_list", "save_clips"],
    "frame_captions": ["sample_frames", "caption_frames"],
    "yolo": ["sample_fps", "detect_object_yolo"],
    "audio_natural": ["ast_timings"],
    "audio_speech": ["asr_timings"],
    "llm": ["describe_scenes"],
    "kg_extract": ["kg_extract"],
    "narrative": ["summarize_scenes"],
    "synopsis": ["synthesize_synopsis"],
    "rag": ["make_embedding"],
}

SCENE_KEYS = {
    "frame_captions": ["frame_captions"],
    "yolo": ["yolo_detections"],
    "audio_natural": ["audio_natural"],
    "audio_speech": ["audio_speech"],
    "llm": ["llm_scene_description"],
    "kg_extract": ["relationships"],
}

TOP_LEVEL_KEYS = {
    "llm": ["knowledge_graph"],
    "narrative": ["narratives"],
    "synopsis": ["synopsis"],
    "rag": ["rag_embedding"],
}


def _normalize_steps(steps: Iterable[str] | None) -> list[str]:
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
    order = {step: idx for idx, step in enumerate(PIPELINE_ORDER)}
    return sorted(steps, key=lambda step: order.get(step, len(order)))


def get_stop_after_step(redo_steps: Iterable[str] | None) -> str | None:
    normalized = _normalize_steps(redo_steps)
    if not normalized:
        return None
    order = {step: idx for idx, step in enumerate(PIPELINE_ORDER)}
    return max(normalized, key=lambda step: order.get(step, -1))


def should_stop_after(current_step: str, stop_after: str | None) -> bool:
    return bool(stop_after) and current_step == stop_after


def _clear_scene_key(scenes: list[dict], key: str) -> bool:
    changed = False
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        if key in scene:
            scene.pop(key, None)
            changed = True
    return changed


def apply_redo(
    checkpoint: dict,
    output_dir: str | Path | None,
    redo_steps: Iterable[str] | None,
    redo_only: bool = False,
) -> tuple[dict, dict]:
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
