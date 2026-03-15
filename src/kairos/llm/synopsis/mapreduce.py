"""Scene chunking and parallel map-reduce summarization."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from kairos.core.utils import print_prefixed
from kairos.llm.synopsis.parsing import NOT_STATED
from kairos.llm.synopsis.prompts import (
    _build_reduce_prompt,
    _build_scene_chunk_summary_prompt,
)

CHUNK_SIZE = 7000


def _mapreduce_log(debug: bool, message: str) -> None:
    """Emit a debug log message for map-reduce summarization.

    Args:
        debug: If ``True``, print the message; otherwise do nothing.
        message: The log message to print.
    """
    if debug:
        print_prefixed("(Synopsis)", message)


SUMMARY_MAX_WORKERS = 6
SUMMARY_REDUCE_GROUP_SIZE = 4


def _normalize_scene_text(value: Any, fallback: str) -> str:
    """Return a cleaned string value, or *fallback* when the value is empty.

    Args:
        value: Arbitrary value (usually a string) to normalise.
        fallback: Default string returned when *value* is empty or not a
            string.

    Returns:
        The stripped string if *value* is a non-empty string, otherwise
        *fallback*.
    """
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _scene_to_narrative_line(scene: dict[str, Any]) -> str:
    """Convert a single scene dict into a one-line narrative sentence.

    The resulting line includes the scene's timecode, visual description
    and spoken dialogue.

    Args:
        scene: Scene dictionary that may contain ``start_timecode``,
            ``llm_scene_description``, and ``audio_speech`` keys.

    Returns:
        A human-readable narrative sentence describing the scene.
    """
    start_timecode = _normalize_scene_text(scene.get("start_timecode"), NOT_STATED)
    llm_scene_description = _normalize_scene_text(
        scene.get("llm_scene_description"), "No visual description."
    )
    audio_speech = _normalize_scene_text(
        scene.get("audio_speech"), "No spoken dialogue."
    )
    return f'At {start_timecode}, {llm_scene_description} It says "{audio_speech}".'


def chunk_scenes(
    scenes: list[dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    debug: bool = False,
) -> list[dict[str, Any]]:
    """Split a list of scene dicts into text chunks of bounded size.

    Each chunk aggregates consecutive scenes until the cumulative
    character length reaches *chunk_size*.  Metadata (scene indices and
    timecodes) are attached to every chunk.

    Args:
        scenes: List of scene dictionaries to chunk.
        chunk_size: Maximum character length per chunk.
        debug: If ``True``, emit debug log messages.

    Returns:
        A list of chunk dictionaries, each containing:
        - ``index``: zero-based chunk index.
        - ``text``: concatenated narrative lines.
        - ``scene_start_idx`` / ``scene_end_idx``: inclusive scene indices.
        - ``start_timecode`` / ``end_timecode``: boundary timecodes.
    """
    scene_count = len(scenes) if isinstance(scenes, list) else 0
    if scene_count == 0:
        _mapreduce_log(debug, "chunk_scenes: no scenes to chunk")
        return []

    chunks: list[dict[str, Any]] = []
    this_chunk = ""
    chunk_start_idx: int | None = None

    for idx, scene in enumerate(scenes):
        scene_obj = scene if isinstance(scene, dict) else {}
        text = _scene_to_narrative_line(scene_obj)
        candidate = f"{this_chunk}\n{text}".strip() if this_chunk else text

        if chunk_start_idx is None:
            chunk_start_idx = idx
        this_chunk = candidate

        if len(this_chunk) >= chunk_size:
            start_scene = (
                scenes[chunk_start_idx]
                if isinstance(scenes[chunk_start_idx], dict)
                else {}
            )
            end_scene = scene_obj
            chunks.append(
                {
                    "index": len(chunks),
                    "text": this_chunk,
                    "scene_start_idx": chunk_start_idx,
                    "scene_end_idx": idx,
                    "start_timecode": start_scene.get("start_timecode"),
                    "end_timecode": end_scene.get("end_timecode")
                    or end_scene.get("start_timecode"),
                }
            )
            this_chunk = ""
            chunk_start_idx = None

    if this_chunk:
        end_idx = len(scenes) - 1
        start_idx = chunk_start_idx if chunk_start_idx is not None else end_idx
        start_scene = scenes[start_idx] if isinstance(scenes[start_idx], dict) else {}
        end_scene = scenes[end_idx] if isinstance(scenes[end_idx], dict) else {}
        chunks.append(
            {
                "index": len(chunks),
                "text": this_chunk,
                "scene_start_idx": start_idx,
                "scene_end_idx": end_idx,
                "start_timecode": start_scene.get("start_timecode"),
                "end_timecode": end_scene.get("end_timecode")
                or end_scene.get("start_timecode"),
            }
        )

    _mapreduce_log(
        debug, f"chunk_scenes: {scene_count} scenes -> {len(chunks)} chunk_scenes"
    )
    return chunks


def chunk_narrative(
    narrative: str,
    chunk_size: int = CHUNK_SIZE,
    debug: bool = False,
) -> list[str]:
    """Split a narrative string into paragraph-aware text chunks.

    Paragraphs are separated by double newlines.  A paragraph that
    exceeds *chunk_size* on its own is split at the character boundary.

    Args:
        narrative: The full narrative text to split.
        chunk_size: Maximum character length per chunk.
        debug: If ``True``, emit debug log messages.

    Returns:
        A list of text chunks, each at most *chunk_size* characters long
        (except when a single paragraph exceeds the limit).
    """
    paragraphs = [p.strip() for p in narrative.split("\n\n") if p.strip()]
    chunks: list[str] = []
    this_chunk = ""

    for para in paragraphs:
        candidate = f"{this_chunk}\n\n{para}".strip() if this_chunk else para
        if len(candidate) <= chunk_size:
            this_chunk = candidate
        else:
            if this_chunk:
                chunks.append(this_chunk)
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size):
                    chunks.append(para[i : i + chunk_size])
                this_chunk = ""
            else:
                this_chunk = para

    if this_chunk:
        chunks.append(this_chunk)
    _mapreduce_log(
        debug,
        f"chunk_narrative: splitting narrative len={len(narrative)} "
        f"to {len(chunks)} chunks",
    )
    return chunks


def condense_chunk(
    call_gpt_fn: Callable[[str], str],
    chunk_text: str,
    pre_carryover_context: str,
    segment_prompt_template: str,
    fallback_prompt_template: str,
    carryover_prompt_template: str,
    debug: bool = False,
) -> tuple[str, str]:
    """Condense a single chunk of text using an LLM with carryover context.

    The function first attempts the primary *segment_prompt_template*,
    falls back to *fallback_prompt_template* on failure, and then
    generates updated carryover context from the result.

    Args:
        call_gpt_fn: Callable that accepts a prompt string and returns
            the LLM response text.
        chunk_text: The chunk of scene text to condense.
        pre_carryover_context: Context carried over from the previous
            chunk (may be empty for the first chunk).
        segment_prompt_template: Primary prompt template with
            ``{carryover_context}`` and ``{scene_chunk}`` placeholders.
        fallback_prompt_template: Fallback prompt template used when
            the primary prompt fails.
        carryover_prompt_template: Prompt template for extracting
            carryover context, with a ``{segment_narrative}``
            placeholder.
        debug: If ``True``, emit debug log messages.

    Returns:
        A 2-tuple of ``(summary, new_carryover_context)``.
    """
    segment_prompt = segment_prompt_template.format(
        carryover_context=pre_carryover_context, scene_chunk=chunk_text
    )
    summary: str | None = None
    try:
        summary = call_gpt_fn(segment_prompt)
    except Exception as exc:
        _mapreduce_log(debug, f"condense_chunk: primary prompt failed: {exc}")
        try:
            fallback_prompt = fallback_prompt_template.format(
                carryover_context=pre_carryover_context, scene_chunk=chunk_text
            )
            summary = call_gpt_fn(fallback_prompt)
        except Exception as exc2:
            _mapreduce_log(debug, f"condense_chunk: fallback prompt failed: {exc2}")
            summary = chunk_text

    carryover_prompt = carryover_prompt_template.format(segment_narrative=summary)
    try:
        new_carryover_context = call_gpt_fn(carryover_prompt)
    except Exception as exc:
        _mapreduce_log(debug, f"condense_chunk: carryover prompt failed: {exc}")
        new_carryover_context = pre_carryover_context
    _mapreduce_log(
        debug, f"    condense_chunk: len={len(chunk_text)} -> len={len(summary)}"
    )
    return summary, new_carryover_context


def parallel_map_summaries(
    call_gpt_fn: Callable[[str], str],
    scene_chunks: list[dict[str, Any]],
    max_workers: int,
    debug: bool = False,
) -> list[dict[str, Any]]:
    """Summarise each scene chunk in parallel (map phase).

    Each chunk is processed independently via *call_gpt_fn*.  Results
    are returned in the original chunk order.

    Args:
        call_gpt_fn: Callable that accepts a prompt string and returns
            the LLM response text.
        scene_chunks: List of chunk dicts as produced by
            :func:`chunk_scenes`.
        max_workers: Maximum number of threads for parallel execution.
        debug: If ``True``, emit debug log messages.

    Returns:
        Ordered list of summary dicts, each containing ``index``,
        ``text``, ``scene_start_idx``, ``scene_end_idx``,
        ``start_timecode``, and ``end_timecode``.
    """
    if not scene_chunks:
        return []
    results: list[dict[str, Any] | None] = [None] * len(scene_chunks)

    def _task(chunk: dict[str, Any]) -> dict[str, Any]:
        prompt = _build_scene_chunk_summary_prompt(chunk)
        try:
            summary = call_gpt_fn(prompt)
        except Exception as exc:
            _mapreduce_log(
                debug, f"parallel_map_summaries: chunk {chunk['index']} failed: {exc}"
            )
            summary = chunk["text"]
        _mapreduce_log(
            debug,
            f"    map_summary: len={len(chunk['text'])} -> len={len(summary.strip())}",
        )
        return {
            "index": chunk["index"],
            "text": summary.strip(),
            "scene_start_idx": chunk["scene_start_idx"],
            "scene_end_idx": chunk["scene_end_idx"],
            "start_timecode": chunk.get("start_timecode"),
            "end_timecode": chunk.get("end_timecode"),
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_task, chunk) for chunk in scene_chunks]
        for future in as_completed(futures):
            item = future.result()
            results[item["index"]] = item

    return [r for r in results if r is not None]


def parallel_reduce_summaries(
    call_gpt_fn: Callable[[str], str],
    summaries: list[dict[str, Any]],
    reduce_group_size: int = SUMMARY_REDUCE_GROUP_SIZE,
    max_workers: int = SUMMARY_MAX_WORKERS,
    debug: bool = False,
) -> dict[str, Any] | None:
    """Hierarchically merge mapped summaries into a single summary (reduce phase).

    Adjacent summaries are grouped and merged in parallel rounds until
    only one summary remains.

    Args:
        call_gpt_fn: Callable that accepts a prompt string and returns
            the LLM response text.
        summaries: Ordered list of summary dicts from the map phase.
        reduce_group_size: Number of adjacent summaries to merge per
            group in each round.
        max_workers: Maximum number of threads for parallel execution.
        debug: If ``True``, emit debug log messages.

    Returns:
        A single merged summary dict, or ``None`` if the input list is
        empty.
    """
    current = summaries
    round_idx = 0
    while len(current) > 1:
        round_idx += 1
        groups = [
            current[i : i + reduce_group_size]
            for i in range(0, len(current), reduce_group_size)
        ]
        reduced: list[dict[str, Any] | None] = [None] * len(groups)

        def _task(
            group_idx: int,
            group_items: list[dict[str, Any]],
            _round: int = round_idx,
        ) -> tuple[int, dict[str, Any]]:
            prompt = _build_reduce_prompt(group_items, _round)
            try:
                merged = call_gpt_fn(prompt).strip()
            except Exception as exc:
                _mapreduce_log(
                    debug, f"parallel_reduce_summaries: group {group_idx} failed: {exc}"
                )
                merged = "\n".join(item["text"] for item in group_items)
            return group_idx, {
                "index": group_idx,
                "text": merged,
                "scene_start_idx": group_items[0]["scene_start_idx"],
                "scene_end_idx": group_items[-1]["scene_end_idx"],
                "start_timecode": group_items[0].get("start_timecode"),
                "end_timecode": group_items[-1].get("end_timecode"),
            }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_task, idx, grp) for idx, grp in enumerate(groups)
            ]
            for future in as_completed(futures):
                group_idx, result = future.result()
                reduced[group_idx] = result
        current = [item for item in reduced if item is not None]
        _mapreduce_log(
            debug,
            f"parallel_reduce_summaries: round={round_idx}, "
            f"groups={len(groups)}, next={len(current)}",
        )

    return current[0] if current else None
