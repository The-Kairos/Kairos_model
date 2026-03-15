"""LLM-powered scene description.

Format raw data, call GPT/Gemini, two-stage map-reduce.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from kairos.core.utils import (
    PROMPTS_DIR,
    apply_gpt_normalization,
    print_prefixed,
    retry_with_backoff,
)
from kairos.llm.client import LLMClient


def describe_flash_scene(
    scene_text: str,
    client: LLMClient,
    prompt_path: str | None = None,
    gpt_temperature: float = 0.3,
    video_path: str | None = None,
) -> str:
    """Generate an LLM summary for a single scene.

    Loads a prompt template from *prompt_path*, injects the normalised
    scene text and video name, then delegates to the LLM *client*.

    Args:
        scene_text: Raw formatted text describing the scene (frames,
            objects, audio, captions).
        client: An :class:`~kairos.llm.client.LLMClient` used for
            generation.
        prompt_path: Path to a prompt template file.  When ``None``,
            defaults to ``PROMPTS_DIR / "describe_scene.txt"``.
        gpt_temperature: Sampling temperature forwarded to the client.
            Defaults to ``0.3``.
        video_path: Optional path to the source video, used to extract
            the video filename for the prompt. Defaults to ``None``.

    Returns:
        str: The LLM-generated scene summary.
    """
    if prompt_path is None:
        prompt_path = str(PROMPTS_DIR / "describe_scene.txt")
    with open(prompt_path, encoding="utf-8") as f:
        template: str = f.read()

    normalized_text: str = apply_gpt_normalization(scene_text)
    video_name: str = Path(video_path).name if video_path else ""
    prompt: str = template.replace("{{SCENE_TEXT}}", normalized_text)
    prompt = prompt.replace("{{VIDEO_NAME}}", video_name)

    return client.generate(
        prompt,
        system="You are a helpful assistant that summarizes visual scenes.",
        max_tokens=2048,
        temperature=gpt_temperature,
    )


def normalize_bbox(bbox: list[float]) -> tuple[float, float, float]:
    """Compute centre coordinates and area from a bounding box.

    Args:
        bbox: A list of four floats ``[x1, y1, x2, y2]`` representing
            the top-left and bottom-right corners.

    Returns:
        tuple[float, float, float]: A tuple of ``(x_center, y_center,
            area)`` where *area* is ``width * height``.
    """
    x1, y1, x2, y2 = bbox
    w: float = max(x2 - x1, 1e-6)
    h: float = max(y2 - y1, 1e-6)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, w * h


def format_single_description(
    captions: list[str],
    yolo: list[dict[str, Any]] | dict[str | int, list[dict[str, Any]]],
) -> str:
    """Format captions and YOLO detections for a single scene into text.

    Supports two YOLO formats:

    * **Track-summary list** (``list``): Each entry is a track summary
      dict.  Formatted via
      :func:`~kairos.video.track_summary.format_track_summaries`.
    * **Legacy per-frame dict** (``dict``): Keys are frame indices
      (``int`` or ``str``), values are lists of detection dicts.

    Args:
        captions: Per-frame caption strings.
        yolo: YOLO detections in either the track-summary list format or
            the legacy per-frame dictionary format.

    Returns:
        str: A multi-line text block describing frames, captions,
            objects and tracks.
    """
    lines: list[str] = []
    if isinstance(yolo, list):
        from kairos.video.track_summary import format_track_summaries

        for idx, cap in enumerate(captions or []):
            lines.append(f"Frame {idx}:")
            lines.append(f'  Caption: "{cap}"')
            lines.append("")
        if yolo:
            lines.append("Tracks:")
            for line in format_track_summaries(yolo, style="narrative"):
                lines.append(f"  - {line}")
        else:
            lines.append("Tracks: none detected.")
        return "\n".join(lines)

    # Legacy per-frame yolo dict format
    frame_count: int = max(
        len(captions), max((int(k) for k in yolo), default=-1) + 1
    )
    for idx in range(frame_count):
        lines.append(f"Frame {idx}:")
        if captions and idx < len(captions):
            lines.append(f'  Caption: "{captions[idx]}"')
        dets: list[dict[str, Any]] = yolo.get(idx, yolo.get(str(idx), [])) or []
        if dets:
            lines.append("  Objects:")
            for det in dets:
                x_center, y_center, area = normalize_bbox(
                    det.get("bbox", [0, 0, 0, 0])
                )
                lines.append(
                    f"    - {det.get('label', 'unknown')} "
                    f"(conf={det.get('confidence', 0.0):.2f}), "
                    f"x_center={x_center:.1f}, "
                    f"y_center={y_center:.1f}, "
                    f"area={area:.1f}"
                )
        else:
            lines.append("  Objects: none detected.")
        lines.append("")
    return "\n".join(lines)


def raw_descriptions(
    scenes: list[dict[str, Any]],
    YOLO_key: str = "yolo_detections",
    FLIP_key: str = "frame_captions",
    ASR_key: str = "audio_natural",
    AST_key: str = "audio_speech",
) -> list[str]:
    """Convert a list of scene dicts into raw formatted text descriptions.

    For each scene the function extracts captions, YOLO detections, and
    audio fields, then formats them via :func:`format_single_description`
    with audio lines appended.

    Args:
        scenes: List of scene dictionaries.
        YOLO_key: Key in each scene dict for YOLO detections.
            Defaults to ``"yolo_detections"``.
        FLIP_key: Key for frame captions. Defaults to
            ``"frame_captions"``.
        ASR_key: Key for natural audio transcription. Defaults to
            ``"audio_natural"``.
        AST_key: Key for speech transcription. Defaults to
            ``"audio_speech"``.

    Returns:
        list[str]: One formatted text string per scene.
    """
    formatted_list: list[str] = []
    for scene in scenes:
        captions: list[str] = scene.get(FLIP_key, []) if FLIP_key else []
        yolo: dict[str, Any] | list[Any] = scene.get(YOLO_key, {}) if YOLO_key else {}
        asr: str = scene.get(ASR_key, "") if ASR_key else ""
        ast: str = scene.get(AST_key, "") if AST_key else ""
        text: str = format_single_description(captions=captions, yolo=yolo)
        if asr:
            text += f"\nAudio transcript: {asr}"
        if ast:
            text += f"\nAudio sounds: {ast}\n"
        formatted_list.append(text)
    return formatted_list


# ---------------------------------------------------------------------------
# Scene description helpers (extracted from describe_scenes)
# ---------------------------------------------------------------------------


def _call_with_retry(
    scene_idx: int,
    scene_text: str,
    prompt_used: str,
    client: LLMClient,
    video_path: str | None,
    max_retries: int,
    cooldown_sec: float,
    debug: bool,
) -> str:
    """Call :func:`describe_flash_scene` with rate-limit retry logic.

    Wraps the generation call in :func:`~kairos.core.utils.retry_with_backoff`
    to handle transient API errors and rate-limit responses.

    Args:
        scene_idx: Zero-based index of the scene (used for logging).
        scene_text: The formatted scene text to describe.
        prompt_used: Path to the prompt template file.
        client: The LLM client to use for generation.
        video_path: Optional source video path.
        max_retries: Maximum number of retry attempts.
        cooldown_sec: Base wait time (in seconds) between retries,
            subject to exponential back-off.
        debug: Whether to emit debug output (currently unused inside
            this helper but forwarded for consistency).

    Returns:
        str: The generated description text.
    """
    return retry_with_backoff(
        lambda: describe_flash_scene(
            scene_text, client, prompt_path=prompt_used, video_path=video_path
        ),
        max_retries=max_retries,
        base_sec=cooldown_sec,
    )


def _generate_with_fallback(
    scene_idx: int,
    scene_text: str,
    primary_prompt: str,
    fallback_prompt: str | None,
    client: LLMClient,
    video_path: str | None,
    max_retries: int,
    rate_cooldown_sec: float,
    post_cooldown_sec: float,
    debug: bool,
) -> str | None:
    """Try the primary prompt then an optional fallback, with cooldown.

    If the primary prompt fails and a *fallback_prompt* is provided, the
    function retries with the fallback.  A post-call cooldown sleep is
    always applied regardless of success or failure.

    Args:
        scene_idx: Zero-based scene index (for logging).
        scene_text: The formatted scene text.
        primary_prompt: Path to the primary prompt template.
        fallback_prompt: Path to the fallback prompt template, or
            ``None`` to skip fallback.
        client: The LLM client.
        video_path: Optional source video path.
        max_retries: Maximum retry attempts per prompt.
        rate_cooldown_sec: Base back-off time for rate-limit retries.
        post_cooldown_sec: Fixed cooldown (seconds) after the call
            completes.
        debug: If ``True``, warnings are printed on failure.

    Returns:
        str | None: The generated description, or ``None`` if both
            attempts fail.
    """
    result: str | None = None
    try:
        result = _call_with_retry(
            scene_idx,
            scene_text,
            primary_prompt,
            client,
            video_path,
            max_retries,
            rate_cooldown_sec,
            debug,
        )
    except Exception as exc:
        if debug:
            print_prefixed("(WARN)", f"Scene {scene_idx} primary failed: {exc}")
        if fallback_prompt:
            try:
                result = _call_with_retry(
                    scene_idx,
                    scene_text,
                    fallback_prompt,
                    client,
                    video_path,
                    max_retries,
                    rate_cooldown_sec,
                    debug,
                )
            except Exception as exc2:
                if debug:
                    print_prefixed(
                        "(WARN)", f"Scene {scene_idx} fallback failed: {exc2}"
                    )
    finally:
        if post_cooldown_sec and post_cooldown_sec > 0:
            time.sleep(post_cooldown_sec)
    return result


def _parallel_map(
    inputs: list[str],
    primary_prompt: str,
    fallback_prompt: str | None,
    max_workers: int,
    client: LLMClient,
    video_path: str | None,
    max_retries: int,
    rate_cooldown_sec: float,
    post_cooldown_sec: float,
    debug: bool,
) -> list[str | None]:
    """Map scene descriptions in parallel (or sequentially if *max_workers* ≤ 1).

    Each input text is processed by :func:`_generate_with_fallback`.
    When *max_workers* is greater than 1, a
    :class:`~concurrent.futures.ThreadPoolExecutor` is used.

    Args:
        inputs: Formatted scene texts to process.
        primary_prompt: Path to the primary prompt template.
        fallback_prompt: Path to the fallback prompt template, or
            ``None``.
        max_workers: Maximum number of parallel threads.
        client: The LLM client.
        video_path: Optional source video path.
        max_retries: Maximum retry attempts per prompt.
        rate_cooldown_sec: Base back-off time for retries.
        post_cooldown_sec: Fixed post-call cooldown (seconds).
        debug: If ``True``, warnings are printed on failure.

    Returns:
        list[str | None]: A list with one result per input (``None``
            for failed scenes), preserving the original order.
    """
    call_kwargs: dict[str, Any] = dict(
        client=client,
        video_path=video_path,
        max_retries=max_retries,
        rate_cooldown_sec=rate_cooldown_sec,
        post_cooldown_sec=post_cooldown_sec,
        debug=debug,
    )
    outputs: list[str | None] = [None] * len(inputs)
    if max_workers <= 1 or len(inputs) <= 1:
        for i, text in enumerate(inputs):
            outputs[i] = _generate_with_fallback(
                i, text, primary_prompt, fallback_prompt, **call_kwargs
            )
        return outputs
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx: dict[Any, int] = {
            executor.submit(
                _generate_with_fallback,
                i,
                text,
                primary_prompt,
                fallback_prompt,
                **call_kwargs,
            ): i
            for i, text in enumerate(inputs)
        }
        for future in as_completed(future_to_idx):
            i: int = future_to_idx[future]
            try:
                outputs[i] = future.result()
            except Exception as exc:
                if debug:
                    print_prefixed("(WARN)", f"Scene {i} worker failed: {exc}")
    return outputs


def _build_short_context(
    short_summaries: list[str | None],
    idx: int,
    hist_size: int,
) -> str:
    """Build a context string from previous short summaries for a scene.

    Used by the second stage of the map-reduce pipeline to give the LLM
    awareness of what happened in recent scenes.

    Args:
        short_summaries: The Stage-1 short summaries (may contain
            ``None`` for failed scenes).
        idx: The zero-based index of the current scene.
        hist_size: How many preceding summaries to include. A value of
            ``0`` disables context.

    Returns:
        str: A formatted context block, or an empty string when no
            usable history is available.
    """
    if hist_size <= 0 or idx <= 0:
        return ""
    start_idx: int = max(0, idx - hist_size)
    lines: list[str] = ["Previous scenes (short summaries):"]
    for j in range(start_idx, idx):
        prev: str = (short_summaries[j] or "").strip()
        if prev:
            lines.append(f"Scene -{idx - j}:\n{prev}")
    return "" if len(lines) == 1 else "\n\n" + "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def describe_scenes(
    scenes: list[dict[str, Any]],
    client: LLMClient,
    hist_size: int = 3,
    YOLO_key: str = "yolo_detections",
    FLIP_key: str = "frame_captions",
    ASR_key: str = "audio_natural",
    AST_key: str = "audio_speech",
    SUMMARY_key: str = "llm_scene_description",
    prompt_path: str | None = None,
    short_prompt_path: str | None = None,
    fallback_prompt_path: str | None = None,
    short_fallback_prompt_path: str | None = None,
    max_workers: int | None = None,
    rate_limit_cooldown_sec: float = 20.0,
    max_rate_limit_retries: int = 4,
    video_path: str | None = None,
    cooldown_sec: float = 5,
    debug: bool = False,
) -> list[dict[str, Any]]:
    """Two-stage (map-reduce) scene description pipeline.

    **Stage 1 (map):** Generate short summaries for every scene in
    parallel using the *short_prompt_path* template.

    **Stage 2 (reduce):** Generate full descriptions using the primary
    prompt, augmented with a rolling window of Stage-1 short summaries
    for temporal context.

    Args:
        scenes: List of scene dictionaries to describe.
        client: The LLM client used for generation.
        hist_size: Number of preceding short summaries to include as
            context in Stage 2. Defaults to ``3``.
        YOLO_key: Scene-dict key for YOLO detections.
            Defaults to ``"yolo_detections"``.
        FLIP_key: Scene-dict key for frame captions.
            Defaults to ``"frame_captions"``.
        ASR_key: Scene-dict key for natural audio.
            Defaults to ``"audio_natural"``.
        AST_key: Scene-dict key for speech audio.
            Defaults to ``"audio_speech"``.
        SUMMARY_key: Key under which the final description is stored in
            the output scene dicts. Defaults to
            ``"llm_scene_description"``.
        prompt_path: Path to the Stage-2 (full) prompt template.
            Defaults to ``PROMPTS_DIR / "describe_scene.txt"``.
        short_prompt_path: Path to the Stage-1 (short) prompt template.
            Defaults to ``PROMPTS_DIR / "describe_scene_short.txt"``.
        fallback_prompt_path: Fallback template for Stage-2.
            Defaults to ``PROMPTS_DIR / "fallback_describe_scene.txt"``.
        short_fallback_prompt_path: Fallback template for Stage-1.
            Defaults to the same path as *fallback_prompt_path*.
        max_workers: Maximum parallel threads. When ``None``, defaults
            to ``min(8, len(scenes))``. Defaults to ``None``.
        rate_limit_cooldown_sec: Base back-off time (seconds) for
            rate-limit retries. Defaults to ``20.0``.
        max_rate_limit_retries: Maximum retry attempts per call.
            Defaults to ``4``.
        video_path: Optional path to the source video file.
            Defaults to ``None``.
        cooldown_sec: Fixed post-call sleep (seconds).
            Defaults to ``5``.
        debug: If ``True``, prints each generated description.
            Defaults to ``False``.

    Returns:
        list[dict[str, Any]]: A list of scene dictionaries (copies of
            the originals) augmented with the *SUMMARY_key* field.
            Scenes whose generation failed are omitted.
    """
    if prompt_path is None:
        prompt_path = str(PROMPTS_DIR / "describe_scene.txt")
    if short_prompt_path is None:
        short_prompt_path = str(PROMPTS_DIR / "describe_scene_short.txt")
    if fallback_prompt_path is None:
        fallback_prompt_path = str(PROMPTS_DIR / "fallback_describe_scene.txt")
    if short_fallback_prompt_path is None:
        short_fallback_prompt_path = fallback_prompt_path

    formatted_scenes: list[str] = raw_descriptions(
        scenes, YOLO_key=YOLO_key, FLIP_key=FLIP_key, ASR_key=ASR_key, AST_key=AST_key
    )

    if max_workers is None:
        max_workers = min(8, max(1, len(formatted_scenes)))
    else:
        max_workers = max(1, int(max_workers))

    map_kwargs: dict[str, Any] = dict(
        max_workers=max_workers,
        client=client,
        video_path=video_path,
        max_retries=max_rate_limit_retries,
        rate_cooldown_sec=rate_limit_cooldown_sec,
        post_cooldown_sec=cooldown_sec,
        debug=debug,
    )

    # Stage 1: short summaries in parallel
    short_summaries: list[str | None] = _parallel_map(
        formatted_scenes, short_prompt_path, short_fallback_prompt_path, **map_kwargs
    )

    # Stage 2: full reports using raw scene + Stage-1 context
    stage2_inputs: list[str] = [
        raw + _build_short_context(short_summaries, i, hist_size)
        for i, raw in enumerate(formatted_scenes)
    ]
    final_summaries: list[str | None] = _parallel_map(
        stage2_inputs, prompt_path, fallback_prompt_path, **map_kwargs
    )

    updated: list[dict[str, Any]] = []
    for idx, (scene, summary) in enumerate(
        zip(scenes, final_summaries, strict=False)
    ):
        if not summary:
            continue
        new_scene: dict[str, Any] = dict(scene)
        new_scene[SUMMARY_key] = summary
        updated.append(new_scene)
        if debug:
            print_prefixed("(GPT4o)", f"----- Scene {idx} -----")
            print_prefixed("(GPT4o)", summary.strip())

    return updated
