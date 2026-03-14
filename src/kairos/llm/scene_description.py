"""LLM-powered scene description.

Format raw data, call GPT/Gemini, two-stage map-reduce.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kairos.core.utils import (
    PROMPTS_DIR,
    apply_gpt_normalization,
    print_prefixed,
    retry_with_backoff,
)


def describe_flash_scene(
    scene_text: str,
    client,
    prompt_path: str | None = None,
    gpt_temperature: float = 0.3,
    video_path: str | None = None,
) -> str:
    """Generate an LLM summary for a single scene."""
    if prompt_path is None:
        prompt_path = str(PROMPTS_DIR / "describe_scene.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    normalized_text = apply_gpt_normalization(scene_text)
    video_name = Path(video_path).name if video_path else ""
    prompt = template.replace("{{SCENE_TEXT}}", normalized_text)
    prompt = prompt.replace("{{VIDEO_NAME}}", video_name)

    return client.generate(
        prompt,
        system="You are a helpful assistant that summarizes visual scenes.",
        max_tokens=2048,
        temperature=gpt_temperature,
    )


def normalize_bbox(bbox):
    x1, y1, x2, y2 = bbox
    w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, w * h


def format_single_description(captions: list, yolo) -> str:
    lines = []
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
    frame_count = max(len(captions), max((int(k) for k in yolo.keys()), default=-1) + 1)
    for idx in range(frame_count):
        lines.append(f"Frame {idx}:")
        if captions and idx < len(captions):
            lines.append(f'  Caption: "{captions[idx]}"')
        dets = yolo.get(idx, yolo.get(str(idx), [])) or []
        if dets:
            lines.append("  Objects:")
            for det in dets:
                x_center, y_center, area = normalize_bbox(det.get("bbox", [0, 0, 0, 0]))
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
    scenes,
    YOLO_key="yolo_detections",
    FLIP_key="frame_captions",
    ASR_key="audio_natural",
    AST_key="audio_speech",
) -> list:
    formatted_list = []
    for scene in scenes:
        captions = scene.get(FLIP_key, []) if FLIP_key else []
        yolo = scene.get(YOLO_key, {}) if YOLO_key else {}
        asr = scene.get(ASR_key, "") if ASR_key else ""
        ast = scene.get(AST_key, "") if AST_key else ""
        text = format_single_description(captions=captions, yolo=yolo)
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
    scene_idx,
    scene_text,
    prompt_used,
    client,
    video_path,
    max_retries,
    cooldown_sec,
    debug,
):
    """Call describe_flash_scene with rate-limit retry logic."""
    return retry_with_backoff(
        lambda: describe_flash_scene(
            scene_text, client, prompt_path=prompt_used, video_path=video_path
        ),
        max_retries=max_retries,
        base_sec=cooldown_sec,
    )


def _generate_with_fallback(
    scene_idx,
    scene_text,
    primary_prompt,
    fallback_prompt,
    client,
    video_path,
    max_retries,
    rate_cooldown_sec,
    post_cooldown_sec,
    debug,
):
    """Try primary prompt, then fallback, with cooldown between calls."""
    result = None
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
    inputs,
    primary_prompt,
    fallback_prompt,
    max_workers,
    client,
    video_path,
    max_retries,
    rate_cooldown_sec,
    post_cooldown_sec,
    debug,
):
    """Map scene descriptions in parallel (or sequentially if max_workers <= 1)."""
    call_kwargs = dict(
        client=client,
        video_path=video_path,
        max_retries=max_retries,
        rate_cooldown_sec=rate_cooldown_sec,
        post_cooldown_sec=post_cooldown_sec,
        debug=debug,
    )
    outputs = [None] * len(inputs)
    if max_workers <= 1 or len(inputs) <= 1:
        for i, text in enumerate(inputs):
            outputs[i] = _generate_with_fallback(
                i, text, primary_prompt, fallback_prompt, **call_kwargs
            )
        return outputs
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
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
            i = future_to_idx[future]
            try:
                outputs[i] = future.result()
            except Exception as exc:
                if debug:
                    print_prefixed("(WARN)", f"Scene {i} worker failed: {exc}")
    return outputs


def _build_short_context(short_summaries, idx, hist_size):
    """Build context string from previous short summaries for a given scene index."""
    if hist_size <= 0 or idx <= 0:
        return ""
    start_idx = max(0, idx - hist_size)
    lines = ["Previous scenes (short summaries):"]
    for j in range(start_idx, idx):
        prev = (short_summaries[j] or "").strip()
        if prev:
            lines.append(f"Scene -{idx - j}:\n{prev}")
    return "" if len(lines) == 1 else "\n\n" + "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def describe_scenes(
    scenes: list,
    client,
    hist_size=3,
    YOLO_key="yolo_detections",
    FLIP_key="frame_captions",
    ASR_key="audio_natural",
    AST_key="audio_speech",
    SUMMARY_key="llm_scene_description",
    prompt_path=None,
    short_prompt_path=None,
    fallback_prompt_path=None,
    short_fallback_prompt_path: str | None = None,
    max_workers: int | None = None,
    rate_limit_cooldown_sec: float = 20.0,
    max_rate_limit_retries: int = 4,
    video_path: str | None = None,
    cooldown_sec: float = 5,
    debug=False,
) -> list:
    """Two-stage (map-reduce) scene description pipeline."""
    if prompt_path is None:
        prompt_path = str(PROMPTS_DIR / "describe_scene.txt")
    if short_prompt_path is None:
        short_prompt_path = str(PROMPTS_DIR / "describe_scene_short.txt")
    if fallback_prompt_path is None:
        fallback_prompt_path = str(PROMPTS_DIR / "fallback_describe_scene.txt")
    if short_fallback_prompt_path is None:
        short_fallback_prompt_path = fallback_prompt_path

    formatted_scenes = raw_descriptions(
        scenes, YOLO_key=YOLO_key, FLIP_key=FLIP_key, ASR_key=ASR_key, AST_key=AST_key
    )

    if max_workers is None:
        max_workers = min(8, max(1, len(formatted_scenes)))
    else:
        max_workers = max(1, int(max_workers))

    map_kwargs = dict(
        max_workers=max_workers,
        client=client,
        video_path=video_path,
        max_retries=max_rate_limit_retries,
        rate_cooldown_sec=rate_limit_cooldown_sec,
        post_cooldown_sec=cooldown_sec,
        debug=debug,
    )

    # Stage 1: short summaries in parallel
    short_summaries = _parallel_map(
        formatted_scenes, short_prompt_path, short_fallback_prompt_path, **map_kwargs
    )

    # Stage 2: full reports using raw scene + Stage-1 context
    stage2_inputs = [
        raw + _build_short_context(short_summaries, i, hist_size)
        for i, raw in enumerate(formatted_scenes)
    ]
    final_summaries = _parallel_map(
        stage2_inputs, prompt_path, fallback_prompt_path, **map_kwargs
    )

    updated = []
    for idx, (scene, summary) in enumerate(zip(scenes, final_summaries)):
        if not summary:
            continue
        new_scene = dict(scene)
        new_scene[SUMMARY_key] = summary
        updated.append(new_scene)
        if debug:
            print_prefixed("(GPT4o)", f"----- Scene {idx} -----")
            print_prefixed("(GPT4o)", summary.strip())

    return updated
