"""LLM-powered scene description: format raw data, call GPT/Gemini, two-stage map-reduce."""

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kairos.utils import apply_gpt_normalization, print_prefixed


def describe_flash_scene(
    scene_text: str,
    client,
    prompt_path: str = "prompts/describe_scene.txt",
    model: str = "gemini-2.5-flash",
    gpt_deployment: str = "gpt-4o-kairos",
    gpt_temperature: float = 0.3,
    video_path: str | None = None,
) -> str:
    """Generate an LLM summary for a single scene."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    normalized_text = apply_gpt_normalization(scene_text)
    video_name = Path(video_path).name if video_path else ""
    prompt = template.replace("{{SCENE_TEXT}}", normalized_text)
    prompt = prompt.replace("{{VIDEO_NAME}}", video_name)

    if "gemini" in model.lower():
        chat = client.chats.create(model=model)
        return chat.send_message(prompt).text.strip()
    elif "gpt" in model.lower():
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes visual scenes."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048, temperature=gpt_temperature, top_p=1.0, model=gpt_deployment,
        )
        return response.choices[0].message.content
    return ""


def normalize_bbox(bbox):
    x1, y1, x2, y2 = bbox
    w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, w * h


def format_single_description(captions: list, yolo) -> str:
    lines = []
    if isinstance(yolo, list):
        from kairos.object_detection import format_track_summaries
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
                    f"    - {det.get('label', 'unknown')} (conf={det.get('confidence', 0.0):.2f}), "
                    f"x_center={x_center:.1f}, y_center={y_center:.1f}, area={area:.1f}"
                )
        else:
            lines.append("  Objects: none detected.")
        lines.append("")
    return "\n".join(lines)


def raw_descriptions(scenes, YOLO_key="yolo_detections", FLIP_key="frame_captions",
                     ASR_key="audio_natural", AST_key="audio_speech") -> list:
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


def describe_scenes(
    scenes: list, client, hist_size=3,
    YOLO_key="yolo_detections", FLIP_key="frame_captions",
    ASR_key="audio_natural", AST_key="audio_speech",
    SUMMARY_key="llm_scene_description",
    model="gemini-2.5-flash",
    prompt_path="prompts/describe_scene.txt",
    short_prompt_path="prompts/describe_scene_short.txt",
    fallback_prompt_path="prompts/fallback_describe_scene.txt",
    short_fallback_prompt_path: str | None = None,
    max_workers: int | None = None,
    rate_limit_cooldown_sec: float = 20.0,
    max_rate_limit_retries: int = 4,
    video_path: str | None = None,
    cooldown_sec: float = 5,
    debug=False,
) -> list:
    """Two-stage (map-reduce) scene description pipeline."""
    if short_fallback_prompt_path is None:
        short_fallback_prompt_path = fallback_prompt_path

    formatted_scenes = raw_descriptions(scenes, YOLO_key=YOLO_key, FLIP_key=FLIP_key, ASR_key=ASR_key, AST_key=AST_key)

    if max_workers is None:
        max_workers = min(8, max(1, len(formatted_scenes)))
    else:
        max_workers = max(1, int(max_workers))

    def _is_rate_limit_error(exc):
        err_text = f"{type(exc).__name__}: {exc}".lower()
        return any(m in err_text for m in ("rate limit", "ratelimit", "too many requests", "429", "quota exceeded", "resource exhausted", "request rate"))

    def _call_with_retry(scene_idx, scene_text, prompt_used):
        attempt = 0
        while True:
            try:
                return describe_flash_scene(scene_text, client, prompt_path=prompt_used, model=model, video_path=video_path)
            except Exception as exc:
                if _is_rate_limit_error(exc) and attempt < max_rate_limit_retries:
                    wait_sec = rate_limit_cooldown_sec * (2 ** attempt) + random.uniform(0.0, 1.0)
                    if debug:
                        print_prefixed("(RATE)", f"Scene {scene_idx} rate-limited; cooling down {wait_sec:.1f}s")
                    time.sleep(wait_sec)
                    attempt += 1
                    continue
                raise

    def _generate_with_fallback(scene_idx, scene_text, primary_prompt, fallback_prompt):
        result = None
        try:
            result = _call_with_retry(scene_idx, scene_text, primary_prompt)
        except Exception as exc:
            if debug:
                print_prefixed("(WARN)", f"Scene {scene_idx} primary failed: {exc}")
            if fallback_prompt:
                try:
                    result = _call_with_retry(scene_idx, scene_text, fallback_prompt)
                except Exception as exc2:
                    if debug:
                        print_prefixed("(WARN)", f"Scene {scene_idx} fallback failed: {exc2}")
        finally:
            if cooldown_sec and cooldown_sec > 0:
                time.sleep(cooldown_sec)
        return result

    def _parallel_map(inputs, primary_prompt, fallback_prompt):
        outputs = [None] * len(inputs)
        if max_workers <= 1 or len(inputs) <= 1:
            for i, text in enumerate(inputs):
                outputs[i] = _generate_with_fallback(i, text, primary_prompt, fallback_prompt)
            return outputs
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_generate_with_fallback, i, text, primary_prompt, fallback_prompt): i
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

    def _build_short_context(short_summaries, idx):
        if hist_size <= 0 or idx <= 0:
            return ""
        start_idx = max(0, idx - hist_size)
        lines = ["Previous scenes (short summaries):"]
        for j in range(start_idx, idx):
            prev = (short_summaries[j] or "").strip()
            if prev:
                lines.append(f"Scene -{idx - j}:\n{prev}")
        return "" if len(lines) == 1 else "\n\n" + "\n\n".join(lines)

    # Stage 1: short summaries in parallel
    short_summaries = _parallel_map(formatted_scenes, short_prompt_path, short_fallback_prompt_path)

    # Stage 2: full reports using raw scene + Stage-1 context
    stage2_inputs = [raw + _build_short_context(short_summaries, i) for i, raw in enumerate(formatted_scenes)]
    final_summaries = _parallel_map(stage2_inputs, prompt_path, fallback_prompt_path)

    updated = []
    for idx, (scene, summary) in enumerate(zip(scenes, final_summaries)):
        if not summary:
            continue
        new_scene = dict(scene)
        new_scene[SUMMARY_key] = summary
        updated.append(new_scene)
        if debug:
            print_prefixed("(GPT4o)", f"----- Scene {idx} -----")
            print(summary.strip())
            print("")

    return updated
