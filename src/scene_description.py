import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from src.debug_utils import apply_gpt_normalization
from src.debug_utils import print_prefixed


def describe_flash_scene(
                        scene_text: str,
                        client,
                        prompt_path="prompts/describe_scene.txt",
                        model = "gemini-2.5-flash",
                        gpt_deployment = "gpt-4o-kairos",
                        gpt_temperature = 0.3,
                        video_path: str | None = None
                         ) -> str:
    """
    Takes ONE raw scene description (string) and returns
    a concise Gemini-generated summary describing:
      - key objects
      - actions
      - spatial relationships
      - temporal relationships
    """

    # Load template prompt from external file
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    # Insert scene text into {{SCENE_TEXT}} placeholder
    normalized_text = apply_gpt_normalization(scene_text)
    video_name = Path(video_path).name if video_path else ""
    prompt = template.replace("{{SCENE_TEXT}}", normalized_text)
    prompt = prompt.replace("{{VIDEO_NAME}}", video_name)

    # Asking LLM
    if "gemini" in model.lower():
        chat = client.chats.create(model=model)
        resp = chat.send_message(prompt)
        answer = resp.text.strip()
    elif "gpt" in model.lower():
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes visual scenes.",
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=2048,
            temperature=gpt_temperature,
            top_p=1.0,
            model=gpt_deployment,
            timeout=60.0,
        )
        answer = response.choices[0].message.content

    return answer


def describe_scenes(
    scenes: list,
    client,
    hist_size = 3,
    YOLO_key="yolo_detections",
    FLIP_key="frame_captions",
    ASR_key: str = "audio_natural",
    AST_key: str = "audio_speech",
    SUMMARY_key: str = "llm_scene_description",
    model= "gemini-2.5-flash",
    prompt_path = "prompts/describe_scene.txt",
    short_prompt_path = "prompts/describe_scene_short.txt",
    fallback_prompt_path = "prompts/fallback_describe_scene.txt",
    short_fallback_prompt_path: str | None = None,
    max_workers: int | None = None,
    rate_limit_cooldown_sec: float = 20.0,
    max_rate_limit_retries: int = 4,
    video_path: str | None = None,
    cooldown_sec: float = 5,
    debug= False,
):
    """
    Two-stage scene description pipeline:
      1) Stage-1 (map): make short per-scene summaries in parallel.
      2) Stage-2 (reduce): generate final scene reports using raw scene text
         plus previous Stage-1 short summaries (instead of previous long reports).

    Returns updated scenes that include `SUMMARY_key`.
    """
    if short_fallback_prompt_path is None:
        short_fallback_prompt_path = fallback_prompt_path

    # First format all scenes using your existing system.
    formatted_scenes = raw_descriptions(
        scenes,
        YOLO_key=YOLO_key,
        FLIP_key=FLIP_key,
        ASR_key=ASR_key,
        AST_key=AST_key,
    )

    if max_workers is None:
        max_workers = min(8, max(1, len(formatted_scenes)))
    else:
        max_workers = max(1, int(max_workers))
    max_rate_limit_retries = max(0, int(max_rate_limit_retries))
    rate_limit_cooldown_sec = max(0.0, float(rate_limit_cooldown_sec))

    def _is_rate_limit_error(exc: Exception) -> bool:
        err_text = f"{type(exc).__name__}: {exc}".lower()
        rate_limit_markers = (
            "rate limit",
            "ratelimit",
            "too many requests",
            "429",
            "quota exceeded",
            "resource exhausted",
            "request rate",
        )
        return any(marker in err_text for marker in rate_limit_markers)

    def _is_responsible_ai_error(exc: Exception) -> bool:
        err_text = f"{type(exc).__name__}: {exc}".lower()
        return (
            "content_filter" in err_text
            or "content filter" in err_text
            or "responsible ai" in err_text
            or "safety system" in err_text
            or "policy_violation" in err_text
        )

    def _call_with_rate_limit_retry(scene_idx: int, scene_text: str, prompt_used: str):
        attempt = 0
        while True:
            try:
                return describe_flash_scene(
                    scene_text,
                    client,
                    prompt_path=prompt_used,
                    model=model,
                    video_path=video_path,
                )
            except Exception as exc:
                if _is_responsible_ai_error(exc):
                    if debug:
                        print_prefixed("(WARN)", f"Scene {scene_idx} blocked by Azure Content Filter. Skipping.")
                    return "Scene omitted due to content filter."
                
                if _is_rate_limit_error(exc) and attempt < max_rate_limit_retries:
                    backoff = rate_limit_cooldown_sec * (2 ** attempt)
                    jitter = random.uniform(0.0, 1.0)
                    wait_sec = backoff + jitter
                    if debug:
                        print_prefixed(
                            "(RATE)",
                            f"Scene {scene_idx} rate-limited on attempt {attempt + 1}; "
                            f"cooling down {wait_sec:.1f}s before retry."
                        )
                    time.sleep(wait_sec)
                    attempt += 1
                    continue
                raise

    def _generate_with_fallback(scene_idx: int, scene_text: str, primary_prompt: str, fallback_prompt: str | None):
        result = None
        try:
            result = _call_with_rate_limit_retry(scene_idx, scene_text, primary_prompt)
        except Exception as exc:
            if debug:
                print_prefixed("(WARN)", f"Scene {scene_idx} primary prompt failed: {exc}")
            if fallback_prompt:
                try:
                    result = _call_with_rate_limit_retry(scene_idx, scene_text, fallback_prompt)
                except Exception as exc2:
                    if debug:
                        print_prefixed("(WARN)", f"Scene {scene_idx} fallback prompt failed: {exc2}")
        finally:
            if cooldown_sec and cooldown_sec > 0:
                time.sleep(cooldown_sec)
        return result

    def _parallel_map(inputs: list[str], primary_prompt: str, fallback_prompt: str | None):
        outputs = [None] * len(inputs)
        if max_workers <= 1 or len(inputs) <= 1:
            for i, input_text in enumerate(inputs):
                outputs[i] = _generate_with_fallback(i, input_text, primary_prompt, fallback_prompt)
            return outputs

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_generate_with_fallback, i, input_text, primary_prompt, fallback_prompt): i
                for i, input_text in enumerate(inputs)
            }
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                try:
                    outputs[i] = future.result()
                except Exception as exc:
                    if debug:
                        print_prefixed("(WARN)", f"Scene {i} worker failed: {exc}")
        return outputs

    def _build_short_context(short_summaries: list[str | None], idx: int) -> str:
        if hist_size <= 0 or idx <= 0:
            return ""
        start_idx = max(0, idx - hist_size)
        context_lines = ["Previous scenes (short summaries):"]
        for j in range(start_idx, idx):
            prev_summary = (short_summaries[j] or "").strip()
            if not prev_summary:
                continue
            distance = idx - j
            context_lines.append(f"Scene -{distance}:\n{prev_summary}")
        if len(context_lines) == 1:
            return ""
        return "\n\n" + "\n\n".join(context_lines)

    # Stage-1 (map): short scene summaries in parallel.
    short_summaries = _parallel_map(
        formatted_scenes,
        primary_prompt=short_prompt_path,
        fallback_prompt=short_fallback_prompt_path,
    )

    # Stage-2 (reduce): full reports using raw scene + Stage-1 context.
    stage2_inputs = []
    for i, raw_text in enumerate(formatted_scenes):
        stage2_inputs.append(raw_text + _build_short_context(short_summaries, i))

    final_summaries = _parallel_map(
        stage2_inputs,
        primary_prompt=prompt_path,
        fallback_prompt=fallback_prompt_path,
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
            print(summary.strip())
            print("")

    return updated

# ================================================================================================
# SCENE DESCRIPTION FORMATTING

def normalize_bbox(bbox):
    """
    Convert [x1, y1, x2, y2] into raw center + area.
    Useful when we do NOT have frame dimensions.
    """
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)

    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    area = w * h

    return x_center, y_center, area

def format_single_description(
    captions: list,
    yolo,
) -> str:
    lines = []

    # If yolo is a list of track summaries (new format)
    if isinstance(yolo, list):
        from src.frame_obj_d_yolo import format_track_summaries

        for idx, cap in enumerate(captions or []):
            lines.append(f"Frame {idx}:")
            lines.append(f'  Caption: "{cap}"')
            lines.append("")

        if yolo:
            lines.append("Tracks:")
            formatted = format_track_summaries(yolo, style="narrative")
            for line in formatted:
                lines.append(f"  - {line}")
        else:
            lines.append("Tracks: none detected.")

        return "\n".join(lines)

    # Legacy per-frame yolo dict format
    frame_count = max(
        len(captions),
        max([int(k) for k in yolo.keys()], default=-1) + 1
    )

    for idx in range(frame_count):
        lines.append(f"Frame {idx}:")

        # ---- Captions ----
        if captions and idx < len(captions):
            cap = captions[idx]
            lines.append(f'  Caption: "{cap}"')

        # ---- YOLO detections ----
        dets = (
            yolo.get(idx)
            if idx in yolo
            else yolo.get(str(idx), [])
        ) or []

        if dets:
            lines.append("  Objects:")

            for det in dets:
                label = det.get("label", "unknown")
                conf = det.get("confidence", 0.0)
                bbox = det.get("bbox", [0, 0, 0, 0])

                x_center, y_center, area = normalize_bbox(bbox)

                obj_str = (
                    f"    - {label} (conf={conf:.2f}), "
                    f"x_center={x_center:.1f}, "
                    f"y_center={y_center:.1f}, "
                    f"area={area:.1f}"
                )
                lines.append(obj_str)
        else:
            lines.append("  Objects: none detected.")

        lines.append("")

    return "\n".join(lines)

def raw_descriptions(
    scenes: list,
    YOLO_key: str = "yolo_detections",
    FLIP_key: str = "frame_captions",
    ASR_key: str = "audio_natural",
    AST_key: str = "audio_speech",
) -> list:
    """
    Outer formatter:
      - Reads scenes
      - Skips YOLO or FLIP keys when None
      - Returns a list of scene description strings
    """

    formatted_list = []

    for scene in scenes:
        captions = scene.get(FLIP_key, []) if FLIP_key else []
        yolo = scene.get(YOLO_key, {}) if YOLO_key else {}
        asr = scene.get(ASR_key, "") if ASR_key else ""
        ast = scene.get(AST_key, "") if AST_key else ""

        single_scene_text = format_single_description(
            captions=captions,
            yolo=yolo,
        )

        if asr: single_scene_text += f"\nAudio transcript: {asr}"
        if ast: single_scene_text += f"\nAudio sounds: {ast}\n"

        formatted_list.append(single_scene_text)
    return formatted_list

def test(
    json_path="./captioned_scenes.json",
    YOLO_key="yolo_detections",
    FLIP_key="frame_captions",
    ASR_key: str = "audio_natural",
    AST_key: str = "audio_speech",
):
    """
    Quick test function for raw_descriptions().
    Loads captioned scenes JSON and prints the formatted descriptions.
    """

    # Load scenes
    with open(json_path, "r", encoding="utf-8") as f:
        scenes = json.load(f)

    # Format scenes
    formatted_scenes = raw_descriptions(
        scenes,
        YOLO_key=YOLO_key,
        FLIP_key=FLIP_key,
        ASR_key=ASR_key,
        AST_key=AST_key,
    )

    # Print preview
    print("=" * 60)
    print("Formatted Scene Descriptions")
    print("=" * 60)

    for i, text in enumerate(formatted_scenes):
        print(f"\n--- Scene {i} ---\n")
        print(text)
        print("\n" + "-" * 60)

    return formatted_scenes

# test()
