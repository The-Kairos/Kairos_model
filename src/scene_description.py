import os
import json
import time
from pathlib import Path

# Calculate absolute path to prompts directory
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "prompts" / "describe_scene.txt"

def describe_flash_scene(
                        scene_text: str,
                        client,
                        prompt_path=None,
                        model = "gemini-2.5-flash",
                        gpt_deployment = "gpt-4o-kairos",
                        gpt_temperature = 0.3
                         ) -> str:
    """
    Takes ONE raw scene description (string) and returns
    a concise Gemini-generated summary.
    """
    if prompt_path is None:
        prompt_path = DEFAULT_PROMPT_PATH

    # Load template prompt from external file
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    # Insert scene text into {{SCENE_TEXT}} placeholder
    prompt = template.replace("{{SCENE_TEXT}}", scene_text)

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
            max_tokens=4096,
            temperature=gpt_temperature,
            top_p=1.0,
            model=gpt_deployment
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
    prompt_path = None,
    debug= False,
):
    if prompt_path is None:
        prompt_path = DEFAULT_PROMPT_PATH
    """
    Takes a list of scene dictionaries.
    Adds a new key to each: llm_scene_description

    Uses the previously built `format_all_scenes()` to generate
    raw scene descriptions.
    """

    # First format all scenes using your existing system
    formatted_scenes = raw_descriptions(
        scenes,
        YOLO_key=YOLO_key,
        FLIP_key=FLIP_key,
        ASR_key=ASR_key,
        AST_key=AST_key,
    )

    updated = []
    previous_summaries = []  # store generated summaries

    for idx, (scene, formatted_text) in enumerate(zip(scenes, formatted_scenes)):

        # ---- build context from last K summaries ----
        if previous_summaries:
            context = "\n\nPrevious scenes:\n"
            for i, s in enumerate(previous_summaries[-hist_size:], start=1):
                context += f"Scene -{len(previous_summaries) - i + 1}:\n{s}\n"
            formatted_text += "\n" + context

        # ---- retry loop for individual scene fusion ----
        max_scene_retries = 5
        summary = "Fusion Failed"
        for attempt in range(max_scene_retries):
            try:
                summary = describe_flash_scene(
                    formatted_text,
                    client,
                    prompt_path=prompt_path,
                    model=model,
                )
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e).upper():
                    wait_time = (attempt + 1) * 30
                    print(f"      [Wait] Scene {idx} Gemini Rate Limit. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_scene_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"      [Error] Scene {idx} Fusion failed: {e}")
                    break

        new_scene = dict(scene)
        new_scene[SUMMARY_key] = summary
        updated.append(new_scene)

        previous_summaries.append(summary)  # save summary

        if debug:
            print("Scene", idx, summary)
        time.sleep(5)

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
