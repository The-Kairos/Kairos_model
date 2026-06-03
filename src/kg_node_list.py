import json
import re
from pathlib import Path

from src.debug_utils import apply_gpt_normalization


NODE_CATEGORIES = (
    "Character",
    "Object",
    "Location",
    "Action",
    "Emotion",
    "Topic",
)


def _default_node_map() -> dict:
    return {category: [] for category in NODE_CATEGORIES}


def _strip_code_fences(text: str) -> str:
    value = (text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def _clean_label(value: str, *, lowercase: bool = False) -> str:
    text = re.sub(r"\s+", " ", (value or "").strip())
    text = text.strip(" -:;,.\t\r\n")
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"n/a", "none", "unknown", "not stated", "not explicitly stated", "unspecified"}:
        return ""
    if lowercase:
        return lowered
    return text


def _normalize_node_map(raw_nodes: dict | None) -> dict:
    normalized = _default_node_map()
    if not isinstance(raw_nodes, dict):
        return normalized

    for category in NODE_CATEGORIES:
        values = raw_nodes.get(category, [])
        if not isinstance(values, list):
            continue

        seen = set()
        cleaned_values = []
        for item in values:
            if not isinstance(item, str):
                continue
            cleaned = _clean_label(item, lowercase=category in {"Action", "Topic", "Emotion"})
            if not cleaned:
                continue
            dedupe_key = cleaned.casefold()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            cleaned_values.append(cleaned)

        normalized[category] = sorted(cleaned_values, key=str.casefold)

    return normalized


def _load_prompt_template() -> str:
    prompt_path = Path("prompts/kg_node_list.txt")
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")

    return (
        "You are extracting a reusable node inventory for a single video's knowledge graph.\n\n"
        "You will be given concatenated short scene summaries from one video.\n"
        "Extract only high-value, reusable, normalized node labels for these categories:\n"
        "Character, Object, Location, Action, Emotion, Topic.\n\n"
        "Requirements:\n"
        "- Return JSON only.\n"
        "- Use exactly these top-level keys: Character, Object, Location, Action, Emotion, Topic.\n"
        "- Each value must be a JSON array of strings.\n"
        "- Normalize duplicates and aliases when the summaries clearly indicate they refer to the same thing.\n"
        "- Prefer specific labels over vague ones.\n"
        "- Exclude Scene.\n"
        "- Exclude relations.\n"
        "- Exclude one-off details that are not likely to be reused.\n"
        "- For actions, use concise verb phrases.\n"
        "- For emotions, use concise emotion labels.\n"
        "- For topics, use recurring discussion or narrative themes.\n"
        "- Keep only likely reusable nodes for a video-level graph.\n\n"
        "VIDEO NAME: {{VIDEO_NAME}}\n\n"
        "SHORT SCENE SUMMARIES:\n"
        "{{SHORT_SUMMARIES}}\n"
    )


def _call_node_extractor(prompt: str, client, model: str, gpt_deployment: str, gpt_temperature: float) -> str:
    if "gemini" in model.lower():
        chat = client.chats.create(model=model)
        resp = chat.send_message(prompt)
        return (resp.text or "").strip()

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Return only valid JSON for the requested node categories.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        response_format={"type": "json_object"},
        max_tokens=2048,
        temperature=gpt_temperature,
        top_p=1.0,
        model=gpt_deployment,
        timeout=60.0,
    )
    return response.choices[0].message.content or ""


def build_kg_node_lists(
    short_summaries: list[str | None],
    client,
    model: str = "gemini-2.5-flash",
    gpt_deployment: str = "gpt-4o-kairos",
    gpt_temperature: float = 0.1,
    video_path: str | None = None,
) -> dict:
    nonempty_summaries = [summary.strip() for summary in short_summaries if isinstance(summary, str) and summary.strip()]
    if not nonempty_summaries:
        return _default_node_map()

    prompt_template = _load_prompt_template()
    joined_summaries = "\n\n".join(
        f"Scene {idx:03d}: {summary}"
        for idx, summary in enumerate(nonempty_summaries)
    )
    normalized_text = apply_gpt_normalization(joined_summaries)
    video_name = Path(video_path).name if video_path else ""

    prompt = prompt_template.replace("{{VIDEO_NAME}}", video_name)
    prompt = prompt.replace("{{SHORT_SUMMARIES}}", normalized_text)

    try:
        raw_response = _call_node_extractor(
            prompt=prompt,
            client=client,
            model=model,
            gpt_deployment=gpt_deployment,
            gpt_temperature=gpt_temperature,
        )
        parsed = json.loads(_strip_code_fences(raw_response))
        return _normalize_node_map(parsed)
    except Exception:
        return _default_node_map()


def format_kg_node_context(node_map: dict | None) -> str:
    normalized = _normalize_node_map(node_map)
    lines = ["Known video-level node candidates:"]
    for category in NODE_CATEGORIES:
        values = normalized.get(category, [])
        rendered = ", ".join(values) if values else "None"
        lines.append(f"{category}: {rendered}")
    return "\n" + "\n".join(lines)
