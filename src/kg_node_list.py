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

RELATION_SPECS = (
    ("HAS_TOPIC", ("Scene",), ("Topic",)),
    ("IS_SHOWN_IN", ("Character", "Object", "Location"), ("Scene",)),
    ("IS_IN", ("Character", "Object"), ("Location",)),
    ("OCCURS_IN", ("Action",), ("Scene",)),
    ("DOES", ("Character",), ("Action",)),
    ("INTERACTS_WITH", ("Character",), ("Object",)),
    ("INVOLVES", ("Action",), ("Object",)),
    ("TARGETS", ("Action",), ("Character", "Object")),
    ("FEELS", ("Character",), ("Emotion",)),
    ("CAUSES", ("Action", "Character", "Object", "Topic"), ("Emotion",)),
    ("SPEAKS_TO", ("Character",), ("Character",)),
    ("MENTIONS", ("Character",), ("Topic", "Character", "Object", "Location")),
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


def _load_relationship_prompt_template() -> str:
    prompt_path = Path("prompts/kg_relationships.txt")
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")

    return (
        "You are extracting knowledge-graph relationships for a single video scene.\n\n"
        "Use only the known node labels provided below.\n"
        "Do not return JSON.\n"
        "Return one relationship per line using this exact format:\n"
        "Character Sheldon <FEELS> Emotion nervousness\n"
        "Character Sheldon <IS_SHOWN_IN> Scene this_scene\n\n"
        "Rules:\n"
        "- Use exactly one of the allowed relation types.\n"
        "- Source format: <Category> <Label>\n"
        "- Target format: <Category> <Label>\n"
        "- Use `Scene this_scene` when the target or source is the current scene.\n"
        "- Use natural labels, not ids.\n"
        "- Do not add bullets, numbering, commentary, markdown, or JSON.\n"
        "- Only emit relationships that are clearly supported by the scene description.\n"
        "- Multiple relationships of the same type are allowed.\n\n"
        "ALLOWED RELATION TYPES:\n"
        "{{RELATION_TYPES}}\n\n"
        "KNOWN NODE LABELS:\n"
        "{{KNOWN_NODE_IDS}}\n\n"
        "SCENE ID: {{SCENE_ID}}\n\n"
        "SCENE DESCRIPTION:\n"
        "{{SCENE_DESCRIPTION}}\n"
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
                "content": "Follow the requested output format exactly and return only the requested content.",
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


def _call_json_extractor(prompt: str, client, model: str, gpt_deployment: str, gpt_temperature: float) -> str:
    return _call_node_extractor(
        prompt=prompt,
        client=client,
        model=model,
        gpt_deployment=gpt_deployment,
        gpt_temperature=gpt_temperature,
    )


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


def _slugify(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")
    return value or "unknown"


def make_node_id(category: str, label: str) -> str:
    prefix_map = {
        "Scene": "scene",
        "Character": "char",
        "Object": "obj",
        "Location": "loc",
        "Action": "action",
        "Emotion": "emotion",
        "Topic": "topic",
    }
    prefix = prefix_map.get(category, category.casefold())
    return f"{prefix}:{_slugify(label)}"


def build_known_node_id_map(node_map: dict | None) -> dict:
    normalized = _normalize_node_map(node_map)
    id_map = {}
    for category in NODE_CATEGORIES:
        for label in normalized.get(category, []):
            node_id = make_node_id(category, label)
            id_map[node_id] = {
                "id": node_id,
                "category": category,
                "label": label,
            }
    return id_map


def _relation_specs_text() -> str:
    lines = []
    for rel_type, source_categories, target_categories in RELATION_SPECS:
        src = "/".join(source_categories)
        tgt = "/".join(target_categories)
        lines.append(f"- {src} <{rel_type}> {tgt}")
    return "\n".join(lines)


def _known_node_id_text(scene_id: str, known_node_ids: dict) -> str:
    lines = ["- Scene: this_scene"]
    for node_id in sorted(known_node_ids):
        meta = known_node_ids[node_id]
        lines.append(f'- {meta["category"]}: {meta["label"]}')
    return "\n".join(lines)


def _allowed_relation_map() -> dict:
    relation_map = {}
    for rel_type, source_categories, target_categories in RELATION_SPECS:
        relation_map[rel_type] = {
            "source": set(source_categories),
            "target": set(target_categories),
        }
    return relation_map


def _build_label_lookup(node_map: dict | None) -> dict:
    normalized = _normalize_node_map(node_map)
    lookup = {}
    for category in NODE_CATEGORIES:
        entries = {}
        for label in normalized.get(category, []):
            entries[label.casefold()] = make_node_id(category, label)
        lookup[category] = entries
    return lookup


def _normalize_scene_label(label: str) -> str:
    lowered = _clean_label(label, lowercase=True)
    if lowered in {"this_scene", "scene", "current scene"}:
        return "this_scene"
    return lowered


def _resolve_label_to_id(category: str, label: str, scene_index: int, label_lookup: dict) -> str | None:
    if category == "Scene":
        normalized_scene = _normalize_scene_label(label)
        if normalized_scene == "this_scene":
            return f"scene:{scene_index}"
        return None

    entries = label_lookup.get(category, {})
    cleaned = _clean_label(label)
    if not cleaned:
        return None

    direct = entries.get(cleaned.casefold())
    if direct:
        return direct

    cleaned_no_paren = re.sub(r"\s*\(.*?\)\s*", " ", cleaned).strip()
    direct = entries.get(cleaned_no_paren.casefold())
    if direct:
        return direct

    for known_label_cf, node_id in entries.items():
        if known_label_cf in cleaned.casefold() or cleaned.casefold() in known_label_cf:
            return node_id

    return None


def _parse_relationship_text(
    raw_text: str,
    scene_index: int,
    known_nodes: dict | None,
) -> list[dict]:
    pattern = re.compile(
        r"^\s*(Scene|Character|Object|Location|Action|Emotion|Topic)\s+(.+?)\s+<([A-Z_]+)>\s+"
        r"(Scene|Character|Object|Location|Action|Emotion|Topic)\s+(.+?)\s*$"
    )
    label_lookup = _build_label_lookup(known_nodes)
    allowed_relation_map = _allowed_relation_map()
    parsed_relationships = []
    seen = set()
    parse_failures = []

    for raw_line in (raw_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if not match:
            parse_failures.append(line)
            continue

        source_category, source_label, rel_type, target_category, target_label = match.groups()
        rel_type = rel_type.strip().upper()
        if rel_type not in allowed_relation_map:
            parse_failures.append(line)
            continue
        if source_category not in allowed_relation_map[rel_type]["source"]:
            parse_failures.append(line)
            continue
        if target_category not in allowed_relation_map[rel_type]["target"]:
            parse_failures.append(line)
            continue

        source_id = _resolve_label_to_id(source_category, source_label, scene_index, label_lookup)
        target_id = _resolve_label_to_id(target_category, target_label, scene_index, label_lookup)
        if not source_id or not target_id:
            parse_failures.append(line)
            continue

        dedupe_key = (rel_type, source_id, target_id)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        parsed_relationships.append(
            {
                "type": rel_type,
                "source_id": source_id,
                "target_id": target_id,
            }
        )

    if parse_failures and not parsed_relationships:
        raise ValueError("Failed to parse any relationship lines.")

    return parsed_relationships


def _validate_relationships(raw_relationships: list, scene_index: int, known_node_ids: dict) -> list[dict]:
    scene_id = f"scene:{scene_index}"
    allowed_node_categories = {scene_id: "Scene"}
    for node_id, meta in known_node_ids.items():
        allowed_node_categories[node_id] = meta["category"]

    allowed_relation_map = _allowed_relation_map()
    validated = []
    seen = set()

    for rel in raw_relationships:
        if not isinstance(rel, dict):
            continue
        rel_type = _clean_label(rel.get("type", "")).upper()
        source_id = _clean_label(rel.get("source_id", ""), lowercase=True)
        target_id = _clean_label(rel.get("target_id", ""), lowercase=True)
        if not rel_type or not source_id or not target_id:
            continue
        if rel_type not in allowed_relation_map:
            continue
        source_category = allowed_node_categories.get(source_id)
        target_category = allowed_node_categories.get(target_id)
        if not source_category or not target_category:
            continue
        if source_category not in allowed_relation_map[rel_type]["source"]:
            continue
        if target_category not in allowed_relation_map[rel_type]["target"]:
            continue
        dedupe_key = (rel_type, source_id, target_id)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        validated.append(
            {
                "type": rel_type,
                "source_id": source_id,
                "target_id": target_id,
            }
        )

    return validated


def extract_scene_relationships(
    scenes: list[dict],
    known_nodes: dict | None,
    client,
    model: str = "gemini-2.5-flash",
    gpt_deployment: str = "gpt-4o-kairos",
    gpt_temperature: float = 0.1,
) -> list[dict]:
    known_node_ids = build_known_node_id_map(known_nodes)
    prompt_template = _load_relationship_prompt_template()
    relation_types = _relation_specs_text()

    updated_scenes = []
    for fallback_idx, scene in enumerate(scenes or []):
        new_scene = dict(scene)
        scene_index = scene.get("scene_index", fallback_idx)
        scene_id = f"scene:{scene_index}"
        scene_description = (scene.get("llm_scene_description") or "").strip()

        if not scene_description:
            new_scene["relationships"] = []
            updated_scenes.append(new_scene)
            continue

        prompt = prompt_template.replace("{{RELATION_TYPES}}", relation_types)
        prompt = prompt.replace("{{KNOWN_NODE_IDS}}", _known_node_id_text(scene_id, known_node_ids))
        prompt = prompt.replace("{{SCENE_ID}}", scene_id)
        prompt = prompt.replace("{{SCENE_DESCRIPTION}}", apply_gpt_normalization(scene_description))

        raw_response = ""
        try:
            raw_response = _call_json_extractor(
                prompt=prompt,
                client=client,
                model=model,
                gpt_deployment=gpt_deployment,
                gpt_temperature=gpt_temperature,
            )
            new_scene["relationships"] = _parse_relationship_text(
                raw_text=_strip_code_fences(raw_response),
                scene_index=scene_index,
                known_nodes=known_nodes,
            )
        except Exception:
            raw_clean = _strip_code_fences(raw_response) if "raw_response" in locals() else ""
            new_scene["relationships"] = [f"ERROR: {raw_clean}"] if raw_clean else []
        updated_scenes.append(new_scene)

    return updated_scenes
