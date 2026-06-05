import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
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
ENTITY_NODE_CATEGORIES = (
    "Character",
    "Object",
    "Location",
    "Emotion",
)
ACTION_TOPIC_CATEGORIES = (
    "Action",
    "Topic",
)
ACTION_TOPIC_CHUNK_SIZE = 200
ENABLE_ACTION_TOPIC_FINAL_NORMALIZATION = False

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
SPATIAL_RELATION_SPECS = (
    ("IN", ("Character", "Object"), ("Location",)),
    ("NEAR", ("Character", "Object"), ("Character", "Object")),
    ("LEFT_OF", ("Character", "Object"), ("Character", "Object")),
    ("IN_FRONT_OF", ("Character", "Object"), ("Character", "Object")),
    ("ON", ("Object",), ("Object", "Location")),
    ("INSIDE", ("Object",), ("Object", "Location")),
)
TEMPORAL_INTERVAL_RELATION_SPECS = (
    ("CONTINUES_INTO", ("Action",), ("Action",)),
    ("AFTER", ("Action",), ("Action",)),
    ("OVERLAPS", ("Action",), ("Action",)),
)
SPATIAL_NODE_CATEGORIES = (
    "Character",
    "Object",
    "Location",
)
TEMPORAL_ACTION_CATEGORIES = (
    "Action",
)
TEMPORAL_WINDOW_SIZE = 6
TEMPORAL_WINDOW_STRIDE = 5
TEMPORAL_ACTION_CONTEXT_REL_TYPES = {
    "OCCURS_IN",
    "DOES",
    "INVOLVES",
    "TARGETS",
}


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


def _load_entity_prompt_template() -> str:
    prompt_path = Path("prompts/kg_node_list_entities.txt")
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")

    return (
        "You are extracting a reusable node inventory for a single video's knowledge graph.\n\n"
        "You will be given concatenated short scene summaries from one video.\n"
        "Extract as many valid reusable nodes as supported by the summaries for each category.\n"
        "Extract only these categories: Character, Object, Location, Emotion.\n\n"
        "Requirements:\n"
        "- Return JSON only.\n"
        "- Use exactly these top-level keys: Character, Object, Location, Emotion.\n"
        "- Each value must be a JSON array of strings.\n"
        "- Normalize duplicates and aliases when the summaries clearly indicate they refer to the same thing.\n"
        "- Prefer specific labels over vague ones.\n"
        "- Exclude Scene.\n"
        "- Exclude relations.\n"
        "- Keep only likely reusable nodes for a video-level graph.\n"
        "- For emotions, use concise emotion labels.\n\n"
        "VIDEO NAME: {{VIDEO_NAME}}\n\n"
        "SHORT SCENE SUMMARIES:\n"
        "{{SHORT_SUMMARIES}}\n"
    )


def _load_action_topic_prompt_template() -> str:
    prompt_path = Path("prompts/kg_node_list_actions_topics.txt")
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")

    return (
        "You are extracting a reusable node inventory for a single video's knowledge graph.\n\n"
        "You will be given a chunk of short scene summaries from one video.\n"
        "Extract as many valid reusable nodes as supported by the summaries for each category.\n"
        "Extract only these categories: Action, Topic.\n\n"
        "Requirements:\n"
        "- Return JSON only.\n"
        "- Use exactly these top-level keys: Action, Topic.\n"
        "- Each value must be a JSON array of strings.\n"
        "- Normalize duplicates and aliases when the summaries clearly indicate they refer to the same thing.\n"
        "- Prefer specific labels over vague ones.\n"
        "- Exclude Scene.\n"
        "- Exclude relations.\n"
        "- Prefer recurring or narratively meaningful actions/topics, but include valid chunk-supported ones.\n"
        "- For actions, use concise verb phrases.\n"
        "- For topics, use recurring discussion or narrative themes.\n\n"
        "VIDEO NAME: {{VIDEO_NAME}}\n"
        "SCENE RANGE: {{SCENE_RANGE}}\n\n"
        "SHORT SCENE SUMMARIES:\n"
        "{{SHORT_SUMMARIES}}\n"
    )


def _load_action_topic_normalize_prompt_template() -> str:
    prompt_path = Path("prompts/kg_node_list_actions_topics_normalize.txt")
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")

    return (
        "You are normalizing merged node candidates for a single video's knowledge graph.\n\n"
        "Normalize duplicates and aliases, and keep as many valid reusable nodes as supported by the candidate lists.\n"
        "Extract only these categories: Action, Topic.\n\n"
        "Requirements:\n"
        "- Return JSON only.\n"
        "- Use exactly these top-level keys: Action, Topic.\n"
        "- Each value must be a JSON array of strings.\n"
        "- Merge duplicates and obvious aliases.\n"
        "- For actions, use concise verb phrases.\n"
        "- For topics, use recurring discussion or narrative themes.\n\n"
        "VIDEO NAME: {{VIDEO_NAME}}\n\n"
        "MERGED ACTION CANDIDATES:\n"
        "{{ACTION_CANDIDATES}}\n\n"
        "MERGED TOPIC CANDIDATES:\n"
        "{{TOPIC_CANDIDATES}}\n"
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


def _load_spatial_relationship_prompt_template() -> str:
    prompt_path = Path("prompts/kg_relationships_spatial.txt")
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")

    return (
        "You are extracting spatial knowledge-graph relationships for a single video scene.\n\n"
        "Use only the known node labels provided below.\n"
        "Do not return JSON.\n"
        "Return one relationship per line using this exact format:\n"
        "Character Jack <IN> Location deck\n"
        "Object suitcase <LEFT_OF> Object chair\n\n"
        "Rules:\n"
        "- Emit only spatial relationships.\n"
        "- Use exactly one of the allowed relation types.\n"
        "- Source format: <Category> <Label>\n"
        "- Target format: <Category> <Label>\n"
        "- Use natural labels, not ids.\n"
        "- Do not add bullets, numbering, commentary, markdown, or JSON.\n"
        "- Use the scene description and YOLO summary together.\n"
        "- If YOLO is empty, rely on the scene description only.\n"
        "- Only emit relationships clearly supported by the evidence.\n\n"
        "ALLOWED RELATION TYPES:\n"
        "{{RELATION_TYPES}}\n\n"
        "KNOWN NODE LABELS:\n"
        "{{KNOWN_NODE_IDS}}\n\n"
        "SCENE ID: {{SCENE_ID}}\n\n"
        "YOLO SUMMARY:\n"
        "{{YOLO_SUMMARY}}\n\n"
        "SCENE DESCRIPTION:\n"
        "{{SCENE_DESCRIPTION}}\n"
    )


def _load_temporal_relationship_prompt_template() -> str:
    prompt_path = Path("prompts/kg_relationships_temporal.txt")
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")

    return (
        "You are extracting temporal interval relationships across a rolling window of video scenes.\n\n"
        "You will be given 6 scene descriptions. The first scene is context-only.\n"
        "Only emit relationships for the last 5 scenes in the window.\n"
        "Use only the known Action node labels provided below.\n"
        "Do not return JSON.\n"
        "Group output by scene header, like:\n"
        "scene 12\n"
        "Action boarding ship <CONTINUES_INTO> Action walking on deck\n"
        "Action searching luggage <AFTER> Action opening trunk\n\n"
        "Rules:\n"
        "- Emit only temporal interval relationships.\n"
        "- Use exactly one of the allowed relation types.\n"
        "- Only use Action nodes in relationship lines.\n"
        "- The first scene in the window is context-only and must not receive output.\n"
        "- Use natural labels, not ids.\n"
        "- Do not add bullets, numbering, commentary, markdown, or JSON.\n"
        "- Only emit relationships clearly supported by the 6-scene context and action relationship context.\n\n"
        "ALLOWED RELATION TYPES:\n"
        "{{RELATION_TYPES}}\n\n"
        "KNOWN ACTION LABELS:\n"
        "{{KNOWN_NODE_IDS}}\n\n"
        "WINDOW SCENES:\n"
        "{{WINDOW_CONTEXT}}\n"
    )


def _call_node_extractor(
    prompt: str,
    client,
    model: str,
    gpt_deployment: str,
    gpt_temperature: float,
    force_json: bool = True,
) -> str:
    if "gemini" in model.lower():
        chat = client.chats.create(model=model)
        resp = chat.send_message(prompt)
        return (resp.text or "").strip()

    request_kwargs = {
        "messages": [
            {
                "role": "system",
                "content": "Follow the requested output format exactly and return only the requested content.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "max_tokens": 2048,
        "temperature": gpt_temperature,
        "top_p": 1.0,
        "model": gpt_deployment,
        "timeout": 60.0,
    }
    if force_json:
        request_kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**request_kwargs)
    return response.choices[0].message.content or ""


def _call_relationship_extractor(prompt: str, client, model: str, gpt_deployment: str, gpt_temperature: float) -> str:
    return _call_node_extractor(
        prompt=prompt,
        client=client,
        model=model,
        gpt_deployment=gpt_deployment,
        gpt_temperature=gpt_temperature,
        force_json=False,
    )


def _resolve_max_workers(task_count: int, max_workers: int | None) -> int:
    if task_count <= 1:
        return 1
    if max_workers is None:
        return min(8, max(1, task_count))
    return max(1, int(max_workers))


def _render_indexed_summaries(indexed_summaries: list[tuple[int, str]]) -> str:
    return "\n\n".join(
        f"Scene {scene_idx:04d}: {summary}"
        for scene_idx, summary in indexed_summaries
    )


def _nonempty_indexed_summaries(short_summaries: list[str | None]) -> list[tuple[int, str]]:
    indexed = []
    for idx, summary in enumerate(short_summaries):
        if not isinstance(summary, str):
            continue
        cleaned = summary.strip()
        if not cleaned:
            continue
        indexed.append((idx, cleaned))
    return indexed


def _chunk_indexed_summaries(indexed_summaries: list[tuple[int, str]], chunk_size: int) -> list[list[tuple[int, str]]]:
    if chunk_size <= 0:
        return [indexed_summaries] if indexed_summaries else []
    return [
        indexed_summaries[start:start + chunk_size]
        for start in range(0, len(indexed_summaries), chunk_size)
    ]


def _merge_node_maps(*node_maps: dict | None) -> dict:
    merged = _default_node_map()
    for node_map in node_maps:
        if not isinstance(node_map, dict):
            continue
        for category in NODE_CATEGORIES:
            values = node_map.get(category, [])
            if isinstance(values, list):
                merged[category].extend(values)
    return _normalize_node_map(merged)


def _filter_node_map(node_map: dict | None, categories: tuple[str, ...]) -> dict:
    normalized = _normalize_node_map(node_map)
    filtered = _default_node_map()
    for category in categories:
        filtered[category] = list(normalized.get(category, []))
    return filtered


def _extract_node_subset(
    prompt: str,
    categories: tuple[str, ...],
    client,
    model: str,
    gpt_deployment: str,
    gpt_temperature: float,
) -> dict:
    raw_response = _call_node_extractor(
        prompt=prompt,
        client=client,
        model=model,
        gpt_deployment=gpt_deployment,
        gpt_temperature=gpt_temperature,
    )
    parsed = json.loads(_strip_code_fences(raw_response))
    subset = {category: parsed.get(category, []) for category in categories}
    return _normalize_node_map(subset)


def _normalize_action_topic_candidates(
    action_topic_map: dict,
    client,
    model: str,
    gpt_deployment: str,
    gpt_temperature: float,
    video_path: str | None = None,
) -> dict:
    normalized_input = _normalize_node_map(action_topic_map)
    actions = normalized_input.get("Action", [])
    topics = normalized_input.get("Topic", [])
    if not actions and not topics:
        return _default_node_map()

    prompt_template = _load_action_topic_normalize_prompt_template()
    video_name = Path(video_path).name if video_path else ""
    prompt = prompt_template.replace("{{VIDEO_NAME}}", video_name)
    prompt = prompt.replace(
        "{{ACTION_CANDIDATES}}",
        "\n".join(f"- {value}" for value in actions) or "None",
    )
    prompt = prompt.replace(
        "{{TOPIC_CANDIDATES}}",
        "\n".join(f"- {value}" for value in topics) or "None",
    )

    try:
        return _extract_node_subset(
            prompt=prompt,
            categories=ACTION_TOPIC_CATEGORIES,
            client=client,
            model=model,
            gpt_deployment=gpt_deployment,
            gpt_temperature=gpt_temperature,
        )
    except Exception:
        return normalized_input


def build_kg_node_lists(
    short_summaries: list[str | None],
    client,
    model: str = "gemini-2.5-flash",
    gpt_deployment: str = "gpt-4o-kairos",
    gpt_temperature: float = 0.1,
    video_path: str | None = None,
) -> dict:
    indexed_summaries = _nonempty_indexed_summaries(short_summaries)
    if not indexed_summaries:
        return _default_node_map()

    video_name = Path(video_path).name if video_path else ""
    full_summary_text = apply_gpt_normalization(_render_indexed_summaries(indexed_summaries))

    entity_prompt = _load_entity_prompt_template()
    entity_prompt = entity_prompt.replace("{{VIDEO_NAME}}", video_name)
    entity_prompt = entity_prompt.replace("{{SHORT_SUMMARIES}}", full_summary_text)

    try:
        entity_nodes = _extract_node_subset(
            prompt=entity_prompt,
            categories=ENTITY_NODE_CATEGORIES,
            client=client,
            model=model,
            gpt_deployment=gpt_deployment,
            gpt_temperature=gpt_temperature,
        )
    except Exception:
        entity_nodes = _default_node_map()

    action_topic_prompt_template = _load_action_topic_prompt_template()
    action_topic_chunks = _chunk_indexed_summaries(indexed_summaries, ACTION_TOPIC_CHUNK_SIZE)
    action_topic_results = []

    for chunk in action_topic_chunks:
        if not chunk:
            continue
        chunk_prompt = action_topic_prompt_template.replace("{{VIDEO_NAME}}", video_name)
        chunk_prompt = chunk_prompt.replace(
            "{{SCENE_RANGE}}",
            f"{chunk[0][0]:04d}-{chunk[-1][0]:04d}",
        )
        chunk_prompt = chunk_prompt.replace(
            "{{SHORT_SUMMARIES}}",
            apply_gpt_normalization(_render_indexed_summaries(chunk)),
        )

        try:
            action_topic_results.append(
                _extract_node_subset(
                    prompt=chunk_prompt,
                    categories=ACTION_TOPIC_CATEGORIES,
                    client=client,
                    model=model,
                    gpt_deployment=gpt_deployment,
                    gpt_temperature=gpt_temperature,
                )
            )
        except Exception:
            continue

    merged_action_topic = _merge_node_maps(*action_topic_results)
    if ENABLE_ACTION_TOPIC_FINAL_NORMALIZATION:
        normalized_action_topic = _normalize_action_topic_candidates(
            merged_action_topic,
            client=client,
            model=model,
            gpt_deployment=gpt_deployment,
            gpt_temperature=gpt_temperature,
            video_path=video_path,
        )
    else:
        normalized_action_topic = merged_action_topic

    return _merge_node_maps(entity_nodes, normalized_action_topic)


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


def _relation_specs_text(relation_specs: tuple = RELATION_SPECS) -> str:
    lines = []
    for rel_type, source_categories, target_categories in relation_specs:
        src = "/".join(source_categories)
        tgt = "/".join(target_categories)
        lines.append(f"- {src} <{rel_type}> {tgt}")
    return "\n".join(lines)


def _known_node_id_text(scene_id: str, known_node_ids: dict, include_scene: bool = True) -> str:
    lines = ["- Scene: this_scene"] if include_scene else []
    for node_id in sorted(known_node_ids):
        meta = known_node_ids[node_id]
        lines.append(f'- {meta["category"]}: {meta["label"]}')
    return "\n".join(lines)


def _allowed_relation_map(relation_specs: tuple = RELATION_SPECS) -> dict:
    relation_map = {}
    for rel_type, source_categories, target_categories in relation_specs:
        relation_map[rel_type] = {
            "source": set(source_categories),
            "target": set(target_categories),
        }
    return relation_map


def _build_id_meta_map(node_map: dict | None) -> dict:
    return build_known_node_id_map(node_map)


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
    relation_specs: tuple = RELATION_SPECS,
) -> list[dict]:
    if not (raw_text or "").strip():
        raise ValueError("Empty LLM relationship output.")

    pattern = re.compile(
        r"^\s*(Scene|Character|Object|Location|Action|Emotion|Topic)\s+(.+?)\s+<([A-Z_]+)>\s+"
        r"(Scene|Character|Object|Location|Action|Emotion|Topic)\s+(.+?)\s*$"
    )
    label_lookup = _build_label_lookup(known_nodes)
    allowed_relation_map = _allowed_relation_map(relation_specs)
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

    if not parsed_relationships:
        if parse_failures:
            raise ValueError("Failed to parse any relationship lines.")
        raise ValueError("No relationships extracted from non-empty LLM output.")

    return parsed_relationships


def _format_yolo_summary(scene: dict) -> str:
    detections = scene.get("yolo_detections", [])
    if not isinstance(detections, list) or not detections:
        return "No YOLO detections available."

    lines = []
    for detection in detections:
        if not isinstance(detection, dict):
            continue
        label = detection.get("label", "unknown")
        track_id = detection.get("track_id", "?")
        confidence = detection.get("confidence_avg", "?")
        start_pos = detection.get("start_pos", "unknown")
        end_pos = detection.get("end_pos", "unknown")
        lines.append(
            f"- track {track_id}: {label} (conf={confidence}) start={start_pos} end={end_pos}"
        )
    return "\n".join(lines) if lines else "No YOLO detections available."


def _format_relationship_for_context(rel: dict, scene_index: int, id_meta_map: dict) -> str | None:
    if not isinstance(rel, dict):
        return None
    rel_type = _clean_label(rel.get("type", "")).upper()
    source_id = rel.get("source_id")
    target_id = rel.get("target_id")
    if not rel_type or not isinstance(source_id, str) or not isinstance(target_id, str):
        return None

    def _render(node_id: str) -> str | None:
        if node_id == f"scene:{scene_index}":
            return f"Scene {scene_index}"
        if node_id.startswith("scene:"):
            return f"Scene {node_id.split(':', 1)[1]}"
        meta = id_meta_map.get(node_id)
        if not meta:
            return None
        return f'{meta["category"]} {meta["label"]}'

    source_text = _render(source_id)
    target_text = _render(target_id)
    if not source_text or not target_text:
        return None
    return f"{source_text} <{rel_type}> {target_text}"


def _format_action_relationship_context(scene: dict, id_meta_map: dict, fallback_idx: int) -> str:
    scene_index = scene.get("scene_index", fallback_idx)
    lines = []
    for rel in scene.get("relationships", []):
        if not isinstance(rel, dict):
            continue
        rel_type = _clean_label(rel.get("type", "")).upper()
        if rel_type not in TEMPORAL_ACTION_CONTEXT_REL_TYPES:
            continue
        rendered = _format_relationship_for_context(rel, scene_index, id_meta_map)
        if rendered:
            lines.append(rendered)
    return "\n".join(lines) if lines else "None"


def _apply_spatial_inverse_relationships(relationships: list[dict]) -> list[dict]:
    expanded = []
    seen = set()

    def _push(rel: dict):
        if not isinstance(rel, dict):
            return
        key = (rel.get("type"), rel.get("source_id"), rel.get("target_id"))
        if key in seen:
            return
        seen.add(key)
        expanded.append(rel)

    for rel in relationships:
        _push(rel)
        rel_type = rel.get("type")
        source_id = rel.get("source_id")
        target_id = rel.get("target_id")
        if rel_type == "LEFT_OF":
            _push({"type": "RIGHT_OF", "source_id": target_id, "target_id": source_id})
        elif rel_type == "IN_FRONT_OF":
            _push({"type": "BEHIND", "source_id": target_id, "target_id": source_id})
        elif rel_type == "NEAR":
            _push({"type": "NEAR", "source_id": target_id, "target_id": source_id})

    return expanded


def _apply_temporal_inverse_relationships(relationships: list[dict]) -> list[dict]:
    expanded = []
    seen = set()

    def _push(rel: dict):
        if not isinstance(rel, dict):
            return
        key = (rel.get("type"), rel.get("source_id"), rel.get("target_id"))
        if key in seen:
            return
        seen.add(key)
        expanded.append(rel)

    for rel in relationships:
        _push(rel)
        if rel.get("type") == "AFTER":
            _push(
                {
                    "type": "BEFORE",
                    "source_id": rel.get("target_id"),
                    "target_id": rel.get("source_id"),
                }
            )

    return expanded


def _add_temporal_occurs_in_relationships(
    relationships: list[dict],
    scene_index: int,
) -> list[dict]:
    expanded = []
    seen = set()
    scene_id = f"scene:{scene_index}"

    def _push(rel: dict):
        if not isinstance(rel, dict):
            return
        key = (rel.get("type"), rel.get("source_id"), rel.get("target_id"))
        if key in seen:
            return
        seen.add(key)
        expanded.append(rel)

    for rel in relationships:
        _push(rel)
        source_id = rel.get("source_id")
        if isinstance(source_id, str) and source_id.startswith("action:"):
            _push(
                {
                    "type": "OCCURS_IN",
                    "source_id": source_id,
                    "target_id": scene_id,
                }
            )

    return expanded


def _parse_temporal_relationship_text(
    raw_text: str,
    known_nodes: dict | None,
    valid_scene_indices: set[int],
) -> tuple[dict[int, list[dict]], list[str]]:
    if not (raw_text or "").strip():
        raise ValueError("Empty LLM temporal relationship output.")

    scene_blocks: dict[int, list[str]] = {}
    errors = []
    current_scene_index = None
    header_pattern = re.compile(r"^\s*scene\s+(\d+)\s*:?\s*$", re.IGNORECASE)

    for raw_line in (raw_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        header_match = header_pattern.match(line)
        if header_match:
            candidate_scene_index = int(header_match.group(1))
            if candidate_scene_index not in valid_scene_indices:
                errors.append(line)
                current_scene_index = None
                continue
            current_scene_index = candidate_scene_index
            scene_blocks.setdefault(current_scene_index, [])
            continue
        if current_scene_index is None:
            errors.append(line)
            continue
        scene_blocks[current_scene_index].append(line)

    parsed_by_scene = {}
    for scene_index, lines in scene_blocks.items():
        if not lines:
            continue
        try:
            parsed_by_scene[scene_index] = _parse_relationship_text(
                raw_text="\n".join(lines),
                scene_index=scene_index,
                known_nodes=known_nodes,
                relation_specs=TEMPORAL_INTERVAL_RELATION_SPECS,
            )
        except Exception:
            errors.extend(lines)

    if not parsed_by_scene:
        if errors:
            raise ValueError("Failed to parse any temporal relationship lines.")
        raise ValueError("No temporal relationships extracted from non-empty LLM output.")

    return parsed_by_scene, errors


def merge_scene_relationships(*relationship_lists: list) -> list[dict]:
    merged = []
    seen = set()
    for relationship_list in relationship_lists:
        if not isinstance(relationship_list, list):
            continue
        for rel in relationship_list:
            if not isinstance(rel, dict):
                continue
            rel_type = _clean_label(rel.get("type", "")).upper()
            source_id = _clean_label(rel.get("source_id", ""), lowercase=True)
            target_id = _clean_label(rel.get("target_id", ""), lowercase=True)
            if not rel_type or not source_id or not target_id:
                continue
            key = (rel_type, source_id, target_id)
            if key in seen:
                continue
            seen.add(key)
            merged.append(
                {
                    "type": rel_type,
                    "source_id": source_id,
                    "target_id": target_id,
                }
            )
    return merged


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
            raw_response = _call_relationship_extractor(
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
        except Exception as exc:
            raw_clean = _strip_code_fences(raw_response)
            if raw_clean:
                new_scene["relationships"] = [f"ERROR: {raw_clean}"]
            else:
                new_scene["relationships"] = [f"ERROR: {type(exc).__name__}: {exc}"]
        updated_scenes.append(new_scene)

    return updated_scenes


def extract_scene_spatial_relationships(
    scenes: list[dict],
    known_nodes: dict | None,
    client,
    model: str = "gemini-2.5-flash",
    gpt_deployment: str = "gpt-4o-kairos",
    gpt_temperature: float = 0.1,
    max_workers: int | None = None,
) -> list[dict]:
    spatial_node_map = _filter_node_map(known_nodes, SPATIAL_NODE_CATEGORIES)
    known_node_ids = build_known_node_id_map(spatial_node_map)
    prompt_template = _load_spatial_relationship_prompt_template()
    relation_types = _relation_specs_text(SPATIAL_RELATION_SPECS)

    def _process_scene(fallback_idx: int, scene: dict) -> dict:
        new_scene = dict(scene)
        scene_index = scene.get("scene_index", fallback_idx)
        scene_id = f"scene:{scene_index}"
        scene_description = (scene.get("llm_scene_description") or "").strip()
        yolo_summary = _format_yolo_summary(scene)
        new_scene["spatial_relationships"] = []
        new_scene["spatial_relationship_errors"] = []

        if not scene_description:
            return new_scene

        prompt = prompt_template.replace("{{RELATION_TYPES}}", relation_types)
        prompt = prompt.replace("{{KNOWN_NODE_IDS}}", _known_node_id_text(scene_id, known_node_ids, include_scene=False))
        prompt = prompt.replace("{{SCENE_ID}}", scene_id)
        prompt = prompt.replace("{{YOLO_SUMMARY}}", apply_gpt_normalization(yolo_summary))
        prompt = prompt.replace("{{SCENE_DESCRIPTION}}", apply_gpt_normalization(scene_description))

        raw_response = ""
        try:
            raw_response = _call_relationship_extractor(
                prompt=prompt,
                client=client,
                model=model,
                gpt_deployment=gpt_deployment,
                gpt_temperature=gpt_temperature,
            )
            parsed = _parse_relationship_text(
                raw_text=_strip_code_fences(raw_response),
                scene_index=scene_index,
                known_nodes=spatial_node_map,
                relation_specs=SPATIAL_RELATION_SPECS,
            )
            new_scene["spatial_relationships"] = _apply_spatial_inverse_relationships(parsed)
        except Exception as exc:
            raw_clean = _strip_code_fences(raw_response)
            if raw_clean:
                new_scene["spatial_relationship_errors"] = [f"ERROR: {raw_clean}"]
            else:
                new_scene["spatial_relationship_errors"] = [f"ERROR: {type(exc).__name__}: {exc}"]
        return new_scene

    input_scenes = list(scenes or [])
    if not input_scenes:
        return []

    resolved_workers = _resolve_max_workers(len(input_scenes), max_workers)
    if resolved_workers <= 1:
        return [_process_scene(idx, scene) for idx, scene in enumerate(input_scenes)]

    updated_scenes = [None] * len(input_scenes)
    with ThreadPoolExecutor(max_workers=resolved_workers) as executor:
        future_to_idx = {
            executor.submit(_process_scene, idx, scene): idx
            for idx, scene in enumerate(input_scenes)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            updated_scenes[idx] = future.result()

    return updated_scenes


def extract_temporal_interval_relationships(
    scenes: list[dict],
    known_nodes: dict | None,
    client,
    model: str = "gemini-2.5-flash",
    gpt_deployment: str = "gpt-4o-kairos",
    gpt_temperature: float = 0.1,
    max_workers: int | None = None,
) -> list[dict]:
    action_node_map = _filter_node_map(known_nodes, TEMPORAL_ACTION_CATEGORIES)
    known_node_ids = build_known_node_id_map(action_node_map)
    full_id_meta_map = _build_id_meta_map(known_nodes)
    prompt_template = _load_temporal_relationship_prompt_template()
    relation_types = _relation_specs_text(TEMPORAL_INTERVAL_RELATION_SPECS)

    updated_scenes = []
    for scene in scenes or []:
        new_scene = dict(scene)
        new_scene["temporal_relationships"] = []
        new_scene["temporal_relationship_errors"] = []
        updated_scenes.append(new_scene)

    if not updated_scenes:
        return updated_scenes

    window_starts = list(range(0, max(len(updated_scenes) - 1, 0), TEMPORAL_WINDOW_STRIDE))

    def _process_window(window_start: int) -> tuple[dict[int, list[dict]], dict[int, list[str]]]:
        window = updated_scenes[window_start:window_start + TEMPORAL_WINDOW_SIZE]
        if len(window) < 2:
            return {}, {}

        window_context_lines = []
        output_scene_indices = set()

        for idx, scene in enumerate(window):
            scene_index = scene.get("scene_index", window_start + idx)
            is_context_only = idx == 0
            if not is_context_only:
                output_scene_indices.add(scene_index)
            scene_description = (scene.get("llm_scene_description") or "").strip() or "No scene description available."
            action_context = _format_action_relationship_context(scene, full_id_meta_map, window_start + idx)
            role_text = "context-only" if is_context_only else "output-scene"
            window_context_lines.append(f"Scene {scene_index} ({role_text})")
            window_context_lines.append("Description:")
            window_context_lines.append(scene_description)
            window_context_lines.append("Existing action relationships:")
            window_context_lines.append(action_context)
            window_context_lines.append("")

        prompt = prompt_template.replace("{{RELATION_TYPES}}", relation_types)
        prompt = prompt.replace(
            "{{KNOWN_NODE_IDS}}",
            _known_node_id_text("scene:this_window", known_node_ids, include_scene=False),
        )
        prompt = prompt.replace(
            "{{WINDOW_CONTEXT}}",
            apply_gpt_normalization("\n".join(window_context_lines).strip()),
        )

        raw_response = ""
        try:
            raw_response = _call_relationship_extractor(
                prompt=prompt,
                client=client,
                model=model,
                gpt_deployment=gpt_deployment,
                gpt_temperature=gpt_temperature,
            )
            parsed_by_scene, parse_errors = _parse_temporal_relationship_text(
                raw_text=_strip_code_fences(raw_response),
                known_nodes=action_node_map,
                valid_scene_indices=output_scene_indices,
            )
            relationships_by_scene = {}
            errors_by_scene = {}
            for scene in window[1:]:
                scene_index = scene.get("scene_index")
                temporal_relationships = parsed_by_scene.get(scene_index, [])
                relationships_by_scene[scene_index] = _add_temporal_occurs_in_relationships(
                    _apply_temporal_inverse_relationships(temporal_relationships),
                    scene_index,
                )
            if parse_errors:
                for scene in window[1:]:
                    scene_index = scene.get("scene_index")
                    errors_by_scene[scene_index] = list(parse_errors)
            return relationships_by_scene, errors_by_scene
        except Exception as exc:
            error_message = _strip_code_fences(raw_response)
            if error_message:
                error_value = f"ERROR: {error_message}"
            else:
                error_value = f"ERROR: {type(exc).__name__}: {exc}"
            errors_by_scene = {}
            for scene in window[1:]:
                errors_by_scene[scene.get("scene_index")] = [error_value]
            return {}, errors_by_scene

    resolved_workers = _resolve_max_workers(len(window_starts), max_workers)
    window_results = []
    if resolved_workers <= 1:
        window_results = [_process_window(window_start) for window_start in window_starts]
    else:
        with ThreadPoolExecutor(max_workers=resolved_workers) as executor:
            future_to_start = {
                executor.submit(_process_window, window_start): window_start
                for window_start in window_starts
            }
            ordered_results = {}
            for future in as_completed(future_to_start):
                window_start = future_to_start[future]
                ordered_results[window_start] = future.result()
            window_results = [ordered_results[window_start] for window_start in window_starts]

    scenes_by_index = {
        scene.get("scene_index", idx): scene
        for idx, scene in enumerate(updated_scenes)
    }
    for relationships_by_scene, errors_by_scene in window_results:
        for scene_index, temporal_relationships in relationships_by_scene.items():
            scene = scenes_by_index.get(scene_index)
            if not scene:
                continue
            scene["temporal_relationships"] = merge_scene_relationships(
                scene.get("temporal_relationships", []),
                temporal_relationships,
            )
        for scene_index, error_values in errors_by_scene.items():
            scene = scenes_by_index.get(scene_index)
            if not scene:
                continue
            scene["temporal_relationship_errors"] = list(scene.get("temporal_relationship_errors", []))
            scene["temporal_relationship_errors"].extend(error_values)

    return updated_scenes
