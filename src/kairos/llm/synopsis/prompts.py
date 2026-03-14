"""Synopsis prompt construction helpers."""

from __future__ import annotations


def _parse_count_range(value, default_min: int, default_max: int) -> tuple[int, int]:
    if isinstance(value, int):
        return max(1, value), max(1, value)
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        try:
            min_count = int(value[0])
            max_count = int(value[1])
            return max(1, min(min_count, max_count)), max(1, max(min_count, max_count))
        except (TypeError, ValueError):
            return default_min, default_max
    if isinstance(value, str):
        raw = value.strip()
        if raw:
            if "-" in raw:
                parts = [p.strip() for p in raw.split("-", 1)]
                if len(parts) == 2:
                    try:
                        min_count = int(parts[0])
                        max_count = int(parts[1])
                        return max(1, min(min_count, max_count)), max(1, max(min_count, max_count))
                    except (TypeError, ValueError):
                        return default_min, default_max
            try:
                single = int(raw)
                return max(1, single), max(1, single)
            except (TypeError, ValueError):
                return default_min, default_max
    return default_min, default_max


def _count_range_label(value, min_count: int, max_count: int) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if min_count == max_count:
        return str(min_count)
    return f"{min_count}-{max_count}"


def _count_rule(key: str, min_count: int, max_count: int, label: str) -> str:
    if min_count == max_count:
        return f"- \"{key}\" must contain exactly {min_count} items.\n"
    return f"- \"{key}\" must contain between {min_count} and {max_count} items ({label}).\n"


def _highlight_count_rule(min_count: int, max_count: int, label: str) -> str:
    return _count_rule("video_highlights", min_count, max_count, label)


def _timeline_count_rule(min_count: int, max_count: int, label: str) -> str:
    return _count_rule("video_timeline", min_count, max_count, label)


def _required_questions_block(required_questions: list[str]) -> str:
    return "\n".join([f"{idx + 1}. {q}" for idx, q in enumerate(required_questions)])


def _format_scene_ranges(items: list[dict], limit: int = 4) -> str:
    if not items:
        return "none"
    ranges = []
    for item in items:
        start_idx = item.get("scene_start_idx")
        end_idx = item.get("scene_end_idx")
        if start_idx is None or end_idx is None:
            continue
        ranges.append(f"{start_idx}->{end_idx}")
    if not ranges:
        return "none"
    if len(ranges) <= limit:
        return ", ".join(ranges)
    head = ", ".join(ranges[:limit])
    return f"{head}, ... (+{len(ranges) - limit} more)"


_SCHEMA_SUMMARY = (
    "Use this exact schema and key names:\n"
    '{ "chat_name": "3-5 word title", "summary": "Single coherent paragraph." }\n'
)

_SCHEMA_HIGHLIGHTS = (
    "Use this exact schema and key names:\n"
    '{ "video_highlights": [ { "start": "00:00:00", "end": "00:00:00", "highlight": "One sentence highlight." } ] }\n'
)

_SCHEMA_TIMELINE = (
    "Use this exact schema and key names:\n"
    '{ "video_timeline": [ { "timestamp": "00:00:00", "event": "3-5 word event" } ] }\n'
)

_SCHEMA_QNA = (
    "Use this exact schema and key names:\n"
    '{ "questions": [ { "question": "string", "answer": "string" } ] }\n'
)

_SCHEMA_MONOLITH = (
    "Use this exact schema and key names:\n"
    "{\n"
    '  "chat_name": "3-5 word title",\n'
    '  "summary": "Single coherent paragraph.",\n'
    '  "video_highlights": [ { "start": "00:00:00", "end": "00:00:00", "highlight": "One sentence highlight." } ],\n'
    '  "video_timeline": [ { "timestamp": "00:00:00", "event": "3-5 word event" } ]\n'
    "}\n"
)


def _build_repair_prompt(
    section: str,
    raw_text: str,
    highlight_min: int,
    highlight_max: int,
    highlight_label: str,
    timeline_min: int,
    timeline_max: int,
    timeline_label: str,
    required_questions_block: str,
    required_questions_count: int,
    extra_questions_count: int,
) -> str:
    header = (
        "You are a strict JSON reformatter.\n"
        "Convert the RAW OUTPUT into ONE valid JSON object only.\n"
        "No markdown, no extra text.\n"
        "If information is missing, use \"Not explicitly stated.\".\n"
    )
    if section == "summary":
        schema = (
            _SCHEMA_SUMMARY
            + "Rules:\n"
            "- If chat_name or summary is missing, set it to \"Not explicitly stated.\".\n"
        )
    elif section == "highlights":
        schema = (
            _SCHEMA_HIGHLIGHTS
            + "Rules:\n"
            f"{_highlight_count_rule(highlight_min, highlight_max, highlight_label)}"
            "- If a start or end timestamp is not explicitly stated, use \"Not explicitly stated\".\n"
        )
    elif section == "timeline":
        schema = (
            _SCHEMA_TIMELINE
            + "Rules:\n"
            f"{_timeline_count_rule(timeline_min, timeline_max, timeline_label)}"
        )
    elif section == "qna_predefined":
        schema = (
            _SCHEMA_QNA
            + "Rules:\n"
            f"- Include exactly {required_questions_count} items.\n"
            "- Questions must be the required questions below in exact order.\n"
            "Required Questions:\n"
            f"{required_questions_block}\n"
        )
    elif section == "qna_generated":
        schema = (
            _SCHEMA_QNA
            + "Rules:\n"
            f"- Include exactly {extra_questions_count} items.\n"
            "- Do not repeat required questions (they are answered elsewhere).\n"
            "- If not enough items, add placeholder questions like \"Additional predicted question N?\".\n"
        )
    else:
        schema = "Use a JSON object.\n"
    return (
        header
        + schema
        + "RAW OUTPUT:\n"
        + (raw_text or "")
    )


def _build_generated_fill_prompt(
    narrative_text: str,
    existing_questions: list[str],
    missing_count: int,
) -> str:
    existing_block = "\n".join([f"- {q}" for q in existing_questions]) if existing_questions else "None"
    return (
        "You are a story detective.\n"
        "Generate additional user-likely questions and answer them from the narrative.\n"
        "Return ONE valid JSON object only. No markdown, no extra text.\n"
        "Use this exact schema:\n"
        '{ "questions": [ { "question": "Question text", "answer": "Answer text" } ] }\n'
        "Rules:\n"
        f"- Return exactly {missing_count} items.\n"
        "- Do not repeat or paraphrase any required questions (they are answered elsewhere).\n"
        "- Do not repeat or paraphrase any existing questions below.\n"
        "- Keep generated questions concrete and useful.\n"
        "- Use only the narrative text.\n"
        "- If a detail is missing, answer \"Not explicitly stated.\".\n"
        "Existing Generated Questions (do not repeat):\n"
        f"{existing_block}\n"
        "INPUT NARRATIVE:\n"
        f"{narrative_text}\n"
    )


def _build_questions_prompt(
    narrative_text: str,
    required_questions: list,
    extra_questions_count: int,
    strict: bool = False,
):
    required_block = "\n".join(
        [f"{idx + 1}. {q}" for idx, q in enumerate(required_questions)]
    )
    total_questions = len(required_questions) + extra_questions_count
    strict_block = (
        "Your response MUST contain only the JSON object described below. "
        "Do not include any other keys or text.\n"
        if strict
        else ""
    )
    return (
        strict_block
        +
        "Return ONE valid JSON object only. No markdown, no extra text.\n"
        "Use this exact schema and key names:\n"
        "{\n"
        '  "questions": [{"question": "Question text", "answer": "Answer text"}]\n'
        "}\n"
        "Rules:\n"
        f'- "questions" must contain exactly {total_questions} items.\n'
        "- The first questions must be the Required Questions in the exact order listed below.\n"
        f"- After that, add exactly {extra_questions_count} new, predicted questions.\n"
        '- Use only the narrative. If a detail is missing, set the answer to "Not explicitly stated."\n'
        "- Do not include any other keys besides questions.\n"
        'Required Questions:\n'
        f"{required_block}\n"
        "INPUT NARRATIVE:\n"
        f"{narrative_text}\n"
    )


def _build_scene_chunk_summary_prompt(chunk: dict) -> str:
    return (
        "You are a story detective.\n"
        "Summarize the scene chunk chronologically and factually.\n"
        "Rules:\n"
        "- Use only the provided chunk text.\n"
        "- Keep names, objects, and quotes precise.\n"
        "- Include key timestamps when present.\n"
        "- Keep it concise while preserving important events.\n"
        f"Chunk metadata: scenes {chunk['scene_start_idx']} to {chunk['scene_end_idx']}, "
        f"time {chunk.get('start_timecode')} to {chunk.get('end_timecode')}.\n\n"
        f"SCENE CHUNK:\n{chunk['text']}\n"
    )


def _build_reduce_prompt(items: list[dict], round_idx: int) -> str:
    blocks = []
    for item in items:
        blocks.append(
            f"[Range scenes {item['scene_start_idx']}-{item['scene_end_idx']}, "
            f"time {item.get('start_timecode')} to {item.get('end_timecode')}]\n"
            f"{item['text']}"
        )
    joined = "\n\n".join(blocks)
    return (
        "You are a story detective.\n"
        "Merge the following adjacent chronological summaries into one coherent summary.\n"
        "Rules:\n"
        "- Keep chronological order.\n"
        "- Remove duplicates only when meaning is preserved.\n"
        "- Do not invent details.\n"
        "- Preserve key entities, objects, and events.\n"
        f"- This is reduce round {round_idx}.\n\n"
        f"INPUT SUMMARIES:\n{joined}\n"
    )


def _build_narrative_consistency_prompt(narrative: str) -> str:
    return (
        "You are a consistency editor for a chronological video narrative.\n"
        "Revise the narrative only to improve consistency and flow.\n"
        "Rules:\n"
        "- Keep chronology intact.\n"
        "- Do not add facts not present in the narrative.\n"
        "- Keep names and object references consistent.\n"
        "- Remove obvious contradictions or duplicates.\n"
        "- Keep output concise.\n\n"
        f"NARRATIVE:\n{narrative}\n"
    )


def _build_safe_section_prompt(
    section: str,
    narrative_text: str,
    highlight_min: int,
    highlight_max: int,
    highlight_label: str,
    timeline_min: int,
    timeline_max: int,
    timeline_label: str,
    required_questions_block: str,
    extra_questions_count: int,
) -> str:
    header = (
        "You are a story detective.\n"
        "Return ONE valid JSON object only. No markdown, no extra text.\n"
        "Base answers ONLY on the provided text. Do not infer or invent details.\n"
        "If information is missing, use \"Not explicitly stated.\"\n"
        "Ensure the report complies with a PG-13 content standard.\n"
    )
    if section == "summary":
        schema = (
            _SCHEMA_SUMMARY
            + "Rules:\n"
            "- \"chat_name\" must be 3-5 words and concrete, not creative.\n"
            "- \"summary\" must be one paragraph.\n"
        )
    elif section == "highlights":
        schema = (
            _SCHEMA_HIGHLIGHTS
            + "Rules:\n"
            f"{_highlight_count_rule(highlight_min, highlight_max, highlight_label)}"
            "- Each highlight is one sentence.\n"
            "- If a start or end timestamp is not explicitly stated, use \"Not explicitly stated\".\n"
        )
    elif section == "timeline":
        schema = (
            _SCHEMA_TIMELINE
            + "Rules:\n"
            f"{_timeline_count_rule(timeline_min, timeline_max, timeline_label)}"
            "- Events must be 3-5 words, chronological order.\n"
            "- If a timestamp is not explicitly stated, use \"Not explicitly stated\".\n"
        )
    elif section == "qna_predefined":
        schema = (
            _SCHEMA_QNA
            + "Rules:\n"
            "- Include only these required questions, in order, no extras:\n"
            f"{required_questions_block}\n"
        )
    elif section == "qna_generated":
        schema = (
            _SCHEMA_QNA
            + "Rules:\n"
            f"- Add exactly {extra_questions_count} additional questions (not in required list).\n"
            "- Do not repeat required questions (they are answered elsewhere).\n"
            "- Use only the narrative.\n"
        )
    elif section == "qna_legacy":
        schema = (
            _SCHEMA_QNA
            + "Rules:\n"
            "- Include the required questions below, in order, then add extra questions.\n"
            f"- Add exactly {extra_questions_count} extra questions.\n"
            "Required Questions:\n"
            f"{required_questions_block}\n"
        )
    elif section == "monolith":
        schema = (
            _SCHEMA_MONOLITH
            + "Rules:\n"
            "- \"chat_name\" must be 3-5 words and concrete, not creative.\n"
            f"{_highlight_count_rule(highlight_min, highlight_max, highlight_label)}"
            f"{_timeline_count_rule(timeline_min, timeline_max, timeline_label)}"
            "- If a start or end timestamp is not explicitly stated, use \"Not explicitly stated\".\n"
        )
    else:
        schema = "Use a JSON object.\n"
    return (
        header
        + schema
        + "INPUT NARRATIVE:\n"
        + narrative_text
    )


def _build_monolith_prompt(
    narrative_text: str,
    highlight_min: int,
    highlight_max: int,
    highlight_label: str,
    timeline_min: int,
    timeline_max: int,
    timeline_label: str,
) -> str:
    return (
        "You are a story detective.\n"
        "Return ONE valid JSON object only. No markdown, no extra text.\n"
        + _SCHEMA_MONOLITH
        + "Rules:\n"
        "- \"chat_name\" must be 3-5 words and concrete, not creative.\n"
        f"{_highlight_count_rule(highlight_min, highlight_max, highlight_label)}"
        f"{_timeline_count_rule(timeline_min, timeline_max, timeline_label)}"
        "- If a start or end timestamp is not explicitly stated, use \"Not explicitly stated\".\n"
        "INPUT NARRATIVE:\n"
        f"{narrative_text}\n"
    )


def _build_consistency_prompt(
    narrative_text: str,
    draft_synopsis: dict,
    highlight_min: int,
    highlight_max: int,
    highlight_label: str,
    timeline_min: int,
    timeline_max: int,
    timeline_label: str,
    required_block: str,
    required_questions_count: int,
    extra_questions_count: int,
) -> str:
    import json
    draft_json = json.dumps(draft_synopsis, ensure_ascii=False)
    consistency_schema = (
        "Use this exact schema and key names:\n"
        "{\n"
        '  "chat_name": "3-5 word title",\n'
        '  "summary": "Single coherent paragraph.",\n'
        '  "video_highlights": [ { "start": "00:00:00", "end": "00:00:00", "highlight": "One sentence highlight." } ],\n'
        '  "video_timeline": [ { "timestamp": "00:00:00", "event": "3-5 word event" } ],\n'
        '  "questions": [ { "question": "Question text", "answer": "Answer text" } ]\n'
        "}\n"
    )
    return (
        "You are a story detective.\n"
        "Return ONE valid JSON object only. No markdown, no extra text.\n"
        "Review the draft synopsis and correct any factual inconsistencies using only the narrative.\n"
        + consistency_schema
        + "Rules:\n"
        f"{_highlight_count_rule(highlight_min, highlight_max, highlight_label)}"
        f"{_timeline_count_rule(timeline_min, timeline_max, timeline_label)}"
        f"- \"questions\" must contain exactly {required_questions_count} items.\n"
        "- Do not add any sections not in the schema.\n"
        "Required Questions (must appear in order within questions):\n"
        f"{required_block}\n"
        "INPUT NARRATIVE:\n"
        f"{narrative_text}\n"
        "DRAFT JSON:\n"
        f"{draft_json}\n"
    )
