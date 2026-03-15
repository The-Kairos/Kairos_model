"""Synopsis prompt construction helpers."""

from __future__ import annotations

from typing import Any

from kairos.llm.synopsis.parsing import NOT_STATED, NOT_STATED_PERIOD


def _parse_count_range(
    value: int | str | tuple[Any, ...] | list[Any] | Any,
    default_min: int,
    default_max: int,
) -> tuple[int, int]:
    """Parse a count specification into a ``(min, max)`` integer range.

    *value* may be a single ``int``, a 2-element tuple/list, or a string
    such as ``"4-6"`` or ``"5"``.

    Args:
        value: The count specification to parse.
        default_min: Returned minimum when *value* cannot be parsed.
        default_max: Returned maximum when *value* cannot be parsed.

    Returns:
        A 2-tuple ``(min_count, max_count)`` with values ≥ 1.
    """
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
                        return max(1, min(min_count, max_count)), max(
                            1, max(min_count, max_count)
                        )
                    except (TypeError, ValueError):
                        return default_min, default_max
            try:
                single = int(raw)
                return max(1, single), max(1, single)
            except (TypeError, ValueError):
                return default_min, default_max
    return default_min, default_max


def _count_range_label(
    value: int | str | Any, min_count: int, max_count: int
) -> str:
    """Return a human-readable label for a count range.

    If *value* is already a non-empty string it is returned as-is;
    otherwise a label such as ``"4"`` or ``"4-6"`` is generated.

    Args:
        value: Original count specification (may be a string label).
        min_count: Minimum of the range.
        max_count: Maximum of the range.

    Returns:
        A string label describing the range.
    """
    if isinstance(value, str) and value.strip():
        return value.strip()
    if min_count == max_count:
        return str(min_count)
    return f"{min_count}-{max_count}"


def _count_rule(key: str, min_count: int, max_count: int, label: str) -> str:
    r"""Build a prompt rule line describing an item-count constraint.

    Args:
        key: JSON key name the rule refers to.
        min_count: Minimum required item count.
        max_count: Maximum allowed item count.
        label: Human-readable label for the range.

    Returns:
        A single prompt rule line ending with ``\\n``.
    """
    if min_count == max_count:
        return f'- "{key}" must contain exactly {min_count} items.\n'
    return (
        f'- "{key}" must contain between {min_count} and {max_count} items ({label}).\n'
    )


def _highlight_count_rule(min_count: int, max_count: int, label: str) -> str:
    r"""Build a prompt rule line for the ``video_highlights`` count.

    Args:
        min_count: Minimum required number of highlights.
        max_count: Maximum allowed number of highlights.
        label: Human-readable label for the range.

    Returns:
        A single prompt rule line ending with ``\\n``.
    """
    return _count_rule("video_highlights", min_count, max_count, label)


def _timeline_count_rule(min_count: int, max_count: int, label: str) -> str:
    r"""Build a prompt rule line for the ``video_timeline`` count.

    Args:
        min_count: Minimum required number of timeline events.
        max_count: Maximum allowed number of timeline events.
        label: Human-readable label for the range.

    Returns:
        A single prompt rule line ending with ``\\n``.
    """
    return _count_rule("video_timeline", min_count, max_count, label)


def _required_questions_block(required_questions: list[str]) -> str:
    """Format a numbered list of required questions for prompt insertion.

    Args:
        required_questions: Ordered list of question strings.

    Returns:
        A newline-separated numbered list (1-indexed).
    """
    return "\n".join([f"{idx + 1}. {q}" for idx, q in enumerate(required_questions)])


def _format_scene_ranges(items: list[dict[str, Any]], limit: int = 4) -> str:
    """Format scene index ranges for debug output.

    Args:
        items: List of dicts with ``scene_start_idx`` and
            ``scene_end_idx`` keys.
        limit: Maximum number of ranges to show before truncating.

    Returns:
        A comma-separated string of ranges such as
        ``"0->4, 5->9, ... (+2 more)"``, or ``"none"`` if empty.
    """
    if not items:
        return "none"
    ranges: list[str] = []
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
    '{ "video_highlights": [ { "start": "00:00:00", '
    '"end": "00:00:00", '
    '"highlight": "One sentence highlight." } ] }\n'
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
    '  "video_highlights": [ { "start": "00:00:00", '
    '"end": "00:00:00", '
    '"highlight": "One sentence highlight." } ],\n'
    '  "video_timeline": [ { "timestamp": "00:00:00", '
    '"event": "3-5 word event" } ]\n'
    "}\n"
)

_SECTION_SCHEMAS: dict[str, str] = {
    "summary": _SCHEMA_SUMMARY,
    "highlights": _SCHEMA_HIGHLIGHTS,
    "timeline": _SCHEMA_TIMELINE,
    "qna_predefined": _SCHEMA_QNA,
    "qna_generated": _SCHEMA_QNA,
    "qna_legacy": _SCHEMA_QNA,
    "monolith": _SCHEMA_MONOLITH,
}

_TIMESTAMP_NOT_STATED_RULE = (
    f'- If a start or end timestamp is not explicitly stated, use "{NOT_STATED}".\n'
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
    """Build a JSON repair prompt for a failed section parse.

    Instructs the model to reformat *raw_text* into the correct JSON
    schema for the given *section*.

    Args:
        section: Section identifier (e.g. ``"summary"``,
            ``"highlights"``, ``"timeline"``, ``"qna_predefined"``,
            ``"qna_generated"``).
        raw_text: The raw model output that failed parsing.
        highlight_min: Minimum required number of highlights.
        highlight_max: Maximum allowed number of highlights.
        highlight_label: Human-readable range label for highlights.
        timeline_min: Minimum required number of timeline events.
        timeline_max: Maximum allowed number of timeline events.
        timeline_label: Human-readable range label for timeline.
        required_questions_block: Formatted block of required questions.
        required_questions_count: Number of required questions.
        extra_questions_count: Number of extra generated questions.

    Returns:
        The fully assembled repair prompt string.
    """
    header = (
        "You are a strict JSON reformatter.\n"
        "Convert the RAW OUTPUT into ONE valid JSON object only.\n"
        "No markdown, no extra text.\n"
        f'If information is missing, use "{NOT_STATED_PERIOD}".\n'
    )
    base_schema = _SECTION_SCHEMAS.get(section, "Use a JSON object.\n")
    if section == "summary":
        rules = (
            f'- If chat_name or summary is missing, set it to "{NOT_STATED_PERIOD}".\n'
        )
    elif section == "highlights":
        rules = (
            f"{_highlight_count_rule(highlight_min, highlight_max, highlight_label)}"
            f"{_TIMESTAMP_NOT_STATED_RULE}"
        )
    elif section == "timeline":
        rules = f"{_timeline_count_rule(timeline_min, timeline_max, timeline_label)}"
    elif section == "qna_predefined":
        rules = (
            f"- Include exactly {required_questions_count} items.\n"
            "- Questions must be the required questions below in exact order.\n"
            "Required Questions:\n"
            f"{required_questions_block}\n"
        )
    elif section == "qna_generated":
        rules = (
            f"- Include exactly {extra_questions_count} items.\n"
            "- Do not repeat required questions (they are answered elsewhere).\n"
            "- If not enough items, add placeholder questions "
            'like "Additional predicted question N?".\n'
        )
    else:
        rules = ""
    schema = base_schema + ("Rules:\n" + rules if rules else "")
    return header + schema + "RAW OUTPUT:\n" + (raw_text or "")


def _build_generated_fill_prompt(
    narrative_text: str,
    existing_questions: list[str],
    missing_count: int,
) -> str:
    """Build a prompt to generate additional Q&A pairs from a narrative.

    Args:
        narrative_text: The narrative text to derive questions from.
        existing_questions: Questions already generated (to avoid
            repetition).
        missing_count: Number of new questions to generate.

    Returns:
        The fully assembled fill prompt string.
    """
    existing_block = (
        "\n".join([f"- {q}" for q in existing_questions])
        if existing_questions
        else "None"
    )
    return (
        "You are a story detective.\n"
        "Generate additional user-likely questions "
        "and answer them from the narrative.\n"
        "Return ONE valid JSON object only. No markdown, no extra text.\n"
        "Use this exact schema:\n"
        '{ "questions": [ { "question": "Question text", '
        '"answer": "Answer text" } ] }\n'
        "Rules:\n"
        f"- Return exactly {missing_count} items.\n"
        "- Do not repeat or paraphrase any required "
        "questions (they are answered elsewhere).\n"
        "- Do not repeat or paraphrase any existing questions below.\n"
        "- Keep generated questions concrete and useful.\n"
        "- Use only the narrative text.\n"
        f'- If a detail is missing, answer "{NOT_STATED_PERIOD}".\n'
        "Existing Generated Questions (do not repeat):\n"
        f"{existing_block}\n"
        "INPUT NARRATIVE:\n"
        f"{narrative_text}\n"
    )


def _build_questions_prompt(
    narrative_text: str,
    required_questions: list[str],
    extra_questions_count: int,
    strict: bool = False,
) -> str:
    """Build a combined prompt for required + generated questions.

    Args:
        narrative_text: The narrative text to answer from.
        required_questions: Ordered list of required question strings.
        extra_questions_count: Number of additional predicted questions.
        strict: If ``True``, add a strictness preamble requiring only
            the described JSON object in the response.

    Returns:
        The fully assembled questions prompt string.
    """
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
        + "Return ONE valid JSON object only. No markdown, no extra text.\n"
        "Use this exact schema and key names:\n"
        "{\n"
        '  "questions": [{"question": "Question text", "answer": "Answer text"}]\n'
        "}\n"
        "Rules:\n"
        f'- "questions" must contain exactly {total_questions} items.\n'
        "- The first questions must be the Required "
        "Questions in the exact order listed below.\n"
        f"- After that, add exactly {extra_questions_count} new, predicted questions.\n"
        "- Use only the narrative. If a detail is missing, "
        f'set the answer to "{NOT_STATED_PERIOD}"\n'
        "- Do not include any other keys besides questions.\n"
        "Required Questions:\n"
        f"{required_block}\n"
        "INPUT NARRATIVE:\n"
        f"{narrative_text}\n"
    )


def _build_scene_chunk_summary_prompt(chunk: dict[str, Any]) -> str:
    """Build a prompt to summarise a single scene chunk.

    Args:
        chunk: A chunk dict with keys ``scene_start_idx``,
            ``scene_end_idx``, ``start_timecode``, ``end_timecode``,
            and ``text``.

    Returns:
        The fully assembled scene-chunk summary prompt string.
    """
    return (
        "You are a story detective.\n"
        "Summarize the scene chunk chronologically and factually.\n"
        "Rules:\n"
        "- Use only the provided chunk text.\n"
        "- Keep names, objects, and quotes precise.\n"
        "- Include key timestamps when present.\n"
        "- Keep it concise while preserving important events.\n"
        f"Chunk metadata: scenes {chunk['scene_start_idx']} "
        f"to {chunk['scene_end_idx']}, "
        f"time {chunk.get('start_timecode')} "
        f"to {chunk.get('end_timecode')}.\n\n"
        f"SCENE CHUNK:\n{chunk['text']}\n"
    )


def _build_reduce_prompt(items: list[dict[str, Any]], round_idx: int) -> str:
    """Build a prompt to merge adjacent summary blocks in a reduce round.

    Args:
        items: List of summary dicts to merge, each containing
            ``scene_start_idx``, ``scene_end_idx``, ``start_timecode``,
            ``end_timecode``, and ``text``.
        round_idx: The current reduce round number (1-indexed).

    Returns:
        The fully assembled reduce prompt string.
    """
    blocks: list[str] = []
    for item in items:
        blocks.append(
            f"[Range scenes {item['scene_start_idx']}-{item['scene_end_idx']}, "
            f"time {item.get('start_timecode')} to {item.get('end_timecode')}]\n"
            f"{item['text']}"
        )
    joined = "\n\n".join(blocks)
    return (
        "You are a story detective.\n"
        "Merge the following adjacent chronological "
        "summaries into one coherent summary.\n"
        "Rules:\n"
        "- Keep chronological order.\n"
        "- Remove duplicates only when meaning is preserved.\n"
        "- Do not invent details.\n"
        "- Preserve key entities, objects, and events.\n"
        f"- This is reduce round {round_idx}.\n\n"
        f"INPUT SUMMARIES:\n{joined}\n"
    )


def _build_narrative_consistency_prompt(narrative: str) -> str:
    """Build a prompt for a narrative consistency editing pass.

    Args:
        narrative: The narrative text to review and revise.

    Returns:
        The fully assembled consistency-editing prompt string.
    """
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
    """Build a safe (PG-13 compliant) prompt for a synopsis section.

    This prompt is used as a fallback when the primary section prompt
    fails (e.g. due to a content filter).

    Args:
        section: Section identifier (e.g. ``"summary"``,
            ``"highlights"``, ``"timeline"``, ``"qna_predefined"``,
            ``"qna_generated"``, ``"qna_legacy"``, ``"monolith"``).
        narrative_text: The narrative text to base the section on.
        highlight_min: Minimum required number of highlights.
        highlight_max: Maximum allowed number of highlights.
        highlight_label: Human-readable range label for highlights.
        timeline_min: Minimum required number of timeline events.
        timeline_max: Maximum allowed number of timeline events.
        timeline_label: Human-readable range label for timeline.
        required_questions_block: Formatted block of required questions.
        extra_questions_count: Number of extra generated questions.

    Returns:
        The fully assembled safe section prompt string.
    """
    header = (
        "You are a story detective.\n"
        "Return ONE valid JSON object only. No markdown, no extra text.\n"
        "Base answers ONLY on the provided text. Do not infer or invent details.\n"
        f'If information is missing, use "{NOT_STATED_PERIOD}"\n'
        "Ensure the report complies with a PG-13 content standard.\n"
    )
    base_schema = _SECTION_SCHEMAS.get(section, "Use a JSON object.\n")
    if section == "summary":
        rules = (
            '- "chat_name" must be 3-5 words and concrete, not creative.\n'
            '- "summary" must be one paragraph.\n'
        )
    elif section == "highlights":
        rules = (
            f"{_highlight_count_rule(highlight_min, highlight_max, highlight_label)}"
            "- Each highlight is one sentence.\n"
            f"{_TIMESTAMP_NOT_STATED_RULE}"
        )
    elif section == "timeline":
        rules = (
            f"{_timeline_count_rule(timeline_min, timeline_max, timeline_label)}"
            "- Events must be 3-5 words, chronological order.\n"
            f'- If a timestamp is not explicitly stated, use "{NOT_STATED}".\n'
        )
    elif section == "qna_predefined":
        rules = (
            "- Include only these required questions, in order, no extras:\n"
            f"{required_questions_block}\n"
        )
    elif section == "qna_generated":
        rules = (
            f"- Add exactly {extra_questions_count} "
            "additional questions (not in required list).\n"
            "- Do not repeat required questions (they are answered elsewhere).\n"
            "- Use only the narrative.\n"
        )
    elif section == "qna_legacy":
        rules = (
            "- Include the required questions below, "
            "in order, then add extra questions.\n"
            f"- Add exactly {extra_questions_count} extra questions.\n"
            "Required Questions:\n"
            f"{required_questions_block}\n"
        )
    elif section == "monolith":
        rules = (
            '- "chat_name" must be 3-5 words and concrete, not creative.\n'
            f"{_highlight_count_rule(highlight_min, highlight_max, highlight_label)}"
            f"{_timeline_count_rule(timeline_min, timeline_max, timeline_label)}"
            f"{_TIMESTAMP_NOT_STATED_RULE}"
        )
    else:
        rules = ""
    schema = base_schema + ("Rules:\n" + rules if rules else "")
    return header + schema + "INPUT NARRATIVE:\n" + narrative_text


def _build_monolith_prompt(
    narrative_text: str,
    highlight_min: int,
    highlight_max: int,
    highlight_label: str,
    timeline_min: int,
    timeline_max: int,
    timeline_label: str,
) -> str:
    """Build a monolithic synopsis prompt (summary + highlights + timeline).

    Args:
        narrative_text: The narrative text to base the synopsis on.
        highlight_min: Minimum required number of highlights.
        highlight_max: Maximum allowed number of highlights.
        highlight_label: Human-readable range label for highlights.
        timeline_min: Minimum required number of timeline events.
        timeline_max: Maximum allowed number of timeline events.
        timeline_label: Human-readable range label for timeline.

    Returns:
        The fully assembled monolith prompt string.
    """
    return (
        "You are a story detective.\n"
        "Return ONE valid JSON object only. No markdown, no extra text.\n"
        + _SCHEMA_MONOLITH
        + "Rules:\n"
        '- "chat_name" must be 3-5 words and concrete, not creative.\n'
        f"{_highlight_count_rule(highlight_min, highlight_max, highlight_label)}"
        f"{_timeline_count_rule(timeline_min, timeline_max, timeline_label)}"
        f"{_TIMESTAMP_NOT_STATED_RULE}"
        "INPUT NARRATIVE:\n"
        f"{narrative_text}\n"
    )


def _build_consistency_prompt(
    narrative_text: str,
    draft_synopsis: dict[str, Any],
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
    """Build a prompt to verify and correct a draft synopsis for consistency.

    Args:
        narrative_text: The source narrative text.
        draft_synopsis: The draft synopsis dict to review.
        highlight_min: Minimum required number of highlights.
        highlight_max: Maximum allowed number of highlights.
        highlight_label: Human-readable range label for highlights.
        timeline_min: Minimum required number of timeline events.
        timeline_max: Maximum allowed number of timeline events.
        timeline_label: Human-readable range label for timeline.
        required_block: Formatted block of required questions.
        required_questions_count: Total number of required questions.
        extra_questions_count: Number of extra generated questions.

    Returns:
        The fully assembled consistency-check prompt string.
    """
    import json

    draft_json = json.dumps(draft_synopsis, ensure_ascii=False)
    consistency_schema = (
        "Use this exact schema and key names:\n"
        "{\n"
        '  "chat_name": "3-5 word title",\n'
        '  "summary": "Single coherent paragraph.",\n'
        '  "video_highlights": [ { "start": "00:00:00", '
        '"end": "00:00:00", '
        '"highlight": "One sentence highlight." } ],\n'
        '  "video_timeline": [ { "timestamp": "00:00:00", '
        '"event": "3-5 word event" } ],\n'
        '  "questions": [ { "question": "Question text", "answer": "Answer text" } ]\n'
        "}\n"
    )
    return (
        "You are a story detective.\n"
        "Return ONE valid JSON object only. No markdown, no extra text.\n"
        "Review the draft synopsis and correct any factual "
        "inconsistencies using only the narrative.\n" + consistency_schema + "Rules:\n"
        f"{_highlight_count_rule(highlight_min, highlight_max, highlight_label)}"
        f"{_timeline_count_rule(timeline_min, timeline_max, timeline_label)}"
        f'- "questions" must contain exactly {required_questions_count} items.\n'
        "- Do not add any sections not in the schema.\n"
        "Required Questions (must appear in order within questions):\n"
        f"{required_block}\n"
        "INPUT NARRATIVE:\n"
        f"{narrative_text}\n"
        "DRAFT JSON:\n"
        f"{draft_json}\n"
    )
