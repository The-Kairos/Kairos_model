"""Synopsis JSON/pipe parsing, validation, and normalization."""

from __future__ import annotations

import json
import re
from typing import Any

from kairos.core.utils import print_prefixed

NOT_STATED = "Not explicitly stated"
NOT_STATED_PERIOD = "Not explicitly stated."


def _parse_json_object(
    text: str, debug: bool = False, context: str = "section"
) -> dict[str, Any]:
    """Attempt to extract and parse a JSON object from *text*.

    If the raw text is not valid JSON, the function looks for the
    outermost ``{`` … ``}`` pair and tries again.

    Args:
        text: Raw text that may contain a JSON object.
        debug: If ``True``, log parse failures.
        context: Label used in debug messages to identify the section.

    Returns:
        The parsed dictionary, or an empty dict on failure.
    """
    if not isinstance(text, str):
        return {}
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
                return obj if isinstance(obj, dict) else {}
            except json.JSONDecodeError as exc:
                if debug:
                    print_prefixed("(Synopsis)", f"{context}: JSON parse failed: {exc}")
                return {}
        if debug:
            print_prefixed("(Synopsis)", f"{context}: JSON parse failed")
        return {}


def _synopsis_fallback(text: str, parse_error: str) -> dict[str, Any]:
    """Return a minimal fallback synopsis dictionary.

    Used when the model output cannot be parsed into a valid synopsis
    structure.

    Args:
        text: Original text to preserve as the ``summary`` value.
        parse_error: Human-readable description of the parse failure.

    Returns:
        A dict with default keys (``chat_name``, ``summary``,
        ``video_highlights``, ``video_timeline``, ``questions``,
        ``parse_error``).
    """
    return {
        "chat_name": NOT_STATED,
        "summary": text.strip() if isinstance(text, str) else "",
        "video_highlights": [],
        "video_timeline": [],
        "questions": [],
        "parse_error": parse_error,
    }


def _parse_synopsis_json(text: str, debug: bool = False) -> dict[str, Any]:
    """Parse a synopsis JSON string, returning a fallback on failure.

    Args:
        text: Raw JSON text from the model.
        debug: If ``True``, log parse failures.

    Returns:
        The parsed synopsis dict, or a fallback dict if parsing fails.
    """
    if not isinstance(text, str):
        return _synopsis_fallback("", "Synopsis output was not a string")
    obj = _parse_json_object(text, debug=debug, context="synopsis")
    if not obj:
        return _synopsis_fallback(text, "Invalid JSON from model")
    return obj


def _parse_pipe_pairs(text: str) -> list[tuple[str, str]]:
    """Parse ``left | right`` pipe-delimited lines into pairs.

    Blank lines and lines without a ``|`` character are skipped.

    Args:
        text: Multi-line text with pipe-delimited pairs.

    Returns:
        A list of ``(left, right)`` string tuples.
    """
    if not isinstance(text, str):
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    pairs: list[tuple[str, str]] = []
    for line in lines:
        if "|" not in line:
            continue
        left, right = line.split("|", 1)
        left = left.strip()
        right = right.strip()
        if not left and not right:
            continue
        pairs.append((left, right))
    return pairs


def _parse_qna_pairs(text: str) -> list[tuple[str, str]]:
    """Parse question/answer pairs from *text*.

    The function first tries pipe-delimited format. If no pipe pairs are
    found it falls back to ``Q: … A: …`` patterns (inline or on
    separate lines).

    Args:
        text: Multi-line text containing question/answer pairs.

    Returns:
        A list of ``(question, answer)`` string tuples.
    """
    pairs = _parse_pipe_pairs(text)
    if pairs:
        return pairs
    if not isinstance(text, str):
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    current_question: str | None = None
    for line in lines:
        inline = re.match(
            r"^\s*q(uestion)?\s*[:\-]\s*(.+?)\s+a(nswer)?\s*[:\-]\s*(.+)$",
            line,
            re.IGNORECASE,
        )
        if inline:
            q_text = inline.group(2).strip()
            a_text = inline.group(4).strip()
            if q_text or a_text:
                pairs.append((q_text, a_text))
            current_question = None
            continue
        q_match = re.match(r"^\s*q(uestion)?\s*[:\-]\s*(.+)$", line, re.IGNORECASE)
        if q_match:
            current_question = q_match.group(2).strip()
            continue
        a_match = re.match(r"^\s*a(nswer)?\s*[:\-]\s*(.+)$", line, re.IGNORECASE)
        if a_match and current_question:
            answer = a_match.group(2).strip()
            pairs.append((current_question, answer))
            current_question = None
    return pairs


def _split_time_range(text: str) -> tuple[str, str]:
    """Split a time-range string into start and end timecodes.

    Supported separators include ``" - "``, ``" – "``, ``" — "``,
    ``" to "``, and ``"-"``.

    Args:
        text: A time-range string (e.g. ``"00:01:00 - 00:02:30"``).

    Returns:
        A 2-tuple ``(start, end)`` of timecode strings.  Missing parts
        are replaced with :data:`NOT_STATED`.
    """
    if not isinstance(text, str) or not text.strip():
        return NOT_STATED, NOT_STATED
    raw = text.strip()
    for sep in (" - ", " – ", " — ", " to ", "-"):  # noqa: RUF001
        if sep in raw:
            left, right = raw.split(sep, 1)
            left = left.strip() if left.strip() else NOT_STATED
            right = right.strip() if right.strip() else NOT_STATED
            return left, right
    return raw, NOT_STATED


def _parse_summary_nonjson(text: str) -> tuple[dict[str, Any], bool]:
    """Parse a summary section from pipe-delimited text.

    Expects the first pipe pair to be ``chat_name | summary``.

    Args:
        text: Raw text from the model.

    Returns:
        A 2-tuple ``(payload, ok)`` where *payload* is a dict with
        ``chat_name`` and ``summary`` keys and *ok* indicates whether
        parsing succeeded.
    """
    pairs = _parse_pipe_pairs(text)
    if not pairs:
        return {}, False
    chat_name, summary = pairs[0]
    if not chat_name or not summary:
        return {}, False
    return {"chat_name": chat_name, "summary": summary}, True


def _parse_items_nonjson(
    text: str, key: str, text_key: str, expected_count: int
) -> tuple[dict[str, Any], bool]:
    """Parse timestamped items from pipe-delimited text.

    Each pipe pair maps to ``{"timestamp": left, text_key: right}``.

    Args:
        text: Raw text from the model.
        key: Top-level key name in the returned dict (e.g.
            ``"video_timeline"``).
        text_key: Key name for the right-hand value in each item dict
            (e.g. ``"event"``).
        expected_count: Minimum number of items required for *ok* to be
            ``True``.

    Returns:
        A 2-tuple ``(payload, ok)`` where *payload* contains the parsed
        items under *key*.
    """
    pairs = _parse_pipe_pairs(text)
    ok = len(pairs) >= expected_count
    items: list[dict[str, str]] = []
    for left, right in pairs[:expected_count]:
        ts = left.strip() if left else NOT_STATED
        value = right.strip() if right else NOT_STATED_PERIOD
        if not ts:
            ts = NOT_STATED
        if not value:
            value = NOT_STATED_PERIOD
        items.append({"timestamp": ts, text_key: value})
    return {key: items}, ok


def _parse_highlights_nonjson(
    text: str, min_count: int, max_count: int
) -> tuple[dict[str, Any], bool]:
    """Parse video highlights from pipe-delimited text.

    Each line may have 2 or 3 pipe-separated parts:
    ``start | end | highlight`` or ``time_range | highlight``.

    Args:
        text: Raw text from the model.
        min_count: Minimum number of highlights required.
        max_count: Maximum number of highlights kept.

    Returns:
        A 2-tuple ``(payload, ok)`` where *payload* contains a
        ``video_highlights`` list and *ok* indicates whether the count
        is within the allowed range.
    """
    if not isinstance(text, str):
        return {}, False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    items: list[dict[str, str]] = []
    for line in lines:
        if "|" not in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        start = end = highlight = ""
        if len(parts) >= 3:
            start = parts[0]
            end = parts[1]
            highlight = "|".join(parts[2:]).strip()
        elif len(parts) == 2:
            start, end = _split_time_range(parts[0])
            highlight = parts[1]
        if not start:
            start = NOT_STATED
        if not end:
            end = NOT_STATED
        if not highlight:
            highlight = NOT_STATED_PERIOD
        items.append({"start": start, "end": end, "highlight": highlight})
    if len(items) > max_count:
        items = items[:max_count]
    ok = min_count <= len(items) <= max_count
    return {"video_highlights": items}, ok


def _parse_timeline_nonjson(
    text: str, min_count: int, max_count: int
) -> tuple[dict[str, Any], bool]:
    """Parse a video timeline from pipe-delimited text.

    Args:
        text: Raw text from the model.
        min_count: Minimum number of timeline events required.
        max_count: Maximum number of timeline events kept.

    Returns:
        A 2-tuple ``(payload, ok)`` where *payload* contains a
        ``video_timeline`` list and *ok* indicates whether the count is
        within the allowed range.
    """
    result, _ = _parse_items_nonjson(text, "video_timeline", "event", max_count)
    items = result.get("video_timeline", [])
    if len(items) > max_count:
        items = items[:max_count]
    ok = min_count <= len(items) <= max_count
    return {"video_timeline": items}, ok


def _parse_questions_nonjson(
    text: str, expected_count: int
) -> tuple[dict[str, Any], bool]:
    """Parse question/answer pairs from non-JSON text.

    Args:
        text: Raw text from the model.
        expected_count: Minimum number of Q&A pairs required for *ok* to
            be ``True``.

    Returns:
        A 2-tuple ``(payload, ok)`` where *payload* contains a
        ``questions`` list and *ok* indicates whether enough pairs were
        found.
    """
    pairs = _parse_qna_pairs(text)
    ok = len(pairs) >= expected_count
    questions: list[dict[str, str]] = []
    for left, right in pairs[:expected_count]:
        question = _clean_question_text(left) if left else ""
        if not question:
            question = NOT_STATED
        answer = right.strip() if right else NOT_STATED_PERIOD
        if not answer:
            answer = NOT_STATED_PERIOD
        questions.append({"question": question, "answer": answer})
    return {"questions": questions}, ok


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_summary_payload(payload: dict[str, Any]) -> bool:
    """Check that *payload* contains valid ``chat_name`` and ``summary`` strings.

    Args:
        payload: Parsed section dictionary.

    Returns:
        ``True`` if both ``chat_name`` and ``summary`` are present and
        are strings.
    """
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("chat_name"), str)
        and isinstance(payload.get("summary"), str)
    )


def _validate_items_payload(
    payload: dict[str, Any], key: str, expected_count: int
) -> bool:
    """Check that *payload[key]* is a list with at least *expected_count* items.

    Args:
        payload: Parsed section dictionary.
        key: Key whose value should be a list.
        expected_count: Minimum required list length.

    Returns:
        ``True`` if the list exists and has enough items.
    """
    if not isinstance(payload, dict):
        return False
    items = payload.get(key)
    return isinstance(items, list) and len(items) >= expected_count


def _validate_list_payload(
    payload: dict[str, Any], key: str, min_count: int, max_count: int
) -> bool:
    """Check that *payload[key]* is a list within the given count range.

    Args:
        payload: Parsed section dictionary.
        key: Key whose value should be a list.
        min_count: Minimum required list length.
        max_count: Maximum allowed list length.

    Returns:
        ``True`` if the list length is in ``[min_count, max_count]``.
    """
    if not isinstance(payload, dict):
        return False
    items = payload.get(key)
    if not isinstance(items, list):
        return False
    return min_count <= len(items) <= max_count


def _validate_highlights_payload(
    payload: dict[str, Any], min_count: int, max_count: int
) -> bool:
    """Validate that the ``video_highlights`` list is within range.

    Args:
        payload: Parsed section dictionary.
        min_count: Minimum required number of highlights.
        max_count: Maximum allowed number of highlights.

    Returns:
        ``True`` if the highlights list length is within range.
    """
    return _validate_list_payload(payload, "video_highlights", min_count, max_count)


def _validate_timeline_payload(
    payload: dict[str, Any], min_count: int, max_count: int
) -> bool:
    """Validate that the ``video_timeline`` list is within range.

    Args:
        payload: Parsed section dictionary.
        min_count: Minimum required number of timeline events.
        max_count: Maximum allowed number of timeline events.

    Returns:
        ``True`` if the timeline list length is within range.
    """
    return _validate_list_payload(payload, "video_timeline", min_count, max_count)


def _validate_questions_payload(
    payload: dict[str, Any], expected_count: int
) -> bool:
    """Validate that enough questions are present in *payload*.

    Args:
        payload: Parsed section dictionary.
        expected_count: Minimum number of questions required.

    Returns:
        ``True`` if at least *expected_count* questions are found.
    """
    questions = _extract_questions(payload)
    return len(questions) >= expected_count


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _clean_question_text(text: str) -> str:
    """Strip leading prefixes and numbering from a question string.

    Removes patterns such as ``Q:``, ``Question -``, numbered bullets,
    and dash/bullet prefixes.

    Args:
        text: Raw question text.

    Returns:
        The cleaned question string, or an empty string if *text* is
        blank.
    """
    if not isinstance(text, str):
        return ""
    s = text.strip()
    if not s:
        return ""
    s = re.sub(r"^\s*(q(uestion)?\s*[:\-])\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*[\(\[]?\d+[\)\].:\-]?\s*", "", s)
    s = re.sub(r"^\s*[-•]+\s*", "", s)
    return s.strip()


def _normalize_question_key(text: str) -> str:
    """Produce a normalised lowercase key for deduplication.

    Args:
        text: Raw question text.

    Returns:
        A lowercased, whitespace-collapsed string suitable for use as a
        deduplication key.
    """
    s = _clean_question_text(text).lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_questions(
    payload: dict[str, Any] | list[Any],
) -> list[dict[str, str]]:
    """Extract and clean question/answer pairs from *payload*.

    Accepts either a dict with a ``questions`` key or a bare list of
    Q&A dicts.

    Args:
        payload: A dict containing a ``questions`` list, or a list of
            Q&A dicts directly.

    Returns:
        A list of dicts with ``question`` and ``answer`` string keys.
    """
    if isinstance(payload, dict):
        questions = payload.get("questions", [])
    elif isinstance(payload, list):
        questions = payload
    else:
        return []
    if not isinstance(questions, list):
        return []
    cleaned: list[dict[str, str]] = []
    for qa in questions:
        if not isinstance(qa, dict):
            continue
        question = qa.get("question")
        answer = qa.get("answer")
        if not isinstance(question, str) or not question.strip():
            continue
        question_text = question.strip()
        if isinstance(answer, str) and answer.strip():
            answer_text = answer.strip()
        else:
            answer_text = NOT_STATED_PERIOD
        cleaned.append({"question": question_text, "answer": answer_text})
    return cleaned


def _count_questions(synopsis: dict[str, Any] | list[Any]) -> int:
    """Count the number of valid questions in *synopsis*.

    A question is considered valid if it has a non-empty ``question``
    string.

    Args:
        synopsis: A dict with a ``questions`` key, or a bare list of Q&A
            dicts.

    Returns:
        The number of valid questions found.
    """
    if isinstance(synopsis, dict):
        questions = synopsis.get("questions", [])
    elif isinstance(synopsis, list):
        questions = synopsis
    else:
        return 0
    if not isinstance(questions, list):
        return 0
    count = 0
    for qa in questions:
        if not isinstance(qa, dict):
            continue
        q = qa.get("question")
        if isinstance(q, str) and q.strip():
            count += 1
    return count


def _normalize_summary_fields(payload: dict[str, Any]) -> tuple[str, str]:
    """Extract and normalise ``chat_name`` and ``summary`` from *payload*.

    Missing or empty values are replaced with :data:`NOT_STATED` /
    :data:`NOT_STATED_PERIOD`.

    Args:
        payload: Parsed section dictionary.

    Returns:
        A 2-tuple ``(chat_name, summary)`` of normalised strings.
    """
    chat_name = payload.get("chat_name") if isinstance(payload, dict) else None
    summary = payload.get("summary") if isinstance(payload, dict) else None
    if not isinstance(chat_name, str) or not chat_name.strip():
        chat_name = NOT_STATED
    if not isinstance(summary, str) or not summary.strip():
        summary = NOT_STATED_PERIOD
    return chat_name.strip(), summary.strip()


def _normalize_highlights(
    payload: dict[str, Any], min_count: int, max_count: int
) -> list[dict[str, str]]:
    """Normalise the ``video_highlights`` list from *payload*.

    Handles varying input shapes (dicts with ``start``/``end``, a single
    ``timestamp`` key, or bare strings).  The list is clamped to
    ``[min_count, max_count]`` and padded with placeholder entries if
    needed.

    Args:
        payload: Parsed section dictionary containing a
            ``video_highlights`` key.
        min_count: Minimum required number of highlights (padding
            applied if short).
        max_count: Maximum number of highlights kept.

    Returns:
        A list of highlight dicts with ``start``, ``end``, and
        ``highlight`` keys.
    """
    raw = payload.get("video_highlights", []) if isinstance(payload, dict) else []
    items: list[dict[str, str]] = []
    if isinstance(raw, list):
        for entry in raw:
            start = end = text = None
            if isinstance(entry, dict):
                if "start" in entry or "end" in entry:
                    start = entry.get("start")
                    end = entry.get("end")
                elif "timestamp" in entry:
                    start = entry.get("timestamp")
                if "highlight" in entry:
                    text = entry.get("highlight")
                elif len(entry) == 1:
                    key, value = next(iter(entry.items()))
                    if start is None:
                        start = key
                    text = value
            elif isinstance(entry, str):
                text = entry
            if not isinstance(start, str) or not start.strip():
                start = NOT_STATED
            if not isinstance(end, str) or not end.strip():
                end = NOT_STATED
            if not isinstance(text, str) or not text.strip():
                text = NOT_STATED_PERIOD
            items.append(
                {"start": start.strip(), "end": end.strip(), "highlight": text.strip()}
            )
    if len(items) > max_count:
        items = items[:max_count]
    while len(items) < min_count:
        items.append(
            {
                "start": NOT_STATED,
                "end": NOT_STATED,
                "highlight": NOT_STATED_PERIOD,
            }
        )
    return items


def _normalize_section_items(
    payload: dict[str, Any],
    key: str,
    text_key: str,
    min_count: int,
    max_count: int | None = None,
) -> list[dict[str, str]]:
    """Normalise a list of timestamped items from *payload*.

    Args:
        payload: Parsed section dictionary.
        key: Top-level key containing the items list (e.g.
            ``"video_timeline"``).
        text_key: Key name for the text value in each item dict.
        min_count: Minimum required item count (padding applied if
            short).
        max_count: Maximum item count kept.  Defaults to *min_count* if
            ``None``.

    Returns:
        A normalised list of dicts, each with ``timestamp`` and
        *text_key* string keys.
    """
    if max_count is None:
        max_count = min_count
    raw = payload.get(key, []) if isinstance(payload, dict) else []
    items: list[dict[str, str]] = []
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                ts = entry.get("timestamp")
                text = entry.get(text_key)
            elif isinstance(entry, str):
                ts = NOT_STATED
                text = entry
            else:
                continue
            if not isinstance(ts, str) or not ts.strip():
                ts = NOT_STATED
            if not isinstance(text, str) or not text.strip():
                text = NOT_STATED_PERIOD
            items.append({"timestamp": ts.strip(), text_key: text.strip()})
    if len(items) > max_count:
        items = items[:max_count]
    while len(items) < min_count:
        items.append({"timestamp": NOT_STATED, text_key: NOT_STATED_PERIOD})
    return items


def _normalize_timeline(
    payload: dict[str, Any], min_count: int, max_count: int
) -> list[dict[str, str]]:
    """Normalise the ``video_timeline`` list from *payload*.

    Args:
        payload: Parsed section dictionary containing a
            ``video_timeline`` key.
        min_count: Minimum required number of timeline events.
        max_count: Maximum number of timeline events kept.

    Returns:
        A normalised list of timeline dicts with ``timestamp`` and
        ``event`` keys.
    """
    return _normalize_section_items(
        payload, "video_timeline", "event", min_count, max_count
    )


def _normalize_predefined_questions(
    questions: list[dict[str, str]], required_questions: list[str]
) -> list[dict[str, str]]:
    """Map parsed Q&A pairs onto the required question list.

    Each required question is matched by its normalised key.  If no
    matching answer is found, the answer defaults to
    :data:`NOT_STATED_PERIOD`.

    Args:
        questions: Parsed Q&A dicts (``question`` / ``answer``).
        required_questions: Ordered list of required question strings.

    Returns:
        A list of Q&A dicts in the same order as *required_questions*.
    """
    answer_by_question: dict[str, str] = {}
    for qa in questions:
        question = qa.get("question") if isinstance(qa, dict) else None
        answer = qa.get("answer") if isinstance(qa, dict) else None
        if not isinstance(question, str) or not question.strip():
            continue
        question_text = _clean_question_text(question)
        if not question_text:
            continue
        key = _normalize_question_key(question_text)
        if key not in answer_by_question:
            answer_by_question[key] = (
                answer
                if isinstance(answer, str) and answer.strip()
                else NOT_STATED_PERIOD
            )

    normalized: list[dict[str, str]] = []
    for required in required_questions:
        key = _normalize_question_key(required)
        answer = answer_by_question.get(key, NOT_STATED_PERIOD)
        normalized.append({"question": required, "answer": answer})
    return normalized


def _normalize_generated_questions(
    questions: list[dict[str, str]],
    required_questions: list[str],
    extra_count: int,
    pad: bool = True,
    exclude_questions: list[str] | None = None,
) -> list[dict[str, str]]:
    """Extract novel generated questions, excluding required ones.

    Deduplicates by normalised question key and optionally pads with
    placeholder entries.

    Args:
        questions: Parsed Q&A dicts from the model.
        required_questions: List of required question strings to
            exclude.
        extra_count: Desired number of generated questions.
        pad: If ``True``, pad the list with placeholder questions to
            reach *extra_count*.
        exclude_questions: Optional additional questions to exclude by
            normalised key.

    Returns:
        A deduplicated, optionally padded list of Q&A dicts.
    """
    required_set = {_normalize_question_key(q) for q in required_questions}
    if exclude_questions:
        required_set.update({_normalize_question_key(q) for q in exclude_questions})
    seen: set[str] = set()
    normalized: list[dict[str, str]] = []
    for qa in questions:
        question = qa.get("question") if isinstance(qa, dict) else None
        answer = qa.get("answer") if isinstance(qa, dict) else None
        if not isinstance(question, str) or not question.strip():
            continue
        question_text = _clean_question_text(question)
        if not question_text:
            continue
        key = _normalize_question_key(question_text)
        if key in required_set or key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "question": question_text,
                "answer": answer.strip()
                if isinstance(answer, str) and answer.strip()
                else NOT_STATED_PERIOD,
            }
        )
        if len(normalized) >= extra_count:
            break
    if pad:
        while len(normalized) < extra_count:
            idx = len(normalized) + 1
            normalized.append(
                {
                    "question": f"Additional predicted question {idx}?",
                    "answer": NOT_STATED_PERIOD,
                }
            )
    return normalized
