"""Synopsis JSON/pipe parsing, validation, and normalization."""

import json
import re

from kairos.core.utils import print_prefixed

NOT_STATED = "Not explicitly stated"
NOT_STATED_PERIOD = "Not explicitly stated."


def _parse_json_object(text: str, debug: bool = False, context: str = "section") -> dict:
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
                obj = json.loads(text[start:end + 1])
                return obj if isinstance(obj, dict) else {}
            except json.JSONDecodeError as exc:
                if debug:
                    print_prefixed("(Synopsis)", f"{context}: JSON parse failed: {exc}")
                return {}
        if debug:
            print_prefixed("(Synopsis)", f"{context}: JSON parse failed")
        return {}


def _synopsis_fallback(text: str, parse_error: str) -> dict:
    return {
        "chat_name": NOT_STATED,
        "summary": text.strip() if isinstance(text, str) else "",
        "video_highlights": [],
        "video_timeline": [],
        "questions": [],
        "parse_error": parse_error,
    }


def _parse_synopsis_json(text: str, debug: bool = False) -> dict:
    if not isinstance(text, str):
        return _synopsis_fallback("", "Synopsis output was not a string")
    obj = _parse_json_object(text, debug=debug, context="synopsis")
    if not obj:
        return _synopsis_fallback(text, "Invalid JSON from model")
    return obj


def _parse_pipe_pairs(text: str) -> list[tuple[str, str]]:
    if not isinstance(text, str):
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    pairs = []
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
    pairs = _parse_pipe_pairs(text)
    if pairs:
        return pairs
    if not isinstance(text, str):
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    current_question = None
    for line in lines:
        inline = re.match(r"^\s*q(uestion)?\s*[:\-]\s*(.+?)\s+a(nswer)?\s*[:\-]\s*(.+)$", line, re.IGNORECASE)
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
    if not isinstance(text, str) or not text.strip():
        return NOT_STATED, NOT_STATED
    raw = text.strip()
    for sep in (" - ", " – ", " — ", " to ", "-"):
        if sep in raw:
            left, right = raw.split(sep, 1)
            left = left.strip() if left.strip() else NOT_STATED
            right = right.strip() if right.strip() else NOT_STATED
            return left, right
    return raw, NOT_STATED


def _parse_summary_nonjson(text: str) -> tuple[dict, bool]:
    pairs = _parse_pipe_pairs(text)
    if not pairs:
        return {}, False
    chat_name, summary = pairs[0]
    if not chat_name or not summary:
        return {}, False
    return {"chat_name": chat_name, "summary": summary}, True


def _parse_items_nonjson(text: str, key: str, text_key: str, expected_count: int) -> tuple[dict, bool]:
    pairs = _parse_pipe_pairs(text)
    ok = len(pairs) >= expected_count
    items = []
    for left, right in pairs[:expected_count]:
        ts = left.strip() if left else NOT_STATED
        value = right.strip() if right else NOT_STATED_PERIOD
        if not ts:
            ts = NOT_STATED
        if not value:
            value = NOT_STATED_PERIOD
        items.append({"timestamp": ts, text_key: value})
    return {key: items}, ok


def _parse_highlights_nonjson(text: str, min_count: int, max_count: int) -> tuple[dict, bool]:
    if not isinstance(text, str):
        return {}, False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    items = []
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


def _parse_timeline_nonjson(text: str, min_count: int, max_count: int) -> tuple[dict, bool]:
    result, _ = _parse_items_nonjson(text, "video_timeline", "event", max_count)
    items = result.get("video_timeline", [])
    if len(items) > max_count:
        items = items[:max_count]
    ok = min_count <= len(items) <= max_count
    return {"video_timeline": items}, ok


def _parse_questions_nonjson(text: str, expected_count: int) -> tuple[dict, bool]:
    pairs = _parse_qna_pairs(text)
    ok = len(pairs) >= expected_count
    questions = []
    for left, right in pairs[:expected_count]:
        question = _clean_question_text(left) if left else ""
        if not question:
            question = NOT_STATED
        answer = right.strip() if right else NOT_STATED_PERIOD
        if not answer:
            answer = NOT_STATED_PERIOD
        questions.append({"question": question, "answer": answer})
    return {"questions": questions}, ok



# Validation

def _validate_summary_payload(payload: dict) -> bool:
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("chat_name"), str)
        and isinstance(payload.get("summary"), str)
    )


def _validate_items_payload(payload: dict, key: str, expected_count: int) -> bool:
    if not isinstance(payload, dict):
        return False
    items = payload.get(key)
    return isinstance(items, list) and len(items) >= expected_count


def _validate_list_payload(payload: dict, key: str, min_count: int, max_count: int) -> bool:
    if not isinstance(payload, dict):
        return False
    items = payload.get(key)
    if not isinstance(items, list):
        return False
    return min_count <= len(items) <= max_count


def _validate_highlights_payload(payload: dict, min_count: int, max_count: int) -> bool:
    return _validate_list_payload(payload, "video_highlights", min_count, max_count)


def _validate_timeline_payload(payload: dict, min_count: int, max_count: int) -> bool:
    return _validate_list_payload(payload, "video_timeline", min_count, max_count)


def _validate_questions_payload(payload: dict, expected_count: int) -> bool:
    questions = _extract_questions(payload)
    return len(questions) >= expected_count


# Normalization

def _clean_question_text(text: str) -> str:
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
    s = _clean_question_text(text).lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_questions(payload) -> list:
    if isinstance(payload, dict):
        questions = payload.get("questions", [])
    elif isinstance(payload, list):
        questions = payload
    else:
        return []
    if not isinstance(questions, list):
        return []
    cleaned = []
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


def _count_questions(synopsis) -> int:
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


def _normalize_summary_fields(payload: dict) -> tuple[str, str]:
    chat_name = payload.get("chat_name") if isinstance(payload, dict) else None
    summary = payload.get("summary") if isinstance(payload, dict) else None
    if not isinstance(chat_name, str) or not chat_name.strip():
        chat_name = NOT_STATED
    if not isinstance(summary, str) or not summary.strip():
        summary = NOT_STATED_PERIOD
    return chat_name.strip(), summary.strip()


def _normalize_highlights(payload: dict, min_count: int, max_count: int) -> list[dict]:
    raw = payload.get("video_highlights", []) if isinstance(payload, dict) else []
    items = []
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
            items.append({"start": start.strip(), "end": end.strip(), "highlight": text.strip()})
    if len(items) > max_count:
        items = items[:max_count]
    while len(items) < min_count:
        items.append({
            "start": NOT_STATED,
            "end": NOT_STATED,
            "highlight": NOT_STATED_PERIOD,
        })
    return items


def _normalize_section_items(payload: dict, key: str, text_key: str,
                             min_count: int, max_count: int | None = None) -> list[dict]:
    if max_count is None:
        max_count = min_count
    raw = payload.get(key, []) if isinstance(payload, dict) else []
    items = []
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


def _normalize_timeline(payload: dict, min_count: int, max_count: int) -> list[dict]:
    return _normalize_section_items(payload, "video_timeline", "event", min_count, max_count)


def _normalize_predefined_questions(questions: list[dict], required_questions: list[str]) -> list[dict]:
    answer_by_question = {}
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
            answer_by_question[key] = answer if isinstance(answer, str) and answer.strip() else NOT_STATED_PERIOD

    normalized = []
    for required in required_questions:
        key = _normalize_question_key(required)
        answer = answer_by_question.get(key, NOT_STATED_PERIOD)
        normalized.append({"question": required, "answer": answer})
    return normalized


def _normalize_generated_questions(
    questions: list[dict],
    required_questions: list[str],
    extra_count: int,
    pad: bool = True,
    exclude_questions: list[str] | None = None,
) -> list[dict]:
    required_set = {_normalize_question_key(q) for q in required_questions}
    if exclude_questions:
        required_set.update({_normalize_question_key(q) for q in exclude_questions})
    seen = set()
    normalized = []
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
        normalized.append({
            "question": question_text,
            "answer": answer.strip() if isinstance(answer, str) and answer.strip() else NOT_STATED_PERIOD,
        })
        if len(normalized) >= extra_count:
            break
    if pad:
        while len(normalized) < extra_count:
            idx = len(normalized) + 1
            normalized.append({
                "question": f"Additional predicted question {idx}?",
                "answer": NOT_STATED_PERIOD,
            })
    return normalized
