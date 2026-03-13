import os
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote
from openai import AzureOpenAI

CHUNK_SIZE = 7000
FINAL_CHUNK_SIZE = CHUNK_SIZE * 5
SUMMARY_MAX_WORKERS = 6
SUMMARY_REDUCE_GROUP_SIZE = 4
GPT_MAX_RETRIES = 6
GPT_RETRY_BASE_SEC = 2.0
HIGHLIGHTS_COUNT = "4-6"
TIMELINE_COUNT = "4-6"
EXTRA_QUESTIONS_COUNT = 15
REQUIRED_QUESTIONS = [
    "What is happening in the video?",
    "What are the key events?",
    "What are the key actions and who performed them?",
    "What are the main conflicts and problems encountered?",
    "Who is the main character? Describe their journey.",
    "List the characters. For each character, describe their appearance, traits, and role in the story.",
    "What are some significant quotes from the video and who said them?",
    "What is the setting? Did it change? How is it related to the story?",
    "How did the video start? Explain the start.",
    "How did the video end? Explain the ending.",
    "What objects are central to the video and when do they appear?",
    "What is the most important thing said or heard?",
    "What is different at the end vs the beginning?",
    "What type of video is this?",
    "What is the goal or intent or theme of the video?",
    "List the moods and tones present, explain each one.",
    "What context is missing or assumed? What would require outside knowledge?",
    "What are key visual descriptions?",
    "What are key audio descriptions?",
    "Are the visual and audio cues noticed throughout the video aligned? If not, how do they differ?",
    "What are prominent visual cues and audio cues noticed throughout the video?",
    "Does the video contain any live action, animation, or special effects?",
]

# ----------------------
# 1. Debug helper
# ----------------------
def _debug_print(enabled: bool, message: str):
    if enabled:
        print(f"(GPT4o) {message}")

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


def _highlight_count_rule(min_count: int, max_count: int, label: str) -> str:
    if min_count == max_count:
        return f"- \"video_highlights\" must contain exactly {min_count} items.\n"
    return f"- \"video_highlights\" must contain between {min_count} and {max_count} items ({label}).\n"


def _timeline_count_rule(min_count: int, max_count: int, label: str) -> str:
    if min_count == max_count:
        return f"- \"video_timeline\" must contain exactly {min_count} items.\n"
    return f"- \"video_timeline\" must contain between {min_count} and {max_count} items ({label}).\n"

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


def _parse_synopsis_json(text: str, debug: bool = False) -> dict:
    if not isinstance(text, str):
        return {
            "chat_name": "Not explicitly stated",
            "summary": "",
            "video_highlights": [],
            "video_timeline": [],
            "questions": [],
            "parse_error": "Synopsis output was not a string",
        }
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                obj = None
        else:
            obj = None
        if obj is None:
            _debug_print(debug, f"synopsis JSON parse failed: {exc}")
            return {
                "chat_name": "Not explicitly stated",
                "summary": text.strip(),
                "video_highlights": [],
            "video_timeline": [],
            "questions": [],
            "parse_error": "Invalid JSON from model",
        }
    if not isinstance(obj, dict):
        return {
            "chat_name": "Not explicitly stated",
            "summary": text.strip(),
            "video_highlights": [],
            "video_timeline": [],
            "questions": [],
            "parse_error": "JSON root was not an object",
        }
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
        ts = left.strip() if left else "Not explicitly stated"
        value = right.strip() if right else "Not explicitly stated."
        if not ts:
            ts = "Not explicitly stated"
        if not value:
            value = "Not explicitly stated."
        items.append({"timestamp": ts, text_key: value})
    return {key: items}, ok


def _split_time_range(text: str) -> tuple[str, str]:
    if not isinstance(text, str) or not text.strip():
        return "Not explicitly stated", "Not explicitly stated"
    raw = text.strip()
    for sep in (" - ", " – ", " — ", " to ", "-"):
        if sep in raw:
            left, right = raw.split(sep, 1)
            left = left.strip() if left.strip() else "Not explicitly stated"
            right = right.strip() if right.strip() else "Not explicitly stated"
            return left, right
    return raw, "Not explicitly stated"


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
            start = "Not explicitly stated"
        if not end:
            end = "Not explicitly stated"
        if not highlight:
            highlight = "Not explicitly stated."
        items.append({"start": start, "end": end, "highlight": highlight})
    if len(items) > max_count:
        items = items[:max_count]
    ok = min_count <= len(items) <= max_count
    return {"video_highlights": items}, ok


def _parse_timeline_nonjson(text: str, min_count: int, max_count: int) -> tuple[dict, bool]:
    pairs = _parse_pipe_pairs(text)
    items = []
    for left, right in pairs:
        ts = left.strip() if left else "Not explicitly stated"
        value = right.strip() if right else "Not explicitly stated."
        if not ts:
            ts = "Not explicitly stated"
        if not value:
            value = "Not explicitly stated."
        items.append({"timestamp": ts, "event": value})
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
            question = "Not explicitly stated"
        answer = right.strip() if right else "Not explicitly stated."
        if not answer:
            answer = "Not explicitly stated."
        questions.append({"question": question, "answer": answer})
    return {"questions": questions}, ok


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


def _validate_highlights_payload(payload: dict, min_count: int, max_count: int) -> bool:
    if not isinstance(payload, dict):
        return False
    items = payload.get("video_highlights")
    if not isinstance(items, list):
        return False
    return min_count <= len(items) <= max_count


def _validate_timeline_payload(payload: dict, min_count: int, max_count: int) -> bool:
    if not isinstance(payload, dict):
        return False
    items = payload.get("video_timeline")
    if not isinstance(items, list):
        return False
    return min_count <= len(items) <= max_count


def _validate_questions_payload(payload: dict, expected_count: int) -> bool:
    questions = _extract_questions(payload)
    return len(questions) >= expected_count


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
            "Use this exact schema and key names:\n"
            '{ "chat_name": "3-5 word title", "summary": "Single coherent paragraph." }\n'
            "Rules:\n"
            "- If chat_name or summary is missing, set it to \"Not explicitly stated.\".\n"
        )
    elif section == "highlights":
        schema = (
            "Use this exact schema and key names:\n"
            '{ "video_highlights": [ { "start": "00:00:00", "end": "00:00:00", "highlight": "One sentence highlight." } ] }\n'
            "Rules:\n"
            f"{_highlight_count_rule(highlight_min, highlight_max, highlight_label)}"
            "- If a start or end timestamp is not explicitly stated, use \"Not explicitly stated\".\n"
        )
    elif section == "timeline":
        schema = (
            "Use this exact schema and key names:\n"
            '{ "video_timeline": [ { "timestamp": "00:00:00", "event": "3-5 word event" } ] }\n'
            "Rules:\n"
            f"{_timeline_count_rule(timeline_min, timeline_max, timeline_label)}"
        )
    elif section == "qna_predefined":
        schema = (
            "Use this exact schema and key names:\n"
            '{ "questions": [ { "question": "string", "answer": "string" } ] }\n'
            "Rules:\n"
            f"- Include exactly {required_questions_count} items.\n"
            "- Questions must be the required questions below in exact order.\n"
            "Required Questions:\n"
            f"{required_questions_block}\n"
        )
    elif section == "qna_generated":
        schema = (
            "Use this exact schema and key names:\n"
            '{ "questions": [ { "question": "string", "answer": "string" } ] }\n'
            "Rules:\n"
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


def _timecode_to_seconds(timecode: str | None):
    if not timecode or not isinstance(timecode, str):
        return None
    tc = timecode.strip()
    if not tc or tc.lower() == "not explicitly stated":
        return None
    parts = tc.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            hours, minutes, seconds = "0", parts[0], parts[1]
        elif len(parts) == 1:
            hours, minutes, seconds = "0", "0", parts[0]
        else:
            return None
        h = int(float(hours))
        m = int(float(minutes))
        s = float(seconds)
        total = (h * 3600) + (m * 60) + s
        return int(round(total))
    except ValueError:
        return None


def _encode_url_path(path: str) -> str:
    return "/".join(quote(part) for part in path.split("/"))


def _build_video_link_base(video_path: str | None, output_dir: str | None) -> str | None:
    if not video_path or not isinstance(video_path, str):
        return None
    base = video_path
    if output_dir:
        try:
            base = os.path.relpath(video_path, start=output_dir)
        except ValueError:
            base = video_path
    base = base.replace("\\", "/")
    return _encode_url_path(base)


def _format_timestamp_markdown(timestamp: str | None, video_link_base: str | None) -> str:
    label = timestamp.strip() if isinstance(timestamp, str) and timestamp.strip() else "Not explicitly stated"
    seconds = _timecode_to_seconds(timestamp)
    if seconds is None:
        return label
    link = f"{video_link_base}#t={seconds}" if video_link_base else f"#t={seconds}"
    return f"[{label}]({link})"


def _format_time_range_markdown(start: str | None, end: str | None, video_link_base: str | None) -> str:
    start_md = _format_timestamp_markdown(start, video_link_base)
    end_md = _format_timestamp_markdown(end, video_link_base)
    if end_md == "Not explicitly stated":
        return start_md
    if start_md == "Not explicitly stated":
        return end_md
    return f"{start_md} - {end_md}"


def _extract_timed_entry(item, text_key: str):
    if not isinstance(item, dict):
        return None, None
    if "timestamp" in item:
        return item.get("timestamp"), item.get(text_key)
    if len(item) == 1:
        timestamp, value = next(iter(item.items()))
        return timestamp, value
    return item.get("timestamp"), item.get(text_key)


def _extract_highlight_entry(item):
    if not isinstance(item, dict):
        return None, None, None
    if "start" in item or "end" in item:
        return item.get("start"), item.get("end"), item.get("highlight")
    if "timestamp" in item:
        return item.get("timestamp"), item.get("end"), item.get("highlight")
    if len(item) == 1:
        key, value = next(iter(item.items()))
        return key, None, value
    return item.get("start"), item.get("end"), item.get("highlight")


def render_synopsis_markdown(synopsis: dict, video_path: str | None = None, output_dir: str | None = None) -> str:
    title = "Video Synopsis"
    if isinstance(synopsis, dict):
        chat_name = synopsis.get("chat_name")
        if isinstance(chat_name, str) and chat_name.strip():
            title = chat_name.strip()

    video_link_base = _build_video_link_base(video_path, output_dir)
    lines = [f"# {title}"]

    summary = synopsis.get("summary") if isinstance(synopsis, dict) else ""
    if isinstance(summary, str) and summary.strip():
        lines.extend(["", "## Summary", summary.strip()])

    highlights = synopsis.get("video_highlights") if isinstance(synopsis, dict) else []
    if isinstance(highlights, list) and highlights:
        lines.extend(["", "## Highlights"])
        for item in highlights:
            if isinstance(item, str) and item.strip():
                lines.append(f"- {item.strip()}")
                continue
            start, end, highlight = _extract_highlight_entry(item)
            ts_md = _format_time_range_markdown(start, end, video_link_base)
            if isinstance(highlight, str) and highlight.strip():
                lines.append(f"- {ts_md}: {highlight.strip()}")
            else:
                lines.append(f"- {ts_md}")

    timeline = synopsis.get("video_timeline") if isinstance(synopsis, dict) else []
    if isinstance(timeline, list) and timeline:
        lines.extend(["", "## Timeline"])
        for entry in timeline:
            ts, event = _extract_timed_entry(entry, "event")
            ts_md = _format_timestamp_markdown(ts, video_link_base)
            if isinstance(event, str) and event.strip():
                lines.append(f"- {ts_md} — {event.strip()}")
            else:
                lines.append(f"- {ts_md}")

    questions = synopsis.get("questions") if isinstance(synopsis, dict) else []
    if isinstance(questions, list) and questions:
        lines.extend(["", "## Questions"])
        for qa in questions:
            if not isinstance(qa, dict):
                continue
            question = qa.get("question")
            answer = qa.get("answer")
            if isinstance(question, str) and question.strip():
                lines.append(f"**Q:** {question.strip()}")
                if isinstance(answer, str) and answer.strip():
                    lines.append(f"**A:** {answer.strip()}")
                else:
                    lines.append("**A:** Not explicitly stated.")
                lines.append("")

    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines).strip() + "\n"


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
            answer_text = "Not explicitly stated."
        cleaned.append({"question": question_text, "answer": answer_text})
    return cleaned


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
                _debug_print(debug, f"{context}: JSON parse failed: {exc}")
                return {}
        _debug_print(debug, f"{context}: JSON parse failed")
        return {}


def _required_questions_block(required_questions: list[str]) -> str:
    return "\n".join([f"{idx + 1}. {q}" for idx, q in enumerate(required_questions)])


def _normalize_section_items(payload: dict, key: str, text_key: str, count: int) -> list[dict]:
    raw = payload.get(key, []) if isinstance(payload, dict) else []
    items = []
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                ts = entry.get("timestamp")
                text = entry.get(text_key)
            elif isinstance(entry, str):
                ts = "Not explicitly stated"
                text = entry
            else:
                continue
            if not isinstance(ts, str) or not ts.strip():
                ts = "Not explicitly stated"
            if not isinstance(text, str) or not text.strip():
                text = "Not explicitly stated."
            items.append({"timestamp": ts.strip(), text_key: text.strip()})
    items = items[:count]
    while len(items) < count:
        items.append({"timestamp": "Not explicitly stated", text_key: "Not explicitly stated."})
    return items


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
                start = "Not explicitly stated"
            if not isinstance(end, str) or not end.strip():
                end = "Not explicitly stated"
            if not isinstance(text, str) or not text.strip():
                text = "Not explicitly stated."
            items.append({"start": start.strip(), "end": end.strip(), "highlight": text.strip()})
    if len(items) > max_count:
        items = items[:max_count]
    while len(items) < min_count:
        items.append({
            "start": "Not explicitly stated",
            "end": "Not explicitly stated",
            "highlight": "Not explicitly stated.",
        })
    return items


def _normalize_timeline(payload: dict, min_count: int, max_count: int) -> list[dict]:
    raw = payload.get("video_timeline", []) if isinstance(payload, dict) else []
    items = []
    if isinstance(raw, list):
        for entry in raw:
            ts = event = None
            if isinstance(entry, dict):
                ts = entry.get("timestamp")
                event = entry.get("event")
            elif isinstance(entry, str):
                event = entry
            if not isinstance(ts, str) or not ts.strip():
                ts = "Not explicitly stated"
            if not isinstance(event, str) or not event.strip():
                event = "Not explicitly stated."
            items.append({"timestamp": ts.strip(), "event": event.strip()})
    if len(items) > max_count:
        items = items[:max_count]
    while len(items) < min_count:
        items.append({"timestamp": "Not explicitly stated", "event": "Not explicitly stated."})
    return items


def _normalize_summary_fields(payload: dict) -> tuple[str, str]:
    chat_name = payload.get("chat_name") if isinstance(payload, dict) else None
    summary = payload.get("summary") if isinstance(payload, dict) else None
    if not isinstance(chat_name, str) or not chat_name.strip():
        chat_name = "Not explicitly stated"
    if not isinstance(summary, str) or not summary.strip():
        summary = "Not explicitly stated."
    return chat_name.strip(), summary.strip()


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
            answer_by_question[key] = answer if isinstance(answer, str) and answer.strip() else "Not explicitly stated."

    normalized = []
    for required in required_questions:
        key = _normalize_question_key(required)
        answer = answer_by_question.get(key, "Not explicitly stated.")
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
            "answer": answer.strip() if isinstance(answer, str) and answer.strip() else "Not explicitly stated.",
        })
        if len(normalized) >= extra_count:
            break
    if pad:
        while len(normalized) < extra_count:
            idx = len(normalized) + 1
            normalized.append({
                "question": f"Additional predicted question {idx}?",
                "answer": "Not explicitly stated.",
            })
    return normalized


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

def _normalize_scene_text(value, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _scene_to_narrative_line(scene: dict) -> str:
    start_timecode = _normalize_scene_text(scene.get("start_timecode"), "Not explicitly stated")
    llm_scene_description = _normalize_scene_text(scene.get("llm_scene_description"), "No visual description.")
    audio_speech = _normalize_scene_text(scene.get("audio_speech"), "No spoken dialogue.")
    return f'At {start_timecode}, {llm_scene_description} It says "{audio_speech}".'

# ----------------------
# 2. Chunk scenes
# ----------------------
def chunk_scenes(scenes: list, chunk_size: int = CHUNK_SIZE, debug: bool = False):
    """
    Break scene dictionaries into <= chunk_size chunks.
    """
    scene_count = len(scenes) if isinstance(scenes, list) else 0
    if scene_count == 0:
        _debug_print(debug, "chunk_scenes: no scenes to chunk")
        return []

    chunks = []
    this_chunk = ""
    chunk_start_idx = None

    for idx, scene in enumerate(scenes):
        scene_obj = scene if isinstance(scene, dict) else {}
        text = _scene_to_narrative_line(scene_obj)
        candidate = f"{this_chunk}\n{text}".strip() if this_chunk else text

        if chunk_start_idx is None:
            chunk_start_idx = idx
        this_chunk = candidate

        if len(this_chunk) >= chunk_size:
            start_scene = scenes[chunk_start_idx] if isinstance(scenes[chunk_start_idx], dict) else {}
            end_scene = scene_obj
            chunks.append({
                "index": len(chunks),
                "text": this_chunk,
                "scene_start_idx": chunk_start_idx,
                "scene_end_idx": idx,
                "start_timecode": start_scene.get("start_timecode"),
                "end_timecode": end_scene.get("end_timecode") or end_scene.get("start_timecode"),
            })
            this_chunk = ""
            chunk_start_idx = None

    if this_chunk:
        end_idx = len(scenes) - 1
        start_idx = chunk_start_idx if chunk_start_idx is not None else end_idx
        start_scene = scenes[start_idx] if isinstance(scenes[start_idx], dict) else {}
        end_scene = scenes[end_idx] if isinstance(scenes[end_idx], dict) else {}
        chunks.append({
            "index": len(chunks),
            "text": this_chunk,
            "scene_start_idx": start_idx,
            "scene_end_idx": end_idx,
            "start_timecode": start_scene.get("start_timecode"),
            "end_timecode": end_scene.get("end_timecode") or end_scene.get("start_timecode"),
        })

    _debug_print(
        debug,
        f"chunk_scenes: {scene_count} scenes -> {len(chunks)} chunk_scenes"
    )
    return chunks

# ----------------------
# 3. GPT call helper
# ----------------------
def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg


def _is_responsible_ai_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "content_filter" in msg
        or "content filter" in msg
        or "responsible ai" in msg
        or "safety system" in msg
        or "policy_violation" in msg
    )


def call_gpt(client, deployment, prompt, retries: int = GPT_MAX_RETRIES, retry_base_sec: float = GPT_RETRY_BASE_SEC):
    """
    Minimal GPT call using AzureOpenAI client
    """
    last_exc = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a precise and reliable assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=16384,  # apparently the max for gpt4o
                temperature=0.2,
                top_p=1.0,
                model=deployment
            )
            message = response.choices[0].message
            content = message.content
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        value = item.get("text")
                        if isinstance(value, str):
                            parts.append(value)
                text = "".join(parts).strip()
            else:
                text = ""
            if not text:
                raise RuntimeError("Model returned empty content")
            return text
        except Exception as exc:
            last_exc = exc
            if _is_responsible_ai_error(exc):
                break
            if attempt >= retries - 1:
                break
            sleep_sec = retry_base_sec * (2 ** attempt)
            if _is_rate_limit_error(exc):
                sleep_sec = max(sleep_sec, 5.0)
            time.sleep(sleep_sec)
    raise last_exc


def call_gpt_safe(
    client,
    deployment,
    prompt: str,
    fallback_text: str,
    debug: bool = False,
    context: str = "call",
    safe_prompt: str | None = None,
    raw_fallback: str | None = None,
):
    try:
        return call_gpt(client, deployment, prompt)
    except Exception as exc:
        _debug_print(debug, f"{context}: primary prompt failed due to API error: {exc}")
        if safe_prompt:
            try:
                return call_gpt(client, deployment, safe_prompt)
            except Exception as exc2:
                _debug_print(debug, f"{context}: safe prompt failed due to API error: {exc2}")
        if isinstance(raw_fallback, str):
            _debug_print(debug, f"{context}: using raw fallback due to API error")
            return raw_fallback
        _debug_print(debug, f"{context}: using fallback due to API error")
        return fallback_text

# ----------------------
# 4. Segment synthesis prompts (loaded from prompts folder)
# ----------------------
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

def load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")

SEGMENT_PROMPT = load_prompt("chunk_summary.txt")
FALLBACK_SEGMENT_PROMPT = load_prompt("fallback_chunk_summary.txt")
CARRYOVER_PROMPT = load_prompt("chunk_summary_carryover.txt")
SYNOPSIS_SUMMARY_PROMPT = load_prompt("synopsis_summary.txt")
SYNOPSIS_HIGHLIGHTS_PROMPT = load_prompt("synopsis_highlight.txt")
SYNOPSIS_TIMELINE_PROMPT = load_prompt("synopsis_timeline.txt")
SYNOPSIS_QNA_PREDEFINED_PROMPT = load_prompt("synopsis_qna_predefined.txt")
SYNOPSIS_QNA_GENERATED_PROMPT = load_prompt("synopsis_qna_generated.txt")


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
            "Use this exact schema and key names:\n"
            '{ "chat_name": "3-5 word title", "summary": "Single coherent paragraph." }\n'
            "Rules:\n"
            "- \"chat_name\" must be 3-5 words and concrete, not creative.\n"
            "- \"summary\" must be one paragraph.\n"
        )
    elif section == "highlights":
        schema = (
            "Use this exact schema and key names:\n"
            '{ "video_highlights": [ { "start": "00:00:00", "end": "00:00:00", "highlight": "One sentence highlight." } ] }\n'
            "Rules:\n"
            f"{_highlight_count_rule(highlight_min, highlight_max, highlight_label)}"
            "- Each highlight is one sentence.\n"
            "- If a start or end timestamp is not explicitly stated, use \"Not explicitly stated\".\n"
        )
    elif section == "timeline":
        schema = (
            "Use this exact schema and key names:\n"
            '{ "video_timeline": [ { "timestamp": "00:00:00", "event": "3-5 word event" } ] }\n'
            "Rules:\n"
            f"{_timeline_count_rule(timeline_min, timeline_max, timeline_label)}"
            "- Events must be 3-5 words, chronological order.\n"
            "- If a timestamp is not explicitly stated, use \"Not explicitly stated\".\n"
        )
    elif section == "qna_predefined":
        schema = (
            "Use this exact schema and key names:\n"
            '{ "questions": [ { "question": "string", "answer": "string" } ] }\n'
            "Rules:\n"
            "- Include only these required questions, in order, no extras:\n"
            f"{required_questions_block}\n"
        )
    elif section == "qna_generated":
        schema = (
            "Use this exact schema and key names:\n"
            '{ "questions": [ { "question": "string", "answer": "string" } ] }\n'
            "Rules:\n"
            f"- Add exactly {extra_questions_count} additional questions (not in required list).\n"
            "- Do not repeat required questions (they are answered elsewhere).\n"
            "- Use only the narrative.\n"
        )
    elif section == "qna_legacy":
        schema = (
            "Use this exact schema and key names:\n"
            '{ "questions": [ { "question": "string", "answer": "string" } ] }\n'
            "Rules:\n"
            "- Include the required questions below, in order, then add extra questions.\n"
            f"- Add exactly {extra_questions_count} extra questions.\n"
            "Required Questions:\n"
            f"{required_questions_block}\n"
        )
    elif section == "monolith":
        schema = (
            "Use this exact schema and key names:\n"
            "{\n"
            '  "chat_name": "3-5 word title",\n'
            '  "summary": "Single coherent paragraph.",\n'
            '  "video_highlights": [ { "start": "00:00:00", "end": "00:00:00", "highlight": "One sentence highlight." } ],\n'
            '  "video_timeline": [ { "timestamp": "00:00:00", "event": "3-5 word event" } ]\n'
            "}\n"
            "Rules:\n"
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

# ----------------------
# 5. Summarize segments with carryover context
# ----------------------
def condense_chunk(client, deployment, chunk_text: str, pre_carryover_context: str, debug: bool = False):
    """
    Summarize a chunk and return (summary, new_carryover_context).
    """
    segment_prompt = SEGMENT_PROMPT.format(
        carryover_context=pre_carryover_context,
        scene_chunk=chunk_text
    )
    summary = None
    try:
        summary = call_gpt(client, deployment, segment_prompt)
    except Exception as exc:
        _debug_print(debug, f"condense_chunk: primary prompt failed: {exc}")
        try:
            fallback_prompt = FALLBACK_SEGMENT_PROMPT.format(
                carryover_context=pre_carryover_context,
                scene_chunk=chunk_text
            )
            summary = call_gpt(client, deployment, fallback_prompt)
        except Exception as exc2:
            _debug_print(debug, f"condense_chunk: fallback prompt failed: {exc2}")
            # Last-resort: preserve input so downstream isn't broken
            summary = chunk_text

    carryover_prompt = CARRYOVER_PROMPT.format(
        segment_narrative=summary
    )
    try:
        new_carryover_context = call_gpt(client, deployment, carryover_prompt)
    except Exception as exc:
        _debug_print(debug, f"condense_chunk: carryover prompt failed: {exc}")
        new_carryover_context = pre_carryover_context
    _debug_print(debug, f"    condense_chunk: len={len(chunk_text)} -> len={len(summary)}")
    return summary, new_carryover_context


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


def _parallel_map_summaries(client, deployment, scene_chunks: list[dict], max_workers: int, debug: bool = False):
    if not scene_chunks:
        return []
    results = [None] * len(scene_chunks)

    def _task(chunk: dict):
        prompt = _build_scene_chunk_summary_prompt(chunk)
        try:
            summary = call_gpt(client, deployment, prompt)
        except Exception as exc:
            _debug_print(debug, f"_parallel_map_summaries: chunk {chunk['index']} failed after retries, fallback to raw chunk: {exc}")
            summary = chunk["text"]
        _debug_print(debug, f"    map_summary: len={len(chunk['text'])} -> len={len(summary.strip())}")
        return {
            "index": chunk["index"],
            "text": summary.strip(),
            "scene_start_idx": chunk["scene_start_idx"],
            "scene_end_idx": chunk["scene_end_idx"],
            "start_timecode": chunk.get("start_timecode"),
            "end_timecode": chunk.get("end_timecode"),
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_task, chunk) for chunk in scene_chunks]
        for future in as_completed(futures):
            item = future.result()
            results[item["index"]] = item

    mapped = [r for r in results if r is not None]
    return mapped


def _parallel_reduce_summaries(
    client,
    deployment,
    summaries: list[dict],
    reduce_group_size: int = SUMMARY_REDUCE_GROUP_SIZE,
    max_workers: int = SUMMARY_MAX_WORKERS,
    debug: bool = False,
):
    current = summaries
    round_idx = 0
    while len(current) > 1:
        round_idx += 1
        groups = [current[i:i + reduce_group_size] for i in range(0, len(current), reduce_group_size)]
        reduced = [None] * len(groups)

        def _task(group_idx: int, group_items: list[dict]):
            prompt = _build_reduce_prompt(group_items, round_idx)
            try:
                merged = call_gpt(client, deployment, prompt).strip()
            except Exception as exc:
                _debug_print(debug, f"_parallel_reduce_summaries: group {group_idx} failed after retries, fallback concatenate: {exc}")
                merged = "\n".join(item["text"] for item in group_items)
            return group_idx, {
                "index": group_idx,
                "text": merged,
                "scene_start_idx": group_items[0]["scene_start_idx"],
                "scene_end_idx": group_items[-1]["scene_end_idx"],
                "start_timecode": group_items[0].get("start_timecode"),
                "end_timecode": group_items[-1].get("end_timecode"),
            }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_task, idx, grp) for idx, grp in enumerate(groups)]
            for future in as_completed(futures):
                group_idx, result = future.result()
                reduced[group_idx] = result
        current = [item for item in reduced if item is not None]
        _debug_print(debug, f"_parallel_reduce_summaries: round={round_idx}, groups={len(groups)}, next={len(current)}")

    return current[0] if current else None

def chunk_narrative(narrative: str, chunk_size: int = CHUNK_SIZE, debug: bool = False):
    """
    Chunk narrative into <= chunk_size blocks, preferring paragraph breaks.
    """
    paragraphs = [p.strip() for p in narrative.split("\n\n") if p.strip()]
    chunks = []
    this_chunk = ""

    for para in paragraphs:
        candidate = f"{this_chunk}\n\n{para}".strip() if this_chunk else para
        if len(candidate) <= chunk_size:
            this_chunk = candidate
        else:
            if this_chunk:
                chunks.append(this_chunk)
            if len(para) > chunk_size:
                # Fallback: hard split a long paragraph
                for i in range(0, len(para), chunk_size):
                    chunks.append(para[i:i + chunk_size])
                this_chunk = ""
            else:
                this_chunk = para

    if this_chunk:
        chunks.append(this_chunk)
    _debug_print(debug, f"chunk_narrative: splitting narrative len={len(narrative)} to {len(chunks)} chunks")
    return chunks

def summarize_scenes(
    client,
    deployment,
    scenes,
    chunk_size: int = CHUNK_SIZE,
    summary_len: int = FINAL_CHUNK_SIZE,
    debug: bool = False,
    output_dir: str | None = None,
    max_workers: int | None = None,
    reduce_group_size: int = SUMMARY_REDUCE_GROUP_SIZE,
):
    """
    Summarize scenes into a narrative, then recursively compress
    until the narrative fits within summary_len.
    """
    scene_chunks = chunk_scenes(scenes, chunk_size, debug=debug)
    if not scene_chunks:
        return {
            "scenes": scenes,
            "narratives": [],
        }

    narratives = []
    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = min(SUMMARY_MAX_WORKERS, max(2, cpu * 2))
    max_workers = max(1, int(max_workers))
    max_workers = min(max_workers, max(1, len(scene_chunks)))
    reduce_group_size = max(2, int(reduce_group_size))

    mapped_summaries = _parallel_map_summaries(
        client=client,
        deployment=deployment,
        scene_chunks=scene_chunks,
        max_workers=max_workers,
        debug=debug,
    )
    narrative = "\n".join(item["text"] for item in mapped_summaries).strip()
    narratives.append({
        "narrative_len": len(narrative),
        "chunk_len": len(mapped_summaries),
        "narrative": narrative
    })
    if debug:
        _debug_print(debug, "summarize_scenes:")
        _debug_print(
            debug,
            f"    narrative_size 1: {len(narrative)} char ({len(mapped_summaries)} chunks)"
        )

    if len(narrative) > summary_len and mapped_summaries:
        reduced = _parallel_reduce_summaries(
            client=client,
            deployment=deployment,
            summaries=mapped_summaries,
            reduce_group_size=reduce_group_size,
            max_workers=max_workers,
            debug=debug,
        )
        if reduced and isinstance(reduced.get("text"), str):
            narrative = reduced["text"].strip()
            narratives.append({
                "narrative_len": len(narrative),
                "chunk_len": 1,
                "narrative": narrative
            })
            if debug:
                _debug_print(debug, f"    narrative_size 2: {len(narrative)} char (tree reduced)")

    if len(narrative) > summary_len:
        final_prompt = _build_narrative_consistency_prompt(narrative)
        try:
            narrative = call_gpt(client, deployment, final_prompt).strip()
            narratives.append({
                "narrative_len": len(narrative),
                "chunk_len": 1,
                "narrative": narrative
            })
            if debug:
                _debug_print(debug, f"    narrative_size 3: {len(narrative)} char (final consistency pass)")
        except Exception as exc:
            _debug_print(debug, f"summarize_scenes: final consistency pass failed: {exc}")

    return {
        "scenes": scenes,
        "narratives": narratives
    }

# ----------------------
# 6. Full narrative synthesis
# ----------------------
def synthesize_synopsis(
    client,
    deployment,
    data: dict,
    debug: bool = False,
    output_dir: str | None = None,
    synopsis_ext: str = "md",
    highlights_count: str | int = HIGHLIGHTS_COUNT,
    timeline_count: str | int = TIMELINE_COUNT,
    extra_questions_count: int = EXTRA_QUESTIONS_COUNT,
    consistency_pass_mode: str = "off",
):
    """
    Produce a final synopsis + Q&A from the narrative.
    """
    narratives = data.get("narratives", [])
    narrative_text = narratives[-1]["narrative"] if narratives else ""
    if not isinstance(narrative_text, str):
        narrative_text = ""

    highlight_min, highlight_max = _parse_count_range(highlights_count, 4, 6)
    highlight_label = _count_range_label(highlights_count, highlight_min, highlight_max)
    timeline_min, timeline_max = _parse_count_range(timeline_count, 4, 6)
    timeline_label = _count_range_label(timeline_count, timeline_min, timeline_max)

    required_split_idx = len(REQUIRED_QUESTIONS) // 2
    required_questions_a = REQUIRED_QUESTIONS[:required_split_idx]
    required_questions_b = REQUIRED_QUESTIONS[required_split_idx:]
    required_block_a = _required_questions_block(required_questions_a)
    required_block_b = _required_questions_block(required_questions_b)
    required_block = _required_questions_block(REQUIRED_QUESTIONS)

    prompts = {
        "summary": SYNOPSIS_SUMMARY_PROMPT.format(
            text=narrative_text,
        ),
        "highlights": SYNOPSIS_HIGHLIGHTS_PROMPT.format(
            text=narrative_text,
            highlights_count=highlight_label,
        ),
        "timeline": SYNOPSIS_TIMELINE_PROMPT.format(
            text=narrative_text,
            timeline_count=timeline_label,
        ),
        "qna_predefined_a": SYNOPSIS_QNA_PREDEFINED_PROMPT.format(
            text=narrative_text,
            required_questions_count=len(required_questions_a),
            required_questions_block=required_block_a,
        ),
        "qna_predefined_b": SYNOPSIS_QNA_PREDEFINED_PROMPT.format(
            text=narrative_text,
            required_questions_count=len(required_questions_b),
            required_questions_block=required_block_b,
        ),
        "qna_generated": SYNOPSIS_QNA_GENERATED_PROMPT.format(
            text=narrative_text,
            extra_questions_count=extra_questions_count,
            required_questions_block=required_block,
        ),
    }

    raw_outputs = {}

    def _call_named(name: str, prompt: str):
        fallback_by_name = {
            "summary": '{"chat_name":"Not explicitly stated","summary":"Not explicitly stated."}',
            "highlights": '{"video_highlights":[]}',
            "timeline": '{"video_timeline":[]}',
            "qna_predefined_a": '{"questions":[]}',
            "qna_predefined_b": '{"questions":[]}',
            "qna_generated": '{"questions":[]}',
        }
        safe_section_name = "qna_predefined" if name.startswith("qna_predefined_") else name
        safe_required_block = required_block_a if name == "qna_predefined_a" else required_block_b if name == "qna_predefined_b" else required_block
        safe_prompt = _build_safe_section_prompt(
            section=safe_section_name,
            narrative_text=narrative_text,
            highlight_min=highlight_min,
            highlight_max=highlight_max,
            highlight_label=highlight_label,
            timeline_min=timeline_min,
            timeline_max=timeline_max,
            timeline_label=timeline_label,
            required_questions_block=safe_required_block,
            extra_questions_count=extra_questions_count,
        )
        def _log_ok(text: str):
            _debug_print(debug, f"synopsis {name} [ok] len={len(text)}")

        def _log_err(exc: Exception):
            _debug_print(debug, f"synopsis {name} [error] {exc}")

        try:
            text = call_gpt(client, deployment, prompt)
            _log_ok(text)
            return name, text
        except Exception as exc:
            _log_err(exc)

        if safe_prompt:
            try:
                text = call_gpt(client, deployment, safe_prompt)
                _log_ok(text)
                return name, text
            except Exception as exc2:
                _log_err(exc2)

        text = fallback_by_name.get(name, "{}")
        _log_ok(text)
        return name, text

    with ThreadPoolExecutor(max_workers=min(8, len(prompts))) as executor:
        future_to_name = {executor.submit(_call_named, name, prompt): name for name, prompt in prompts.items()}
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                _, text = future.result()
                raw_outputs[name] = text
            except Exception as exc:
                _debug_print(debug, f"synthesize_synopsis: section '{name}' call failed: {exc}")


    had_errors = False

    summary_text = raw_outputs.get("summary", "")
    summary_payload, summary_ok = _parse_summary_nonjson(summary_text)
    if not summary_ok:
        json_payload = _parse_json_object(summary_text, debug=debug, context="summary_json_fallback")
        if _validate_summary_payload(json_payload):
            summary_payload = json_payload
            summary_ok = True

    highlights_text = raw_outputs.get("highlights", "")
    highlights_payload, highlights_ok = _parse_highlights_nonjson(
        highlights_text,
        highlight_min,
        highlight_max,
    )
    if not highlights_ok:
        json_payload = _parse_json_object(highlights_text, debug=debug, context="highlights_json_fallback")
        if _validate_highlights_payload(json_payload, highlight_min, highlight_max):
            highlights_payload = json_payload
            highlights_ok = True

    timeline_text = raw_outputs.get("timeline", "")
    timeline_payload, timeline_ok = _parse_timeline_nonjson(
        timeline_text,
        timeline_min,
        timeline_max,
    )
    if not timeline_ok:
        json_payload = _parse_json_object(timeline_text, debug=debug, context="timeline_json_fallback")
        if _validate_timeline_payload(json_payload, timeline_min, timeline_max):
            timeline_payload = json_payload
            timeline_ok = True

    qna_predefined_a_text = raw_outputs.get("qna_predefined_a", "")
    qna_predefined_a_payload, qna_predefined_a_ok = _parse_questions_nonjson(
        qna_predefined_a_text,
        len(required_questions_a),
    )
    if not qna_predefined_a_ok:
        json_payload = _parse_json_object(qna_predefined_a_text, debug=debug, context="qna_predefined_a_json_fallback")
        if _validate_questions_payload(json_payload, len(required_questions_a)):
            qna_predefined_a_payload = json_payload
            qna_predefined_a_ok = True

    qna_predefined_b_text = raw_outputs.get("qna_predefined_b", "")
    qna_predefined_b_payload, qna_predefined_b_ok = _parse_questions_nonjson(
        qna_predefined_b_text,
        len(required_questions_b),
    )
    if not qna_predefined_b_ok:
        json_payload = _parse_json_object(qna_predefined_b_text, debug=debug, context="qna_predefined_b_json_fallback")
        if _validate_questions_payload(json_payload, len(required_questions_b)):
            qna_predefined_b_payload = json_payload
            qna_predefined_b_ok = True

    qna_generated_text = raw_outputs.get("qna_generated", "")
    qna_generated_payload, qna_generated_ok = _parse_questions_nonjson(
        qna_generated_text,
        extra_questions_count,
    )
    if not qna_generated_ok:
        json_payload = _parse_json_object(qna_generated_text, debug=debug, context="qna_generated_json_fallback")
        if _validate_questions_payload(json_payload, extra_questions_count):
            qna_generated_payload = json_payload
            qna_generated_ok = True

    repair_requests = {}
    if not summary_ok:
        repair_requests["summary"] = summary_text
    if not highlights_ok:
        repair_requests["highlights"] = highlights_text
    if not timeline_ok:
        repair_requests["timeline"] = timeline_text
    if not qna_predefined_a_ok:
        repair_requests["qna_predefined_a"] = qna_predefined_a_text
    if not qna_predefined_b_ok:
        repair_requests["qna_predefined_b"] = qna_predefined_b_text
    if not qna_generated_ok:
        repair_requests["qna_generated"] = qna_generated_text

    if repair_requests:
        had_errors = True

        def _repair_task(name: str, raw_text: str, required_block_all: str = required_block):
            section = "qna_predefined" if name.startswith("qna_predefined_") else name
            required_block = (
                required_block_a if name == "qna_predefined_a"
                else required_block_b if name == "qna_predefined_b"
                else required_block_all
            )
            required_count = (
                len(required_questions_a) if name == "qna_predefined_a"
                else len(required_questions_b) if name == "qna_predefined_b"
                else len(REQUIRED_QUESTIONS)
            )
            prompt = _build_repair_prompt(
                section=section,
                raw_text=raw_text,
                highlight_min=highlight_min,
                highlight_max=highlight_max,
                highlight_label=highlight_label,
                timeline_min=timeline_min,
                timeline_max=timeline_max,
                timeline_label=timeline_label,
                required_questions_block=required_block,
                required_questions_count=required_count,
                extra_questions_count=extra_questions_count,
            )
            if debug:
                _debug_print(debug, f"synopsis {name} parse failed, repairing")
            fallback_by_name = {
                "summary": '{"chat_name":"Not explicitly stated","summary":"Not explicitly stated."}',
                "highlights": '{"video_highlights":[]}',
                "timeline": '{"video_timeline":[]}',
                "qna_predefined_a": '{"questions":[]}',
                "qna_predefined_b": '{"questions":[]}',
                "qna_generated": '{"questions":[]}',
            }
            text = call_gpt_safe(
                client,
                deployment,
                prompt=prompt,
                fallback_text=fallback_by_name.get(name, "{}"),
                debug=debug,
                context=f"synthesize_synopsis:repair:{name}",
                raw_fallback=fallback_by_name.get(name, "{}"),
            )
            return name, text

        with ThreadPoolExecutor(max_workers=min(6, len(repair_requests))) as executor:
            futures = [executor.submit(_repair_task, name, raw_text) for name, raw_text in repair_requests.items()]
            for future in as_completed(futures):
                name, text = future.result()
                payload = _parse_json_object(text, debug=debug, context=f"{name}_repair")
                if name == "summary" and _validate_summary_payload(payload):
                    summary_payload = payload
                    summary_ok = True
                elif name == "highlights" and _validate_highlights_payload(payload, highlight_min, highlight_max):
                    highlights_payload = payload
                    highlights_ok = True
                elif name == "timeline" and _validate_timeline_payload(payload, timeline_min, timeline_max):
                    timeline_payload = payload
                    timeline_ok = True
                elif name == "qna_predefined_a" and _validate_questions_payload(payload, len(required_questions_a)):
                    qna_predefined_a_payload = payload
                    qna_predefined_a_ok = True
                elif name == "qna_predefined_b" and _validate_questions_payload(payload, len(required_questions_b)):
                    qna_predefined_b_payload = payload
                    qna_predefined_b_ok = True
                elif name == "qna_generated" and _validate_questions_payload(payload, extra_questions_count):
                    qna_generated_payload = payload
                    qna_generated_ok = True

    base_ok = summary_ok and highlights_ok and timeline_ok
    if not base_ok:
        had_errors = True
        _debug_print(debug, "synthesize_synopsis: falling back to monolithic synopsis for base sections")
        monolith_fallback = '{"chat_name":"Not explicitly stated","summary":"Not explicitly stated.","video_highlights":[],"video_timeline":[]}'
        monolith_prompt = (
            "You are a story detective.\n"
            "Return ONE valid JSON object only. No markdown, no extra text.\n"
            "Use this exact schema and key names:\n"
            "{\n"
            '  "chat_name": "3-5 word title",\n'
            '  "summary": "Single coherent paragraph.",\n'
            '  "video_highlights": [ { "start": "00:00:00", "end": "00:00:00", "highlight": "One sentence highlight." } ],\n'
            '  "video_timeline": [ { "timestamp": "00:00:00", "event": "3-5 word event" } ]\n'
            "}\n"
            "Rules:\n"
            "- \"chat_name\" must be 3-5 words and concrete, not creative.\n"
            f"{_highlight_count_rule(highlight_min, highlight_max, highlight_label)}"
            f"{_timeline_count_rule(timeline_min, timeline_max, timeline_label)}"
            "- If a start or end timestamp is not explicitly stated, use \"Not explicitly stated\".\n"
            "INPUT NARRATIVE:\n"
            f"{narrative_text}\n"
        )
        synopsis_text = call_gpt_safe(
            client,
            deployment,
            prompt=monolith_prompt,
            fallback_text=monolith_fallback,
            debug=debug,
            context="synthesize_synopsis:monolith_fallback",
            safe_prompt=_build_safe_section_prompt(
                section="monolith",
                narrative_text=narrative_text,
                highlight_min=highlight_min,
                highlight_max=highlight_max,
                highlight_label=highlight_label,
                timeline_min=timeline_min,
                timeline_max=timeline_max,
                timeline_label=timeline_label,
                required_questions_block=required_block,
                extra_questions_count=extra_questions_count,
            ),
            raw_fallback=monolith_fallback,
        )
        monolith = _parse_synopsis_json(synopsis_text, debug=debug)
        summary_payload = {"chat_name": monolith.get("chat_name"), "summary": monolith.get("summary")}
        highlights_payload = {"video_highlights": monolith.get("video_highlights", [])}
        timeline_payload = {"video_timeline": monolith.get("video_timeline", [])}
        summary_ok = highlights_ok = timeline_ok = True

    chat_name, summary = _normalize_summary_fields(summary_payload)
    video_highlights = _normalize_highlights(highlights_payload, highlight_min, highlight_max)
    video_timeline = _normalize_timeline(timeline_payload, timeline_min, timeline_max)

    predefined_questions = _normalize_predefined_questions(
        _extract_questions(qna_predefined_a_payload),
        required_questions_a,
    ) + _normalize_predefined_questions(
        _extract_questions(qna_predefined_b_payload),
        required_questions_b,
    )
    generated_questions = _normalize_generated_questions(
        _extract_questions(qna_generated_payload),
        REQUIRED_QUESTIONS,
        extra_questions_count,
        pad=False,
    )
    missing_generated = extra_questions_count - len(generated_questions)
    if missing_generated > 0:
        had_errors = True
        existing_generated_questions = [
            q.get("question") for q in generated_questions
            if isinstance(q, dict) and isinstance(q.get("question"), str)
        ]
        fill_prompt = _build_generated_fill_prompt(
            narrative_text=narrative_text,
            existing_questions=existing_generated_questions,
            missing_count=missing_generated,
        )
        fill_text = call_gpt_safe(
            client,
            deployment,
            prompt=fill_prompt,
            fallback_text='{"questions":[]}',
            debug=debug,
            context="synthesize_synopsis:fill_generated",
            raw_fallback='{"questions":[]}',
        )
        fill_payload = _parse_json_object(fill_text, debug=debug, context="qna_generated_fill")
        fill_questions = _normalize_generated_questions(
            _extract_questions(fill_payload),
            REQUIRED_QUESTIONS,
            missing_generated,
            pad=False,
            exclude_questions=existing_generated_questions,
        )
        generated_questions.extend(fill_questions)
        missing_generated = extra_questions_count - len(generated_questions)
        if missing_generated > 0:
            if debug:
                _debug_print(debug, "synthesize_synopsis: generated questions still short, padding placeholders")
            while len(generated_questions) < extra_questions_count:
                idx = len(generated_questions) + 1
                generated_questions.append({
                    "question": f"Additional predicted question {idx}?",
                    "answer": "Not explicitly stated.",
                })
    questions = predefined_questions + generated_questions
    required_total = len(REQUIRED_QUESTIONS) + extra_questions_count
    questions_fallback_used = False
    if _count_questions(questions) < required_total:
        had_errors = True
        questions_fallback_used = True
        _debug_print(debug, "synthesize_synopsis: retrying legacy questions prompt")
        questions_fallback = '{"questions":[]}'
        questions_prompt = _build_questions_prompt(
            narrative_text=narrative_text,
            required_questions=REQUIRED_QUESTIONS,
            extra_questions_count=extra_questions_count,
            strict=True,
        )
        questions_text = call_gpt_safe(
            client,
            deployment,
            questions_prompt,
            fallback_text=questions_fallback,
            debug=debug,
            context="synthesize_synopsis:legacy_questions",
            safe_prompt=_build_safe_section_prompt(
                section="qna_legacy",
                narrative_text=narrative_text,
                highlight_min=highlight_min,
                highlight_max=highlight_max,
                highlight_label=highlight_label,
                timeline_min=timeline_min,
                timeline_max=timeline_max,
                timeline_label=timeline_label,
                required_questions_block=required_block,
                extra_questions_count=extra_questions_count,
            ),
            raw_fallback=questions_fallback,
        )
        questions_payload = _parse_synopsis_json(questions_text, debug=debug)
        fallback_questions = _extract_questions(questions_payload)
        questions = _normalize_predefined_questions(fallback_questions, REQUIRED_QUESTIONS) + _normalize_generated_questions(
            fallback_questions,
            REQUIRED_QUESTIONS,
            extra_questions_count,
        )

    draft_synopsis = {
        "chat_name": chat_name,
        "summary": summary,
        "video_highlights": video_highlights,
        "video_timeline": video_timeline,
        "questions": questions,
    }

    synopsis_json = draft_synopsis
    mode = (consistency_pass_mode or "off").lower()
    do_consistency = mode == "always" or (mode == "on_error" and had_errors)
    if do_consistency:
        try:
            consistency_prompt = (
                "You are a story detective.\n"
                "Return ONE valid JSON object only. No markdown, no extra text.\n"
                "Review the draft synopsis and correct any factual inconsistencies using only the narrative.\n"
                "Use this exact schema and key names:\n"
                "{\n"
                '  "chat_name": "3-5 word title",\n'
                '  "summary": "Single coherent paragraph.",\n'
                '  "video_highlights": [ { "start": "00:00:00", "end": "00:00:00", "highlight": "One sentence highlight." } ],\n'
                '  "video_timeline": [ { "timestamp": "00:00:00", "event": "3-5 word event" } ],\n'
                '  "questions": [ { "question": "Question text", "answer": "Answer text" } ]\n'
                "}\n"
                "Rules:\n"
                f"{_highlight_count_rule(highlight_min, highlight_max, highlight_label)}"
                f"{_timeline_count_rule(timeline_min, timeline_max, timeline_label)}"
                f"- \"questions\" must contain exactly {len(REQUIRED_QUESTIONS) + extra_questions_count} items.\n"
                "- Do not add any sections not in the schema.\n"
                "Required Questions (must appear in order within questions):\n"
                f"{required_block}\n"
                "INPUT NARRATIVE:\n"
                f"{narrative_text}\n"
                "DRAFT JSON:\n"
                f"{json.dumps(draft_synopsis, ensure_ascii=False)}\n"
            )
            consistency_text = call_gpt_safe(
                client,
                deployment,
                consistency_prompt,
                fallback_text=json.dumps(draft_synopsis, ensure_ascii=False),
                debug=debug,
                context="synthesize_synopsis:consistency_pass",
                safe_prompt=(
                    "Return ONE valid JSON object only. No markdown, no extra text.\n"
                    "Return the JSON below exactly as-is, with no changes:\n"
                    f"{json.dumps(draft_synopsis, ensure_ascii=False)}"
                ),
                raw_fallback=json.dumps(draft_synopsis, ensure_ascii=False),
            )
            consistency_payload = _parse_synopsis_json(consistency_text, debug=debug)
            c_chat_name, c_summary = _normalize_summary_fields(consistency_payload)
            c_highlights = _normalize_highlights(consistency_payload, highlight_min, highlight_max)
            c_timeline = _normalize_timeline(consistency_payload, timeline_min, timeline_max)
            c_questions = _extract_questions(consistency_payload)
            c_predefined = _normalize_predefined_questions(c_questions, REQUIRED_QUESTIONS)
            c_generated = _normalize_generated_questions(c_questions, REQUIRED_QUESTIONS, extra_questions_count)
            synopsis_json = {
                "chat_name": c_chat_name,
                "summary": c_summary,
                "video_highlights": c_highlights,
                "video_timeline": c_timeline,
                "questions": c_predefined + c_generated,
            }
        except Exception as exc:
            _debug_print(debug, f"synthesize_synopsis: consistency pass failed: {exc}")

    synopsis_text = json.dumps(synopsis_json, ensure_ascii=False)
    synopsis_md = render_synopsis_markdown(
        synopsis_json,
        video_path=data.get("video_path"),
        output_dir=output_dir,
    )

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = synopsis_ext.lstrip(".") if synopsis_ext else "md"
        synopsis_path = out_dir / f"synopsis.{ext}"
        if ext.lower() == "md":
            synopsis_path.write_text(synopsis_md, encoding="utf-8")
        elif ext.lower() == "json":
            synopsis_path.write_text(
                json.dumps(synopsis_json, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            synopsis_path.write_text(synopsis_text, encoding="utf-8")
        _debug_print(debug, f"synopsis is saved in {synopsis_path}")

    return {
        "scenes": data.get("scenes", []),
        "narratives": narratives,
        "synopsis": synopsis_json
    }

# ----------------------
# 7. Example usage
# ----------------------
def test(log_path):
    endpoint = os.getenv("GPT_ENDPOINT")
    deployment = os.getenv("GPT_DEPLOYMENT")
    subscription_key = os.getenv("GPT_KEY")
    api_version = os.getenv("GPT_VERSION")

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )
    with open(log_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    data = summarize_scenes(client, deployment, logs.get("scenes"), debug=True, output_dir="logs/synonsis_test")
    result = synthesize_synopsis(client, deployment, data, debug=True, output_dir="logs/synonsis_test")
    return result

# test(r"logs\pasta_hist3_gpt-4o_20260129_105840.json")
