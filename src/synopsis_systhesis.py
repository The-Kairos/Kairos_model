import os
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 7000
FINAL_CHUNK_SIZE = CHUNK_SIZE * 5
SUMMARY_MAX_WORKERS = 6
SUMMARY_REDUCE_GROUP_SIZE = 4
GPT_MAX_RETRIES = 6
GPT_RETRY_BASE_SEC = 2.0
HIGHLIGHTS_COUNT = 4
TIMELINE_COUNT = 10
CLIPS_COUNT = 5
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
            "suggested_clips": [],
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
                "suggested_clips": [],
                "questions": [],
                "parse_error": "Invalid JSON from model",
            }
    if not isinstance(obj, dict):
        return {
            "chat_name": "Not explicitly stated",
            "summary": text.strip(),
            "video_highlights": [],
            "video_timeline": [],
            "suggested_clips": [],
            "questions": [],
            "parse_error": "JSON root was not an object",
        }
    return obj


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


def _extract_timed_entry(item, text_key: str):
    if not isinstance(item, dict):
        return None, None
    if "timestamp" in item:
        return item.get("timestamp"), item.get(text_key)
    if len(item) == 1:
        timestamp, value = next(iter(item.items()))
        return timestamp, value
    return item.get("timestamp"), item.get(text_key)


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
            ts, highlight = _extract_timed_entry(item, "highlight")
            ts_md = _format_timestamp_markdown(ts, video_link_base)
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

    clips = synopsis.get("suggested_clips") if isinstance(synopsis, dict) else []
    if isinstance(clips, list) and clips:
        lines.extend(["", "## Suggested Clips"])
        for entry in clips:
            ts, desc = _extract_timed_entry(entry, "description")
            ts_md = _format_timestamp_markdown(ts, video_link_base)
            if isinstance(desc, str) and desc.strip():
                lines.append(f"- {ts_md}: {desc.strip()}")
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


def _normalize_generated_questions(questions: list[dict], required_questions: list[str], extra_count: int) -> list[dict]:
    required_set = {_normalize_question_key(q) for q in required_questions}
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
SYNOPSIS_MONOLITH_PROMPT = load_prompt("synposis_rag.txt")
SYNOPSIS_SUMMARY_PROMPT = load_prompt("synopsis_summary.txt")
SYNOPSIS_HIGHLIGHTS_PROMPT = load_prompt("synopsis_highlight.txt")
SYNOPSIS_TIMELINE_PROMPT = load_prompt("synopsis_timeline.txt")
SYNOPSIS_CLIPS_PROMPT = load_prompt("synopsis_clips.txt")
SYNOPSIS_QNA_PREDEFINED_PROMPT = load_prompt("synopsis_qna_predefined.txt")
SYNOPSIS_QNA_GENERATED_PROMPT = load_prompt("synopsis_qna_generated.txt")
SYNOPSIS_CONSISTENCY_PROMPT = load_prompt("synopsis_consistency_pass.txt")


def _build_safe_section_prompt(
    section: str,
    narrative_text: str,
    highlights_count: int,
    timeline_count: int,
    clips_count: int,
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
            '{ "video_highlights": [ { "timestamp": "00:00:00", "highlight": "One sentence highlight." } ] }\n'
            "Rules:\n"
            f"- \"video_highlights\" must contain exactly {highlights_count} items.\n"
            "- Each highlight is one sentence.\n"
            "- If a timestamp is not explicitly stated, use \"Not explicitly stated\".\n"
        )
    elif section == "timeline":
        schema = (
            "Use this exact schema and key names:\n"
            '{ "video_timeline": [ { "timestamp": "00:00:00", "event": "3-5 word event" } ] }\n'
            "Rules:\n"
            f"- \"video_timeline\" must contain exactly {timeline_count} items.\n"
            "- Events must be 3-5 words, chronological order.\n"
            "- If a timestamp is not explicitly stated, use \"Not explicitly stated\".\n"
        )
    elif section == "clips":
        schema = (
            "Use this exact schema and key names:\n"
            '{ "suggested_clips": [ { "timestamp": "00:00:00", "description": "Two sentences about significance of the clip" } ] }\n'
            "Rules:\n"
            f"- \"suggested_clips\" must contain exactly {clips_count} items.\n"
            "- Each description is exactly 2 sentences.\n"
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
            "- Do not repeat required questions.\n"
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
            '  "video_highlights": [ { "timestamp": "00:00:00", "highlight": "One sentence highlight." } ],\n'
            '  "video_timeline": [ { "timestamp": "00:00:00", "event": "3-5 word event" } ],\n'
            '  "suggested_clips": [ { "timestamp": "00:00:00", "description": "Two sentences about significance of the clip" } ]\n'
            "}\n"
            "Rules:\n"
            "- \"chat_name\" must be 3-5 words and concrete, not creative.\n"
            f"- \"video_highlights\" must contain exactly {highlights_count} items.\n"
            f"- \"video_timeline\" must contain exactly {timeline_count} items.\n"
            f"- \"suggested_clips\" must contain exactly {clips_count} items.\n"
            "- If a timestamp is not explicitly stated, use \"Not explicitly stated\".\n"
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
    highlights_count: int = HIGHLIGHTS_COUNT,
    timeline_count: int = TIMELINE_COUNT,
    clips_count: int = CLIPS_COUNT,
    extra_questions_count: int = EXTRA_QUESTIONS_COUNT,
):
    """
    Produce a final synopsis + Q&A from the narrative.
    """
    narratives = data.get("narratives", [])
    narrative_text = narratives[-1]["narrative"] if narratives else ""
    if not isinstance(narrative_text, str):
        narrative_text = ""

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
            highlights_count=highlights_count,
        ),
        "timeline": SYNOPSIS_TIMELINE_PROMPT.format(
            text=narrative_text,
            timeline_count=timeline_count,
        ),
        "clips": SYNOPSIS_CLIPS_PROMPT.format(
            text=narrative_text,
            clips_count=clips_count,
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
            "clips": '{"suggested_clips":[]}',
            "qna_predefined_a": '{"questions":[]}',
            "qna_predefined_b": '{"questions":[]}',
            "qna_generated": '{"questions":[]}',
        }
        safe_section_name = "qna_predefined" if name.startswith("qna_predefined_") else name
        safe_required_block = required_block_a if name == "qna_predefined_a" else required_block_b if name == "qna_predefined_b" else required_block
        safe_prompt = _build_safe_section_prompt(
            section=safe_section_name,
            narrative_text=narrative_text,
            highlights_count=highlights_count,
            timeline_count=timeline_count,
            clips_count=clips_count,
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

    summary_payload = _parse_json_object(raw_outputs.get("summary"), debug=debug, context="summary")
    highlights_payload = _parse_json_object(raw_outputs.get("highlights"), debug=debug, context="highlights")
    timeline_payload = _parse_json_object(raw_outputs.get("timeline"), debug=debug, context="timeline")
    clips_payload = _parse_json_object(raw_outputs.get("clips"), debug=debug, context="clips")
    qna_predefined_a_payload = _parse_json_object(raw_outputs.get("qna_predefined_a"), debug=debug, context="qna_predefined_a")
    qna_predefined_b_payload = _parse_json_object(raw_outputs.get("qna_predefined_b"), debug=debug, context="qna_predefined_b")
    qna_generated_payload = _parse_json_object(raw_outputs.get("qna_generated"), debug=debug, context="qna_generated")

    missing_base_sections = any(
        not isinstance(raw_outputs.get(key), str) or not raw_outputs.get(key, "").strip()
        for key in ("summary", "highlights", "timeline", "clips")
    )
    base_payload_invalid = (
        not isinstance(summary_payload.get("summary"), str)
        or not isinstance(highlights_payload.get("video_highlights"), list)
        or not isinstance(timeline_payload.get("video_timeline"), list)
        or not isinstance(clips_payload.get("suggested_clips"), list)
    )
    missing_base_sections = missing_base_sections or base_payload_invalid
    if missing_base_sections:
        _debug_print(debug, "synthesize_synopsis: falling back to monolithic synopsis for base sections")
        monolith_fallback = '{"chat_name":"Not explicitly stated","summary":"Not explicitly stated.","video_highlights":[],"video_timeline":[],"suggested_clips":[]}'
        synopsis_text = call_gpt_safe(
            client,
            deployment,
            prompt=SYNOPSIS_MONOLITH_PROMPT.format(
                text=narrative_text,
                highlights_count=highlights_count,
                timeline_count=timeline_count,
                clips_count=clips_count,
            ),
            fallback_text=monolith_fallback,
            debug=debug,
            context="synthesize_synopsis:monolith_fallback",
            safe_prompt=_build_safe_section_prompt(
                section="monolith",
                narrative_text=narrative_text,
                highlights_count=highlights_count,
                timeline_count=timeline_count,
                clips_count=clips_count,
                required_questions_block=required_block,
                extra_questions_count=extra_questions_count,
            ),
            raw_fallback=monolith_fallback,
        )
        monolith = _parse_synopsis_json(synopsis_text, debug=debug)
        summary_payload = {"chat_name": monolith.get("chat_name"), "summary": monolith.get("summary")}
        highlights_payload = {"video_highlights": monolith.get("video_highlights", [])}
        timeline_payload = {"video_timeline": monolith.get("video_timeline", [])}
        clips_payload = {"suggested_clips": monolith.get("suggested_clips", [])}

    chat_name, summary = _normalize_summary_fields(summary_payload)
    video_highlights = _normalize_section_items(highlights_payload, "video_highlights", "highlight", highlights_count)
    video_timeline = _normalize_section_items(timeline_payload, "video_timeline", "event", timeline_count)
    suggested_clips = _normalize_section_items(clips_payload, "suggested_clips", "description", clips_count)

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
    )
    questions = predefined_questions + generated_questions
    required_total = len(REQUIRED_QUESTIONS) + extra_questions_count
    if _count_questions(questions) < required_total:
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
                highlights_count=highlights_count,
                timeline_count=timeline_count,
                clips_count=clips_count,
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
        "suggested_clips": suggested_clips,
        "questions": questions,
    }

    synopsis_json = draft_synopsis
    try:
        consistency_prompt = SYNOPSIS_CONSISTENCY_PROMPT.format(
            text=narrative_text,
            draft_json=json.dumps(draft_synopsis, ensure_ascii=False),
            highlights_count=highlights_count,
            timeline_count=timeline_count,
            clips_count=clips_count,
            required_questions_count=len(REQUIRED_QUESTIONS),
            extra_questions_count=extra_questions_count,
            required_questions_block=required_block,
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
        c_highlights = _normalize_section_items(consistency_payload, "video_highlights", "highlight", highlights_count)
        c_timeline = _normalize_section_items(consistency_payload, "video_timeline", "event", timeline_count)
        c_clips = _normalize_section_items(consistency_payload, "suggested_clips", "description", clips_count)
        c_questions = _extract_questions(consistency_payload)
        c_predefined = _normalize_predefined_questions(c_questions, REQUIRED_QUESTIONS)
        c_generated = _normalize_generated_questions(c_questions, REQUIRED_QUESTIONS, extra_questions_count)
        synopsis_json = {
            "chat_name": c_chat_name,
            "summary": c_summary,
            "video_highlights": c_highlights,
            "video_timeline": c_timeline,
            "suggested_clips": c_clips,
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
