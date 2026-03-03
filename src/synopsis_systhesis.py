import os
import json
from pathlib import Path
from urllib.parse import quote
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 7000
FINAL_CHUNK_SIZE = CHUNK_SIZE * 5
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

# ----------------------
# 2. Chunk scenes
# ----------------------
def chunk_scenes(scenes: list, chunk_size: int = CHUNK_SIZE, debug: bool = False):
    """
    Break scene dictionaries into <= chunk_size chunks.
    """
    scene_count = len(scenes) if scenes else 0
    chunks = []
    this_chunk = ""

    for scene in scenes:
        start_timecode = scene.get("start_timecode")
        audio_speech = scene.get("audio_speech")
        llm_scene_description = scene.get("llm_scene_description")

        this_chunk += f'At {start_timecode}, {llm_scene_description}. It says "{audio_speech}".'

        if len(this_chunk) >= chunk_size:
            chunks.append(this_chunk)
            this_chunk = ""

    if this_chunk:
        chunks.append(this_chunk)

    _debug_print(debug, f"chunk_scenes: {scene_count} scenes turned into {len(chunks)} chunk_scenes")
    return chunks

# ----------------------
# 3. GPT call helper
# ----------------------
def call_gpt(client, deployment, prompt):
    """
    Minimal GPT call using AzureOpenAI client
    """
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a precise and reliable assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=16384,  # apparently the max for gpt4o
        temperature=1.0,
        top_p=1.0,
        model=deployment
    )
    text = response.choices[0].message.content.strip()
    return text

# ----------------------
# 4. Segment synthesis prompts (loaded from prompts folder)
# ----------------------
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

def load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")

SEGMENT_PROMPT = load_prompt("chunk_summary.txt")
FALLBACK_SEGMENT_PROMPT = load_prompt("fallback_chunk_summary.txt")
CARRYOVER_PROMPT = load_prompt("chunk_summary_carryover.txt")
SYSTHESIS_PROMPT = load_prompt("synposis_rag.txt")

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
    _debug_print(debug, f"    condense_chunk: chunk_len={len(chunk_text)} condensed into len={len(summary)}")
    return summary, new_carryover_context

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

def summarize_scenes(client, deployment, scenes, chunk_size: int = CHUNK_SIZE, summary_len: int = FINAL_CHUNK_SIZE, debug: bool = False, output_dir: str | None = None):
    """
    Summarize scenes into a narrative, then recursively compress
    until the narrative fits within summary_len.
    """
    scene_chunks = chunk_scenes(scenes, chunk_size, debug=debug,)
    if scene_chunks:
        raw_narrative = "\n".join(scene_chunks).strip()
        if len(raw_narrative) <= summary_len:
            narratives = [{
                "narrative_len": len(raw_narrative),
                "chunk_len": len(scene_chunks),
                "narrative": raw_narrative
            }]
            if output_dir:
                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"narrative_1_len_{len(raw_narrative)}.txt"
                out_path.write_text(raw_narrative, encoding="utf-8")
            return {
                "scenes": scenes,
                "narratives": narratives
            }
    narratives = []
    pre_carryover_context = "None"

    narrative = ""
    for scene in scene_chunks:
        summary, pre_carryover_context = condense_chunk(client, deployment, scene, pre_carryover_context, debug=debug)
        narrative += summary + "\n"

    narratives.append({
        "narrative_len": len(narrative),
        "chunk_len": len(scene_chunks),
        "narrative": narrative.strip()
    })

    if debug:
        _debug_print(debug, "summarize_scenes:")
        _debug_print(
            debug,
            f"    narrative_size 1: {len(narrative)} char ({len(scene_chunks)} chunks)"
        )

    round_index = 1
    while len(narrative) > summary_len:
        round_index += 1
        narrative_chunks = chunk_narrative(narrative, chunk_size, debug=debug)
        narrative = ""
        for chunk in narrative_chunks:
            summary, pre_carryover_context = condense_chunk(client, deployment, chunk, pre_carryover_context, debug=debug)
            narrative += summary + "\n"

        narratives.append({
            "narrative_len": len(narrative),
            "chunk_len": len(narrative_chunks),
            "narrative": narrative.strip()
        })

        if debug:
            _debug_print(
                debug,
                f"    narrative_size {round_index}: {len(narrative)} char ({len(narrative_chunks)} chunks)"
            )

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, item in enumerate(narratives, start=1):
            out_path = out_dir / f"narrative_{i}_len_{item['narrative_len']}.txt"
            out_path.write_text(item["narrative"], encoding="utf-8")

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
    synopsis_text = call_gpt(
        client,
        deployment,
        SYSTHESIS_PROMPT.format(
            text=narrative_text,
            highlights_count=highlights_count,
            timeline_count=timeline_count,
            clips_count=clips_count,
        ),
    )
    synopsis_json = _parse_synopsis_json(synopsis_text, debug=debug)

    required_total = len(REQUIRED_QUESTIONS) + extra_questions_count
    questions_prompt = _build_questions_prompt(
        narrative_text=narrative_text,
        required_questions=REQUIRED_QUESTIONS,
        extra_questions_count=extra_questions_count,
        strict=False,
    )
    questions_text = call_gpt(client, deployment, questions_prompt)
    questions_payload = _parse_synopsis_json(questions_text, debug=debug)
    questions = _extract_questions(questions_payload)
    if _count_questions(questions) < required_total:
        _debug_print(
            debug,
            "synopsis questions incomplete; retrying with strict questions prompt",
        )
        questions_prompt = _build_questions_prompt(
            narrative_text=narrative_text,
            required_questions=REQUIRED_QUESTIONS,
            extra_questions_count=extra_questions_count,
            strict=True,
        )
        questions_text = call_gpt(client, deployment, questions_prompt)
        questions_payload = _parse_synopsis_json(questions_text, debug=debug)
        questions = _extract_questions(questions_payload)

    synopsis_json["questions"] = questions
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
