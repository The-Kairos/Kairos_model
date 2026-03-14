"""Video synopsis orchestration: scene summarization and structured synopsis generation."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kairos.core.utils import load_prompt
from kairos.llm.synopsis.parsing import (
    _debug_print,
    _parse_synopsis_json,
    _parse_json_object,
    _parse_summary_nonjson,
    _parse_highlights_nonjson,
    _parse_timeline_nonjson,
    _parse_questions_nonjson,
    _validate_summary_payload,
    _validate_highlights_payload,
    _validate_timeline_payload,
    _validate_questions_payload,
    _extract_questions,
    _count_questions,
    _normalize_summary_fields,
    _normalize_highlights,
    _normalize_timeline,
    _normalize_predefined_questions,
    _normalize_generated_questions,
)
from kairos.llm.synopsis.prompts import (
    _parse_count_range,
    _count_range_label,
    _required_questions_block,
    _highlight_count_rule,
    _timeline_count_rule,
    _build_repair_prompt,
    _build_generated_fill_prompt,
    _build_questions_prompt,
    _build_safe_section_prompt,
    _build_narrative_consistency_prompt,
)
from kairos.llm.synopsis.render import render_synopsis_markdown
from kairos.llm.synopsis.mapreduce import (
    chunk_scenes,
    chunk_narrative,
    parallel_map_summaries,
    parallel_reduce_summaries,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Prompt templates (loaded once at import time)
# ---------------------------------------------------------------------------
SEGMENT_PROMPT = load_prompt("chunk_summary.txt")
FALLBACK_SEGMENT_PROMPT = load_prompt("fallback_chunk_summary.txt")
CARRYOVER_PROMPT = load_prompt("chunk_summary_carryover.txt")
SYNOPSIS_SUMMARY_PROMPT = load_prompt("synopsis_summary.txt")
SYNOPSIS_HIGHLIGHTS_PROMPT = load_prompt("synopsis_highlight.txt")
SYNOPSIS_TIMELINE_PROMPT = load_prompt("synopsis_timeline.txt")
SYNOPSIS_QNA_PREDEFINED_PROMPT = load_prompt("synopsis_qna_predefined.txt")
SYNOPSIS_QNA_GENERATED_PROMPT = load_prompt("synopsis_qna_generated.txt")

# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

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


def call_gpt(client, prompt, retries: int = GPT_MAX_RETRIES, retry_base_sec: float = GPT_RETRY_BASE_SEC):
    """Call LLM with retries via the LLMClient protocol."""
    last_exc = None
    for attempt in range(retries):
        try:
            return client.generate(prompt, max_tokens=16384, temperature=0.2)
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
    prompt: str,
    fallback_text: str,
    debug: bool = False,
    context: str = "call",
    safe_prompt: str | None = None,
    raw_fallback: str | None = None,
):
    try:
        return call_gpt(client, prompt)
    except Exception as exc:
        _debug_print(debug, f"{context}: primary prompt failed due to API error: {exc}")
        if safe_prompt:
            try:
                return call_gpt(client, safe_prompt)
            except Exception as exc2:
                _debug_print(debug, f"{context}: safe prompt failed due to API error: {exc2}")
        if isinstance(raw_fallback, str):
            _debug_print(debug, f"{context}: using raw fallback due to API error")
            return raw_fallback
        _debug_print(debug, f"{context}: using fallback due to API error")
        return fallback_text


# ---------------------------------------------------------------------------
# Scene summarization (map-reduce)
# ---------------------------------------------------------------------------

def summarize_scenes(
    client,
    scenes,
    chunk_size: int = CHUNK_SIZE,
    summary_len: int = FINAL_CHUNK_SIZE,
    debug: bool = False,
    output_dir: str | None = None,
    max_workers: int | None = None,
    reduce_group_size: int = SUMMARY_REDUCE_GROUP_SIZE,
):
    """Summarize scenes into a narrative via parallel map-reduce."""
    scene_chunks = chunk_scenes(scenes, chunk_size, debug=debug)
    if not scene_chunks:
        return {"scenes": scenes, "narratives": []}

    narratives = []
    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = min(SUMMARY_MAX_WORKERS, max(2, cpu * 2))
    max_workers = max(1, int(max_workers))
    max_workers = min(max_workers, max(1, len(scene_chunks)))
    reduce_group_size = max(2, int(reduce_group_size))

    call_fn = lambda prompt: call_gpt(client, prompt)

    mapped_summaries = parallel_map_summaries(
        call_gpt_fn=call_fn,
        scene_chunks=scene_chunks,
        max_workers=max_workers,
        debug=debug,
    )
    narrative = "\n".join(item["text"] for item in mapped_summaries).strip()
    narratives.append({
        "narrative_len": len(narrative),
        "chunk_len": len(mapped_summaries),
        "narrative": narrative,
    })
    if debug:
        _debug_print(debug, "summarize_scenes:")
        _debug_print(debug, f"    narrative_size 1: {len(narrative)} char ({len(mapped_summaries)} chunks)")

    if len(narrative) > summary_len and mapped_summaries:
        reduced = parallel_reduce_summaries(
            call_gpt_fn=call_fn,
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
                "narrative": narrative,
            })
            if debug:
                _debug_print(debug, f"    narrative_size 2: {len(narrative)} char (tree reduced)")

    if len(narrative) > summary_len:
        final_prompt = _build_narrative_consistency_prompt(narrative)
        try:
            narrative = call_gpt(client, final_prompt).strip()
            narratives.append({
                "narrative_len": len(narrative),
                "chunk_len": 1,
                "narrative": narrative,
            })
            if debug:
                _debug_print(debug, f"    narrative_size 3: {len(narrative)} char (final consistency pass)")
        except Exception as exc:
            _debug_print(debug, f"summarize_scenes: final consistency pass failed: {exc}")

    return {"scenes": scenes, "narratives": narratives}


# ---------------------------------------------------------------------------
# Full synopsis synthesis
# ---------------------------------------------------------------------------

def synthesize_synopsis(
    client,
    data: dict,
    debug: bool = False,
    output_dir: str | None = None,
    synopsis_ext: str = "md",
    highlights_count: str | int = HIGHLIGHTS_COUNT,
    timeline_count: str | int = TIMELINE_COUNT,
    extra_questions_count: int = EXTRA_QUESTIONS_COUNT,
    consistency_pass_mode: str = "off",
):
    """Produce a final synopsis + Q&A from the narrative."""
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
        "summary": SYNOPSIS_SUMMARY_PROMPT.format(text=narrative_text),
        "highlights": SYNOPSIS_HIGHLIGHTS_PROMPT.format(text=narrative_text, highlights_count=highlight_label),
        "timeline": SYNOPSIS_TIMELINE_PROMPT.format(text=narrative_text, timeline_count=timeline_label),
        "qna_predefined_a": SYNOPSIS_QNA_PREDEFINED_PROMPT.format(
            text=narrative_text, required_questions_count=len(required_questions_a),
            required_questions_block=required_block_a,
        ),
        "qna_predefined_b": SYNOPSIS_QNA_PREDEFINED_PROMPT.format(
            text=narrative_text, required_questions_count=len(required_questions_b),
            required_questions_block=required_block_b,
        ),
        "qna_generated": SYNOPSIS_QNA_GENERATED_PROMPT.format(
            text=narrative_text, extra_questions_count=extra_questions_count,
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

        try:
            text = call_gpt(client, prompt)
            _debug_print(debug, f"synopsis {name} [ok] len={len(text)}")
            return name, text
        except Exception as exc:
            _debug_print(debug, f"synopsis {name} [error] {exc}")

        if safe_prompt:
            try:
                text = call_gpt(client, safe_prompt)
                _debug_print(debug, f"synopsis {name} [ok] len={len(text)}")
                return name, text
            except Exception as exc2:
                _debug_print(debug, f"synopsis {name} [error] {exc2}")

        text = fallback_by_name.get(name, "{}")
        _debug_print(debug, f"synopsis {name} [ok] len={len(text)}")
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

    # --- Parse raw outputs ---
    summary_text = raw_outputs.get("summary", "")
    summary_payload, summary_ok = _parse_summary_nonjson(summary_text)
    if not summary_ok:
        json_payload = _parse_json_object(summary_text, debug=debug, context="summary_json_fallback")
        if _validate_summary_payload(json_payload):
            summary_payload = json_payload
            summary_ok = True

    highlights_text = raw_outputs.get("highlights", "")
    highlights_payload, highlights_ok = _parse_highlights_nonjson(highlights_text, highlight_min, highlight_max)
    if not highlights_ok:
        json_payload = _parse_json_object(highlights_text, debug=debug, context="highlights_json_fallback")
        if _validate_highlights_payload(json_payload, highlight_min, highlight_max):
            highlights_payload = json_payload
            highlights_ok = True

    timeline_text = raw_outputs.get("timeline", "")
    timeline_payload, timeline_ok = _parse_timeline_nonjson(timeline_text, timeline_min, timeline_max)
    if not timeline_ok:
        json_payload = _parse_json_object(timeline_text, debug=debug, context="timeline_json_fallback")
        if _validate_timeline_payload(json_payload, timeline_min, timeline_max):
            timeline_payload = json_payload
            timeline_ok = True

    qna_predefined_a_text = raw_outputs.get("qna_predefined_a", "")
    qna_predefined_a_payload, qna_predefined_a_ok = _parse_questions_nonjson(qna_predefined_a_text, len(required_questions_a))
    if not qna_predefined_a_ok:
        json_payload = _parse_json_object(qna_predefined_a_text, debug=debug, context="qna_predefined_a_json_fallback")
        if _validate_questions_payload(json_payload, len(required_questions_a)):
            qna_predefined_a_payload = json_payload
            qna_predefined_a_ok = True

    qna_predefined_b_text = raw_outputs.get("qna_predefined_b", "")
    qna_predefined_b_payload, qna_predefined_b_ok = _parse_questions_nonjson(qna_predefined_b_text, len(required_questions_b))
    if not qna_predefined_b_ok:
        json_payload = _parse_json_object(qna_predefined_b_text, debug=debug, context="qna_predefined_b_json_fallback")
        if _validate_questions_payload(json_payload, len(required_questions_b)):
            qna_predefined_b_payload = json_payload
            qna_predefined_b_ok = True

    qna_generated_text = raw_outputs.get("qna_generated", "")
    qna_generated_payload, qna_generated_ok = _parse_questions_nonjson(qna_generated_text, extra_questions_count)
    if not qna_generated_ok:
        json_payload = _parse_json_object(qna_generated_text, debug=debug, context="qna_generated_json_fallback")
        if _validate_questions_payload(json_payload, extra_questions_count):
            qna_generated_payload = json_payload
            qna_generated_ok = True

    # --- Repair pass ---
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
            req_block = (
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
                required_questions_block=req_block,
                required_questions_count=required_count,
                extra_questions_count=extra_questions_count,
            )
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

    # --- Monolith fallback for base sections ---
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

    # --- Normalize all sections ---
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

    # --- Fill missing generated questions ---
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
            _debug_print(debug, "synthesize_synopsis: generated questions still short, padding placeholders")
            while len(generated_questions) < extra_questions_count:
                idx = len(generated_questions) + 1
                generated_questions.append({
                    "question": f"Additional predicted question {idx}?",
                    "answer": "Not explicitly stated.",
                })

    # --- Legacy questions fallback ---
    questions = predefined_questions + generated_questions
    required_total = len(REQUIRED_QUESTIONS) + extra_questions_count
    if _count_questions(questions) < required_total:
        had_errors = True
        _debug_print(debug, "synthesize_synopsis: retrying legacy questions prompt")
        questions_prompt = _build_questions_prompt(
            narrative_text=narrative_text,
            required_questions=REQUIRED_QUESTIONS,
            extra_questions_count=extra_questions_count,
            strict=True,
        )
        questions_text = call_gpt_safe(
            client,
            questions_prompt,
            fallback_text='{"questions":[]}',
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
            raw_fallback='{"questions":[]}',
        )
        questions_payload = _parse_synopsis_json(questions_text, debug=debug)
        fallback_questions = _extract_questions(questions_payload)
        questions = _normalize_predefined_questions(fallback_questions, REQUIRED_QUESTIONS) + _normalize_generated_questions(
            fallback_questions, REQUIRED_QUESTIONS, extra_questions_count,
        )

    # --- Build draft synopsis ---
    draft_synopsis = {
        "chat_name": chat_name,
        "summary": summary,
        "video_highlights": video_highlights,
        "video_timeline": video_timeline,
        "questions": questions,
    }

    # --- Optional consistency pass ---
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

    # --- Save output ---
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
        "synopsis": synopsis_json,
    }
