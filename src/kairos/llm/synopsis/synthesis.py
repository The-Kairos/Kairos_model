"""Synopsis synthesis: LLM orchestration for scene summarization and structured synopsis generation."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kairos.core.utils import (
    is_rate_limit_error,
    load_prompt,
    print_prefixed,
    retry_with_backoff,
)
from kairos.llm.synopsis.mapreduce import (
    CHUNK_SIZE,
    SUMMARY_MAX_WORKERS,
    SUMMARY_REDUCE_GROUP_SIZE,
    chunk_scenes,
    parallel_map_summaries,
    parallel_reduce_summaries,
)
from kairos.llm.synopsis.parsing import (
    NOT_STATED,
    NOT_STATED_PERIOD,
    _count_questions,
    _extract_questions,
    _normalize_generated_questions,
    _normalize_highlights,
    _normalize_predefined_questions,
    _normalize_summary_fields,
    _normalize_timeline,
    _parse_highlights_nonjson,
    _parse_json_object,
    _parse_questions_nonjson,
    _parse_summary_nonjson,
    _parse_synopsis_json,
    _parse_timeline_nonjson,
    _validate_highlights_payload,
    _validate_questions_payload,
    _validate_summary_payload,
    _validate_timeline_payload,
)
from kairos.llm.synopsis.prompts import (
    _build_generated_fill_prompt,
    _build_narrative_consistency_prompt,
    _build_questions_prompt,
    _build_repair_prompt,
    _build_safe_section_prompt,
    _count_range_label,
    _parse_count_range,
    _required_questions_block,
)
from kairos.llm.synopsis.render import render_synopsis_markdown

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FINAL_CHUNK_SIZE = CHUNK_SIZE * 5
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

_SECTION_FALLBACKS = {
    "summary": f'{{"chat_name":"{NOT_STATED}","summary":"{NOT_STATED_PERIOD}"}}',
    "highlights": '{"video_highlights":[]}',
    "timeline": '{"video_timeline":[]}',
    "qna_predefined_a": '{"questions":[]}',
    "qna_predefined_b": '{"questions":[]}',
    "qna_generated": '{"questions":[]}',
}

# ---------------------------------------------------------------------------
# Prompt templates (loaded once at import time)
# ---------------------------------------------------------------------------
SEGMENT_PROMPT = load_prompt("chunk_summary.txt")
FALLBACK_SEGMENT_PROMPT = load_prompt("fallback_chunk_summary.txt")


def _synopsis_log(debug: bool, message: str):
    """Unified debug logging for synopsis modules."""
    if debug:
        print_prefixed("(Synopsis)", message)


CARRYOVER_PROMPT = load_prompt("chunk_summary_carryover.txt")
SYNOPSIS_SUMMARY_PROMPT = load_prompt("synopsis_summary.txt")
SYNOPSIS_HIGHLIGHTS_PROMPT = load_prompt("synopsis_highlight.txt")
SYNOPSIS_TIMELINE_PROMPT = load_prompt("synopsis_timeline.txt")
SYNOPSIS_QNA_PREDEFINED_PROMPT = load_prompt("synopsis_qna_predefined.txt")
SYNOPSIS_QNA_GENERATED_PROMPT = load_prompt("synopsis_qna_generated.txt")

# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------


def _is_responsible_ai_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "content_filter" in msg
        or "content filter" in msg
        or "responsible ai" in msg
        or "safety system" in msg
        or "policy_violation" in msg
    )


def call_gpt(
    client,
    prompt,
    retries: int = GPT_MAX_RETRIES,
    retry_base_sec: float = GPT_RETRY_BASE_SEC,
):
    """Call LLM with retries via the LLMClient protocol."""

    def _is_retryable(exc):
        if _is_responsible_ai_error(exc):
            return False
        return is_rate_limit_error(exc)

    return retry_with_backoff(
        lambda: client.generate(prompt, max_tokens=16384, temperature=0.2),
        max_retries=retries - 1,
        base_sec=retry_base_sec,
        is_retryable=_is_retryable,
    )


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
        _synopsis_log(
            debug, f"{context}: primary prompt failed due to API error: {exc}"
        )
        if safe_prompt:
            try:
                return call_gpt(client, safe_prompt)
            except Exception as exc2:
                _synopsis_log(
                    debug, f"{context}: safe prompt failed due to API error: {exc2}"
                )
        if isinstance(raw_fallback, str):
            _synopsis_log(debug, f"{context}: using raw fallback due to API error")
            return raw_fallback
        _synopsis_log(debug, f"{context}: using fallback due to API error")
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

    def call_fn(prompt: str) -> str:
        return call_gpt(client, prompt)

    mapped_summaries = parallel_map_summaries(
        call_gpt_fn=call_fn,
        scene_chunks=scene_chunks,
        max_workers=max_workers,
        debug=debug,
    )
    narrative = "\n".join(item["text"] for item in mapped_summaries).strip()
    narratives.append(
        {
            "narrative_len": len(narrative),
            "chunk_len": len(mapped_summaries),
            "narrative": narrative,
        }
    )
    if debug:
        _synopsis_log(debug, "summarize_scenes:")
        _synopsis_log(
            debug,
            f"    narrative_size 1: {len(narrative)} char ({len(mapped_summaries)} chunks)",
        )

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
            narratives.append(
                {
                    "narrative_len": len(narrative),
                    "chunk_len": 1,
                    "narrative": narrative,
                }
            )
            if debug:
                _synopsis_log(
                    debug, f"    narrative_size 2: {len(narrative)} char (tree reduced)"
                )

    if len(narrative) > summary_len:
        final_prompt = _build_narrative_consistency_prompt(narrative)
        try:
            narrative = call_gpt(client, final_prompt).strip()
            narratives.append(
                {
                    "narrative_len": len(narrative),
                    "chunk_len": 1,
                    "narrative": narrative,
                }
            )
            if debug:
                _synopsis_log(
                    debug,
                    f"    narrative_size 3: {len(narrative)} char (final consistency pass)",
                )
        except Exception as exc:
            _synopsis_log(
                debug, f"summarize_scenes: final consistency pass failed: {exc}"
            )

    return {"scenes": scenes, "narratives": narratives}


# ---------------------------------------------------------------------------
# Synopsis synthesis helpers
# ---------------------------------------------------------------------------


def _parse_section(
    raw_text, parse_fn, parse_args, validate_fn, validate_args, debug, context
):
    """Parse a raw LLM output with non-JSON parser, falling back to JSON extraction."""
    payload, ok = parse_fn(raw_text, *parse_args)
    if not ok:
        json_payload = _parse_json_object(
            raw_text, debug=debug, context=f"{context}_json_fallback"
        )
        if validate_fn(json_payload, *validate_args):
            return json_payload, True
    return payload, ok


def _call_all_sections(client, prompts, narrative_text, safe_prompt_kwargs, debug):
    """Call LLM for all synopsis sections in parallel, with safe-prompt fallback."""
    raw_outputs = {}

    def _call_named(name, prompt):
        safe_section_name = (
            "qna_predefined" if name.startswith("qna_predefined_") else name
        )
        safe_prompt = _build_safe_section_prompt(
            section=safe_section_name,
            narrative_text=narrative_text,
            **safe_prompt_kwargs,
        )
        try:
            text = call_gpt(client, prompt)
            _synopsis_log(debug, f"synopsis {name} [ok] len={len(text)}")
            return name, text
        except Exception as exc:
            _synopsis_log(debug, f"synopsis {name} [error] {exc}")
        if safe_prompt:
            try:
                text = call_gpt(client, safe_prompt)
                _synopsis_log(debug, f"synopsis {name} [ok] len={len(text)}")
                return name, text
            except Exception as exc2:
                _synopsis_log(debug, f"synopsis {name} [error] {exc2}")
        text = _SECTION_FALLBACKS.get(name, "{}")
        _synopsis_log(debug, f"synopsis {name} [ok] len={len(text)}")
        return name, text

    with ThreadPoolExecutor(max_workers=min(8, len(prompts))) as executor:
        future_to_name = {
            executor.submit(_call_named, name, prompt): name
            for name, prompt in prompts.items()
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                _, text = future.result()
                raw_outputs[name] = text
            except Exception as exc:
                _synopsis_log(
                    debug, f"synthesize_synopsis: section '{name}' call failed: {exc}"
                )

    return raw_outputs


def _repair_failed_sections(client, parsed, repair_configs, debug):
    """Re-call LLM for sections that failed parsing, then re-validate."""
    repair_requests = {name: raw for name, (_, ok, raw) in parsed.items() if not ok}
    if not repair_requests:
        return parsed, False

    def _repair_task(name, raw_text):
        cfg = repair_configs[name]
        prompt = _build_repair_prompt(
            section=cfg["section"], raw_text=raw_text, **cfg["repair_kwargs"]
        )
        _synopsis_log(debug, f"synopsis {name} parse failed, repairing")
        text = call_gpt_safe(
            client,
            prompt=prompt,
            fallback_text=_SECTION_FALLBACKS.get(name, "{}"),
            debug=debug,
            context=f"synthesize_synopsis:repair:{name}",
            raw_fallback=_SECTION_FALLBACKS.get(name, "{}"),
        )
        return name, text

    with ThreadPoolExecutor(max_workers=min(6, len(repair_requests))) as executor:
        futures = [
            executor.submit(_repair_task, name, raw_text)
            for name, raw_text in repair_requests.items()
        ]
        for future in as_completed(futures):
            name, text = future.result()
            cfg = repair_configs[name]
            payload = _parse_json_object(text, debug=debug, context=f"{name}_repair")
            if cfg["validate_fn"](payload, *cfg["validate_args"]):
                parsed[name] = (payload, True, parsed[name][2])

    return parsed, True


def _apply_monolith_fallback(
    client,
    narrative_text,
    highlight_min,
    highlight_max,
    highlight_label,
    timeline_min,
    timeline_max,
    timeline_label,
    safe_prompt_kwargs,
    debug,
):
    """Fall back to a single monolithic LLM call for summary + highlights + timeline."""
    from kairos.llm.synopsis.prompts import _build_monolith_prompt

    _synopsis_log(
        debug,
        "synthesize_synopsis: falling back to monolithic synopsis for base sections",
    )
    monolith_fallback = f'{{"chat_name":"{NOT_STATED}","summary":"{NOT_STATED_PERIOD}","video_highlights":[],"video_timeline":[]}}'
    monolith_prompt = _build_monolith_prompt(
        narrative_text=narrative_text,
        highlight_min=highlight_min,
        highlight_max=highlight_max,
        highlight_label=highlight_label,
        timeline_min=timeline_min,
        timeline_max=timeline_max,
        timeline_label=timeline_label,
    )
    synopsis_text = call_gpt_safe(
        client,
        prompt=monolith_prompt,
        fallback_text=monolith_fallback,
        debug=debug,
        context="synthesize_synopsis:monolith_fallback",
        safe_prompt=_build_safe_section_prompt(
            section="monolith", narrative_text=narrative_text, **safe_prompt_kwargs
        ),
        raw_fallback=monolith_fallback,
    )
    monolith = _parse_synopsis_json(synopsis_text, debug=debug)
    return (
        {"chat_name": monolith.get("chat_name"), "summary": monolith.get("summary")},
        {"video_highlights": monolith.get("video_highlights", [])},
        {"video_timeline": monolith.get("video_timeline", [])},
    )


def _fill_missing_generated(
    client, generated_questions, narrative_text, extra_questions_count, debug
):
    """Fill missing generated questions via LLM, then pad with placeholders if still short."""
    missing = extra_questions_count - len(generated_questions)
    if missing <= 0:
        return generated_questions, False

    existing = [
        q.get("question")
        for q in generated_questions
        if isinstance(q, dict) and isinstance(q.get("question"), str)
    ]
    fill_prompt = _build_generated_fill_prompt(
        narrative_text=narrative_text,
        existing_questions=existing,
        missing_count=missing,
    )
    fill_text = call_gpt_safe(
        client,
        prompt=fill_prompt,
        fallback_text=_SECTION_FALLBACKS["qna_generated"],
        debug=debug,
        context="synthesize_synopsis:fill_generated",
        raw_fallback=_SECTION_FALLBACKS["qna_generated"],
    )
    fill_payload = _parse_json_object(
        fill_text, debug=debug, context="qna_generated_fill"
    )
    fill_questions = _normalize_generated_questions(
        _extract_questions(fill_payload),
        REQUIRED_QUESTIONS,
        missing,
        pad=False,
        exclude_questions=existing,
    )
    generated_questions = generated_questions + fill_questions

    if len(generated_questions) < extra_questions_count:
        _synopsis_log(
            debug,
            "synthesize_synopsis: generated questions still short, padding placeholders",
        )
        while len(generated_questions) < extra_questions_count:
            idx = len(generated_questions) + 1
            generated_questions.append(
                {
                    "question": f"Additional predicted question {idx}?",
                    "answer": "Not explicitly stated.",
                }
            )

    return generated_questions, True


def _run_consistency_pass(
    client,
    draft_synopsis,
    narrative_text,
    highlight_min,
    highlight_max,
    highlight_label,
    timeline_min,
    timeline_max,
    timeline_label,
    required_block,
    extra_questions_count,
    debug,
):
    """Optionally re-check the draft synopsis for factual consistency."""
    from kairos.llm.synopsis.prompts import _build_consistency_prompt

    try:
        consistency_prompt = _build_consistency_prompt(
            narrative_text=narrative_text,
            draft_synopsis=draft_synopsis,
            highlight_min=highlight_min,
            highlight_max=highlight_max,
            highlight_label=highlight_label,
            timeline_min=timeline_min,
            timeline_max=timeline_max,
            timeline_label=timeline_label,
            required_block=required_block,
            required_questions_count=len(REQUIRED_QUESTIONS) + extra_questions_count,
            extra_questions_count=extra_questions_count,
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
        payload = _parse_synopsis_json(consistency_text, debug=debug)
        c_chat_name, c_summary = _normalize_summary_fields(payload)
        c_highlights = _normalize_highlights(payload, highlight_min, highlight_max)
        c_timeline = _normalize_timeline(payload, timeline_min, timeline_max)
        c_questions = _extract_questions(payload)
        c_predefined = _normalize_predefined_questions(c_questions, REQUIRED_QUESTIONS)
        c_generated = _normalize_generated_questions(
            c_questions, REQUIRED_QUESTIONS, extra_questions_count
        )
        return {
            "chat_name": c_chat_name,
            "summary": c_summary,
            "video_highlights": c_highlights,
            "video_timeline": c_timeline,
            "questions": c_predefined + c_generated,
        }
    except Exception as exc:
        _synopsis_log(debug, f"synthesize_synopsis: consistency pass failed: {exc}")
        return draft_synopsis


# ---------------------------------------------------------------------------
# Full synopsis synthesis
# ---------------------------------------------------------------------------


def _build_section_prompts(
    narrative_text,
    highlight_label,
    timeline_label,
    required_questions_a,
    required_questions_b,
    required_block_a,
    required_block_b,
    required_block,
    extra_questions_count,
):
    """Build the prompt dict for all synopsis sections."""
    return {
        "summary": SYNOPSIS_SUMMARY_PROMPT.format(text=narrative_text),
        "highlights": SYNOPSIS_HIGHLIGHTS_PROMPT.format(
            text=narrative_text, highlights_count=highlight_label
        ),
        "timeline": SYNOPSIS_TIMELINE_PROMPT.format(
            text=narrative_text, timeline_count=timeline_label
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


def _build_repair_configs(
    parsed,
    section_parse_configs,
    required_questions_a,
    required_questions_b,
    required_block_a,
    required_block_b,
    required_block,
    highlight_min,
    highlight_max,
    highlight_label,
    timeline_min,
    timeline_max,
    timeline_label,
    extra_questions_count,
):
    """Build repair config dict for sections that failed parsing."""
    repair_configs = {}
    for name, (_, ok, _) in parsed.items():
        if not ok:
            section = "qna_predefined" if name.startswith("qna_predefined_") else name
            req_block = (
                required_block_a
                if name == "qna_predefined_a"
                else required_block_b
                if name == "qna_predefined_b"
                else required_block
            )
            req_count = (
                len(required_questions_a)
                if name == "qna_predefined_a"
                else len(required_questions_b)
                if name == "qna_predefined_b"
                else len(REQUIRED_QUESTIONS)
            )
            _, _, validate_fn, validate_args = section_parse_configs[name]
            repair_configs[name] = {
                "section": section,
                "repair_kwargs": dict(
                    highlight_min=highlight_min,
                    highlight_max=highlight_max,
                    highlight_label=highlight_label,
                    timeline_min=timeline_min,
                    timeline_max=timeline_max,
                    timeline_label=timeline_label,
                    required_questions_block=req_block,
                    required_questions_count=req_count,
                    extra_questions_count=extra_questions_count,
                ),
                "validate_fn": validate_fn,
                "validate_args": validate_args,
            }
    return repair_configs


def _save_synopsis_output(synopsis_json, synopsis_md, output_dir, synopsis_ext, debug):
    """Write synopsis to disk in the requested format."""
    if not output_dir:
        return
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = synopsis_ext.lstrip(".") if synopsis_ext else "md"
    synopsis_path = out_dir / f"synopsis.{ext}"
    if ext.lower() == "md":
        synopsis_path.write_text(synopsis_md, encoding="utf-8")
    elif ext.lower() == "json":
        synopsis_path.write_text(
            json.dumps(synopsis_json, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    else:
        synopsis_path.write_text(
            json.dumps(synopsis_json, ensure_ascii=False), encoding="utf-8"
        )
    if debug:
        _synopsis_log(debug, f"synopsis saved in {synopsis_path}")


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

    safe_prompt_kwargs = dict(
        highlight_min=highlight_min,
        highlight_max=highlight_max,
        highlight_label=highlight_label,
        timeline_min=timeline_min,
        timeline_max=timeline_max,
        timeline_label=timeline_label,
        required_questions_block=required_block,
        extra_questions_count=extra_questions_count,
    )

    # --- Step 1: Call LLM for all sections in parallel ---
    prompts = _build_section_prompts(
        narrative_text,
        highlight_label,
        timeline_label,
        required_questions_a,
        required_questions_b,
        required_block_a,
        required_block_b,
        required_block,
        extra_questions_count,
    )
    raw_outputs = _call_all_sections(
        client, prompts, narrative_text, safe_prompt_kwargs, debug
    )

    # --- Step 2: Parse all sections ---
    section_parse_configs = {
        "summary": (_parse_summary_nonjson, (), _validate_summary_payload, ()),
        "highlights": (
            _parse_highlights_nonjson,
            (highlight_min, highlight_max),
            _validate_highlights_payload,
            (highlight_min, highlight_max),
        ),
        "timeline": (
            _parse_timeline_nonjson,
            (timeline_min, timeline_max),
            _validate_timeline_payload,
            (timeline_min, timeline_max),
        ),
        "qna_predefined_a": (
            _parse_questions_nonjson,
            (len(required_questions_a),),
            _validate_questions_payload,
            (len(required_questions_a),),
        ),
        "qna_predefined_b": (
            _parse_questions_nonjson,
            (len(required_questions_b),),
            _validate_questions_payload,
            (len(required_questions_b),),
        ),
        "qna_generated": (
            _parse_questions_nonjson,
            (extra_questions_count,),
            _validate_questions_payload,
            (extra_questions_count,),
        ),
    }
    parsed = {}
    for name, (
        parse_fn,
        parse_args,
        validate_fn,
        validate_args,
    ) in section_parse_configs.items():
        raw_text = raw_outputs.get(name, "")
        payload, ok = _parse_section(
            raw_text, parse_fn, parse_args, validate_fn, validate_args, debug, name
        )
        parsed[name] = (payload, ok, raw_text)

    # --- Step 3: Repair failed sections ---
    repair_configs = _build_repair_configs(
        parsed,
        section_parse_configs,
        required_questions_a,
        required_questions_b,
        required_block_a,
        required_block_b,
        required_block,
        highlight_min,
        highlight_max,
        highlight_label,
        timeline_min,
        timeline_max,
        timeline_label,
        extra_questions_count,
    )
    parsed, had_errors = _repair_failed_sections(client, parsed, repair_configs, debug)

    # --- Step 4: Monolith fallback for base sections ---
    summary_payload, summary_ok = parsed["summary"][0], parsed["summary"][1]
    highlights_payload, highlights_ok = parsed["highlights"][0], parsed["highlights"][1]
    timeline_payload, timeline_ok = parsed["timeline"][0], parsed["timeline"][1]

    if not (summary_ok and highlights_ok and timeline_ok):
        had_errors = True
        summary_payload, highlights_payload, timeline_payload = (
            _apply_monolith_fallback(
                client,
                narrative_text,
                highlight_min,
                highlight_max,
                highlight_label,
                timeline_min,
                timeline_max,
                timeline_label,
                safe_prompt_kwargs,
                debug,
            )
        )

    # --- Step 5: Normalize all sections ---
    chat_name, summary = _normalize_summary_fields(summary_payload)
    video_highlights = _normalize_highlights(
        highlights_payload, highlight_min, highlight_max
    )
    video_timeline = _normalize_timeline(timeline_payload, timeline_min, timeline_max)

    predefined_questions = _normalize_predefined_questions(
        _extract_questions(parsed["qna_predefined_a"][0]),
        required_questions_a,
    ) + _normalize_predefined_questions(
        _extract_questions(parsed["qna_predefined_b"][0]),
        required_questions_b,
    )
    generated_questions = _normalize_generated_questions(
        _extract_questions(parsed["qna_generated"][0]),
        REQUIRED_QUESTIONS,
        extra_questions_count,
        pad=False,
    )

    # --- Step 6: Fill missing generated questions ---
    generated_questions, fill_had_errors = _fill_missing_generated(
        client,
        generated_questions,
        narrative_text,
        extra_questions_count,
        debug,
    )
    had_errors = had_errors or fill_had_errors

    # --- Step 7: Legacy questions fallback ---
    questions = predefined_questions + generated_questions
    required_total = len(REQUIRED_QUESTIONS) + extra_questions_count
    if _count_questions(questions) < required_total:
        had_errors = True
        _synopsis_log(debug, "synthesize_synopsis: retrying legacy questions prompt")
        questions_prompt = _build_questions_prompt(
            narrative_text=narrative_text,
            required_questions=REQUIRED_QUESTIONS,
            extra_questions_count=extra_questions_count,
            strict=True,
        )
        questions_text = call_gpt_safe(
            client,
            questions_prompt,
            fallback_text=_SECTION_FALLBACKS["qna_generated"],
            debug=debug,
            context="synthesize_synopsis:legacy_questions",
            safe_prompt=_build_safe_section_prompt(
                section="qna_legacy",
                narrative_text=narrative_text,
                **safe_prompt_kwargs,
            ),
            raw_fallback=_SECTION_FALLBACKS["qna_generated"],
        )
        questions_payload = _parse_synopsis_json(questions_text, debug=debug)
        fallback_questions = _extract_questions(questions_payload)
        questions = _normalize_predefined_questions(
            fallback_questions, REQUIRED_QUESTIONS
        ) + _normalize_generated_questions(
            fallback_questions,
            REQUIRED_QUESTIONS,
            extra_questions_count,
        )

    # --- Step 8: Build draft and optional consistency pass ---
    draft_synopsis = {
        "chat_name": chat_name,
        "summary": summary,
        "video_highlights": video_highlights,
        "video_timeline": video_timeline,
        "questions": questions,
    }

    synopsis_json = draft_synopsis
    mode = (consistency_pass_mode or "off").lower()
    if mode == "always" or (mode == "on_error" and had_errors):
        synopsis_json = _run_consistency_pass(
            client,
            draft_synopsis,
            narrative_text,
            highlight_min,
            highlight_max,
            highlight_label,
            timeline_min,
            timeline_max,
            timeline_label,
            required_block,
            extra_questions_count,
            debug,
        )

    # --- Step 9: Save output ---
    synopsis_md = render_synopsis_markdown(
        synopsis_json,
        video_path=data.get("video_path"),
        output_dir=output_dir,
    )
    _save_synopsis_output(synopsis_json, synopsis_md, output_dir, synopsis_ext, debug)

    return {
        "scenes": data.get("scenes", []),
        "narratives": narratives,
        "synopsis": synopsis_json,
    }
