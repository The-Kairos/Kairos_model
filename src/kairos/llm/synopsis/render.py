"""Synopsis markdown rendering."""

from __future__ import annotations

import os
from urllib.parse import quote

from kairos.llm.synopsis.parsing import NOT_STATED, NOT_STATED_PERIOD


def _timecode_to_seconds(timecode: str | None):
    if not timecode or not isinstance(timecode, str):
        return None
    tc = timecode.strip()
    if not tc or tc.lower() == NOT_STATED.lower():
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
    label = timestamp.strip() if isinstance(timestamp, str) and timestamp.strip() else NOT_STATED
    seconds = _timecode_to_seconds(timestamp)
    if seconds is None:
        return label
    link = f"{video_link_base}#t={seconds}" if video_link_base else f"#t={seconds}"
    return f"[{label}]({link})"


def _format_time_range_markdown(start: str | None, end: str | None, video_link_base: str | None) -> str:
    start_md = _format_timestamp_markdown(start, video_link_base)
    end_md = _format_timestamp_markdown(end, video_link_base)
    if end_md == NOT_STATED:
        return start_md
    if start_md == NOT_STATED:
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
                    lines.append(f"**A:** {NOT_STATED_PERIOD}")
                lines.append("")

    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines).strip() + "\n"
