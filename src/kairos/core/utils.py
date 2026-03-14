"""Shared utility functions: printing, timecodes, prompt loading, normalization."""

import json
import re
from pathlib import Path

SECTION_LINE = "=" * 40

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def print_section(title: str) -> None:
    print(SECTION_LINE)
    print(title)
    print(SECTION_LINE)


def print_prefixed(prefix: str, message: str, indent: int = 0) -> None:
    pad = " " * indent
    print(f"{prefix} {pad}{message}")


def format_timecode(seconds: float | None) -> str:
    if seconds is None:
        return "??:??:??.???"
    try:
        ms_total = int(round(float(seconds) * 1000))
    except (TypeError, ValueError):
        return "??:??:??.???"
    sec_total, ms = divmod(ms_total, 1000)
    mins_total, sec = divmod(sec_total, 60)
    hrs, mins = divmod(mins_total, 60)
    return f"{hrs:02d}:{mins:02d}:{sec:02d}.{ms:03d}"


def load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def apply_gpt_normalization(text: str, filename: str = "gpt_normalizations.json") -> str:
    """Normalize text before sending to GPT using word-boundary replacements."""
    path = PROMPTS_DIR / filename
    if not path.exists():
        return text

    with open(path, "r", encoding="utf-8-sig") as f:
        mapping = json.load(f)

    for src, dst in mapping.items():
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text, flags=re.IGNORECASE)
    return text


def see_first_scene(df):
    print("Printing first captioned scene:")
    print("{")
    for key in df[0]:
        if key == "frames":
            continue
        print(f"{key}, {df[0][key]},")
    print("}")


def see_scenes_cuts(df):
    print_prefixed("(PysceneDetect)", f"Found {len(df)} scenes.")
    for idx, s in enumerate(df):
        scene_index = s.get("scene_index", idx)
        scene_label = (
            f"{int(scene_index):03d}"
            if isinstance(scene_index, (int, float))
            else str(scene_index)
        )
        start_tc = s.get("start_timecode") or format_timecode(s.get("start_seconds"))
        end_tc = s.get("end_timecode") or format_timecode(s.get("end_seconds"))
        print_prefixed(
            "(PysceneDetect)",
            f"Scene {scene_label}: {start_tc} -> {end_tc} ({s['duration_seconds']:.2f} sec)",
            indent=4,
        )


def is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception indicates an API rate limit error."""
    err_text = f"{type(exc).__name__}: {exc}".lower()
    return any(m in err_text for m in (
        "429", "rate limit", "ratelimit", "too many requests",
        "quota exceeded", "resource exhausted", "request rate",
    ))
