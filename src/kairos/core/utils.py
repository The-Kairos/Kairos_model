"""Shared utility functions.

Printing, timecodes, prompt loading, normalization, retry, and helpers.
"""

import json
import random
import re
import time
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


def apply_gpt_normalization(
    text: str, filename: str = "gpt_normalizations.json"
) -> str:
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
    print_prefixed("(Debug)", "Printing first captioned scene:")
    print_prefixed("(Debug)", "{")
    for key in df[0]:
        if key == "frames":
            continue
        print_prefixed("(Debug)", f"{key}, {df[0][key]},")
    print_prefixed("(Debug)", "}")


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
            f"Scene {scene_label}: {start_tc} -> {end_tc} "
            f"({s['duration_seconds']:.2f} sec)",
            indent=4,
        )


def is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception indicates an API rate limit error."""
    err_text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        m in err_text
        for m in (
            "429",
            "rate limit",
            "ratelimit",
            "too many requests",
            "quota exceeded",
            "resource exhausted",
            "request rate",
        )
    )


def retry_with_backoff(
    fn,
    *,
    max_retries: int = 3,
    base_sec: float = 2.0,
    is_retryable=None,
    jitter: bool = True,
):
    """Call *fn()* with exponential backoff on retryable errors.

    Args:
        fn: Zero-argument callable to attempt.
        max_retries: Maximum number of retry attempts
            (total calls = max_retries + 1 at most).
        base_sec: Base delay in seconds; doubles each attempt.
        is_retryable: Predicate ``(Exception) -> bool``.
            Defaults to ``is_rate_limit_error``.
        jitter: Add random jitter (0-1 s) to the sleep time.

    Returns:
        The return value of *fn()* on success.

    Raises:
        The last exception if all attempts fail or the error is not retryable.
    """
    if is_retryable is None:
        is_retryable = is_rate_limit_error
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if not is_retryable(exc) or attempt >= max_retries:
                raise
            sleep_sec = base_sec * (2**attempt)
            if jitter:
                sleep_sec += random.uniform(0.0, 1.0)
            time.sleep(sleep_sec)
    raise last_exc


def flatten(values) -> list:
    """Flatten a list of items which may themselves be lists/tuples.

    >>> flatten([[1, 2], 3, [4]])
    [1, 2, 3, 4]
    """
    flat: list = []
    if not values:
        return flat
    for value in values:
        if isinstance(value, (list, tuple)):
            flat.extend(value)
        else:
            flat.append(value)
    return flat
