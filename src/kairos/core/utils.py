"""Shared utility functions.

Printing, timecodes, prompt loading, normalization, retry, and helpers.

This module collects small, stateless helpers that are used across the
Kairos pipeline.  Nothing here depends on heavy ML libraries so it can
be imported early without pulling in PyTorch, OpenCV, etc.

Functions
---------
print_section
    Print a section header surrounded by separator lines.
print_prefixed
    Print a message with a bracketed prefix and optional indent.
format_timecode
    Convert seconds to ``HH:MM:SS.mmm`` timecode strings.
load_prompt
    Load a prompt template from the ``prompts/`` package directory.
apply_gpt_normalization
    Apply word-boundary find-and-replace rules before sending text to GPT.
see_first_scene
    Debug-print the first captioned scene from a scene list.
see_scenes_cuts
    Print a summary of detected scene cuts.
is_rate_limit_error
    Detect API rate-limit errors from exception text.
retry_with_backoff
    Call a callable with exponential backoff on retryable errors.
flatten
    Flatten a list whose elements may themselves be lists or tuples.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

SECTION_LINE: str = "=" * 40
"""Separator line used by :func:`print_section`."""

PROMPTS_DIR: Path = Path(__file__).resolve().parents[1] / "prompts"
"""Absolute path to the ``kairos/prompts/`` directory."""


def print_section(title: str) -> None:
    """Print a section header surrounded by separator lines.

    Outputs three lines to *stdout*: a separator, the *title*, and another
    separator.  Useful for visually splitting pipeline log output.

    Args:
        title: The section heading text to display.

    Example::

        >>> print_section("Frame Sampling")
        ========================================
        Frame Sampling
        ========================================
    """
    print(SECTION_LINE)
    print(title)
    print(SECTION_LINE)


def print_prefixed(prefix: str, message: str, indent: int = 0) -> None:
    """Print a message with a bracketed prefix and optional indent.

    Args:
        prefix: A short tag printed before the message, e.g.
            ``"(Debug)"`` or ``"(PysceneDetect)"``.
        message: The message body to display.
        indent: Number of leading spaces inserted between the prefix and
            the message.  Defaults to ``0``.

    Example::

        >>> print_prefixed("(Info)", "Processing complete", indent=4)
        (Info)     Processing complete
    """
    pad: str = " " * indent
    print(f"{prefix} {pad}{message}")


def format_timecode(seconds: float | None) -> str:
    """Format a duration in seconds as an ``HH:MM:SS.mmm`` timecode string.

    Args:
        seconds: The number of seconds to format.  If ``None`` or not
            convertible to a float, the function returns a placeholder
            string ``"??:??:??.???"``.

    Returns:
        A zero-padded timecode string, e.g. ``"00:01:23.456"``.

    Examples::

        >>> format_timecode(83.456)
        '00:01:23.456'
        >>> format_timecode(None)
        '??:??:??.???'
    """
    if seconds is None:
        return "??:??:??.???"
    try:
        ms_total: int = round(float(seconds) * 1000)
    except (TypeError, ValueError):
        return "??:??:??.???"
    sec_total, ms = divmod(ms_total, 1000)
    mins_total, sec = divmod(sec_total, 60)
    hrs, mins = divmod(mins_total, 60)
    return f"{hrs:02d}:{mins:02d}:{sec:02d}.{ms:03d}"


def load_prompt(filename: str) -> str:
    """Load a prompt template from the ``prompts/`` package directory.

    Args:
        filename: Name of the template file inside ``kairos/prompts/``
            (e.g. ``"scene_description.txt"``).

    Returns:
        The full text content of the prompt file (UTF-8).

    Raises:
        FileNotFoundError: If the requested file does not exist.
    """
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def apply_gpt_normalization(
    text: str, filename: str = "gpt_normalizations.json"
) -> str:
    """Normalize text before sending to GPT using word-boundary replacements.

    Loads a JSON mapping of ``{source: replacement}`` pairs from
    :data:`PROMPTS_DIR` and applies each substitution with
    case-insensitive word-boundary matching.

    Args:
        text: The input text to normalize.
        filename: Name of the JSON mapping file inside the prompts
            directory.  Defaults to ``"gpt_normalizations.json"``.

    Returns:
        The normalized text with all matching words replaced.  If the
        mapping file does not exist, *text* is returned unchanged.
    """
    path: Path = PROMPTS_DIR / filename
    if not path.exists():
        return text

    with open(path, encoding="utf-8-sig") as f:
        mapping: dict[str, str] = json.load(f)

    for src, dst in mapping.items():
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text, flags=re.IGNORECASE)
    return text


def see_first_scene(df: list[dict[str, Any]]) -> None:
    """Debug-print the key-value pairs of the first captioned scene.

    Iterates over the keys of ``df[0]``, skipping ``"frames"``
    (which contains large NumPy arrays), and prints each key-value pair
    using :func:`print_prefixed`.

    Args:
        df: A list of scene dictionaries.  Only the first element
            (``df[0]``) is printed.
    """
    print_prefixed("(Debug)", "Printing first captioned scene:")
    print_prefixed("(Debug)", "{")
    for key in df[0]:
        if key == "frames":
            continue
        print_prefixed("(Debug)", f"{key}, {df[0][key]},")
    print_prefixed("(Debug)", "}")


def see_scenes_cuts(df: list[dict[str, Any]]) -> None:
    """Print a one-line summary for each detected scene cut.

    For every scene dictionary in *df*, outputs its index, start/end
    timecodes, and duration using the ``(PysceneDetect)`` prefix.

    Args:
        df: A list of scene dictionaries, each expected to contain
            at least ``"duration_seconds"`` and optionally
            ``"scene_index"``, ``"start_timecode"``, ``"end_timecode"``,
            ``"start_seconds"``, and ``"end_seconds"``.
    """
    print_prefixed("(PysceneDetect)", f"Found {len(df)} scenes.")
    for idx, s in enumerate(df):
        scene_index: Any = s.get("scene_index", idx)
        scene_label: str = (
            f"{int(scene_index):03d}"
            if isinstance(scene_index, (int, float))
            else str(scene_index)
        )
        start_tc: str = s.get("start_timecode") or format_timecode(
            s.get("start_seconds")
        )
        end_tc: str = s.get("end_timecode") or format_timecode(
            s.get("end_seconds")
        )
        print_prefixed(
            "(PysceneDetect)",
            f"Scene {scene_label}: {start_tc} -> {end_tc} "
            f"({s['duration_seconds']:.2f} sec)",
            indent=4,
        )


def is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception indicates an API rate-limit error.

    Inspects the string representation of *exc* (including its class
    name) for common rate-limit indicators such as HTTP 429, "rate
    limit", "too many requests", and similar phrases.

    Args:
        exc: The caught exception to inspect.

    Returns:
        ``True`` if the exception text matches a known rate-limit
        pattern, ``False`` otherwise.
    """
    err_text: str = f"{type(exc).__name__}: {exc}".lower()
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
    fn: Callable[[], Any],
    *,
    max_retries: int = 3,
    base_sec: float = 2.0,
    is_retryable: Callable[[Exception], bool] | None = None,
    jitter: bool = True,
) -> Any:
    """Call *fn()* with exponential backoff on retryable errors.

    Args:
        fn: Zero-argument callable to attempt.
        max_retries: Maximum number of retry attempts
            (total calls = ``max_retries + 1`` at most).
        base_sec: Base delay in seconds; doubles each attempt.
        is_retryable: Predicate ``(Exception) -> bool`` that decides
            whether a given exception warrants a retry.  Defaults to
            :func:`is_rate_limit_error`.
        jitter: If ``True``, add random jitter (0–1 s) to each sleep
            duration to reduce thundering-herd effects.

    Returns:
        The return value of *fn()* on success.

    Raises:
        Exception: The last exception raised by *fn()* if all attempts
            fail or if the error is not retryable.
    """
    if is_retryable is None:
        is_retryable = is_rate_limit_error
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if not is_retryable(exc) or attempt >= max_retries:
                raise
            sleep_sec: float = base_sec * (2 ** attempt)
            if jitter:
                sleep_sec += random.uniform(0.0, 1.0)
            time.sleep(sleep_sec)
    raise last_exc  # type: ignore[misc]


def flatten(values: list[Any] | None) -> list[Any]:
    """Flatten a list of items which may themselves be lists or tuples.

    Only performs one level of flattening — nested structures deeper
    than one level are not recursively expanded.

    Args:
        values: An iterable of items.  Items that are ``list`` or
            ``tuple`` instances are expanded inline; all other items
            are kept as-is.  If ``None`` or empty, an empty list is
            returned.

    Returns:
        A new flat list containing all individual elements.

    Examples::

        >>> flatten([[1, 2], 3, [4]])
        [1, 2, 3, 4]
        >>> flatten(None)
        []
    """
    flat: list[Any] = []
    if not values:
        return flat
    for value in values:
        if isinstance(value, (list, tuple)):
            flat.extend(value)
        else:
            flat.append(value)
    return flat
