"""Hallucination filtering and text cleaning for Whisper transcription output.

Whisper occasionally produces hallucinated segments — repeated phrases,
emoji-laden strings, or segments with very low confidence.  This module
provides utilities to strip such artefacts and return only plausible
transcription segments.
"""

from __future__ import annotations

import re
import unicodedata


def _strip_emoji_symbols(text: str) -> str:
    """Strip emoji and Unicode symbol characters from *text*.

    Uses the ``emoji`` library when available for comprehensive emoji
    removal; otherwise falls back to filtering out characters in the
    Unicode *So* (Symbol, other) and *Sk* (Symbol, modifier) categories.

    Args:
        text: Input string potentially containing emoji or symbol
            characters.

    Returns:
        A copy of *text* with all emoji and symbol characters removed.
    """
    try:
        import emoji

        text = emoji.replace_emoji(text, replace="")
    except Exception:
        pass
    return "".join(c for c in text if unicodedata.category(c) not in ("So", "Sk"))


def clean_repetitive_text(text: str) -> str:
    """Clean repetitive phrases and words from transcription text.

    Whisper may stutter, producing the same phrase or word multiple times
    in a row.  This function collapses consecutive duplicate phrases
    (delimited by punctuation) and consecutive duplicate words into a
    single occurrence.

    Args:
        text: Raw transcription text that may contain repeated content.

    Returns:
        Cleaned text with consecutive duplicate phrases and words removed.
        Returns the original *text* unchanged when it is empty or falsy.
    """
    if not text:
        return text
    text = re.sub(r"\s+", " ", text).strip()
    phrases: list[str] = re.split(r"([.?!,]+)", text)
    cleaned_phrases: list[str] = []
    last_p: str | None = None
    i: int = 0
    while i < len(phrases):
        p: str = phrases[i]
        punct: str = phrases[i + 1] if i + 1 < len(phrases) else ""
        p_norm: str = p.strip().lower()
        if p_norm:
            if p_norm == last_p:
                if (
                    punct
                    and cleaned_phrases
                    and not re.search(r"[.?!,]$", cleaned_phrases[-1])
                ):
                    cleaned_phrases[-1] = cleaned_phrases[-1].rstrip() + punct
            else:
                cleaned_phrases.append(p.strip() + punct)
                last_p = p_norm
        i += 2
    text = " ".join(cleaned_phrases).strip()
    words: list[str] = text.split()
    if not words:
        return text
    cleaned_words: list[str] = [words[0]]
    for w in words[1:]:
        w_norm: str = w.lower().strip(".,!?")
        last_norm: str = cleaned_words[-1].lower().strip(".,!?")
        if w_norm == last_norm and len(w_norm) > 0:
            if re.search(r"[.,!?]$", w):
                cleaned_words[-1] = (
                    cleaned_words[-1].rstrip(".,!?") + re.search(r"[.,!?]+$", w).group()
                )
        else:
            cleaned_words.append(w)
    return " ".join(cleaned_words)


def filter_hallucinations(
    segments: list[dict], primary_lang: str | None = None
) -> list[dict]:
    """Filter Whisper hallucinated segments from a transcription result.

    A segment is considered hallucinated and dropped when any of the
    following conditions hold:

    * More than 15 % of its characters are "special" (not alphanumeric,
      whitespace, or basic punctuation).
    * Its ``avg_logprob`` is below ``-1.2``.
    * Its ``no_speech_prob`` exceeds ``0.8``.
    * After cleaning, the text is empty or an exact duplicate of a
      previously seen segment.

    Remaining segments have their ``text`` field cleaned via
    :func:`clean_repetitive_text` and :func:`_strip_emoji_symbols`.

    Args:
        segments: List of Whisper segment dicts, each containing at least
            a ``"text"`` key and optionally ``"avg_logprob"`` and
            ``"no_speech_prob"`` keys.
        primary_lang: ISO-639 language code of the expected primary
            language.  Currently reserved for future language-specific
            filtering rules.

    Returns:
        A filtered list of segment dicts with cleaned ``"text"`` values.
    """
    final: list[dict] = []
    seen_texts: set[str] = set()
    for seg in segments:
        text: str = _strip_emoji_symbols(seg["text"].strip())
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        special_count: int = sum(
            1 for c in text if not (c.isalnum() or c.isspace() or c in ".,!?'-")
        )
        if len(text) > 0 and special_count / len(text) > 0.15:
            continue
        if seg.get("avg_logprob", 0) < -1.2:
            continue
        if seg.get("no_speech_prob", 0) > 0.8:
            continue
        text = clean_repetitive_text(text)
        text_lower: str = text.lower().strip(".,!? ")
        if not text_lower or text_lower in seen_texts:
            continue
        if len(text_lower) > 2:
            seen_texts.add(text_lower)
        seg["text"] = text
        final.append(seg)
    return final
