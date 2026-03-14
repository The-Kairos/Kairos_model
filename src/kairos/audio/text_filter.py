"""Hallucination filtering and text cleaning for Whisper transcription output."""

import re
import unicodedata


def _strip_emoji_symbols(text: str) -> str:
    try:
        import emoji
        text = emoji.replace_emoji(text, replace="")
    except Exception:
        pass
    return "".join(c for c in text if unicodedata.category(c) not in ("So", "Sk"))


def clean_repetitive_text(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"\s+", " ", text).strip()
    phrases = re.split(r"([.?!,]+)", text)
    cleaned_phrases = []
    last_p = None
    i = 0
    while i < len(phrases):
        p = phrases[i]
        punct = phrases[i + 1] if i + 1 < len(phrases) else ""
        p_norm = p.strip().lower()
        if p_norm:
            if p_norm == last_p:
                if punct and cleaned_phrases and not re.search(r"[.?!,]$", cleaned_phrases[-1]):
                    cleaned_phrases[-1] = cleaned_phrases[-1].rstrip() + punct
            else:
                cleaned_phrases.append(p.strip() + punct)
                last_p = p_norm
        i += 2
    text = " ".join(cleaned_phrases).strip()
    words = text.split()
    if not words:
        return text
    cleaned_words = [words[0]]
    for w in words[1:]:
        w_norm = w.lower().strip(".,!?")
        last_norm = cleaned_words[-1].lower().strip(".,!?")
        if w_norm == last_norm and len(w_norm) > 0:
            if re.search(r"[.,!?]$", w):
                cleaned_words[-1] = cleaned_words[-1].rstrip(".,!?") + re.search(r"[.,!?]+$", w).group()
        else:
            cleaned_words.append(w)
    return " ".join(cleaned_words)


def filter_hallucinations(segments: list, primary_lang: str = None) -> list:
    final = []
    seen_texts = set()
    for seg in segments:
        text = _strip_emoji_symbols(seg["text"].strip())
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        special_count = sum(1 for c in text if not (c.isalnum() or c.isspace() or c in ".,!?'-"))
        if len(text) > 0 and special_count / len(text) > 0.15:
            continue
        if seg.get("avg_logprob", 0) < -1.2:
            continue
        if seg.get("no_speech_prob", 0) > 0.8:
            continue
        text = clean_repetitive_text(text)
        text_lower = text.lower().strip(".,!? ")
        if not text_lower or text_lower in seen_texts:
            continue
        if len(text_lower) > 2:
            seen_texts.add(text_lower)
        seg["text"] = text
        final.append(seg)
    return final
