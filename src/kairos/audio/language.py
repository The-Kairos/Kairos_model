"""Language detection using Whisper."""

from __future__ import annotations

import gc

import numpy as np
import whisper

from kairos.core.utils import print_prefixed


def detect_languages(
    audio: np.ndarray, sr: int, speech_regions: list[dict], debug: bool = False
) -> dict[str, object]:
    """Detect spoken languages in audio using OpenAI Whisper.

    Samples up to 10 speech regions, runs Whisper's language detection
    on each 30-second (max) chunk, and tallies the results. The
    Whisper ``"base"`` model is loaded on CPU, used for detection, and
    then explicitly deleted to free memory.

    Args:
        audio: 1-D float array containing the full audio waveform.
        sr: Sampling rate of ``audio`` in Hz.
        speech_regions: List of dicts with ``"start_sec"`` and
            ``"end_sec"`` keys identifying speech segments (as returned
            by :func:`kairos.audio.vad.detect_speech_regions`). An
            empty list causes an early return with ``primary_language``
            set to ``None``.
        debug: If ``True``, emit a summary of detected languages via
            :func:`~kairos.core.utils.print_prefixed`.
            Defaults to ``False``.

    Returns:
        A dictionary with the following keys:

        * **primary_language** (*str | None*) – ISO 639-1 code of the
          most frequently detected language, or ``None`` if no speech
          regions were provided or none were long enough to analyse.
        * **detected_languages** (*dict[str, int]*) – Mapping of
          language code to the number of sampled regions in which it
          was detected.
        * **is_multilingual** (*bool*) – ``True`` when at least one
          non-primary language was detected in two or more regions.
    """
    if not speech_regions:
        return {
            "primary_language": None,
            "detected_languages": {},
            "is_multilingual": False,
        }
    model = whisper.load_model("base", device="cpu")
    sample_indices = np.linspace(
        0, len(speech_regions) - 1, min(10, len(speech_regions)), dtype=int
    )
    detected_counts: dict[str, int] = {}
    for idx in sample_indices:
        region = speech_regions[idx]
        start = int(region["start_sec"] * sr)
        end = min(start + 30 * sr, int(region["end_sec"] * sr))
        if end - start < sr:
            continue
        chunk = whisper.pad_or_trim(audio[start:end])
        mel = whisper.log_mel_spectrogram(chunk).to("cpu")
        _, probs = model.detect_language(mel)
        lang = max(probs, key=probs.get)
        detected_counts[lang] = detected_counts.get(lang, 0) + 1
    del model
    gc.collect()

    if not detected_counts:
        return {
            "primary_language": None,
            "detected_languages": {},
            "is_multilingual": False,
        }
    sorted_langs = sorted(detected_counts.items(), key=lambda x: x[1], reverse=True)
    primary: str = sorted_langs[0][0]
    is_multilingual: bool = any(count >= 2 for _, count in sorted_langs[1:])
    if debug:
        print_prefixed(
            "(AudioDetector)",
            f"Languages: {detected_counts}, "
            f"Primary: {primary}, "
            f"Multilingual: {is_multilingual}",
        )
    return {
        "primary_language": primary,
        "detected_languages": detected_counts,
        "is_multilingual": is_multilingual,
    }
