"""Language detection using Whisper."""

import gc

import numpy as np
import whisper

from kairos.core.utils import print_prefixed


def detect_languages(
    audio: np.ndarray, sr: int, speech_regions: list, debug: bool = False
) -> dict:
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
    detected_counts = {}
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
    primary = sorted_langs[0][0]
    is_multilingual = any(count >= 2 for _, count in sorted_langs[1:])
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
