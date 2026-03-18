"""
audio_speech.py - Per-scene WAV transcription via Whisper/Azure API.

Provides extract_speech_asr_api for the vlms_light and vlms_heavy pipelines.
Reads a WAV file, transcribes with Azure Whisper API (or local fallback), returns (text, timings).
"""

import numpy as np
import scipy.io.wavfile as wav


def extract_speech_asr_api(wav_path: str, enable_logs: bool = False) -> tuple:
    """
    Transcribe a single WAV file (e.g. scene audio) using Whisper or Azure API.

    Returns:
        (speech_text: str, timings: dict)
    """
    sr, audio_int = wav.read(wav_path)
    if audio_int.ndim > 1:
        audio_int = audio_int.mean(axis=1)
    # Normalize to float32 [-1, 1]; handle int16 and int32
    if audio_int.dtype in (np.int16, np.int32):
        scale = 32768.0 if audio_int.dtype == np.int16 else 2147483648.0
        audio_np = (audio_int.astype(np.float32) / scale)
    else:
        audio_np = audio_int.astype(np.float32)

    if len(audio_np) < sr * 0.1:  # < 0.1 s
        return "", {"total_time_sec": 0, "transcribe_time_sec": 0}

    from src.audio_whisper_parallel import transcribe_full_video

    use_api = True  # Prefer Azure when credentials exist
    result = transcribe_full_video(
        audio_np,
        sr,
        model_size="base",
        use_vad=False,
        force_cpu=True,
        debug=enable_logs,
        silero_model=None,
        get_speech_ts_fn=None,
        use_api=use_api,
        language=None,
    )

    text = result.get("full_text", "").strip()
    timings = {
        "total_time_sec": result.get("total_time_sec", 0),
        "transcribe_time_sec": result.get("transcribe_time_sec", 0),
    }
    return text, timings
