"""Audio processing subpackage for the Kairos video-understanding pipeline.

This package contains all audio-related modules used during the Kairos
analysis pipeline.  It handles the full lifecycle of audio data — from
raw extraction and pre-scanning through classification, transcription,
and post-processing.

Submodules
----------
classifier
    Audio Set Tagging (AST) classification for labelling non-speech
    audio events (e.g. music, applause, ambient noise).
extraction
    FFmpeg-based extraction of audio tracks from video files into
    temporary WAV segments for downstream processing.
language
    Language identification utilities used to select the correct
    Whisper model or to tag the detected spoken language.
prescan
    Lightweight pre-scan pass that computes per-scene audio statistics
    (RMS energy, spectral features) to decide which scenes contain
    meaningful audio before invoking heavier models.
rms
    Root-mean-square (RMS) energy computation on raw audio waveforms,
    used to detect silence and estimate loudness.
spectral
    Spectral-feature extraction (e.g. spectral centroid, bandwidth)
    for audio characterization and filtering.
text_filter
    Post-transcription text cleanup — removes filler words, normalises
    whitespace, and applies project-specific substitution rules.
transcription
    High-level transcription orchestration that coordinates Whisper
    calls, merges segments, and attaches results to scene objects.
vad
    Voice Activity Detection (VAD) wrappers used to locate speech
    regions within an audio track before transcription.
whisper_api
    Thin client around the OpenAI Whisper API for cloud-based
    speech-to-text when local inference is not available.

Usage
-----
Import individual submodules as needed::

    from kairos.audio import transcription, classifier

Or access them via the package after importing it::

    import kairos.audio
    kairos.audio.prescan.run(scene_list, audio_path)

All public submodule names are listed in ``__all__`` so that
``from kairos.audio import *`` works predictably (though explicit
imports are preferred).
"""

from __future__ import annotations

__all__ = [
    "classifier",
    "extraction",
    "language",
    "prescan",
    "rms",
    "spectral",
    "text_filter",
    "transcription",
    "vad",
    "whisper_api",
]
