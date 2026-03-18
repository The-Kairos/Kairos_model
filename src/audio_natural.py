"""
audio_natural.py - Per-scene AST (natural sounds) classification.

Provides extract_sounds for the vlms_light and vlms_heavy pipelines.
Runs a full audio scan, then AST per-scene, mutating scenes with audio_natural.
"""

from src.audio_detector import scan_audio
from src.audio_MIT_ast_parallel import extract_sounds_optimized


def extract_sounds(video_path: str, scenes: list, debug: bool = False) -> None:
    """
    Add audio_natural (AST labels) to each scene. Mutates scenes in place.

    Runs scan_audio to get scan_result, then extract_sounds_optimized.
    """
    scan_result = scan_audio(video_path, scenes, target_sr=16000, debug=debug)
    extract_sounds_optimized(
        scenes,
        scan_result,
        target_sr=16000,
        max_workers=4,
        use_processes=False,
        force_cpu=True,
        debug=debug,
    )
