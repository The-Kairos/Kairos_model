"""Kairos: automated video understanding pipeline.

Kairos is a platform for long-form video analysis that combines visual
scene detection, frame captioning, object tracking, audio transcription,
and LLM-powered summarization into a single orchestrated pipeline.

This top-level package re-exports the most commonly used symbols so that
downstream code can simply write ``from kairos import PipelineConfig``.
"""

__version__ = "0.1.0"

from kairos.config import PipelineConfig
from kairos.core.exceptions import KairosError
from kairos.core.models import ModelRegistry
from kairos.core.scene import Scene

__all__ = [
    "KairosError",
    "ModelRegistry",
    "PipelineConfig",
    "Scene",
    "__version__",
]
