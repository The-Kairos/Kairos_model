"""Kairos: automated video understanding pipeline."""

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
