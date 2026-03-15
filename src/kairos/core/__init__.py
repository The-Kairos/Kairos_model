"""Core infrastructure.

Checkpoint I/O, logging, pipeline orchestration, redo, utilities.

New modules added by the enhancement pass:
- :mod:`kairos.core.exceptions` — custom exception hierarchy
- :mod:`kairos.core.models` — thread-safe ML model registry
- :mod:`kairos.core.scene` — typed ``Scene`` dataclass
- :mod:`kairos.core.timing` — ``@timed_stage`` decorator and timing reports
"""

from kairos.core.exceptions import (
    KairosConfigError,
    KairosError,
    KairosIOError,
    KairosLLMError,
    KairosModelError,
    KairosRAGError,
)
from kairos.core.models import ModelRegistry
from kairos.core.scene import Scene

__all__ = [
    "KairosConfigError",
    "KairosError",
    "KairosIOError",
    "KairosLLMError",
    "KairosModelError",
    "KairosRAGError",
    "ModelRegistry",
    "Scene",
]
