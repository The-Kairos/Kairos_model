"""Core infrastructure: checkpoint I/O, logging, pipeline orchestration, redo, utilities.

New modules added by the enhancement pass:
- :mod:`kairos.core.exceptions` — custom exception hierarchy
- :mod:`kairos.core.models` — thread-safe ML model registry
- :mod:`kairos.core.scene` — typed ``Scene`` dataclass
- :mod:`kairos.core.timing` — ``@timed_stage`` decorator and timing reports
"""

from kairos.core.exceptions import (  # noqa: F401
    KairosConfigError,
    KairosError,
    KairosIOError,
    KairosLLMError,
    KairosModelError,
    KairosRAGError,
)
from kairos.core.models import ModelRegistry  # noqa: F401
from kairos.core.scene import Scene  # noqa: F401

__all__ = [
    "KairosError",
    "KairosConfigError",
    "KairosIOError",
    "KairosLLMError",
    "KairosModelError",
    "KairosRAGError",
    "ModelRegistry",
    "Scene",
]
