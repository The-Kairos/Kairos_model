"""Kairos exception hierarchy.

All custom exceptions inherit from :class:`KairosError` so callers can
catch the entire family with a single ``except KairosError`` clause.
"""


class KairosError(Exception):
    """Base exception for all Kairos errors."""


class KairosConfigError(KairosError):
    """Invalid or inconsistent pipeline configuration."""


class KairosModelError(KairosError):
    """Failure loading or running an ML model (BLIP, YOLO, AST, Whisper …)."""


class KairosLLMError(KairosError):
    """Failure calling an external LLM API (OpenAI, Gemini, Claude)."""


class KairosIOError(KairosError):
    """File / network I/O error (missing video, bad checkpoint, etc.)."""


class KairosRAGError(KairosError):
    """Failure during RAG embedding, retrieval, or answer generation."""
