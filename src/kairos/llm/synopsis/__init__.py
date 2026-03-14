"""Video synopsis orchestration.

Scene summarization and structured synopsis generation.
"""

from kairos.llm.synopsis.synthesis import (  # noqa: F401
    call_gpt,
    call_gpt_safe,
    summarize_scenes,
    synthesize_synopsis,
)

__all__ = ["call_gpt", "call_gpt_safe", "summarize_scenes", "synthesize_synopsis"]
