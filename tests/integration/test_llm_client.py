"""Integration tests for kairos.llm.client.

Verifies that LLM client construction works for each supported provider
(Gemini, OpenAI, Claude) when the required environment variables are set,
and that all concrete clients satisfy the LLMClient protocol.
"""

import os

import pytest

from kairos.llm.client import (
    ClaudeLLMClient,
    GeminiLLMClient,
    OpenAILLMClient,
    build_llm_client,
)

pytestmark = pytest.mark.integration


def test_build_llm_client_gemini() -> None:
    """Verify that build_llm_client('gemini') returns a valid GeminiLLMClient."""
    if not os.getenv("GEMINI_PROJECT"):
        pytest.skip("GEMINI_PROJECT not set")
    client = build_llm_client("gemini")
    assert client is not None
    assert isinstance(client, GeminiLLMClient)
    assert isinstance(client.model, str)
    assert len(client.model) > 0


def test_build_llm_client_openai() -> None:
    """Verify that build_llm_client('openai') returns a valid OpenAILLMClient."""
    if not os.getenv("OPENAI_KEY"):
        pytest.skip("OPENAI_KEY not set")
    client = build_llm_client("openai")
    assert client is not None
    assert isinstance(client, OpenAILLMClient)
    assert isinstance(client.model, str)


def test_build_llm_client_claude() -> None:
    """Verify that build_llm_client('claude') returns a valid ClaudeLLMClient."""
    if not os.getenv("CLAUDE_PROJECT") and not os.getenv("GEMINI_PROJECT"):
        pytest.skip("CLAUDE_PROJECT / GEMINI_PROJECT not set")
    client = build_llm_client("claude")
    assert client is not None
    assert isinstance(client, ClaudeLLMClient)
    assert isinstance(client.model, str)


def test_llm_client_protocol() -> None:
    """Verify concrete clients satisfy the LLMClient protocol (generate + model)."""
    assert isinstance(GeminiLLMClient, type)
    assert isinstance(OpenAILLMClient, type)
    assert isinstance(ClaudeLLMClient, type)
    # Protocol structural check — each class has generate() and model
    for cls in (GeminiLLMClient, OpenAILLMClient, ClaudeLLMClient):
        assert hasattr(cls, "generate")
        assert hasattr(cls, "model")
