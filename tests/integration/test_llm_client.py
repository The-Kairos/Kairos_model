"""Integration tests for kairos.llm.client."""

import os

import pytest

from kairos.llm.client import build_llm_client, is_gemini_client

pytestmark = pytest.mark.integration


def test_build_llm_client_gemini():
    if not os.getenv("GEMINI_PROJECT"):
        pytest.skip("GEMINI_PROJECT not set")
    client, model_name, deployment = build_llm_client("gemini")
    assert client is not None
    assert isinstance(model_name, str)
    assert len(model_name) > 0
    assert is_gemini_client(client) or hasattr(client, "models")


def test_build_llm_client_openai():
    if not os.getenv("OPENAI_KEY"):
        pytest.skip("OPENAI_KEY not set")
    client, model_name, deployment = build_llm_client("openai")
    assert client is not None
    assert isinstance(model_name, str)
    assert hasattr(client, "chat")


def test_is_gemini_client_with_mock():
    class FakeGemini:
        models = True

    class FakeOpenAI:
        chat = True

    assert is_gemini_client(FakeGemini()) is True
    assert is_gemini_client(FakeOpenAI()) is False
