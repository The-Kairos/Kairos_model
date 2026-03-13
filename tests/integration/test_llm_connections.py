"""Integration tests for LLM API connections.

Converted from benchmarks/gpt4o_connection.py and benchmarks/gemini_connection.py.
"""

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def openai_env():
    endpoint = os.getenv("OPENAI_ENDPOINT")
    deployment = os.getenv("OPENAI_DEPLOYMENT")
    api_key = os.getenv("OPENAI_KEY")
    if not all([endpoint, deployment, api_key]):
        pytest.skip("OpenAI env vars not set (OPENAI_ENDPOINT, OPENAI_DEPLOYMENT, OPENAI_KEY)")
    return endpoint, deployment, api_key


@pytest.fixture
def gemini_env():
    project = os.getenv("GEMINI_PROJECT")
    if not project:
        pytest.skip("GEMINI_PROJECT env var not set")
    return project


def test_openai_connection(openai_env):
    from openai import OpenAI

    endpoint, deployment, api_key = openai_env
    client = OpenAI(base_url=endpoint, api_key=api_key)
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one word."},
        ],
        max_completion_tokens=50,
        model=deployment,
    )
    content = response.choices[0].message.content
    assert content is not None
    assert len(content.strip()) > 0


def test_gemini_connection(gemini_env):
    from google import genai

    project = gemini_env
    location = os.getenv("GEMINI_LOCATION", "us-central1")
    client = genai.Client(vertexai=True, project=project, location=location)
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    chat = client.chats.create(model=model)
    response = chat.send_message("Say hello in one word.")
    assert response.text is not None
    assert len(response.text.strip()) > 0
