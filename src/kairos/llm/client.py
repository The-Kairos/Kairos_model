"""LLM client abstraction: Protocol + Gemini/OpenAI/Claude implementations."""

from __future__ import annotations

import os
import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Unified interface for LLM generation."""

    @property
    def model(self) -> str: ...

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str: ...


class GeminiLLMClient:
    """Gemini via Vertex AI (google-genai SDK)."""

    def __init__(self, client, model: str):
        self._client = client
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        contents = prompt
        if system:
            contents = f"{system}\n\n{prompt}"
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
        )
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned empty content")
        return text


class OpenAILLMClient:
    """OpenAI / Azure OpenAI (openai SDK)."""

    def __init__(self, client, model: str):
        self._client = client
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        system_msg = system or "You are a precise and reliable assistant."
        kwargs = dict(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            model=self._model,
        )
        model_ver = re.search(r"(\d+)", self._model or "")
        if model_ver and int(model_ver.group(1)) >= 5:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
            kwargs["top_p"] = 1.0
        response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    value = item.get("text")
                    if isinstance(value, str):
                        parts.append(value)
            text = "".join(parts).strip()
        else:
            text = ""
        if not text:
            raise RuntimeError("OpenAI returned empty content")
        return text


class ClaudeLLMClient:
    """Claude via Vertex AI (anthropic[vertex] SDK)."""

    def __init__(self, client, model: str):
        self._client = client
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        kwargs = dict(
            model=self._model,
            max_tokens=max_tokens,
            messages=messages,
        )
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        text = response.content[0].text.strip()
        if not text:
            raise RuntimeError("Claude returned empty content")
        return text


def build_llm_client(llm: str | None = None) -> LLMClient:
    """Build an LLMClient from environment variables.

    Args:
        llm: "gemini", "openai", or "claude" to force a backend.
             None falls back to LLM_BACKEND env var (default: "openai").

    Returns:
        An LLMClient instance.
    """
    backend = (llm or os.getenv("LLM_BACKEND", "openai")).lower()

    if backend == "gemini":
        from google import genai
        project = os.getenv("GEMINI_PROJECT", "prj-udst-prod-oussama-1")
        location = os.getenv("GEMINI_LOCATION", "us-central1")
        client = genai.Client(vertexai=True, project=project, location=location)
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return GeminiLLMClient(client, model)

    if backend == "claude":
        from anthropic import AnthropicVertex
        region = os.getenv("CLAUDE_LOCATION", "us-east5")
        project = os.getenv("CLAUDE_PROJECT", os.getenv("GEMINI_PROJECT", "prj-udst-prod-oussama-1"))
        client = AnthropicVertex(region=region, project_id=project)
        model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
        return ClaudeLLMClient(client, model)

    # default: openai
    from openai import OpenAI
    endpoint = os.getenv("OPENAI_ENDPOINT")
    api_key = os.getenv("OPENAI_KEY")
    client = OpenAI(base_url=endpoint, api_key=api_key)
    model = os.getenv("OPENAI_MODEL", os.getenv("OPENAI_DEPLOYMENT", "gpt-4o"))
    return OpenAILLMClient(client, model)
