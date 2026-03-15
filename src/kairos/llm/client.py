"""LLM client abstraction: Protocol + Gemini/OpenAI/Claude implementations."""

from __future__ import annotations

import os
import re
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Unified interface for LLM generation.

    Any object that exposes a ``model`` property and a ``generate`` method
    with the signature below satisfies this protocol and can be used
    interchangeably throughout the Kairos pipeline.
    """

    @property
    def model(self) -> str:
        """Return the model identifier string.

        Returns:
            str: The model name or deployment ID used by this client.
        """
        ...

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The user prompt to send to the LLM.
            system: An optional system-level instruction prepended to the
                conversation. Defaults to ``None``.
            max_tokens: Maximum number of tokens to generate.
                Defaults to ``2048``.
            temperature: Sampling temperature. Lower values produce more
                deterministic output. Defaults to ``0.3``.

        Returns:
            str: The generated text, stripped of leading/trailing whitespace.
        """
        ...


class GeminiLLMClient:
    """Gemini via Vertex AI (google-genai SDK).

    Wraps a ``google.genai.Client`` instance and delegates generation to
    the Vertex AI Gemini endpoint.

    Attributes:
        _client: The underlying ``google.genai.Client``.
        _model: Gemini model identifier (e.g. ``"gemini-2.5-flash"``).
    """

    def __init__(self, client: Any, model: str) -> None:
        """Initialise the Gemini LLM client.

        Args:
            client: A ``google.genai.Client`` instance configured for
                Vertex AI.
            model: The Gemini model identifier to use for generation.
        """
        self._client = client
        self._model = model

    @property
    def model(self) -> str:
        """Return the Gemini model identifier.

        Returns:
            str: The model name passed at construction time.
        """
        return self._model

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        """Generate text using the Gemini model.

        If a *system* instruction is provided it is prepended to the
        prompt separated by two newlines, since the Gemini
        ``generate_content`` API does not have a dedicated system field.

        Args:
            prompt: The user prompt.
            system: Optional system instruction prepended to the prompt.
                Defaults to ``None``.
            max_tokens: Maximum tokens to generate (currently unused by
                the underlying SDK call but kept for protocol
                compatibility). Defaults to ``2048``.
            temperature: Sampling temperature (currently unused by the
                underlying SDK call but kept for protocol compatibility).
                Defaults to ``0.3``.

        Returns:
            str: The generated text, stripped of whitespace.

        Raises:
            RuntimeError: If the model returns empty content.
        """
        contents: str = prompt
        if system:
            contents = f"{system}\n\n{prompt}"
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
        )
        text: str = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned empty content")
        return text


class OpenAILLMClient:
    """OpenAI / Azure OpenAI (openai SDK).

    Wraps an ``openai.OpenAI`` client and delegates generation to the
    chat completions endpoint.

    Attributes:
        _client: The underlying ``openai.OpenAI`` client.
        _model: Model or deployment name (e.g. ``"gpt-4o"``).
    """

    def __init__(self, client: Any, model: str) -> None:
        """Initialise the OpenAI LLM client.

        Args:
            client: An ``openai.OpenAI`` client instance.
            model: The model or deployment name to use.
        """
        self._client = client
        self._model = model

    @property
    def model(self) -> str:
        """Return the OpenAI model identifier.

        Returns:
            str: The model name passed at construction time.
        """
        return self._model

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        """Generate text using the OpenAI chat completions API.

        For models whose version number is ≥ 5 (e.g. ``o1``, ``o3``),
        only ``max_completion_tokens`` is passed (no ``temperature`` /
        ``top_p``) to comply with the API constraints of reasoning models.

        Args:
            prompt: The user message content.
            system: Optional system message. Defaults to
                ``"You are a precise and reliable assistant."``.
            max_tokens: Maximum number of tokens to generate.
                Defaults to ``2048``.
            temperature: Sampling temperature. Defaults to ``0.3``.

        Returns:
            str: The generated text, stripped of whitespace.

        Raises:
            RuntimeError: If the model returns empty content.
        """
        system_msg: str = system or "You are a precise and reliable assistant."
        kwargs: dict[str, Any] = dict(
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
            text: str = content.strip()
        elif isinstance(content, list):
            parts: list[str] = []
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
    """Claude via Vertex AI (anthropic[vertex] SDK).

    Wraps an ``AnthropicVertex`` client and delegates generation to
    Claude's messages endpoint.

    Attributes:
        _client: The underlying ``AnthropicVertex`` client.
        _model: Claude model identifier (e.g. ``"claude-sonnet-4-6"``).
    """

    def __init__(self, client: Any, model: str) -> None:
        """Initialise the Claude LLM client.

        Args:
            client: An ``anthropic.AnthropicVertex`` client instance.
            model: The Claude model identifier to use.
        """
        self._client = client
        self._model = model

    @property
    def model(self) -> str:
        """Return the Claude model identifier.

        Returns:
            str: The model name passed at construction time.
        """
        return self._model

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        """Generate text using the Claude messages API.

        Args:
            prompt: The user message content.
            system: Optional system instruction. Defaults to ``None``.
            max_tokens: Maximum tokens to generate. Defaults to ``2048``.
            temperature: Sampling temperature (not forwarded to the API
                but kept for protocol compatibility). Defaults to ``0.3``.

        Returns:
            str: The generated text, stripped of whitespace.

        Raises:
            RuntimeError: If the model returns empty content.
        """
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = dict(
            model=self._model,
            max_tokens=max_tokens,
            messages=messages,
        )
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        text: str = response.content[0].text.strip()
        if not text:
            raise RuntimeError("Claude returned empty content")
        return text


def _build_gemini_raw_client() -> Any:
    """Build a raw Gemini ``genai.Client`` from environment variables.

    Uses ``GEMINI_PROJECT`` (default ``"prj-udst-prod-oussama-1"``) and
    ``GEMINI_LOCATION`` (default ``"us-central1"``) from the environment.

    Returns:
        Any: A ``google.genai.Client`` configured for Vertex AI.
    """
    from google import genai

    project: str = os.getenv("GEMINI_PROJECT", "prj-udst-prod-oussama-1")
    location: str = os.getenv("GEMINI_LOCATION", "us-central1")
    return genai.Client(vertexai=True, project=project, location=location)


def get_embedding_client() -> Any:
    """Return a raw Gemini ``genai.Client`` suitable for embedding calls.

    This is a convenience wrapper around :func:`_build_gemini_raw_client`.

    Returns:
        Any: A ``google.genai.Client`` configured for Vertex AI.
    """
    return _build_gemini_raw_client()


def build_llm_client(llm: str | None = None) -> LLMClient:
    """Build an :class:`LLMClient` from environment variables.

    The function selects the appropriate backend and instantiates the
    corresponding client class, reading API keys, endpoints, model names
    and other configuration from the process environment.

    Args:
        llm: One of ``"gemini"``, ``"openai"``, or ``"claude"`` to force
            a backend. When ``None``, falls back to the ``LLM_BACKEND``
            environment variable (default: ``"openai"``).

    Returns:
        LLMClient: A fully-initialised LLM client ready for generation.
    """
    backend: str = (llm or os.getenv("LLM_BACKEND", "openai")).lower()

    if backend == "gemini":
        client = _build_gemini_raw_client()
        model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return GeminiLLMClient(client, model)

    if backend == "claude":
        from anthropic import AnthropicVertex

        region: str = os.getenv("CLAUDE_LOCATION", "us-east5")
        project: str = os.getenv(
            "CLAUDE_PROJECT", os.getenv("GEMINI_PROJECT", "prj-udst-prod-oussama-1")
        )
        client = AnthropicVertex(region=region, project_id=project)
        model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
        return ClaudeLLMClient(client, model)

    # default: openai
    from openai import OpenAI

    endpoint: str | None = os.getenv("OPENAI_ENDPOINT")
    api_key: str | None = os.getenv("OPENAI_KEY")
    client = OpenAI(base_url=endpoint, api_key=api_key)
    model = os.getenv("OPENAI_MODEL", os.getenv("OPENAI_DEPLOYMENT", "gpt-4o"))
    return OpenAILLMClient(client, model)
