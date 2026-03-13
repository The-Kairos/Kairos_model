"""LLM client factory: build Gemini or OpenAI client from environment."""

import os


def build_llm_client(llm: str | None = None):
    """Build the LLM client, model name, and deployment from environment variables.

    Args:
        llm: "gemini" or "openai" to force a backend. None falls back to LLM_BACKEND env var.

    Returns:
        (client, model_name, deployment) tuple.
    """
    if llm is not None:
        use_gemini = llm == "gemini"
    else:
        backend = os.getenv("LLM_BACKEND", "openai").lower()
        use_gemini = backend == "gemini"

    if use_gemini:
        from google import genai
        project = os.getenv("GEMINI_PROJECT", "prj-udst-prod-oussama-1")
        location = os.getenv("GEMINI_LOCATION", "us-central1")
        client = genai.Client(vertexai=True, project=project, location=location)
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        deployment = model_name
    else:
        from openai import OpenAI
        endpoint = os.getenv("OPENAI_ENDPOINT")
        deployment = os.getenv("OPENAI_DEPLOYMENT")
        api_key = os.getenv("OPENAI_KEY")
        client = OpenAI(
            base_url=endpoint,
            api_key=api_key,
        )
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o")

    return client, model_name, deployment


def is_gemini_client(client) -> bool:
    """Check if the client is a Gemini client (vs OpenAI)."""
    return hasattr(client, "models") and not hasattr(client, "chat")
