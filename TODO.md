# TODO

## Open Items

### Authentication

- [ ] **Add API key authentication for Gemini and Claude**
  Currently, both Gemini and Claude are accessed exclusively through Google Cloud Vertex AI
  using Application Default Credentials (service account). Users without a GCP service
  account cannot use these backends. Implement an alternative API key path:
  - **Gemini:** Support `GEMINI_API_KEY` env var → `genai.Client(api_key=...)` (drop `vertexai=True`)
  - **Claude:** Support `ANTHROPIC_API_KEY` env var → `anthropic.Anthropic(api_key=...)` (instead of `AnthropicVertex`)
  - Auto-detect: if API key env var is set, use direct API; otherwise fall back to Vertex AI ADC
  - Update `build_llm_client()` in `src/kairos/llm/client.py`
  - Update `get_embedding_client()` for Gemini embeddings
  - Update docs (README, configuration.md, models.md) to reflect both auth paths
