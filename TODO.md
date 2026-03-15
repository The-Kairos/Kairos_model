# TODO

Comprehensive backlog for making Kairos production-ready and deployable.
Items are grouped by priority and domain.

---

## 🔴 Critical — Deployment Blockers

### Containerisation ([Issue #8](https://github.com/The-Kairos/Kairos_model/issues/8))

- [ ] **Parallelize** 
- [ ] **Add `Dockerfile`** — multi-stage build with conda env, CUDA runtime, and
  pip-installed Kairos package. Target: reproducible GPU-enabled image.
- [ ] **Add `docker-compose.yml`** — service definition with NVIDIA runtime /
  GPU passthrough, volume mounts for `data/`, and `.env` file binding.
- [ ] Add `make docker-build` and `make docker-run` targets to `Makefile` once
  Docker files exist.

### Repository Hygiene ([Issue #9](https://github.com/The-Kairos/Kairos_model/issues/9), [Issue #10](https://github.com/The-Kairos/Kairos_model/issues/10))

- [ ] **Track `.env.example` in git** — the `.gitignore` exclusion (`!.env.example`)
  is now in place; ensure the file is staged and committed so new clones get it.
- [ ] **Remove large binary `models/yolov8s.pt` from git history** (#9) — either:
  - Move to Git LFS (`git lfs track "models/*.pt"`), or
  - Add a download-on-first-run script (preferred for open-source distribution),
    and gitignore model weights.
- [ ] **Remove `src/kairos/tmp_whisper/` and evaluate `var/`** (#10) — empty temp
  directory tracked in git. Whisper temp files should be created at runtime in a
  system temp dir or configurable path, not shipped in the package.

---

## 🟡 Medium — Adoption & Quality

### Authentication ([Issue #7](https://github.com/The-Kairos/Kairos_model/issues/7))

- [ ] **Add API key authentication for Gemini and Claude**
  Currently, both Gemini and Claude are accessed exclusively through Google Cloud
  Vertex AI using Application Default Credentials (service account). Users without
  a GCP service account cannot use these backends.
  - **Gemini:** Support `GEMINI_API_KEY` env var → `genai.Client(api_key=...)` (drop `vertexai=True`)
  - **Claude:** Support `ANTHROPIC_API_KEY` env var → `anthropic.Anthropic(api_key=...)` (instead of `AnthropicVertex`)
  - Auto-detect: if API key env var is set, use direct API; otherwise fall back to Vertex AI ADC
  - Update `build_llm_client()` in `src/kairos/llm/client.py`
  - Update `get_embedding_client()` for Gemini embeddings
  - Update docs (README, `docs/configuration.md`, `docs/models.md`) to reflect both auth paths

### Operational

- [ ] **Add `kairos health` CLI command** — verify GPU availability, model weights
  present, LLM backend connectivity, disk space, and Python environment.
- [ ] **Wire `kairos --version`** — `__version__` exists in `__init__.py`; add
  `--version` flag to the CLI argument parser.

### Testing

- [ ] **Increase unit test coverage** — target ≥80% line coverage; add
  `pytest-cov` reporting to CI (`make test-cov`).
- [ ] **Add smoke tests** — lightweight tests that verify imports, config
  validation, and CLI `--help` work without GPU or API keys.

### Security

- [ ] **Add dependency audit to CI** — run `pip-audit` or `safety check` in a
  GitHub Actions step to catch known vulnerabilities.

---

## 🟢 Nice-to-Have — Professional Polish

### Documentation

- [ ] **Add `CONTRIBUTING.md`** — contribution guidelines, branch naming, commit
  conventions, PR template, and local development setup.
- [ ] **Add deployment guide** — step-by-step instructions for Docker, GCP VM,
  and bare-metal GPU deployment.
- [ ] **Add PR and issue templates** — `.github/ISSUE_TEMPLATE/` and
  `.github/pull_request_template.md` for consistent project governance.

### Observability

- [ ] **Structured JSON logging** — add an optional `--log-format json` flag for
  machine-parseable logs (useful for production monitoring).
- [ ] **Pipeline metrics export** — expose step timings and resource usage as
  JSON or Prometheus-compatible metrics.

### Performance

- [ ] **Benchmark & profile pipeline stages** — identify bottlenecks in scene
  detection, BLIP captioning, YOLO inference, and LLM calls.
- [ ] **Batch LLM calls** — evaluate batching scene descriptions to reduce
  API round-trips and improve throughput.

### Code Quality

- [ ] **Remove legacy `src/kairos/main.py`** — consolidate into `cli/app.py` or
  clearly document its role vs. `__main__.py`.
- [ ] **Evaluate `var/` directory** — empty tracked directory; remove or document
  its intended purpose.

---

## ✅ Completed

- [X] Restructure into `src/` layout with `pyproject.toml` packaging
- [X] Domain sub-packages (`audio/`, `video/`, `llm/`, `cli/`, `core/`)
- [X] Protocol-based LLM client abstraction (Gemini, OpenAI, Claude)
- [X] `PipelineConfig` dataclass with validation and presets
- [X] Checkpoint & redo system for pipeline resumability
- [X] Comprehensive pytest test suite (unit / integration / model / e2e)
- [X] GitHub Actions CI (Ruff lint + pytest matrix 3.10–3.12)
- [X] SOLID / DRY / KISS refactoring passes
- [X] Full type annotations and Google-style docstrings
- [X] `from __future__ import annotations` across all modules
- [X] 9-page documentation (architecture, pipeline, config, CLI, RAG, etc.)
- [X] `Makefile` with `lint`, `format`, `test`, `build`, `clean` targets
- [X] `.pre-commit-config.yaml` (Ruff + file hygiene hooks)
- [X] `CHANGELOG.md` for v0.1.0
- [X] Close duplicate GitHub issue #6
- [X] Fix `.gitignore` to allow `.env.example` and `.pre-commit-config.yaml`
