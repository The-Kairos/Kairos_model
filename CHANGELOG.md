# Changelog

All notable changes to Kairos will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-03-15

First structured release — transforms the research prototype into an installable,
testable, and documented Python package. **No changes to core ML pipeline logic.**

### Added

- **Package structure** — `src/` layout with `pyproject.toml`, CLI entry points
  (`kairos`, `kairos-download`, `kairos-report`, `kairos-compare`), and domain
  sub-packages (`audio/`, `video/`, `llm/`, `cli/`, `core/`).
- **Configuration system** — `PipelineConfig` dataclass with validation, four
  presets (`default`, `fast`, `motion`, `static`), and `.env` support.
- **LLM client abstraction** — Protocol-based `LLMClient` with implementations
  for Gemini (Vertex AI), OpenAI/Azure, and Claude (Vertex AI).
- **Checkpoint & redo system** — JSON-based pipeline resumability with transitive
  and non-transitive redo (`--redo` / `--redo-only`).
- **Test suite** — 16 unit tests, 2 integration tests, 3 model tests, and 1 e2e
  test, structured with pytest markers (`unit`, `integration`, `model`, `e2e`).
- **CI/CD** — GitHub Actions workflows for Ruff linting and pytest across Python
  3.10/3.11/3.12 matrix, with concurrency groups.
- **Documentation** — 9 doc pages: architecture, pipeline, configuration, CLI,
  RAG, API reference, models, benchmarks, monitoring.
- **Makefile** — Standard `make lint`, `make test`, `make build` targets.
- **Pre-commit hooks** — Ruff lint/format, trailing whitespace, YAML/TOML checks.

### Changed

- **SOLID/DRY/KISS refactoring** — Extracted Protocol abstractions, consolidated
  retry logic, deduplicated prompts and parsing, simplified functions.
- **Strict linting** — Enforced Ruff rules (E, F, I, W, UP, B, SIM, RUF, D, ANN)
  with Google-style docstrings and full type annotations.
- **`from __future__ import annotations`** — Added to all source modules for
  modern annotation evaluation.
- **Repository layout** — Moved model weights to `models/`, pipeline outputs to
  `data/processed/`, consolidated logs.

### Removed

- Flat `requirements.txt` (replaced by `pyproject.toml` dependencies).
- Stale tracked files (`tmp_whisper/` artifacts, raw log JSONs).
- Legacy entry points consolidated into CLI sub-commands.
