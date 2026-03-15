.PHONY: lint format test test-unit test-integration test-e2e build clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Quality ──────────────────────────────────────────────────────────

lint: ## Run Ruff linter (check only)
	ruff check src/ tests/

format: ## Auto-format code with Ruff
	ruff format src/ tests/
	ruff check --fix src/ tests/

# ── Testing ──────────────────────────────────────────────────────────

test: ## Run all tests (unit + integration + model + e2e)
	pytest tests/ -v

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests (requires API keys)
	pytest tests/integration/ -v -m integration

test-e2e: ## Run end-to-end pipeline test (requires GPU + API keys)
	pytest tests/e2e/ -v -m e2e

test-model: ## Run model tests (requires GPU / model weights)
	pytest tests/model/ -v -m model

# ── Build ────────────────────────────────────────────────────────────

build: ## Build the package (sdist + wheel)
	python -m build

clean: ## Remove build artifacts, caches, and temp files
	rm -rf build/ dist/ src/*.egg-info .ruff_cache/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
