
SHELL=/bin/bash

.PHONY: venv install install-release lint format check test clean help

# maturin installs polars-hash, not uv: skip the project, and don't prune it.
venv:  ## Sync virtual environment from uv.lock
	uv sync --no-install-project --inexact

install: venv ## Install polars-hash in develop mode
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop

install-release: venv ## Install polars-hash in release mode
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop --release

lint: venv ## Run linter
	unset CONDA_PREFIX && \
	source .venv/bin/activate && ruff check --fix . && cargo check

format: venv ## Run formatters
	unset CONDA_PREFIX && \
	source .venv/bin/activate && ruff format . && cargo fmt

check: venv ## Verify types, lint and formatting without editing files
	unset CONDA_PREFIX && \
	source .venv/bin/activate && \
	mypy polars_hash && \
	ruff check --diff . && \
	ruff format --check --diff

test: install ## Run pytests
	unset CONDA_PREFIX && \
	source .venv/bin/activate && pytest tests/

clean: ## Clean venv and clean cargo
	-@rm -r .venv
	-@cargo clean

help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort
