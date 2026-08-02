
# maturin refuses to run with both VIRTUAL_ENV and CONDA_PREFIX set, and
# `uv run` always sets VIRTUAL_ENV. Dropping it here beats `unset` in every
# recipe: it needs no shell builtin, so the recipes also run on Windows.
unexport CONDA_PREFIX

.PHONY: venv install install-release lint format check test clean help

# maturin installs polars-hash, not uv: skip the project, and don't prune it.
venv:  ## Sync virtual environment from uv.lock
	uv sync --no-install-project --inexact

install: venv ## Install polars-hash in develop mode
	uv run --no-sync maturin develop

install-release: venv ## Install polars-hash in release mode
	uv run --no-sync maturin develop --release

lint: venv ## Run linter
	uv run --no-sync ruff check --fix .
	cargo check

format: venv ## Run formatters
	uv run --no-sync ruff format .
	cargo fmt

check: venv ## Verify types, lint and formatting without editing files
	uv run --no-sync mypy polars_hash
	uv run --no-sync ruff check --diff .
	uv run --no-sync ruff format --check --diff

test: install ## Run pytests
	uv run --no-sync pytest tests/

clean: ## Clean venv and clean cargo
	-@rm -r .venv
	-@cargo clean

help: SHELL=/bin/bash
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort
