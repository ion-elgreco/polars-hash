
# maturin refuses to run with both VIRTUAL_ENV and CONDA_PREFIX set, and
# `uv run` always sets VIRTUAL_ENV. Dropping it here beats `unset` in every
# recipe: it needs no shell builtin, so the recipes also run on Windows.
unexport CONDA_PREFIX

.PHONY: venv install install-release lint format check test clean \
	docs-build docs-serve docs-deploy docs-set-default help

# maturin installs polars-hash, not uv: skip the project, and don't prune it.
venv:  ## Sync virtual environment from uv.lock
	uv sync --no-install-project --inexact

install: venv ## Install polars-hash in develop mode
	uv run --no-sync maturin develop

install-release: venv ## Install polars-hash in release mode
	uv run --no-sync maturin develop --release

lint: venv ## Run linter
	uv run --no-sync ruff check --fix .
	cargo clippy --all-targets

format: venv ## Run formatters
	uv run --no-sync ruff format .
	cargo fmt

check: venv ## Verify types, lint and formatting without editing files
	uv run --no-sync mypy polars_hash
	uv run --no-sync ruff check --diff .
	uv run --no-sync ruff format --check --diff
	cargo fmt --check
	cargo clippy --all-targets -- -D warnings

test: install ## Run pytests
	uv run --no-sync pytest tests/

clean: ## Clean venv and clean cargo
	-@rm -r .venv
	-@cargo clean

# uvx runs the docs toolchain in a throwaway environment, so no `docs-*` target
# depends on `venv`.
DOCS_CONFIG := ../mkdocs.yml

docs-build: ## Build the documentation site into ../site
	uvx zensical build -f $(DOCS_CONFIG) --strict

docs-serve: ## Serve the documentation with live reload on :8000
	uvx zensical serve -f $(DOCS_CONFIG)

# squidfunk's fork: zensical's versioning provider expects fixes that haven't
# landed upstream yet. Bump when zensical updates its versioning guidance.
MIKE_PKG := git+https://github.com/squidfunk/mike.git@2d4ad799442f4592db8ad53b179bfb33db8c69ac
MIKE := uvx --from "$(MIKE_PKG)" --with zensical mike

# Pass PUSH= for a local dry-run that leaves gh-pages unpushed.
PUSH ?= --push

docs-deploy: ## Deploy versioned docs to gh-pages (VERSION=x.y.z)
	@test -n "$(VERSION)" || { echo "VERSION is required, e.g. make docs-deploy VERSION=0.6.1"; exit 1; }
	$(MIKE) deploy $(PUSH) --update-aliases -F $(DOCS_CONFIG) $(VERSION) latest

docs-set-default: ## Point the gh-pages root redirect at `latest`
	$(MIKE) set-default $(PUSH) -F $(DOCS_CONFIG) latest

help: SHELL=/bin/bash
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort
