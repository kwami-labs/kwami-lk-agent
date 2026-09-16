.PHONY: help install dev create deploy test test-unit test-contract test-integration test-cov test-e2e lint format typecheck check clean

help:
	@echo "Kwami LiveKit Agent - Development Commands"
	@echo ""
	@echo "Agent Commands:"
	@echo "  make install          - Install agent dependencies (incl. dev tools)"
	@echo "  make dev              - Run agent locally (dev mode)"
	@echo "  make deploy           - Deploy agent to LiveKit Cloud"
	@echo ""
	@echo "Testing:"
	@echo "  make test             - Run the offline suite (unit + contract + integration)"
	@echo "  make test-unit        - Unit tests only"
	@echo "  make test-contract    - Contract tests only (our assumptions vs the installed SDKs)"
	@echo "  make test-integration - Integration tests only"
	@echo "  make test-cov         - Offline suite with a coverage report"
	@echo "  make test-e2e         - End-to-end tests against REAL providers (needs .env keys)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             - Run linter"
	@echo "  make format           - Format code"
	@echo "  make typecheck        - Run mypy"
	@echo "  make check            - lint + typecheck + test (what CI runs)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            - Remove caches and virtual envs"

# =============================================================================
# Agent
# =============================================================================

install:
	cd agent && uv sync --extra dev

# The package is `src` (see agent/Dockerfile), not `agent`.
dev:
	cd agent && uv run python -m src.main dev

create:
	cd agent && lk agent create .

deploy:
	cd agent && lk agent deploy

# =============================================================================
# Testing
# =============================================================================

test:
	cd agent && uv run python -m pytest tests/ -v

test-unit:
	cd agent && uv run python -m pytest tests/unit/ -v

test-contract:
	cd agent && uv run python -m pytest tests/contract/ -v

test-integration:
	cd agent && uv run python -m pytest tests/integration/ -v

test-cov:
	cd agent && uv run python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

# `live` tests are deselected by default in pyproject; -m live opts back in.
test-e2e:
	cd agent && uv run python -m pytest tests/e2e/ -v -m live

# =============================================================================
# Code Quality
# =============================================================================

lint:
	cd agent && uv run ruff check . && uv run ruff format --check .

format:
	cd agent && uv run ruff format . && uv run ruff check --fix .

typecheck:
	cd agent && uv run mypy

check: lint typecheck test

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf agent/.venv agent/__pycache__ agent/**/__pycache__
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
