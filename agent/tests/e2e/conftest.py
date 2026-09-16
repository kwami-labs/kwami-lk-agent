"""Fixtures for the real-provider end-to-end suite.

These tests are marked `live`: they cost money, need network, and are excluded
from the default run by `addopts = -m 'not live'`. CI runs them nightly and on
main, never as a PR gate, because real providers are nondeterministic and a
flaky gate is worse than no gate.

`.env` is loaded here rather than in the root conftest so that only this suite
sees real credentials -- the offline suites deliberately run with every
provider variable stripped.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

# Load before any fixture runs, so the autouse `isolated_env` exemption for
# `live` tests has real values to leave in place. The file lives at the repo
# root, one level above the `agent/` package.
_HERE = Path(__file__).resolve()
for _candidate in (_HERE.parents[3] / ".env", _HERE.parents[2] / ".env"):
    if _candidate.is_file():
        load_dotenv(_candidate)
        break

# Cheapest model that still reliably drives tool calls.
E2E_LLM_MODEL = "gpt-4o-mini"


def require_env(*names: str) -> dict[str, str]:
    """Skip the test unless every named credential is present."""
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        pytest.skip(f"live test needs {', '.join(missing)}")
    return {name: os.environ[name] for name in names}


@pytest.fixture
def openai_key() -> str:
    return require_env("OPENAI_API_KEY")["OPENAI_API_KEY"]


@pytest.fixture
def zep_key() -> str:
    return require_env("ZEP_API_KEY")["ZEP_API_KEY"]


@pytest.fixture
def real_llm(openai_key: str) -> Any:
    """A real LLM, used both to drive the agent and to judge its answers."""
    from livekit.plugins import openai as openai_plugin

    return openai_plugin.LLM(model=E2E_LLM_MODEL, api_key=openai_key)


@pytest.fixture
def judge_llm(openai_key: str) -> Any:
    from livekit.plugins import openai as openai_plugin

    return openai_plugin.LLM(model=E2E_LLM_MODEL, api_key=openai_key)
