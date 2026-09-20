"""Shared pytest configuration for the Kwami agent test suite.

Deliberately does NOT stub out `livekit` or `zep_cloud`.

The previous version of this file replaced both packages with ``MagicMock`` and
redefined ``livekit.agents.Agent`` as an empty class. Against mocks, any
disagreement between this codebase and the installed SDK passes silently --
which is exactly how a wrong ``on_enter`` signature, a hook that does not
exist, a tool list that gets wiped, and five nonexistent Zep methods all
shipped to production. Both packages are real dependencies and are installed;
tests import them for real so that contract drift shows up as a red build.

Fakes belong to *our* boundaries (the ports), not to the SDKs.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import pytest

# Make `src.*` importable without installing the package.
AGENT_DIR = Path(__file__).parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))


from src.settings import ENV_VAR_NAMES

# Variables the SDKs read directly, which `Settings` therefore never sees. They
# still have to be cleared: a stray LIVEKIT_URL reaches livekit-agents at import
# time, with no Settings field to hold it.
SDK_ENV_VARS = (
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
)

# Every variable the code reads from the environment. Cleared by default so a
# developer's exported keys can never change a test outcome -- `test_default_config`
# used to fail on any machine with ZEP_API_KEY set.
#
# Derived from `Settings.ENV_VAR_NAMES` rather than restated, because restating
# it is how the two lists drifted: this tuple was hand-maintained and missed
# BROWSERBASE_API_KEY, BROWSERBASE_PROJECT_ID, ELEVENLABS_API_KEY,
# KWAMI_ALLOW_BROWSER_JS and KWAMI_BROWSER_PROVIDER. With those exported the
# suite reported `2 failed, 1905 passed` -- and one of the two was
# `test_js_execution_is_refused_by_default`, so whether the browser's
# JavaScript security default held was a property of the developer's shell.
PROVIDER_ENV_VARS = tuple(sorted({*ENV_VAR_NAMES, *SDK_ENV_VARS}))


@pytest.fixture(autouse=True)
def reset_settings():
    """Drop the cached Settings around every test.

    Settings are resolved once per process and memoised. Without this, the
    first test to touch them would freeze that snapshot for the whole run, and
    a test that sets an environment variable would have no effect.
    """
    from src.settings import set_settings

    set_settings(None)
    yield
    set_settings(None)


@pytest.fixture(autouse=True)
def reset_shared_http():
    """Drop the process-wide HTTP pool around every test.

    `adapters.http.shared_client` is a module-level singleton, so without this a
    client built under one test's respx mock would be reused by the next -- and
    a closed one would resurrect as a real client pointed at the internet.
    """
    import src.adapters.http as http_module

    http_module._shared = None
    yield
    http_module._shared = None


@pytest.fixture(autouse=True)
def reset_logging_config():
    """Undo any logging configuration a test installs.

    `configure_logging` wraps the record factory and sets a formatter on every
    root handler -- including pytest's own caplog handler. Without this, one
    test switching to JSON changed how every later test's records were
    formatted, which is exactly the kind of order-dependent failure that is
    miserable to diagnose.
    """
    factory = logging.getLogRecordFactory()
    root = logging.getLogger()
    handlers = root.handlers[:]
    formatters = [(h, h.formatter) for h in handlers]
    yield
    logging.setLogRecordFactory(factory)
    root.handlers = handlers
    for handler, formatter in formatters:
        handler.setFormatter(formatter)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Remove every provider credential from the environment.

    Tests that need a key set one explicitly. `live` tests are exempt: they are
    supposed to talk to real providers.
    """
    if request.node.get_closest_marker("live"):
        return
    for name in PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def all_tools_available():
    """Credentials for every gated built-in, so all forty are registered.

    `KwamiAgent` withholds built-ins whose credential is absent (see
    `domain/tool_gating.py`), and the offline suite strips every credential --
    so by default an agent built in a test has 21 tools, not 40. Tests about
    *client tools not displacing built-ins* need the full set to be meaningful;
    they are not tests about gating, and should not accidentally become them.
    """
    from src.settings import Settings, set_settings

    set_settings(
        Settings(
            serpapi_key="test-key-not-real",
            tavily_api_key="test-key-not-real",
            zep_api_key="test-key-not-real",
            browserbase_api_key="test-key-not-real",
            browserbase_project_id="test-project",
        )
    )
    yield
    set_settings(None)


@pytest.fixture
def env_setting(monkeypatch: pytest.MonkeyPatch):
    """Set an environment variable and make it take effect immediately.

    Settings are memoised per process, so a bare `monkeypatch.setenv` is
    invisible to anything that has already resolved them. This sets the
    variable and drops the cache.
    """
    from src.settings import set_settings

    def _set(name: str, value: str | None) -> None:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
        set_settings(None)

    return _set


@pytest.fixture
def fake_key(monkeypatch: pytest.MonkeyPatch):
    """Set one provider credential to a syntactically valid dummy value."""

    def _set(*names: str, value: str = "test-key-not-real") -> None:
        for name in names:
            monkeypatch.setenv(name, value)

    return _set


@pytest.fixture
def live_credentials() -> dict[str, str]:
    """Credentials for `live` tests, skipping the test when they are absent."""
    required = ("OPENAI_API_KEY",)
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.skip(f"live test needs {', '.join(missing)}")
    return {name: os.environ[name] for name in required}


class RecordingRoom:
    """Minimal stand-in for `rtc.Room` that records published data messages.

    Only models the surface the agent actually uses, so a change in what the
    agent expects from a room shows up here rather than being absorbed by a mock.
    """

    def __init__(self, identity: str = "agent-test") -> None:
        self.published: list[dict[str, Any]] = []
        self.local_participant = _LocalParticipant(identity, self.published)
        self.remote_participants: dict[str, Any] = {}
        self.disconnected = False

    async def disconnect(self) -> None:
        self.disconnected = True


class _LocalParticipant:
    def __init__(self, identity: str, sink: list[dict[str, Any]]) -> None:
        self.identity = identity
        self._sink = sink

    async def publish_data(self, payload: bytes, *args: Any, **kwargs: Any) -> None:
        import json

        self._sink.append(json.loads(payload.decode()))


@pytest.fixture
def room() -> RecordingRoom:
    return RecordingRoom()
