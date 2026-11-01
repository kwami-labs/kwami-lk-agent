"""The voice never searches: the brain does, with Tavily when Gemini cannot.

Two rules from the user (2026-10-09, enforced 2026-10-18):

* With the brain on, every search goes through `delegate_task`. The voice's own
  `web_search`, `deep_research` and `product_search` are withheld and the
  prompt says to delegate. With the brain off they come back.
* The `search` role is Gemini with Google Search. When it fails -- its quota
  answered 429 on 2026-10-18 -- the search runs on Tavily, written up by Claude
  or GPT, instead of failing or paying for a second native search.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from src.brain import Bridge
from src.brain.config import BrainConfig, BrainModel, default_brain
from src.brain.ledger import Ledger
from src.brain.providers import BrainAnswer, BrainRequest
from src.brain.router import can_delegate, tavily_writer
from src.domain.prompt import SEARCH_DELEGATION_NOTE
from src.domain.tool_gating import BRAIN_SEARCH_TOOLS
from src.settings import Settings, set_settings

BRAIN_KEYS = {"ANTHROPIC_API_KEY": "a", "GOOGLE_API_KEY": "g"}

#: Credentials for every search tool, so only the brain can withhold them.
SEARCH_CREDENTIALS = {"tavily_api_key": "t", "serpapi_key": "s"}


def _names(agent: Any) -> set[str]:
    return {t.info.name for t in agent.tools}


# -- the brain decides whether the voice searches ----------------------------------


def test_the_brain_delegates_with_a_remote_model_only() -> None:
    config = default_brain()
    assert can_delegate(config, BRAIN_KEYS)
    assert not can_delegate(config, {})
    config.enabled = False
    assert not can_delegate(config, BRAIN_KEYS)
    # Only the local executor: it writes code, it does not search.
    local_only = BrainConfig(models=[BrainModel("cc", "local", "claude-sonnet-5-5", "code")])
    assert not can_delegate(local_only, BRAIN_KEYS)


def test_with_the_brain_on_the_voice_has_no_search_tool() -> None:
    from src.agent import KwamiAgent

    set_settings(Settings(**SEARCH_CREDENTIALS, provider_keys=dict(BRAIN_KEYS)))
    try:
        agent = KwamiAgent()
        names = _names(agent)
        assert not (BRAIN_SEARCH_TOOLS & names)
        assert "delegate_task" in names
        assert SEARCH_DELEGATION_NOTE in agent.instructions
        assert "web_search" not in {_tool.info.name for _tool in agent.available_builtin_tools()}
    finally:
        set_settings(None)


def test_without_a_brain_the_voice_keeps_searching() -> None:
    from src.agent import KwamiAgent

    set_settings(Settings(**SEARCH_CREDENTIALS))
    try:
        agent = KwamiAgent()
        assert _names(agent) >= BRAIN_SEARCH_TOOLS
        assert SEARCH_DELEGATION_NOTE not in agent.instructions
    finally:
        set_settings(None)


class _BridgeStub:
    def __init__(self, delegates: bool) -> None:
        self.on = delegates

    def delegates(self) -> bool:
        return self.on


async def test_switching_the_brain_off_hands_searching_back() -> None:
    from src.agent import KwamiAgent

    set_settings(Settings(**SEARCH_CREDENTIALS, provider_keys=dict(BRAIN_KEYS)))
    try:
        agent = KwamiAgent()
        assert not (BRAIN_SEARCH_TOOLS & _names(agent))

        bridge = _BridgeStub(delegates=False)
        agent._bridge = lambda: bridge  # type: ignore[method-assign]
        assert await agent.refresh_brain_gating() is True
        assert _names(agent) >= BRAIN_SEARCH_TOOLS
        assert SEARCH_DELEGATION_NOTE not in agent.instructions

        # Nothing changed: no second update.
        assert await agent.refresh_brain_gating() is False

        bridge.on = True
        assert await agent.refresh_brain_gating() is True
        assert not (BRAIN_SEARCH_TOOLS & _names(agent))
        assert SEARCH_DELEGATION_NOTE in agent.instructions
    finally:
        set_settings(None)


# -- the search role falls back to Tavily ------------------------------------------


def test_the_tavily_writer_is_claude_or_gpt_never_gemini() -> None:
    config = default_brain()
    writer = tavily_writer(config, BRAIN_KEYS, "research", "simple", frozenset({"gemini-flash"}))
    assert writer is not None and writer.provider == "anthropic"
    only_gemini = BrainConfig(models=[m for m in config.models if m.provider == "google"])
    assert tavily_writer(only_gemini, BRAIN_KEYS, "research", "simple") is None


class _Session:
    user_state = "listening"
    agent_state = "listening"

    def __init__(self) -> None:
        self.replies: list[str] = []

    def generate_reply(self, *, instructions: str, allow_interruptions: bool = True) -> None:
        self.replies.append(instructions)

    async def interrupt(self) -> None:
        return None


class _Publisher:
    async def publish(self, payload: dict[str, Any]) -> bool:
        return True


def _bridge(tmp_path: Path, tavily: str, gemini_ok: bool) -> tuple[Bridge, list]:
    bridge = Bridge(keys=BRAIN_KEYS, tavily_key=tavily, ledger=Ledger(tmp_path))
    bridge.config.split = False
    calls: list[tuple[str, BrainRequest]] = []

    async def fake_ask(model: BrainModel, request: BrainRequest) -> BrainAnswer:
        calls.append((model.id, request))
        if model.provider == "google" and not gemini_ok:
            return BrainAnswer(
                "La cuota de Gemini está agotada ahora mismo.", model.id, model.model, ok=False
            )
        return BrainAnswer(f"respuesta de {model.id}", model.id, model.model, 0.01)

    bridge._ask = fake_ask  # type: ignore[method-assign]
    bridge.attach(_Session(), _Publisher())
    return bridge, calls


async def _run(bridge: Bridge) -> Any:
    task = bridge.start(
        task="noticias de hoy", title="Noticias", kind="research", difficulty="simple"
    )
    assert not isinstance(task, str)
    assert task.handle is not None
    await asyncio.wait_for(task.handle, timeout=5)
    return task


async def test_a_search_goes_to_gemini_with_google(tmp_path: Path) -> None:
    bridge, calls = _bridge(tmp_path, tavily="t", gemini_ok=True)
    task = await _run(bridge)
    assert task.status == "done"
    assert [c[0] for c in calls] == ["gemini-flash"]
    assert calls[0][1].native_search is True


async def test_a_spent_gemini_quota_searches_on_tavily(tmp_path: Path) -> None:
    bridge, calls = _bridge(tmp_path, tavily="t", gemini_ok=False)
    task = await _run(bridge)

    assert task.status == "done"
    assert [c[0] for c in calls] == ["gemini-flash", "claude-sonnet"]
    rescue = calls[1][1]
    assert rescue.native_search is False, "the writer must not pay for its own search"
    assert rescue.tavily_key == "t"
    assert task.models == ["Gemini 3.5 Flash", "Claude Sonnet 5.5 + Tavily"]


async def test_without_tavily_the_usual_fallback_still_answers(tmp_path: Path) -> None:
    bridge, calls = _bridge(tmp_path, tavily="", gemini_ok=False)
    task = await _run(bridge)
    assert task.status == "done"
    assert [c[0] for c in calls] == ["gemini-flash", "claude-sonnet"]
    assert calls[1][1].tavily_key == ""
