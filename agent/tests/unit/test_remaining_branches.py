"""The branches the obvious tests walk past.

Statement coverage is reached long before branch coverage, and what is left
over at the end is rarely uniform. These are the *other* sides of conditions
that the happy path only ever takes one way: a provider list that is already
correct so nothing changes, an optional collaborator that is absent, a response
whose outer shape is wrong rather than its inner one.

They are cheap tests, but they are not decorative. Several pin the "do nothing"
half of a guard, and a guard whose no-op branch is untested is one refactor away
from doing something.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from src.agent import KwamiAgent
from src.domain import KwamiConfig, KwamiSoulConfig, build_system_prompt
from src.handlers.realtime import apply_realtime_fields
from src.runtime.container import AgentDeps
from src.runtime.dispatch import DataMessageRouter
from src.session import SessionState
from src.tools.knowledge import QUOTE_URL, TAVILY_SEARCH_URL


class Ctx:
    def __init__(self, deps: Any = None) -> None:
        self.userdata = deps


class FakeSession:
    def __init__(self) -> None:
        self.agent: Any = None

    def update_agent(self, agent: Any) -> None:
        self.agent = agent


@pytest.fixture
def agent() -> KwamiAgent:
    return KwamiAgent(config=KwamiConfig())


# -- prompt -----------------------------------------------------------------


def test_a_soul_with_nothing_set_gets_no_optional_sections() -> None:
    """Every field in the soul is optional; an empty one must be omitted.

    Each of these is its own branch, and each renders a line the model reads as
    an instruction — an empty "Conversation style:" is worse than no line.
    """
    bare = build_system_prompt(
        KwamiSoulConfig(
            name="K",
            personality="helpful",
            traits=[],
            conversation_style="",
            emotional_tone="",  # type: ignore[arg-type]
            response_length="",  # type: ignore[arg-type]
        )
    )

    assert "Key traits" not in bare
    assert "Conversation style" not in bare


# -- realtime ---------------------------------------------------------------


def test_resending_the_model_already_in_use_changes_nothing() -> None:
    """The frontend re-syncs on connect; an unchanged value must not rebuild."""
    voice = KwamiConfig().voice
    voice.realtime_provider = "openai"
    voice.realtime_model = "gpt-realtime"

    changed = apply_realtime_fields(voice, {"realtime_model": "openai/gpt-realtime"})

    assert changed == set(), "an unchanged model was reported as a change"


# -- dispatch ---------------------------------------------------------------


async def test_a_tool_result_is_routed_to_the_agent(agent: KwamiAgent) -> None:
    """The client's reply to a `tool_call`; without this route it hangs 30s."""
    import asyncio

    config = KwamiConfig()
    config.tools = [
        {"name": "set_ui_control", "description": "x", "parameters": {"type": "object"}}
    ]
    live = KwamiAgent(config=config)
    pending: asyncio.Future = asyncio.Future()
    live.client_tools.pending_calls["call-1"] = pending

    state = SessionState(current_agent=live)
    router = DataMessageRouter(session=FakeSession(), state=state, room=None)

    handled = router.handle(
        {"type": "tool_result", "toolCallId": "call-1", "result": json.dumps({"ok": True})}
    )

    assert handled == "tool_result"
    assert pending.done(), "the waiting tool call was never resolved"


# -- session ----------------------------------------------------------------


async def test_a_shared_memory_is_not_closed_on_a_swap() -> None:
    """Closing it would break the *incoming* agent, which is still using it."""

    class Memory:
        def __init__(self) -> None:
            self.closed = False
            self.tracker: Any = None

        async def close(self) -> None:
            self.closed = True

        def set_usage_tracker(self, tracker: Any) -> None:
            self.tracker = tracker

    import asyncio

    memory = Memory()
    old = KwamiAgent(config=KwamiConfig(), memory=memory)
    new = KwamiAgent(config=KwamiConfig(), memory=memory)
    state = SessionState(current_agent=old)

    state.prepare_handoff(new)
    await asyncio.gather(*state._cleanup_tasks, return_exceptions=True)

    assert not memory.closed, "the shared memory was closed under the new agent"


async def test_a_provider_with_no_close_at_all_is_skipped() -> None:
    """Not every object hung off the agent is closable."""

    class Opaque:
        pass

    agent = KwamiAgent(config=KwamiConfig(), tts=Opaque())
    state = SessionState()

    # The assertion is that this returns rather than raising.
    await state._cleanup_agent_voice_pipeline(agent, wait_for_release=False)


async def test_cleanup_of_an_agent_without_memory() -> None:
    """Memory is optional; teardown must not assume it."""
    state = SessionState(current_agent=KwamiAgent(config=KwamiConfig()))
    state.user_identity = "user-1"
    state.room_name = "room-1"

    await state.cleanup()


# -- knowledge --------------------------------------------------------------


@respx.mock
async def test_research_skips_results_that_are_not_objects(agent: KwamiAgent, env_setting) -> None:
    """Provider output is untrusted; a string in the results list must not raise."""
    env_setting("TAVILY_API_KEY", "tvly-test")
    respx.post(TAVILY_SEARCH_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "answer": "An answer.",
                "results": ["not an object", {"title": "T", "url": "https://a.com/1"}],
            },
        )
    )

    result = await agent.deep_research(None, "fusion power")

    assert "An answer." in result


@respx.mock
async def test_research_keeps_a_result_with_no_text(agent: KwamiAgent, env_setting) -> None:
    """A bare URL is still a source, even with nothing to quote from it."""
    env_setting("TAVILY_API_KEY", "tvly-test")
    respx.post(TAVILY_SEARCH_URL).mock(
        return_value=httpx.Response(
            200,
            json={"answer": "An answer.", "results": [{"url": "https://a.com/1"}]},
        )
    )

    result = await agent.deep_research(None, "fusion power")

    assert "from 1 sources" in result


@respx.mock
async def test_research_without_a_memory_writer(agent: KwamiAgent, env_setting) -> None:
    """`_remember_in_background` comes from another mixin and may be absent."""
    env_setting("TAVILY_API_KEY", "tvly-test")
    respx.post(TAVILY_SEARCH_URL).mock(
        return_value=httpx.Response(
            200,
            json={"answer": "An answer.", "results": [{"title": "T", "url": "https://a.com/1"}]},
        )
    )
    agent._remember_in_background = "not callable"  # type: ignore[assignment]

    assert "An answer." in await agent.deep_research(None, "fusion power")


def test_publishing_research_without_a_publisher(agent: KwamiAgent) -> None:
    """A publish failure must not kill the tool that produced the answer."""
    agent._publisher = "not callable"  # type: ignore[assignment]

    agent._publish_research(Ctx(), "a topic", [{"title": "T", "url": "https://a.com"}])


async def test_publishing_research_creates_the_task_set(agent: KwamiAgent) -> None:
    """The set is lazily created; without it the task could be collected mid-flight."""
    published: list[dict[str, Any]] = []

    class Publisher:
        async def publish(self, message: dict[str, Any]) -> bool:
            published.append(message)
            return True

    agent._publisher = lambda context=None: Publisher()  # type: ignore[assignment]
    del agent._background_tasks

    agent._publish_research(Ctx(), "a topic", [{"title": "T", "url": "https://a.com"}])

    assert agent._background_tasks, "the task was left without a strong reference"
    import asyncio

    await asyncio.gather(*agent._background_tasks, return_exceptions=True)
    assert published


@pytest.mark.parametrize(
    "payload",
    [["not", "a", "dict"], {"chart": {"result": ["not a dict"]}}],
    ids=["outer-is-a-list", "first-result-is-a-string"],
)
@respx.mock
async def test_a_quote_response_with_the_wrong_outer_shape(agent: KwamiAgent, payload: Any) -> None:
    """The endpoint is undocumented; the *container* can be wrong too."""
    respx.get(QUOTE_URL.format(symbol="TSLA")).mock(return_value=httpx.Response(200, json=payload))

    quote = await agent.get_market_quote(None, "TSLA")

    assert "error" in quote
    assert "price" not in quote


# -- media ------------------------------------------------------------------


async def test_now_playing_without_a_panel(agent: KwamiAgent) -> None:
    async def _none() -> Any:
        return None

    agent._get_browser_session = _none  # type: ignore[method-assign]

    info = await agent.get_now_playing(None)

    assert info["playing"] is False
    assert "no browser panel" in info["reason"]


async def test_now_playing_with_braces_but_invalid_json(agent: KwamiAgent) -> None:
    """A payload that looks extractable and then will not parse."""

    class Session:
        is_active = True

        async def evaluate_js(self, expression: str) -> str:
            return "Result: {'title': 'single quotes are not JSON'}"

    async def _get() -> Any:
        return Session()

    agent._get_browser_session = _get  # type: ignore[method-assign]

    info = await agent.get_now_playing(None)

    assert info["playing"] is False
    assert "could not read the player" in info["reason"]


# -- pipeline control -------------------------------------------------------


async def test_changing_realtime_model_within_the_same_provider_keeps_the_voice() -> None:
    """The voice is only reset when the *provider* changes, not the model."""
    from src.runtime.reconfigure import Reconfigurator

    config = KwamiConfig()
    config.voice.pipeline_type = "realtime"
    config.voice.realtime_provider = "openai"
    config.voice.realtime_model = "gpt-realtime"
    config.voice.realtime_voice = "cedar"

    class FakeRealtime:
        model = "gpt-realtime"

        def session(self) -> Any: ...
        def update_options(self, **kwargs: Any) -> None: ...

    agent = KwamiAgent(config=config, llm=FakeRealtime())
    state = SessionState(current_agent=agent)

    def create_agent_fn(cfg, vad, memory=None, skip_greeting=False):
        return KwamiAgent(config=cfg, llm=FakeRealtime())

    deps = AgentDeps(
        reconfigure=Reconfigurator(state=state, vad=None, create_agent_fn=create_agent_fn)
    )

    new_agent, _ = await agent.change_ai_model(Ctx(deps), "gpt-realtime-mini")

    assert new_agent.kwami_config.voice.realtime_voice == "cedar", "the voice was reset needlessly"


async def test_pipeline_status_without_a_model_name(agent: KwamiAgent) -> None:
    """Some plugins do not expose `.model`; the report must still be produced."""

    class Opaque:
        pass

    quiet = KwamiAgent(config=KwamiConfig(), llm=Opaque())

    status = await quiet.get_pipeline_status(Ctx(AgentDeps()))

    assert "model" not in status["running"]
    assert status["can_reconfigure"] is False


# -- trading ----------------------------------------------------------------


async def test_preparing_a_trade_without_a_quote_tool() -> None:
    """`get_market_quote` comes from another mixin and may be absent."""
    from src.domain import KwamiConfig as Config

    agent = KwamiAgent(config=Config())
    agent.get_market_quote = "not callable"  # type: ignore[assignment]

    summary = await agent.prepare_trade(None, "TSLA", "buy", 10)

    assert summary["order"] == "buy 10.0 TSLA at market"
    assert "warning" in summary


async def test_submitting_without_a_browser_getter(
    agent: KwamiAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No browser mixin at all is the same answer as no browser open."""

    async def quote(context: Any, symbol: str) -> dict[str, Any]:
        return {"price": 100.0}

    monkeypatch.setattr(agent, "get_market_quote", quote)
    summary = await agent.prepare_trade(None, "TSLA", "buy", 10)
    agent._get_browser_session = "not callable"  # type: ignore[assignment]

    result = await agent.submit_trade(None, summary["confirmation_code"])

    assert "isn't open" in result


# -- the last few branches --------------------------------------------------


async def test_a_provider_whose_close_is_a_coroutine() -> None:
    """Some providers offer an async `close()` and no `aclose()` at all."""
    closed: list[str] = []

    class AsyncCloseOnly:
        async def close(self) -> None:
            closed.append("tts")

    agent = KwamiAgent(config=KwamiConfig(), tts=AsyncCloseOnly())
    state = SessionState()

    await state._cleanup_agent_voice_pipeline(agent, wait_for_release=False)

    assert closed == ["tts"], "an async close() was called without being awaited, or skipped"


async def test_cleanup_with_no_agent_still_reports_usage() -> None:
    """A session can end before any agent is installed; the usage is still real."""

    class Reporter:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def report(self, *, user_id: str, session_id: str, tracker: Any) -> bool:
            self.calls.append(user_id)
            return True

    reporter = Reporter()
    state = SessionState(current_agent=None)
    state.user_identity = "user-1"
    state.room_name = "room-1"
    state.usage_reporter = reporter  # type: ignore[assignment]
    state.usage_tracker.record_external_usage("tool", "test/tool", units_used=1.0)

    await state.cleanup()

    assert reporter.calls == ["user-1"], "usage was dropped because there was no agent"


@respx.mock
async def test_research_shows_one_card_per_publisher(agent: KwamiAgent, env_setting) -> None:
    """Two different pages from one outlet is one source, not two.

    Distinct from the identical-URL case: these are separate articles, so the
    URL guard does not catch them. Four angles on one topic return the same few
    outlets repeatedly, and a panel of duplicates reads as breadth the research
    does not have.
    """
    env_setting("TAVILY_API_KEY", "tvly-test")
    respx.post(TAVILY_SEARCH_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "answer": "An answer.",
                "results": [
                    {"title": "First", "url": "https://nature.com/a", "content": "one"},
                    {"title": "Second", "url": "https://nature.com/b", "content": "two"},
                ],
            },
        )
    )

    result = await agent.deep_research(None, "fusion power")

    assert "from 1 sources" in result, "two articles from one outlet counted as two sources"
    assert "Second" in result, "the second article's text was dropped along with its card"


def test_publishing_no_research_sources_does_nothing(agent: KwamiAgent) -> None:
    """An empty result set must not put an empty panel on screen."""
    published: list[dict[str, Any]] = []

    class Publisher:
        async def publish(self, message: dict[str, Any]) -> bool:  # pragma: no cover
            published.append(message)
            return True

    agent._publisher = lambda context=None: Publisher()  # type: ignore[assignment]

    agent._publish_research(Ctx(), "a topic", [])

    assert published == []


# -- the mixins composed differently ----------------------------------------
#
# `MediaToolsMixin` and `TradingToolsMixin` both reach for `navigate_to`, which
# `AgentToolsMixin` supplies. These guards were carrying `# pragma: no cover`
# with the reason "the mixin is always combined" — true today, and the wrong
# use of a pragma. The composition is our own code, not the environment, so the
# branch is reachable from a test and the pragma was excluding a real path from
# the report rather than acknowledging an unreachable one.


async def test_playing_media_without_the_navigation_mixin() -> None:
    """A refusal, not an AttributeError, if the mixins are ever composed apart."""
    agent = KwamiAgent(config=KwamiConfig())
    agent.navigate_to = None  # type: ignore[assignment]

    result = await agent.play_media(None, "a song", kind="music")

    assert "can't open the browser panel" in result


async def test_opening_a_trade_ticket_without_the_navigation_mixin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def quote(context: Any, symbol: str) -> dict[str, Any]:
        return {"price": 100.0}

    agent = KwamiAgent(config=KwamiConfig())
    monkeypatch.setattr(agent, "get_market_quote", quote)
    await agent.prepare_trade(None, "TSLA", "buy", 10)
    agent.navigate_to = None  # type: ignore[assignment]

    result = await agent.open_trade_ticket(None, broker="trading212")

    assert "can't open the browser panel" in result
