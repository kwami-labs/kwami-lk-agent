"""The refusals, the degradations and the metering that must never raise.

Every branch here is one the happy-path suites do not reach: a malformed
argument, a dead feed, a provider that cannot be built, a usage tracker that
throws. They share one property worth stating once — none of them may cost the
user their turn. A voice agent that raises inside a tool call goes silent, and
silence is indistinguishable from a broken connection.

Metering is the subtler case. It is wrapped in `try/except` everywhere on
purpose: billing is a side effect of a tool call, and a tracker that throws
must lose the *charge*, not the answer the user asked for.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from src.agent import KwamiAgent
from src.domain import KwamiConfig
from src.handlers.realtime import (
    apply_live_realtime_options,
    has_realtime_keys,
    normalize_pipeline_type,
    realtime_model_of,
)
from src.runtime.container import AgentDeps
from src.runtime.reconfigure import Reconfigurator, reconfigurator_from_context
from src.session import SessionState
from src.tools.knowledge import QUOTE_URL, TAVILY_SEARCH_URL, _domain
from src.tools.media import resolve_service
from src.tools.trading import BROKER_URLS


class ExplodingTracker:
    """A usage tracker that fails. Billing must never cost the answer."""

    def record_external_usage(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("metering backend is down")


class FakeRealtimeModel:
    def __init__(self) -> None:
        self.model = "gpt-realtime"
        self.updates: list[dict[str, Any]] = []

    def session(self) -> Any:  # pragma: no cover - presence is the signal
        raise NotImplementedError

    def update_options(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class Ctx:
    def __init__(self, deps: Any) -> None:
        self.userdata = deps


@pytest.fixture
def agent() -> KwamiAgent:
    return KwamiAgent(config=KwamiConfig())


def _realtime_agent(**voice: Any) -> KwamiAgent:
    config = KwamiConfig()
    config.voice.pipeline_type = "realtime"
    config.voice.realtime_provider = "openai"
    config.voice.realtime_model = "gpt-realtime"
    config.voice.realtime_voice = "marin"
    for key, value in voice.items():
        setattr(config.voice, key, value)
    return KwamiAgent(config=config, llm=FakeRealtimeModel())


# -- metering must never cost the answer ------------------------------------


@respx.mock
async def test_a_failing_tracker_does_not_lose_a_quote(agent: KwamiAgent) -> None:
    respx.get(QUOTE_URL.format(symbol="TSLA")).mock(
        return_value=httpx.Response(
            200,
            json={
                "chart": {
                    "result": [
                        {"meta": {"symbol": "TSLA", "regularMarketPrice": 100.0, "currency": "USD"}}
                    ]
                }
            },
        )
    )
    agent.usage_tracker = ExplodingTracker()

    quote = await agent.get_market_quote(None, "TSLA")

    assert quote["price"] == 100.0, "a broken meter swallowed the answer"


@respx.mock
async def test_a_failing_tracker_does_not_lose_research(agent: KwamiAgent, env_setting) -> None:
    env_setting("TAVILY_API_KEY", "tvly-test")
    respx.post(TAVILY_SEARCH_URL).mock(
        return_value=httpx.Response(
            200,
            json={"answer": "An answer.", "results": [{"title": "T", "url": "https://a.com/1"}]},
        )
    )
    agent.usage_tracker = ExplodingTracker()

    result = await agent.deep_research(None, "fusion power")

    assert "An answer." in result


@respx.mock
async def test_research_survives_the_whole_client_failing(
    agent: KwamiAgent, env_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not one angle failing -- the HTTP client itself refusing to start."""
    env_setting("TAVILY_API_KEY", "tvly-test")

    class Broken:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("no event loop for you")

    monkeypatch.setattr("src.tools.knowledge.httpx.AsyncClient", Broken)

    result = await agent.deep_research(None, "fusion power")

    assert "couldn't complete the research" in result


def test_a_malformed_url_has_no_domain() -> None:
    """Used for deduplication; it must not raise on provider junk."""
    assert _domain("http://[") == ""
    assert _domain("") == ""


# -- media refusals ---------------------------------------------------------


def test_an_unrecognised_kind_falls_back_to_music() -> None:
    key, _ = resolve_service("interpretive dance", None)
    assert key == resolve_service("music", None)[0]


async def test_playing_on_an_unknown_service_is_refused(agent: KwamiAgent) -> None:
    result = await agent.play_media(None, "a song", kind="music", service="pandora")

    assert "can't play on 'pandora'" in result
    assert "youtube" in result, "the refusal did not say what it can use"


@pytest.mark.parametrize("level", ["loud", None, [0.5]])
async def test_a_non_numeric_volume_is_refused(agent: KwamiAgent, level: Any) -> None:
    result = await agent.set_playback_volume(None, level)
    assert "between 0 and 1" in result


async def test_volume_without_a_panel_says_so(agent: KwamiAgent) -> None:
    async def _none() -> Any:
        return None

    agent._get_browser_session = _none  # type: ignore[method-assign]

    assert "Nothing is playing" in await agent.set_playback_volume(None, 0.5)


async def test_media_tools_without_a_browser_getter(agent: KwamiAgent) -> None:
    """`_media_session` tolerates an agent assembled without the browser mixin."""
    monkey = object()
    agent._get_browser_session = monkey  # type: ignore[assignment]

    assert await agent._media_session() is None


class _NoMediaSession:
    is_active = True

    def __init__(self) -> None:
        self.evaluated: list[str] = []

    async def evaluate_js(self, expression: str) -> str:
        self.evaluated.append(expression)
        return "Result: no-media"


async def test_controls_report_a_page_with_no_player(agent: KwamiAgent) -> None:
    session = _NoMediaSession()

    async def _get() -> Any:
        return session

    agent._get_browser_session = _get  # type: ignore[method-assign]

    assert "can't find anything playing" in await agent.control_playback(None, "pause")
    assert "can't find anything playing" in await agent.set_playback_volume(None, 0.4)
    assert (await agent.get_now_playing(None))["playing"] is False


class _BadJsonSession:
    is_active = True

    def __init__(self, payload: str) -> None:
        self._payload = payload

    async def evaluate_js(self, expression: str) -> str:
        return f"Result: {self._payload}"


@pytest.mark.parametrize(
    "payload",
    ["not json at all", '{"title": "unterminated'],
    ids=["no-braces", "malformed-json"],
)
async def test_an_unreadable_player_is_reported_not_guessed(
    agent: KwamiAgent, payload: str
) -> None:
    session = _BadJsonSession(payload)

    async def _get() -> Any:
        return session

    agent._get_browser_session = _get  # type: ignore[method-assign]

    info = await agent.get_now_playing(None)

    assert info["playing"] is False
    assert "could not read the player" in info["reason"]


# -- trading refusals -------------------------------------------------------


@pytest.mark.parametrize("quantity", ["a few", None, [1]])
async def test_a_non_numeric_quantity_is_refused(agent: KwamiAgent, quantity: Any) -> None:
    result = await agent.prepare_trade(None, "TSLA", "buy", quantity)
    assert result["error"] == "How many?"


async def test_a_quote_that_raises_does_not_block_the_read_back(
    agent: KwamiAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead feed loses the estimate, not the order."""

    async def explode(context: Any, symbol: str) -> dict[str, Any]:
        raise RuntimeError("feed down")

    monkeypatch.setattr(agent, "get_market_quote", explode)

    summary = await agent.prepare_trade(None, "TSLA", "buy", 10)

    assert summary["order"] == "buy 10.0 TSLA at market"
    assert "warning" in summary


async def test_a_limit_order_reads_its_price_back(
    agent: KwamiAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def quote(context: Any, symbol: str) -> dict[str, Any]:
        return {"price": 100.0, "currency": "USD"}

    monkeypatch.setattr(agent, "get_market_quote", quote)

    summary = await agent.prepare_trade(None, "TSLA", "buy", 10, "limit", 95.0)

    assert summary["limit_price"] == 95.0
    assert "limit of 95.0" in summary["order"]
    assert summary["estimated_value"] == 950.0, "priced off the market, not the limit"


async def test_a_ticket_with_no_broker_and_no_url_asks(
    agent: KwamiAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def quote(context: Any, symbol: str) -> dict[str, Any]:
        return {"price": 100.0}

    monkeypatch.setattr(agent, "get_market_quote", quote)
    await agent.prepare_trade(None, "TSLA", "buy", 10)

    assert "Which broker?" in await agent.open_trade_ticket(None)


async def test_a_user_supplied_url_opens(
    agent: KwamiAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only an address the user said -- never one lifted off a page."""
    navigated: list[str] = []

    async def quote(context: Any, symbol: str) -> dict[str, Any]:
        return {"price": 100.0}

    async def navigate(context: Any, url: str) -> str:
        navigated.append(url)
        return f"Opening {url}."

    monkeypatch.setattr(agent, "get_market_quote", quote)
    monkeypatch.setattr(agent, "navigate_to", navigate)
    await agent.prepare_trade(None, "TSLA", "buy", 10)

    await agent.open_trade_ticket(None, url="https://my-broker.example/trade")

    assert navigated == ["https://my-broker.example/trade"]
    assert "https://my-broker.example/trade" not in BROKER_URLS.values()


async def test_cancelling_nothing_says_so(agent: KwamiAgent) -> None:
    assert "nothing pending" in await agent.cancel_prepared_trade(None)


# -- pipeline control refusals ----------------------------------------------


async def test_an_unintelligible_model_request_asks_which_provider(
    agent: KwamiAgent,
) -> None:
    state = SessionState(current_agent=agent)

    def create_agent_fn(config, vad, memory=None, skip_greeting=False):  # pragma: no cover
        raise AssertionError("a rebuild was attempted for an unparseable request")

    deps = AgentDeps(
        reconfigure=Reconfigurator(state=state, vad=None, create_agent_fn=create_agent_fn)
    )

    result = await agent.change_ai_model(Ctx(deps), "   ")

    assert "couldn't work out" in result


async def test_listing_models_follows_the_pipeline(agent: KwamiAgent) -> None:
    standard = await agent.list_available_models(Ctx(AgentDeps()))
    assert standard["pipeline"] == "standard"
    assert "anthropic" in standard["providers"]

    realtime = await _realtime_agent().list_available_models(Ctx(AgentDeps()))
    assert realtime["pipeline"] == "realtime"
    assert "anthropic" not in realtime["providers"], "a chat-only provider leaked in"


async def test_listing_providers_includes_spoken_aliases(agent: KwamiAgent) -> None:
    providers = await agent.list_model_providers(Ctx(AgentDeps()))
    assert "claude" in providers
    assert "gemini" in providers


async def test_switching_pipeline_without_a_handle(agent: KwamiAgent) -> None:
    result = await agent.switch_pipeline_mode(Ctx(AgentDeps()), "realtime")
    assert "can't change" in result.lower()


async def test_asking_for_the_voice_already_in_use(agent: KwamiAgent) -> None:
    realtime = _realtime_agent(realtime_voice="cedar")

    assert "already using cedar" in await realtime.change_realtime_voice(Ctx(AgentDeps()), "cedar")


async def test_a_realtime_voice_change_with_no_live_model_and_no_handle() -> None:
    """The live push cannot happen and there is nothing to rebuild with."""
    config = KwamiConfig()
    config.voice.pipeline_type = "realtime"
    config.voice.realtime_provider = "openai"
    config.voice.realtime_voice = "marin"
    agent = KwamiAgent(config=config)  # no realtime model attached

    result = await agent.change_realtime_voice(Ctx(AgentDeps()), "cedar")

    assert "can't change" in result.lower()
    assert agent.kwami_config.voice.realtime_voice == "marin", "the config was left mutated"


async def test_a_realtime_voice_change_rebuilds_when_it_cannot_go_live() -> None:
    """No live model, but a reconfigure handle: rebuild rather than refuse."""
    config = KwamiConfig()
    config.voice.pipeline_type = "realtime"
    config.voice.realtime_provider = "openai"
    config.voice.realtime_voice = "marin"
    agent = KwamiAgent(config=config)
    built: list[Any] = []
    state = SessionState(current_agent=agent)

    def create_agent_fn(config, vad, memory=None, skip_greeting=False):
        new_agent = KwamiAgent(config=config, llm=FakeRealtimeModel())
        built.append(new_agent)
        return new_agent

    deps = AgentDeps(
        reconfigure=Reconfigurator(state=state, vad=None, create_agent_fn=create_agent_fn)
    )

    new_agent, message = await agent.change_realtime_voice(Ctx(deps), "cedar")

    assert new_agent is built[0]
    assert new_agent.kwami_config.voice.realtime_voice == "cedar"
    assert "cedar" in message


def test_a_switch_with_no_requested_model_is_not_a_failure(agent: KwamiAgent) -> None:
    """An empty `wanted_model` cannot be compared, so it is not called a failure."""
    from src.tools.pipeline_control import _switch_took_effect

    assert _switch_took_effect(agent, "") is True


def test_a_model_that_reports_nothing_is_not_called_a_failure() -> None:
    """Realtime models and some plugins do not expose `.model`."""
    from src.tools.pipeline_control import _switch_took_effect

    class Opaque:
        pass

    agent = KwamiAgent(config=KwamiConfig(), llm=Opaque())
    assert _switch_took_effect(agent, "claude-3-5-sonnet-latest") is True


# -- the reconfigure handle -------------------------------------------------


def test_the_current_voice_of_an_empty_state() -> None:
    reconfigurator = Reconfigurator(state=SessionState(), vad=None, create_agent_fn=lambda *a: None)
    assert reconfigurator.current_voice() is None


def test_the_current_voice_of_a_live_state() -> None:
    agent = KwamiAgent(config=KwamiConfig())
    reconfigurator = Reconfigurator(
        state=SessionState(current_agent=agent), vad=None, create_agent_fn=lambda *a: None
    )
    assert reconfigurator.current_voice() is agent.kwami_config.voice


def test_a_handle_without_a_factory_is_not_available() -> None:
    assert Reconfigurator(state=SessionState(), create_agent_fn=None).is_available is False
    assert Reconfigurator(state=None, create_agent_fn=lambda *a: None).is_available is False


def test_resolving_a_handle_from_a_context_without_deps() -> None:
    assert reconfigurator_from_context(None) is None
    assert reconfigurator_from_context(Ctx(None)) is None
    assert reconfigurator_from_context(Ctx(AgentDeps())) is None


# -- realtime helpers -------------------------------------------------------


@pytest.mark.parametrize("value", [None, 42, [], {}, "   ", "nonsense"])
def test_an_unusable_pipeline_type_is_none(value: Any) -> None:
    assert normalize_pipeline_type(value) is None


@pytest.mark.parametrize("value", [None, 42, "a string", []])
def test_realtime_keys_on_a_non_dict(value: Any) -> None:
    assert has_realtime_keys(value) is False


def test_what_counts_as_a_realtime_model() -> None:
    """A plain LLM also has `update_options`; only the realtime one has `session`."""

    class PlainLLM:
        def update_options(self, **kwargs: Any) -> None: ...

    class NoOptions:
        def session(self) -> Any: ...

    assert realtime_model_of(KwamiAgent(config=KwamiConfig(), llm=PlainLLM())) is None
    assert realtime_model_of(KwamiAgent(config=KwamiConfig(), llm=NoOptions())) is None
    assert realtime_model_of(KwamiAgent(config=KwamiConfig())) is None
    assert realtime_model_of(_realtime_agent()) is not None


def test_a_live_push_with_nothing_to_push_succeeds() -> None:
    """No overlap with the live-updatable set is success, not failure."""
    realtime = _realtime_agent()

    assert apply_live_realtime_options(realtime, {"realtime_model"}) is True
    assert realtime.llm.updates == []


def test_a_live_push_without_a_realtime_model_fails() -> None:
    assert (
        apply_live_realtime_options(KwamiAgent(config=KwamiConfig()), {"realtime_voice"}) is False
    )
