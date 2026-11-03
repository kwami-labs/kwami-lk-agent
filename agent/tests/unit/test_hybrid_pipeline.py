"""The hybrid pipeline: a realtime model thinks, the session's TTS speaks.

Only realtime models that can answer in text can do it. Gemini's Live models
answer only in audio (1007 for a text session, measured on 2026-10-22 with the
voice lab), so "Gemini Live + a TTS" is corrected to Gemini Live with its own
voice instead of being built into a silent agent.
"""

from __future__ import annotations

from typing import Any

import pytest
from livekit.agents.types import NOT_GIVEN

from src.domain import KwamiConfig
from src.domain.pipelines import realtime_speaks_text, resolve_pipeline
from src.factories import realtime as realtime_module
from src.factories.realtime import create_realtime_model
from src.handlers.config_handler import handle_full_config
from src.handlers.realtime import normalize_pipeline_type
from src.runtime import pipeline as pipeline_module
from src.runtime.turn_taking import agent_turn_handling

from .test_pipeline_control import _realtime_agent, _wire
from .test_voice_resilience import ErrorStub, Event, FakeAgent, FakeState, RecordingPublisher

# -- which models can ----------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("openai", "gpt-realtime"),
        ("openai", "gpt-realtime-mini"),
        ("openai", "openai/gpt-4o-realtime-preview"),
    ],
)
def test_openai_realtime_models_answer_in_text(provider: str, model: str) -> None:
    assert realtime_speaks_text(provider, model)


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("google", "gemini-3.1-flash-live-preview"),
        ("gemini", "gemini-3.8-live"),
        ("google", "gemini-2.5-flash-native-audio-preview-12-2025"),
        ("openai", ""),
        (None, None),
    ],
)
def test_gemini_live_and_blanks_do_not(provider: Any, model: Any) -> None:
    assert not realtime_speaks_text(provider, model)


def test_a_hybrid_request_on_gemini_live_becomes_realtime() -> None:
    choice = resolve_pipeline("hybrid", "google", "gemini-3.1-flash-live-preview")

    assert choice.pipeline == "realtime"
    assert choice.adjusted
    assert "audio" in choice.reason


@pytest.mark.parametrize(
    ("pipeline", "provider", "model"),
    [
        ("hybrid", "openai", "gpt-realtime"),
        ("realtime", "google", "gemini-3.8-live"),
        ("standard", "google", "gemini-3.8-live"),
    ],
)
def test_combinations_that_work_are_left_alone(pipeline: str, provider: str, model: str) -> None:
    choice = resolve_pipeline(pipeline, provider, model)

    assert choice.pipeline == pipeline
    assert not choice.adjusted


# -- the wire --------------------------------------------------------------------


@pytest.mark.parametrize("spelling", ["hybrid", "Hybrid", "realtime-tts", "half-cascade"])
def test_hybrid_spellings_are_read(spelling: str) -> None:
    assert normalize_pipeline_type(spelling) == "hybrid"


async def test_the_sdk_wire_type_hybrid_reaches_the_config() -> None:
    from src.session import SessionState

    class Session:
        def update_agent(self, agent: Any) -> None:
            self.agent = agent

    configs: list[KwamiConfig] = []

    def recorder(config, vad, memory=None, skip_greeting=False):
        configs.append(config)
        return object()

    await handle_full_config(
        Session(), SessionState(current_agent=None), {"voice": {"type": "hybrid"}}, None, recorder
    )

    assert configs[-1].voice.pipeline_type == "hybrid"


# -- building ----------------------------------------------------------------------


class Recorded:
    def __init__(self, kind: str, voice: Any) -> None:
        self.kind = kind
        self.pipeline = voice.pipeline_type


@pytest.fixture
def factories(monkeypatch: pytest.MonkeyPatch) -> list[Recorded]:
    built: list[Recorded] = []

    def make(kind: str):
        def factory(voice: Any) -> Recorded:
            built.append(Recorded(kind, voice))
            return built[-1]

        return factory

    monkeypatch.setattr(pipeline_module, "create_realtime_model", make("realtime"))
    monkeypatch.setattr(pipeline_module, "create_tts", make("tts"))
    monkeypatch.setattr(pipeline_module, "with_tts_fallback", lambda tts, *a: tts)
    monkeypatch.setattr(pipeline_module, "KwamiAgent", _CapturingAgent)
    return built


class _CapturingAgent:
    def __init__(self, config: Any, **parts: Any) -> None:
        self.kwami_config = config
        self.parts = parts


def _config(pipeline: str, provider: str, model: str) -> KwamiConfig:
    config = KwamiConfig()
    config.voice.pipeline_type = pipeline  # type: ignore[assignment]
    config.voice.realtime_provider = provider
    config.voice.realtime_model = model
    return config


def test_hybrid_builds_the_realtime_model_and_a_tts(factories: list[Recorded]) -> None:
    agent = pipeline_module.create_agent_from_config(
        _config("hybrid", "openai", "gpt-realtime"), vad=None
    )

    assert [b.kind for b in factories] == ["realtime", "tts"]
    assert agent.parts["llm"].pipeline == "hybrid"  # so it is asked for text
    assert agent.parts["tts"] is factories[1]
    assert "stt" not in agent.parts


def test_hybrid_on_gemini_live_is_built_with_its_own_voice(
    factories: list[Recorded], caplog: pytest.LogCaptureFixture
) -> None:
    agent = pipeline_module.create_agent_from_config(
        _config("hybrid", "google", "gemini-3.1-flash-live-preview"), vad=None
    )

    assert [b.kind for b in factories] == ["realtime"]
    assert agent.kwami_config.voice.pipeline_type == "realtime"
    assert "adjusted to realtime" in caplog.text


class _OpenAIRealtime:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}
        self.realtime = self

    def RealtimeModel(self, **kwargs: Any) -> str:  # noqa: N802
        self.kwargs = kwargs
        return "openai-realtime"


@pytest.mark.parametrize(("pipeline", "modalities"), [("hybrid", ["text"]), ("realtime", None)])
def test_openai_answers_in_text_only_on_hybrid(
    monkeypatch: pytest.MonkeyPatch, pipeline: str, modalities: Any
) -> None:
    plugin = _OpenAIRealtime()
    monkeypatch.setattr(realtime_module, "openai", plugin)
    config = _config(pipeline, "openai", "gpt-realtime").voice

    create_realtime_model(config)

    assert plugin.kwargs["modalities"] == (modalities or config.realtime_modalities)


class _GoogleRealtime:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}
        self.beta = self
        self.realtime = self

    def RealtimeModel(self, **kwargs: Any) -> str:  # noqa: N802
        self.kwargs = kwargs
        return "google-realtime"


def test_gemini_is_asked_for_text_on_hybrid_and_left_alone_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the voice lab asks Gemini for: it answers 1007, which is the point."""
    from google.genai import types

    plugin = _GoogleRealtime()
    monkeypatch.setattr(realtime_module, "google", plugin)

    create_realtime_model(_config("hybrid", "google", "gemini-3.8-live").voice)
    assert plugin.kwargs["modalities"] == [types.Modality.TEXT]

    create_realtime_model(_config("realtime", "google", "gemini-3.8-live").voice)
    assert "modalities" not in plugin.kwargs


def test_the_realtime_model_takes_the_turns_on_hybrid_too() -> None:
    assert agent_turn_handling("hybrid") is NOT_GIVEN


# -- the agent's own switch ----------------------------------------------------------


async def test_switching_to_hybrid_on_openai_realtime() -> None:
    agent = _realtime_agent()
    ctx, _ = _wire(agent)

    new_agent, message = await agent.switch_pipeline_mode(ctx, "hybrid")

    assert new_agent.kwami_config.voice.pipeline_type == "hybrid"
    assert "hybrid" in message


async def test_switching_to_hybrid_on_gemini_live_is_refused() -> None:
    agent = _realtime_agent(realtime_provider="google", realtime_model="gemini-3.8-live")
    ctx, built = _wire(agent)

    result = await agent.switch_pipeline_mode(ctx, "hybrid")

    assert isinstance(result, str)
    assert "own voice" in result
    assert built == []


# -- a dead socket -----------------------------------------------------------------


async def test_a_dead_realtime_model_on_hybrid_is_rescued_too() -> None:
    import asyncio

    from .test_voice_resilience import EmittingSession

    session = EmittingSession()
    state = FakeState(FakeAgent("hybrid"))
    publisher = RecordingPublisher()

    def create_agent_fn(config: Any, vad: Any, memory: Any, skip_greeting: bool) -> Any:
        return FakeAgent(config.voice.pipeline_type)

    from src.runtime.recovery import install_realtime_rescue

    install_realtime_rescue(session, state, None, create_agent_fn, publisher)
    error = ErrorStub("realtime_model_error", recoverable=False)
    session.handlers["error"](Event(error))
    await asyncio.gather(*state.spawned)

    assert state.current_agent.kwami_config.voice.pipeline_type == "standard"
    assert publisher.messages[0]["from"] == "hybrid"


def test_the_brain_status_tells_the_app_which_models_can(tmp_path) -> None:
    from src.brain.bridge import Bridge
    from src.brain.ledger import Ledger

    status = Bridge(keys={}, ledger=Ledger(tmp_path)).status()

    assert status["hybridRealtime"]["google"] == []
    assert "gpt-realtime" in status["hybridRealtime"]["openai"]


# -- a character switched to another TTS on Live + TTS ---------------------------


async def test_a_tts_change_on_hybrid_rebuilds_with_the_new_tts() -> None:
    """The panel sends the realtime and TTS fields together; the TTS half counts."""
    from src.handlers.realtime import update_realtime

    agent = _realtime_agent(pipeline_type="hybrid", tts_provider="elevenlabs")
    built: list[Any] = []

    class State:
        def update_agent(self, session: Any, new_agent: Any) -> None:
            built.append(new_agent)

    def create_agent_fn(config, vad, memory=None, skip_greeting=False):
        return config

    agent._memory = None
    await update_realtime(
        None,
        State(),
        agent,
        {
            "realtimeProvider": "openai",
            "realtimeModel": "gpt-realtime",
            "realtimeVoice": "marin",
            "tts_provider": "openai",
            "tts_model": "tts-1",
            "tts_voice": "onyx",
        },
        None,
        create_agent_fn,
    )

    assert len(built) == 1
    assert built[0].voice.tts_provider == "openai"
    assert built[0].voice.tts_voice == "onyx"
    assert built[0].voice.pipeline_type == "hybrid"


async def test_on_realtime_the_tts_half_is_still_ignored() -> None:
    from src.handlers.realtime import update_realtime

    agent = _realtime_agent(tts_provider="elevenlabs")
    built: list[Any] = []

    class State:
        def update_agent(self, session: Any, new_agent: Any) -> None:
            built.append(new_agent)

    agent._memory = None
    await update_realtime(
        None,
        State(),
        agent,
        {"realtimeProvider": "openai", "realtimeModel": "gpt-realtime", "tts_provider": "openai"},
        None,
        lambda config, *a, **k: config,
    )

    assert built == []
