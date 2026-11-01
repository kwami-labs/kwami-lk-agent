"""What keeps the voice talking when a provider fails (2026-10-20).

Each test pins a failure seen in production: a free-tier Gemini quota leaving
the standard pipeline mute, a text model handed to Gemini Live closing the
session with 1008, typed chat messages dropped by the router, and a dead
realtime model taking the whole call down with it.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from src.domain import KwamiConfig
from src.factories import create_llm, create_stt, create_tts
from src.factories.fallback import (
    ELEVENLABS_FEMALE,
    ELEVENLABS_MALE,
    is_female_voice,
    primary_of,
    with_llm_fallback,
    with_stt_fallback,
    with_tts_fallback,
)
from src.handlers.realtime import is_realtime_model
from src.runtime.dispatch import MAX_TYPED_CHARS, DataMessageRouter
from src.runtime.recovery import install_realtime_rescue

# -- realtime models ---------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("google", "gemini-3.1-flash-live-preview"),
        ("google", "google/gemini-3.1-flash-live-preview"),
        ("google", "gemini-2.5-flash-native-audio-preview-12-2025"),
        ("openai", "gpt-realtime"),
        ("openai", "gpt-4o-realtime-preview"),
    ],
)
def test_realtime_models_are_recognised(provider: str, model: str) -> None:
    assert is_realtime_model(provider, model)


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("google", "gemini-3.5-flash"),
        ("openai", "gpt-4.1-mini"),
        ("anthropic", "claude-haiku-5-5"),
        ("google", ""),
        (None, None),
    ],
)
def test_text_models_are_not_realtime_ones(provider: Any, model: Any) -> None:
    assert not is_realtime_model(provider, model)


# -- fallback chains ---------------------------------------------------------


def _voice(**changes: Any) -> Any:
    return replace(KwamiConfig().voice, **changes)


def test_without_other_keys_the_llm_is_left_alone(env_setting) -> None:
    env_setting("GOOGLE_API_KEY", "test-key-not-real")
    voice = _voice(llm_provider="google", llm_model="gemini-3.5-flash")
    primary = create_llm(voice)

    assert with_llm_fallback(primary, voice, create_llm) is primary


def test_the_llm_falls_back_to_the_providers_with_a_key(env_setting) -> None:
    """A spent Gemini quota moves the turn to OpenAI, then Claude."""
    for name in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        env_setting(name, "test-key-not-real")
    voice = _voice(llm_provider="google", llm_model="gemini-3.5-flash")
    primary = create_llm(voice)

    chain = with_llm_fallback(primary, voice, create_llm)

    members = chain._llm_instances
    assert members[0] is primary
    assert [type(m).__module__.split(".")[2] for m in members[1:]] == ["openai", "anthropic"]
    assert primary_of(chain) is primary


def test_the_configured_provider_is_never_its_own_backup(env_setting) -> None:
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        env_setting(name, "test-key-not-real")
    voice = _voice(llm_provider="openai", llm_model="gpt-4.1-mini")

    chain = with_llm_fallback(create_llm(voice), voice, create_llm)

    assert len(chain._llm_instances) == 2


def test_the_tts_backup_keeps_the_characters_register(env_setting) -> None:
    for name in ("GOOGLE_API_KEY", "ELEVEN_API_KEY", "CARTESIA_API_KEY"):
        env_setting(name, "test-key-not-real")
    voice = _voice(tts_provider="gemini", tts_voice="Aoede", tts_style="Habla como Friday")

    chain = with_tts_fallback(create_tts(voice), voice, create_tts)

    members = chain._tts_instances
    assert len(members) == 3
    eleven = members[1]
    assert primary_of(eleven)._opts.voice_id == ELEVENLABS_FEMALE


@pytest.mark.parametrize(
    ("voice", "female"),
    [
        ("Charon", False),
        ("Aoede", True),
        ("kore", True),
        (ELEVENLABS_MALE, False),
        (ELEVENLABS_FEMALE, True),
        ("", False),
        (None, False),
    ],
)
def test_voice_register(voice: Any, female: bool) -> None:
    assert is_female_voice(voice) is female


def test_the_stt_falls_back_too(env_setting) -> None:
    for name in ("GOOGLE_API_KEY", "ASSEMBLYAI_API_KEY", "DEEPGRAM_API_KEY"):
        env_setting(name, "test-key-not-real")
    voice = _voice(stt_provider="deepgram", stt_model="nova-3", stt_language="es")

    chain = with_stt_fallback(create_stt(voice), voice, create_stt, vad=None)

    assert len(chain._stt_instances) == 2


def test_primary_of_a_plain_component_is_itself() -> None:
    component = object()
    assert primary_of(component) is component


# -- typed messages ----------------------------------------------------------


class RecordingSession:
    def __init__(self, *, interrupt_raises: bool = False) -> None:
        self.replies: list[dict[str, Any]] = []
        self.interrupts = 0
        self._interrupt_raises = interrupt_raises

    def interrupt(self, **_: Any) -> None:
        self.interrupts += 1
        if self._interrupt_raises:
            raise RuntimeError("nothing to interrupt")

    def generate_reply(self, **kwargs: Any) -> None:
        self.replies.append(kwargs)


def _router(session: Any) -> DataMessageRouter:
    return DataMessageRouter(session=session, state=object())


def test_a_typed_message_is_answered() -> None:
    """The chat box sends {type: text}; it had no route and was dropped."""
    session = RecordingSession()

    handled = _router(session).handle({"type": "text", "text": "  ¿Qué hora es?  "})

    assert handled == "text"
    assert session.replies == [{"user_input": "¿Qué hora es?"}]
    assert session.interrupts == 1


def test_a_typed_message_answers_even_with_nothing_to_interrupt() -> None:
    session = RecordingSession(interrupt_raises=True)

    _router(session).handle({"type": "text", "text": "Hola"})

    assert session.replies == [{"user_input": "Hola"}]


@pytest.mark.parametrize("text", ["", "   ", None, 42])
def test_an_empty_typed_message_is_ignored(text: Any) -> None:
    session = RecordingSession()

    _router(session).handle({"type": "text", "text": text})

    assert session.replies == []


def test_a_long_typed_message_is_capped() -> None:
    session = RecordingSession()

    _router(session).handle({"type": "text", "text": "a" * (MAX_TYPED_CHARS + 50)})

    assert len(session.replies[0]["user_input"]) == MAX_TYPED_CHARS


def test_the_interrupt_button_interrupts() -> None:
    session = RecordingSession()

    assert _router(session).handle({"type": "interrupt"}) == "interrupt"
    assert session.interrupts == 1


def test_a_failed_reply_does_not_escape_the_router() -> None:
    class Broken(RecordingSession):
        def generate_reply(self, **kwargs: Any) -> None:
            raise RuntimeError("session closing")

    _router(Broken()).handle({"type": "text", "text": "Hola"})


# -- realtime rescue ---------------------------------------------------------


class EmittingSession:
    def __init__(self) -> None:
        self.handlers: dict[str, Any] = {}

    def on(self, event: str):
        def register(fn: Any) -> Any:
            self.handlers[event] = fn
            return fn

        return register


class ErrorStub:
    def __init__(self, type_: str, recoverable: bool) -> None:
        self.type = type_
        self.recoverable = recoverable
        self.error = Exception("1008 models/x is not supported for bidiGenerateContent")


class Event:
    def __init__(self, error: Any) -> None:
        self.error = error


class FakeAgent:
    def __init__(self, pipeline: str) -> None:
        self.kwami_config = KwamiConfig()
        self.kwami_config.voice.pipeline_type = pipeline
        self._memory = None


class FakeState:
    def __init__(self, agent: Any) -> None:
        self.current_agent = agent
        self.spawned: list[Any] = []
        self.swaps: list[Any] = []

    def spawn(self, coro: Any, *, name: str) -> Any:
        task = asyncio.ensure_future(coro)
        self.spawned.append(task)
        return task

    async def run_serialized(self, coro: Any) -> None:
        await coro

    def update_agent(self, session: Any, agent: Any) -> None:
        self.swaps.append(agent)
        self.current_agent = agent


class RecordingPublisher:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def publish(self, message: dict[str, Any]) -> None:
        self.messages.append(message)


async def test_a_dead_realtime_model_moves_the_call_to_the_standard_pipeline() -> None:
    session = EmittingSession()
    state = FakeState(FakeAgent("realtime"))
    publisher = RecordingPublisher()
    built: list[Any] = []

    def create_agent_fn(config: Any, vad: Any, memory: Any, skip_greeting: bool) -> Any:
        agent = FakeAgent(config.voice.pipeline_type)
        built.append((config, skip_greeting))
        return agent

    install_realtime_rescue(session, state, None, create_agent_fn, publisher)
    error = ErrorStub("realtime_model_error", recoverable=False)
    session.handlers["error"](Event(error))
    await asyncio.gather(*state.spawned)

    assert error.recoverable is True  # the framework no longer closes the session
    assert built[0][0].voice.pipeline_type == "standard"
    assert built[0][1] is True
    assert state.current_agent.kwami_config.voice.pipeline_type == "standard"
    assert publisher.messages[0]["type"] == "voice_fallback"


@pytest.mark.parametrize(
    ("type_", "recoverable", "pipeline"),
    [
        ("realtime_model_error", True, "realtime"),
        ("tts_error", False, "realtime"),
        ("realtime_model_error", False, "standard"),
    ],
)
async def test_other_errors_are_left_to_the_framework(
    type_: str, recoverable: bool, pipeline: str
) -> None:
    session = EmittingSession()
    state = FakeState(FakeAgent(pipeline))
    install_realtime_rescue(session, state, None, lambda *a, **k: None)
    error = ErrorStub(type_, recoverable=recoverable)

    session.handlers["error"](Event(error))

    assert error.recoverable is recoverable
    assert state.spawned == []


async def test_a_second_failure_during_the_rescue_starts_no_second_rescue() -> None:
    session = EmittingSession()
    state = FakeState(FakeAgent("realtime"))
    release = asyncio.Event()

    def create_agent_fn(config: Any, vad: Any, memory: Any, skip_greeting: bool) -> Any:
        return FakeAgent(config.voice.pipeline_type)

    async def slow_serialized(coro: Any) -> None:
        await release.wait()
        await coro

    state.run_serialized = slow_serialized  # type: ignore[method-assign]
    install_realtime_rescue(session, state, None, create_agent_fn)
    first, second = (
        ErrorStub("realtime_model_error", False),
        ErrorStub("realtime_model_error", False),
    )

    session.handlers["error"](Event(first))
    session.handlers["error"](Event(second))
    release.set()
    await asyncio.gather(*state.spawned)

    assert second.recoverable is True
    assert len(state.spawned) == 1
    assert len(state.swaps) == 1


async def test_a_model_that_died_before_greeting_leaves_the_greeting_owed() -> None:
    class Deps:
        greeting_owed = False

    session = EmittingSession()
    session.userdata = Deps()  # type: ignore[attr-defined]
    state = FakeState(FakeAgent("realtime"))

    def create_agent_fn(config: Any, vad: Any, memory: Any, skip_greeting: bool) -> Any:
        return FakeAgent(config.voice.pipeline_type)

    install_realtime_rescue(session, state, None, create_agent_fn)
    session.handlers["error"](Event(ErrorStub("realtime_model_error", False)))
    await asyncio.gather(*state.spawned)

    assert session.userdata.greeting_owed is True  # type: ignore[attr-defined]


def test_gemini_backs_itself_up_through_livekit_inference(env_setting) -> None:
    """A spent free-tier Google quota does not apply to Inference, billed by LiveKit."""
    for name in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        env_setting(name, "test-key-not-real")
    voice = _voice(llm_provider="google", llm_model="gemini-3.5-flash")

    chain = with_llm_fallback(create_llm(voice), voice, create_llm)

    backup = chain._llm_instances[1]
    assert type(backup).__module__ == "livekit.agents.inference.llm"
    assert backup.model == "google/gemini-2.5-flash"


def test_inference_is_only_a_backup_for_gemini(env_setting) -> None:
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        env_setting(name, "test-key-not-real")
    voice = _voice(llm_provider="openai", llm_model="gpt-4.1-mini")

    chain = with_llm_fallback(create_llm(voice), voice, create_llm)

    assert all("inference" not in type(m).__module__ for m in chain._llm_instances)


def test_an_inference_model_needs_its_provider_prefix(env_setting) -> None:
    env_setting("OPENAI_API_KEY", "test-key-not-real")
    voice = _voice(llm_provider="inference", llm_model="gemini-2.5-flash")
    # No "provider/" in the id: not a model Inference can route, so OpenAI.
    assert "openai" in type(create_llm(voice)).__module__
