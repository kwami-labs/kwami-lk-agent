"""One model of each kind is live at a time: switching STT, TTS or LLM stops the old one.

The user's rule (2026-10-18): the model picked loads, and the others neither
hear the audio nor bill. This runs a real `AgentSession` (no room) through the
same config messages the app sends, with stand-in components that subclass the
framework's own STT, LLM and TTS -- so the session subscribes to them, and
unsubscribes from them, exactly as it does with a plugin.

After each switch every component ever built reports usage. Only the live ones
may reach `UsageTracker` (what the credits report bills) and the session's own
usage collector, and every replaced one must have been closed, which is what
shuts a provider's socket.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import pytest
from livekit.agents import AgentSession, llm, stt, tts
from livekit.agents.metrics import LLMMetrics, STTMetrics, TTSMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from src.agent import KwamiAgent
from src.domain import KwamiConfig
from src.handlers.config_handler import handle_config_update
from src.runtime import AgentDeps, route_metrics
from src.session import create_session_state


class _Billing:
    """What every stand-in shares: a name, a close flag, and a way to bill."""

    kind = ""

    def _setup(self, provider: str, model: str) -> None:
        self._provider_name = provider
        self._model_name = model
        self.closed = False

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return self._provider_name

    @property
    def tag(self) -> str:
        return f"{self._provider_name}/{self._model_name}"

    async def aclose(self) -> None:
        self.closed = True

    def bill(self) -> None:
        meta = Metadata(model_name=self._model_name, model_provider=self._provider_name)
        common: dict[str, Any] = {
            "label": self.tag,
            "request_id": f"req-{self.tag}",
            "timestamp": time.time(),
            "metadata": meta,
        }
        if self.kind == "stt":
            metrics: Any = STTMetrics(**common, duration=0.0, audio_duration=6.0, streamed=True)
        elif self.kind == "llm":
            metrics = LLMMetrics(
                **common,
                duration=0.2,
                ttft=0.1,
                cancelled=False,
                completion_tokens=20,
                prompt_tokens=100,
                prompt_cached_tokens=0,
                total_tokens=120,
                tokens_per_second=100.0,
            )
        else:
            metrics = TTSMetrics(
                **common,
                ttfb=0.1,
                duration=0.5,
                audio_duration=2.0,
                cancelled=False,
                characters_count=40,
                streamed=False,
            )
        self.emit("metrics_collected", metrics)  # type: ignore[attr-defined]


class _SilentStream(stt.RecognizeStream):
    """Takes audio and transcribes nothing: there is no room, so none arrives."""

    async def _run(self) -> None:
        async for _ in self._input_ch:
            pass


class FakeSTT(_Billing, stt.STT):
    kind = "stt"

    def __init__(self, provider: str, model: str) -> None:
        stt.STT.__init__(
            self, capabilities=stt.STTCapabilities(streaming=True, interim_results=False)
        )
        self._setup(provider, model)

    async def _recognize_impl(self, buffer: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def stream(self, **kwargs: Any) -> stt.RecognizeStream:
        return _SilentStream(stt=self, conn_options=DEFAULT_API_CONNECT_OPTIONS)


class FakeLLM(_Billing, llm.LLM):
    kind = "llm"

    def __init__(self, provider: str, model: str) -> None:
        llm.LLM.__init__(self)
        self._setup(provider, model)

    def chat(self, **kwargs: Any) -> Any:
        raise NotImplementedError


class FakeTTS(_Billing, tts.TTS):
    kind = "tts"

    def __init__(self, provider: str, model: str) -> None:
        tts.TTS.__init__(
            self,
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )
        self._setup(provider, model)

    def synthesize(self, text: str, **kwargs: Any) -> Any:
        raise NotImplementedError


@dataclass
class Factory:
    """`create_agent_from_config` with stand-in components, remembering each one."""

    built: list[_Billing] = field(default_factory=list)

    def __call__(
        self, config: KwamiConfig, vad: Any, memory: Any = None, skip_greeting: bool = False
    ) -> KwamiAgent:
        voice = config.voice
        parts = (
            FakeSTT(voice.stt_provider, voice.stt_model),
            FakeLLM(voice.llm_provider, voice.llm_model),
            FakeTTS(voice.tts_provider, voice.tts_model),
        )
        self.built.extend(parts)
        stt_, llm_, tts_ = parts
        return KwamiAgent(
            config, vad=None, memory=memory, stt=stt_, llm=llm_, tts=tts_, skip_greeting=True
        )


def _live(agent: KwamiAgent) -> set[str]:
    return {agent.stt.tag, agent.llm.tag, agent.tts.tag}  # type: ignore[union-attr]


def _billed(tracker: Any) -> set[str]:
    return {entry.model_id for entry in tracker._usage.values()}


def _session_billed(session: AgentSession) -> set[str]:
    return {f"{u.provider}/{u.model}" for u in session.usage.model_usage if u.provider}


async def _settle(state: Any, agent: KwamiAgent) -> None:
    """Wait until the session runs `agent` and the replaced one is closed."""
    for _ in range(200):
        if state.current_agent is agent and getattr(agent, "_activity", None) is not None:
            break
        await asyncio.sleep(0.01)
    tasks = list(state._cleanup_tasks)
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)


def _bill_everything(factory: Factory) -> None:
    for part in factory.built:
        part.bill()


@pytest.fixture
async def running() -> Any:
    factory = Factory()
    config = KwamiConfig()
    config.voice.stt_provider, config.voice.stt_model = "deepgram", "nova-3"
    config.voice.llm_provider, config.voice.llm_model = "openai", "gpt-4.1-mini"
    config.voice.tts_provider, config.voice.tts_model = "openai", "tts-1"
    agent = factory(config, None)

    # No room and no VAD: nothing here needs turn detection.
    session: AgentSession = AgentSession(
        vad=None, userdata=AgentDeps(), turn_handling={"turn_detection": None}
    )
    state = create_session_state(initial_agent=agent, room_name="room-1")

    @session.on("metrics_collected")
    def _on_metrics(event: Any) -> None:
        route_metrics(state.usage_tracker, event.metrics)

    await session.start(agent=agent)
    await _settle(state, agent)
    try:
        yield factory, session, state
    finally:
        await session.aclose()


async def _switch(session: Any, state: Any, factory: Factory, message: dict[str, Any]) -> Any:
    before = state.current_agent
    await handle_config_update(session, state, message, None, factory)
    agent = state.current_agent
    assert agent is not before, f"{message} did not rebuild the agent"
    await _settle(state, agent)
    return agent


async def test_switching_each_model_leaves_only_the_new_one_billing(running: Any) -> None:
    factory, session, state = running
    tracker = state.usage_tracker

    switches = [
        {"updateType": "voice", "config": {"stt_provider": "assemblyai", "stt_model": "u3-pro"}},
        {
            "updateType": "voice",
            "config": {"tts_provider": "elevenlabs", "tts_model": "eleven_flash_v2_5"},
        },
        {"updateType": "llm", "config": {"provider": "groq", "model": "qwen/qwen3.8-27b"}},
    ]
    for message in switches:
        tracker._usage.clear()
        agent = await _switch(session, state, factory, message)

        _bill_everything(factory)

        live = _live(agent)
        assert _billed(tracker) == live, f"after {message}: billed {_billed(tracker)}"
        assert live <= _session_billed(session)
        for part in factory.built:
            if part.tag in live and any(part is p for p in (agent.stt, agent.llm, agent.tts)):
                assert not part.closed
            else:
                assert part.closed, f"{part.tag} was replaced but never closed"

    assert _live(state.current_agent) == {
        "assemblyai/u3-pro",
        "groq/qwen/qwen3.8-27b",
        "elevenlabs/eleven_flash_v2_5",
    }


async def test_a_new_tts_model_on_the_same_provider_replaces_the_old_one(running: Any) -> None:
    """ElevenLabs turbo to flash, Gemini flash TTS to pro: the model is fixed when
    the engine is built, so the old one kept speaking -- and billing -- while the
    panel showed the new one."""
    factory, session, state = running
    agent = await _switch(
        session, state, factory, {"updateType": "voice", "config": {"tts_model": "gpt-4o-mini-tts"}}
    )
    state.usage_tracker._usage.clear()
    _bill_everything(factory)
    assert agent.tts.tag == "openai/gpt-4o-mini-tts"  # type: ignore[union-attr]
    assert "openai/tts-1" not in _billed(state.usage_tracker)


async def test_one_message_changing_tts_and_stt_applies_both(running: Any) -> None:
    """`updateVoiceLive` carries TTS and STT together. A TTS rebuild used to build
    the new agent from the old STT settings, dropping the STT change."""
    factory, session, state = running
    agent = await _switch(
        session,
        state,
        factory,
        {
            "updateType": "voice",
            "config": {
                "tts_provider": "cartesia",
                "tts_model": "sonic-3",
                "stt_provider": "assemblyai",
                "stt_model": "universal-streaming-multilingual",
                "stt_language": "es",
            },
        },
    )
    assert agent.tts.tag == "cartesia/sonic-3"  # type: ignore[union-attr]
    assert agent.stt.tag == "assemblyai/universal-streaming-multilingual"  # type: ignore[union-attr]
    assert agent.kwami_config.voice.stt_language == "es"
    state.usage_tracker._usage.clear()
    _bill_everything(factory)
    assert _billed(state.usage_tracker) == _live(agent)
