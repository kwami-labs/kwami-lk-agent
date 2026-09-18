"""Partial config updates: `updateType` messages from a live frontend.

`handle_config_update` and `handle_full_config` both catch every exception and
only log, so a test that gets the shape wrong passes green while asserting
nothing. Everything here asserts on the resulting agent or config, never on the
absence of a raise.

The agent is a real KwamiAgent wherever the path touches instructions or tools,
because those go through the framework.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from src.agent import KwamiAgent
from src.domain import KwamiConfig
from src.handlers.config_handler import (
    _update_stt_if_needed,
    _update_tts_options,
    handle_config_update,
    update_memory,
    update_soul,
    update_tools,
)
from src.memory.context import MemoryContext
from src.session import SessionState


class FakeSession:
    def __init__(self) -> None:
        self.agent: Any = None

    def update_agent(self, agent: Any) -> None:
        self.agent = agent


class RecordingTTS:
    """Stands in for a plugin TTS that supports live option updates."""

    def __init__(self, provider: str = "", model: str = "") -> None:
        self.provider = provider
        self.model = model
        self.updates: list[dict[str, Any]] = []
        self.fail = False

    def update_options(self, **kwargs: Any) -> None:
        if self.fail:
            raise RuntimeError("plugin refused the update")
        self.updates.append(kwargs)


class RecordingSTT:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update_options(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class FakeAgent:
    """Only the surface the TTS/STT option paths reach for."""

    def __init__(self, config: KwamiConfig | None = None, tts: Any = None, stt: Any = None):
        self.kwami_config = config or KwamiConfig()
        self.tts = tts
        self.stt = stt
        self._memory = None


def tool_names(agent: Any) -> set[str]:
    """FunctionTool exposes its name through `info`, not an attribute."""
    from livekit.agents.llm.tool_context import get_function_info

    return {get_function_info(tool).name for tool in agent.tools}


@pytest.fixture
def captured() -> dict:
    return {}


@pytest.fixture
def create_agent_fn(captured: dict):
    def _create(config, vad, memory=None, skip_greeting=False):
        captured["config"] = config
        captured["memory"] = memory
        captured["skip_greeting"] = skip_greeting
        return FakeAgent(config)

    return _create


def state_with(agent: Any) -> SessionState:
    state = SessionState()
    state.current_agent = agent
    return state


# =============================================================================
# handle_config_update dispatch
# =============================================================================


async def test_an_update_for_a_foreign_agent_is_ignored(create_agent_fn) -> None:
    """A non-KwamiAgent means the session is mid-swap; applying would write
    config onto an object that is about to be discarded."""
    state = state_with(FakeAgent())

    await handle_config_update(
        FakeSession(),
        state,
        {"updateType": "soul", "config": {"name": "Ada"}},
        vad=None,
        create_agent_fn=create_agent_fn,
    )

    assert state.current_agent.kwami_config.soul.name != "Ada"


async def test_an_unknown_update_type_is_ignored() -> None:
    agent = KwamiAgent()
    state = state_with(agent)

    await handle_config_update(
        FakeSession(),
        state,
        {"updateType": "nonesuch", "config": {}},
        vad=None,
        create_agent_fn=lambda *a, **k: None,
    )

    assert state.current_agent is agent


async def test_a_soul_update_is_dispatched() -> None:
    agent = KwamiAgent()
    state = state_with(agent)

    await handle_config_update(
        FakeSession(),
        state,
        {"updateType": "soul", "config": {"name": "Ada"}},
        vad=None,
        create_agent_fn=lambda *a, **k: None,
    )

    assert agent.kwami_config.soul.name == "Ada"


async def test_the_legacy_persona_update_type_still_works() -> None:
    agent = KwamiAgent()
    state = state_with(agent)

    await handle_config_update(
        FakeSession(),
        state,
        {"updateType": "persona", "config": {"name": "Ada"}},
        vad=None,
        create_agent_fn=lambda *a, **k: None,
    )

    assert agent.kwami_config.soul.name == "Ada"


async def test_a_memory_update_is_dispatched() -> None:
    agent = KwamiAgent()
    state = state_with(agent)

    await handle_config_update(
        FakeSession(),
        state,
        {"updateType": "memory", "config": {"includeFacts": False}},
        vad=None,
        create_agent_fn=lambda *a, **k: None,
    )

    assert agent.kwami_config.memory.include_facts is False


async def test_a_failing_update_is_logged_not_raised(caplog) -> None:
    """A raise here would propagate into the data-channel handler."""
    agent = KwamiAgent()
    state = state_with(agent)

    with caplog.at_level(logging.ERROR):
        await handle_config_update(
            FakeSession(),
            state,
            {"updateType": "soul", "config": None},
            vad=None,
            create_agent_fn=lambda *a, **k: None,
        )

    assert "Error updating soul" in caplog.text


async def test_a_pipeline_update_with_no_type_is_a_no_op(caplog) -> None:
    agent = KwamiAgent()
    state = state_with(agent)

    with caplog.at_level(logging.WARNING):
        await handle_config_update(
            FakeSession(),
            state,
            {"updateType": "pipeline", "config": {}},
            vad=None,
            create_agent_fn=lambda *a, **k: None,
        )

    assert "no usable pipeline type" in caplog.text


# =============================================================================
# update_soul
# =============================================================================


@pytest.mark.parametrize(
    ("key", "value", "attr"),
    [
        ("name", "Ada", "name"),
        ("personality", "dry", "personality"),
        ("systemPrompt", "be brief", "system_prompt"),
        ("system_prompt", "be brief", "system_prompt"),
        ("traits", ["warm"], "traits"),
        ("conversationStyle", "casual", "conversation_style"),
        ("conversation_style", "casual", "conversation_style"),
        ("responseLength", "short", "response_length"),
        ("response_length", "short", "response_length"),
        ("emotionalTone", "upbeat", "emotional_tone"),
        ("emotional_tone", "upbeat", "emotional_tone"),
    ],
)
async def test_every_soul_field_updates_under_both_spellings(
    key: str, value: Any, attr: str
) -> None:
    agent = KwamiAgent()

    await update_soul(FakeSession(), agent, {key: value})

    assert getattr(agent.kwami_config.soul, attr) == value


async def test_emotional_traits_must_be_a_mapping() -> None:
    agent = KwamiAgent()
    original = agent.kwami_config.soul.emotional_traits

    await update_soul(FakeSession(), agent, {"emotionalTraits": ["not", "a", "dict"]})

    assert agent.kwami_config.soul.emotional_traits == original


async def test_emotional_traits_are_applied_when_a_mapping() -> None:
    agent = KwamiAgent()

    await update_soul(FakeSession(), agent, {"emotionalTraits": {"warmth": 0.8}})

    assert agent.kwami_config.soul.emotional_traits == {"warmth": 0.8}


async def test_a_soul_update_rewrites_the_instructions() -> None:
    agent = KwamiAgent()

    await update_soul(FakeSession(), agent, {"name": "Ada", "personality": "dry and precise"})

    assert "Ada" in agent.instructions
    assert "dry and precise" in agent.instructions


async def test_an_empty_soul_update_changes_nothing() -> None:
    agent = KwamiAgent()
    before = agent.instructions

    await update_soul(FakeSession(), agent, {})

    assert agent.instructions == before


async def test_memory_context_survives_a_soul_update() -> None:
    """A live persona change must not silently drop what the agent remembers."""
    agent = KwamiAgent()
    agent._last_memory_context = MemoryContext(facts=["the user likes tea"])

    await update_soul(FakeSession(), agent, {"name": "Ada"})

    assert "the user likes tea" in agent.instructions


async def test_an_unreadable_memory_context_does_not_block_the_soul_update() -> None:
    class Unrenderable:
        def to_system_prompt_addition(self) -> str:
            raise RuntimeError("cannot render")

    agent = KwamiAgent()
    agent._last_memory_context = Unrenderable()

    await update_soul(FakeSession(), agent, {"name": "Ada"})

    assert "Ada" in agent.instructions


# =============================================================================
# update_tools
# =============================================================================


@pytest.mark.parametrize(
    "payload",
    [pytest.param([], id="empty"), pytest.param(None, id="null"), pytest.param({}, id="dict")],
)
async def test_an_unusable_tools_payload_is_skipped(payload: Any, caplog) -> None:
    agent = KwamiAgent()
    before = len(agent.tools)

    with caplog.at_level(logging.WARNING):
        await update_tools(agent, payload)

    assert len(agent.tools) == before
    assert "empty or non-list" in caplog.text


async def test_client_tools_are_registered_alongside_the_builtins() -> None:
    """The defect this guards: assigning client tools straight onto the agent
    deleted every built-in for the rest of the session."""
    agent = KwamiAgent()
    builtin_count = len(agent.tools)

    await update_tools(
        agent, [{"name": "set_theme", "description": "change theme", "parameters": {}}]
    )

    names = tool_names(agent)
    assert "set_theme" in names
    assert "web_search" in names
    assert len(agent.tools) == builtin_count + 1


async def test_re_registering_replaces_the_previous_client_tools() -> None:
    agent = KwamiAgent()

    await update_tools(agent, [{"name": "first", "description": "d", "parameters": {}}])
    await update_tools(agent, [{"name": "second", "description": "d", "parameters": {}}])

    names = tool_names(agent)
    assert "second" in names
    assert "first" not in names


async def test_a_tools_failure_is_logged_not_raised(caplog) -> None:
    agent = KwamiAgent()

    with caplog.at_level(logging.ERROR):
        await update_tools(agent, [{"not": "a tool definition"}])

    assert agent.tools


# =============================================================================
# update_memory
# =============================================================================


async def test_memory_retrieval_settings_are_applied() -> None:
    agent = KwamiAgent()

    await update_memory(
        agent, {"maxContextMessages": 20, "includeFacts": False, "minFactRelevance": 0.8}
    )

    cfg = agent.kwami_config.memory
    assert cfg.max_context_messages == 20
    assert cfg.include_facts is False
    assert cfg.min_fact_relevance == 0.8


@pytest.mark.parametrize(
    ("sent", "expected"),
    [(0, 1), (-5, 1), (999, 50), (25, 25)],
)
async def test_the_context_message_count_is_clamped(sent: int, expected: int) -> None:
    """A 999-message context is a bill, not a feature."""
    agent = KwamiAgent()

    await update_memory(agent, {"maxContextMessages": sent})

    assert agent.kwami_config.memory.max_context_messages == expected


@pytest.mark.parametrize(
    ("sent", "expected"),
    [(-1.0, 0.0), (5.0, 1.0), (0.3, 0.3)],
)
async def test_the_relevance_threshold_is_clamped(sent: float, expected: float) -> None:
    agent = KwamiAgent()

    await update_memory(agent, {"minFactRelevance": sent})

    assert agent.kwami_config.memory.min_fact_relevance == expected


async def test_an_unparseable_context_count_is_warned(caplog) -> None:
    agent = KwamiAgent()
    before = agent.kwami_config.memory.max_context_messages

    with caplog.at_level(logging.WARNING):
        await update_memory(agent, {"maxContextMessages": "lots"})

    assert agent.kwami_config.memory.max_context_messages == before
    assert "Invalid maxContextMessages" in caplog.text


async def test_an_unparseable_relevance_is_warned(caplog) -> None:
    agent = KwamiAgent()

    with caplog.at_level(logging.WARNING):
        await update_memory(agent, {"minFactRelevance": "high"})

    assert "Invalid minFactRelevance" in caplog.text


async def test_an_empty_memory_update_changes_nothing() -> None:
    agent = KwamiAgent()
    before = agent.kwami_config.memory.max_context_messages

    await update_memory(agent, {})

    assert agent.kwami_config.memory.max_context_messages == before


async def test_the_live_memory_instance_sees_the_new_settings() -> None:
    """Otherwise the change applies only to the next agent, not this session."""

    class Mem:
        config = None

    agent = KwamiAgent()
    agent._memory = Mem()

    await update_memory(agent, {"includeFacts": False})

    assert agent._memory.config is agent.kwami_config.memory


# =============================================================================
# _update_tts_options
# =============================================================================


async def test_no_tts_means_no_option_update() -> None:
    await _update_tts_options(FakeAgent(tts=None), {"tts_voice": "nova"}, "nova", False)


async def test_a_voice_update_reaches_the_plugin() -> None:
    tts = RecordingTTS()
    agent = FakeAgent(tts=tts)

    await _update_tts_options(agent, {}, "some-voice", False)

    assert tts.updates == [{"voice": "some-voice"}]
    assert agent.kwami_config.voice.tts_voice == "some-voice"


async def test_the_direct_elevenlabs_plugin_uses_voice_id() -> None:
    """inference.TTS always takes `voice`; only the direct plugin takes
    `voice_id`, and sending the wrong one is a TypeError mid-session."""
    tts = RecordingTTS(provider="elevenlabs")
    agent = FakeAgent(tts=tts)

    await _update_tts_options(agent, {}, "21m00Tcm4TlvDq8ikWAM", True)

    assert tts.updates == [{"voice_id": "21m00Tcm4TlvDq8ikWAM"}]


class OpenAITTS(RecordingTTS):
    """Its module decides the provider, exactly as the real plugin's does."""


OpenAITTS.__module__ = "livekit.plugins.openai.tts"


async def test_an_invalid_voice_for_openai_is_skipped(caplog) -> None:
    """A Rime voice left over from a provider fallback must not be sent to
    OpenAI, where it is rejected on every turn."""
    tts = OpenAITTS()
    agent = FakeAgent(tts=tts)

    with caplog.at_level(logging.WARNING):
        await _update_tts_options(agent, {}, "orion", False)

    assert tts.updates == []
    assert "not valid for current OpenAI TTS" in caplog.text


async def test_a_speed_update_reaches_the_plugin() -> None:
    tts = RecordingTTS()
    agent = FakeAgent(tts=tts)

    await _update_tts_options(agent, {"tts_speed": 1.25}, None, False)

    assert tts.updates == [{"speed": 1.25}]
    assert agent.kwami_config.voice.tts_speed == 1.25


async def test_a_zero_speed_is_honoured() -> None:
    """`number` is falsy-safe; a truthiness guard would read 0 as absent."""
    tts = RecordingTTS()
    agent = FakeAgent(tts=tts)

    await _update_tts_options(agent, {"tts_speed": 0}, None, False)

    assert tts.updates == [{"speed": 0}]


async def test_nothing_to_update_touches_the_plugin() -> None:
    tts = RecordingTTS()

    await _update_tts_options(FakeAgent(tts=tts), {}, None, False)

    assert tts.updates == []


async def test_a_plugin_that_refuses_the_update_is_warned(caplog) -> None:
    tts = RecordingTTS()
    tts.fail = True
    agent = FakeAgent(tts=tts)

    with caplog.at_level(logging.WARNING):
        await _update_tts_options(agent, {}, "some-voice", False)

    assert "Failed to update TTS options" in caplog.text


async def test_a_plugin_without_update_options_is_left_alone() -> None:
    class Static:
        provider = ""

    await _update_tts_options(FakeAgent(tts=Static()), {}, "some-voice", False)


# =============================================================================
# _update_stt_if_needed
# =============================================================================


async def test_an_stt_provider_change_recreates_the_agent(captured, create_agent_fn) -> None:
    agent = FakeAgent()
    state = state_with(agent)

    await _update_stt_if_needed(
        FakeSession(), state, agent, {"stt_provider": "openai"}, None, create_agent_fn
    )

    assert captured["config"].voice.stt_provider == "openai"
    assert captured["skip_greeting"] is True


async def test_an_stt_model_change_recreates_the_agent(captured, create_agent_fn) -> None:
    agent = FakeAgent()
    state = state_with(agent)

    await _update_stt_if_needed(
        FakeSession(), state, agent, {"stt_model": "deepgram/nova-3"}, None, create_agent_fn
    )

    assert captured["config"].voice.stt_model == "nova-3"


async def test_the_language_travels_with_an_stt_recreation(captured, create_agent_fn) -> None:
    agent = FakeAgent()
    state = state_with(agent)

    await _update_stt_if_needed(
        FakeSession(),
        state,
        agent,
        {"stt_provider": "openai", "stt_language": "fr"},
        None,
        create_agent_fn,
    )

    assert captured["config"].voice.stt_language == "fr"


async def test_a_language_only_change_updates_options_in_place(create_agent_fn) -> None:
    """No provider change means no rebuild -- a rebuild would drop the session's
    browser and pending tool calls for a language tweak."""
    stt = RecordingSTT()
    agent = FakeAgent(stt=stt)
    state = state_with(agent)

    await _update_stt_if_needed(
        FakeSession(), state, agent, {"stt_language": "fr"}, None, create_agent_fn
    )

    assert stt.updates == [{"language": "fr"}]


async def test_an_unchanged_stt_does_nothing(create_agent_fn) -> None:
    stt = RecordingSTT()
    agent = FakeAgent(stt=stt)
    state = state_with(agent)

    await _update_stt_if_needed(
        FakeSession(),
        state,
        agent,
        {"stt_provider": agent.kwami_config.voice.stt_provider},
        None,
        create_agent_fn,
    )

    assert stt.updates == []


async def test_an_stt_without_update_options_is_left_alone(create_agent_fn) -> None:
    class Static:
        pass

    agent = FakeAgent(stt=Static())
    state = state_with(agent)

    await _update_stt_if_needed(
        FakeSession(), state, agent, {"stt_language": "fr"}, None, create_agent_fn
    )


async def test_no_stt_at_all_is_tolerated(create_agent_fn) -> None:
    agent = FakeAgent(stt=None)
    state = state_with(agent)

    await _update_stt_if_needed(
        FakeSession(), state, agent, {"stt_language": "fr"}, None, create_agent_fn
    )
