"""The `config` message: the whole identity and pipeline in one payload.

This is the message that turns the placeholder agent into the user's Kwami, and
it is wrapped in a catch-all that only logs. So every test here asserts on the
config the agent factory was handed, never on the absence of a raise.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from src.domain import KwamiConfig
from src.handlers.config_handler import handle_full_config, update_llm, update_voice
from src.session import SessionState


class FakeSession:
    def __init__(self) -> None:
        self.agent: Any = None

    def update_agent(self, agent: Any) -> None:
        self.agent = agent


class FakeAgent:
    def __init__(self, config: KwamiConfig | None = None) -> None:
        self.kwami_config = config or KwamiConfig()
        self._memory = None
        self._browser_session = None
        self.client_tools = type("T", (), {"pending_calls": {}})()
        self.room = None
        self.tts = None
        self.stt = None


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


async def run_config(message: dict, create_agent_fn, state: SessionState | None = None):
    state = state or SessionState()
    await handle_full_config(
        FakeSession(), state, message, vad=None, create_agent_fn=create_agent_fn
    )
    return state


# =============================================================================
# Soul
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
async def test_every_soul_field_arrives(
    captured, create_agent_fn, key: str, value: Any, attr: str
) -> None:
    await run_config({"soul": {key: value}}, create_agent_fn)

    assert getattr(captured["config"].soul, attr) == value


async def test_the_legacy_persona_key_is_still_read(captured, create_agent_fn) -> None:
    await run_config({"persona": {"name": "Ada"}}, create_agent_fn)

    assert captured["config"].soul.name == "Ada"


async def test_soul_wins_over_persona(captured, create_agent_fn) -> None:
    await run_config({"soul": {"name": "New"}, "persona": {"name": "Old"}}, create_agent_fn)

    assert captured["config"].soul.name == "New"


async def test_an_empty_system_prompt_is_honoured(captured, create_agent_fn) -> None:
    """Checked against None, not truthiness: clearing the prompt is a real
    instruction, and a truthiness guard would silently ignore it."""
    await run_config({"soul": {"systemPrompt": ""}}, create_agent_fn)

    assert captured["config"].soul.system_prompt == ""


async def test_emotional_traits_must_be_a_mapping(captured, create_agent_fn) -> None:
    default = KwamiConfig().soul.emotional_traits

    await run_config({"soul": {"emotionalTraits": ["not", "a", "dict"]}}, create_agent_fn)

    assert captured["config"].soul.emotional_traits == default


async def test_emotional_traits_are_applied_when_a_mapping(
    captured, create_agent_fn
) -> None:
    await run_config({"soul": {"emotionalTraits": {"warmth": 0.9}}}, create_agent_fn)

    assert captured["config"].soul.emotional_traits == {"warmth": 0.9}


# =============================================================================
# Identity and tools
# =============================================================================


async def test_the_kwami_id_sets_the_user_identity(captured, create_agent_fn) -> None:
    """Usage reporting keys off user_identity; without this a web session that
    never resolved a participant is never billed."""
    state = await run_config({"kwamiId": "kwami-7"}, create_agent_fn)

    assert captured["config"].kwami_id == "kwami-7"
    assert state.user_identity == "kwami-7"


async def test_an_existing_user_identity_is_not_overwritten(
    captured, create_agent_fn
) -> None:
    state = SessionState()
    state.user_identity = "already-known"

    await run_config({"kwamiId": "kwami-7"}, create_agent_fn, state)

    assert state.user_identity == "already-known"


async def test_the_identity_falls_back_to_the_participant(
    captured, create_agent_fn
) -> None:
    state = SessionState()
    state.user_identity = "participant-3"

    await run_config({}, create_agent_fn, state)

    assert captured["config"].kwami_id == "participant-3"


async def test_the_kwami_name_arrives(captured, create_agent_fn) -> None:
    await run_config({"kwamiName": "Ada"}, create_agent_fn)

    assert captured["config"].kwami_name == "Ada"


async def test_client_tools_arrive(captured, create_agent_fn) -> None:
    tools = [{"name": "set_theme", "description": "d", "parameters": {}}]

    await run_config({"tools": tools}, create_agent_fn)

    assert captured["config"].tools == tools


async def test_a_non_list_tools_payload_is_ignored(captured, create_agent_fn) -> None:
    await run_config({"tools": {"not": "a list"}}, create_agent_fn)

    assert captured["config"].tools == []


# =============================================================================
# Greeting
# =============================================================================


async def test_the_first_config_lets_the_new_agent_greet(
    captured, create_agent_fn
) -> None:
    """The first config arrives right after the placeholder starts and kills
    its greeting mid-flight, so the reconfigured agent must greet."""
    state = await run_config({}, create_agent_fn)

    assert captured["skip_greeting"] is False
    assert state.greeting_delivered is True


async def test_a_later_config_does_not_greet_again(captured, create_agent_fn) -> None:
    state = SessionState()
    state.greeting_delivered = True

    await run_config({}, create_agent_fn, state)

    assert captured["skip_greeting"] is True


# =============================================================================
# Memory
# =============================================================================


async def test_memory_settings_arrive(captured, create_agent_fn, env_setting) -> None:
    env_setting("ZEP_API_KEY", None)

    await run_config(
        {
            "kwamiId": "kwami-7",
            "memory": {
                "enabled": True,
                "maxContextMessages": 15,
                "includeFacts": False,
                "minFactRelevance": 0.7,
            },
        },
        create_agent_fn,
    )

    memory_cfg = captured["config"].memory
    assert memory_cfg.enabled is True
    assert memory_cfg.max_context_messages == 15
    assert memory_cfg.include_facts is False
    assert memory_cfg.min_fact_relevance == 0.7


async def test_the_memory_user_id_defaults_to_the_kwami_id(
    captured, create_agent_fn, env_setting
) -> None:
    """Each kwami gets its own graph; sharing one would leak memories between
    personas."""
    env_setting("ZEP_API_KEY", None)

    await run_config(
        {"kwamiId": "kwami_auth_7", "memory": {"enabled": True}}, create_agent_fn
    )

    assert captured["config"].memory.user_id == "kwami_auth_7"


async def test_memory_disabled_in_the_message_creates_none(
    captured, create_agent_fn, env_setting
) -> None:
    env_setting("ZEP_API_KEY", "zep-key")

    await run_config({"memory": {"enabled": False}}, create_agent_fn)

    assert captured["memory"] is None


async def test_no_memory_section_and_no_key_means_no_memory(
    captured, create_agent_fn, env_setting
) -> None:
    env_setting("ZEP_API_KEY", None)

    await run_config({}, create_agent_fn)

    assert captured["memory"] is None


async def test_an_existing_memory_is_reused(
    captured, create_agent_fn, env_setting
) -> None:
    """Rebuilding the Zep client on every config message leaked a connection
    pool and re-ran the whole user/thread/ontology setup."""
    env_setting("ZEP_API_KEY", "zep-key")

    class ExistingMemory:
        is_initialized = True

        def __init__(self, config: Any) -> None:
            self.config = config

    state = SessionState()
    config = KwamiConfig()
    config.memory.enabled = True
    config.memory.user_id = "kwami-7"
    existing = ExistingMemory(config.memory)
    state.current_agent = FakeAgent(config)
    state.current_agent._memory = existing

    await run_config(
        {"kwamiId": "kwami-7", "memory": {"enabled": True}}, create_agent_fn, state
    )

    assert captured["memory"] is existing


# =============================================================================
# Failure
# =============================================================================


async def test_a_broken_message_leaves_the_placeholder_live(caplog) -> None:
    """The catch-all: a config that cannot be applied must not take the
    session down, it must leave the default agent answering."""
    session = FakeSession()

    def explode(*args: Any, **kwargs: Any):
        raise RuntimeError("factory is broken")

    with caplog.at_level(logging.ERROR):
        await handle_full_config(
            session, SessionState(), {}, vad=None, create_agent_fn=explode
        )

    assert session.agent is None
    assert "Failed to process full config" in caplog.text


# =============================================================================
# update_voice: TTS provider switching
# =============================================================================


async def test_an_explicit_provider_change_rebuilds(captured, create_agent_fn) -> None:
    agent = FakeAgent()
    state = SessionState()
    state.current_agent = agent

    await update_voice(
        FakeSession(), state, agent, {"tts_provider": "cartesia"}, None, create_agent_fn
    )

    assert captured["config"].voice.tts_provider == "cartesia"
    assert captured["skip_greeting"] is True


async def test_switching_provider_clears_the_old_model_and_voice(
    captured, create_agent_fn
) -> None:
    """Cartesia's "sonic-3" carried over to OpenAI, where it is invalid, and
    Rime's "astra" carried over to ElevenLabs, where it does not exist."""
    agent = FakeAgent()
    agent.kwami_config.voice.tts_provider = "cartesia"
    agent.kwami_config.voice.tts_model = "sonic-3"
    agent.kwami_config.voice.tts_voice = "some-uuid"
    state = SessionState()
    state.current_agent = agent

    await update_voice(
        FakeSession(), state, agent, {"tts_provider": "openai"}, None, create_agent_fn
    )

    assert captured["config"].voice.tts_model == ""
    assert captured["config"].voice.tts_voice == ""


async def test_an_explicit_model_survives_the_switch(captured, create_agent_fn) -> None:
    agent = FakeAgent()
    state = SessionState()
    state.current_agent = agent

    await update_voice(
        FakeSession(), state, agent,
        {"tts_provider": "cartesia", "tts_model": "cartesia/sonic-2"},
        None, create_agent_fn,
    )

    assert captured["config"].voice.tts_model == "sonic-2"


async def test_a_provider_is_detected_from_the_model_alone(
    captured, create_agent_fn
) -> None:
    agent = FakeAgent()
    state = SessionState()
    state.current_agent = agent

    await update_voice(
        FakeSession(), state, agent, {"tts_model": "sonic-2"}, None, create_agent_fn
    )

    assert captured["config"].voice.tts_provider == "cartesia"


async def test_a_speed_change_rebuilds_for_providers_that_need_it(
    captured, create_agent_fn
) -> None:
    """ElevenLabs and Rime cannot take a live speed update, so the agent has to
    be rebuilt rather than silently ignoring the change."""
    agent = FakeAgent()
    agent.kwami_config.voice.tts_provider = "elevenlabs"
    agent.kwami_config.voice.tts_speed = 1.0
    state = SessionState()
    state.current_agent = agent

    await update_voice(
        FakeSession(), state, agent, {"tts_speed": 1.4}, None, create_agent_fn
    )

    assert captured["config"].voice.tts_speed == 1.4


async def test_an_unchanged_speed_does_not_rebuild(create_agent_fn, captured) -> None:
    agent = FakeAgent()
    agent.kwami_config.voice.tts_provider = "elevenlabs"
    agent.kwami_config.voice.tts_speed = 1.0
    state = SessionState()
    state.current_agent = agent

    await update_voice(
        FakeSession(), state, agent, {"tts_speed": 1.0}, None, create_agent_fn
    )

    assert "config" not in captured


async def test_a_speed_change_on_a_live_updatable_provider_does_not_rebuild(
    create_agent_fn, captured
) -> None:
    agent = FakeAgent()
    agent.kwami_config.voice.tts_provider = "openai"
    state = SessionState()
    state.current_agent = agent

    await update_voice(
        FakeSession(), state, agent, {"tts_speed": 1.4}, None, create_agent_fn
    )

    assert "config" not in captured


# =============================================================================
# update_llm
# =============================================================================


async def test_an_llm_change_rebuilds_with_the_new_settings(
    captured, create_agent_fn
) -> None:
    agent = FakeAgent()
    state = SessionState()
    state.current_agent = agent

    await update_llm(
        FakeSession(), state, agent,
        {"provider": "anthropic", "model": "anthropic/claude-sonnet-4", "temperature": 0.2},
        None, create_agent_fn,
    )

    voice = captured["config"].voice
    assert voice.llm_provider == "anthropic"
    assert voice.llm_model == "claude-sonnet-4"
    assert voice.llm_temperature == 0.2
    assert captured["skip_greeting"] is True


async def test_a_zero_temperature_survives_an_llm_update(
    captured, create_agent_fn
) -> None:
    agent = FakeAgent()
    state = SessionState()
    state.current_agent = agent

    await update_llm(
        FakeSession(), state, agent, {"temperature": 0}, None, create_agent_fn
    )

    assert captured["config"].voice.llm_temperature == 0


async def test_max_tokens_arrives_under_both_spellings(
    captured, create_agent_fn
) -> None:
    agent = FakeAgent()
    state = SessionState()
    state.current_agent = agent

    await update_llm(
        FakeSession(), state, agent, {"max_tokens": 512}, None, create_agent_fn
    )

    assert captured["config"].voice.llm_max_tokens == 512
