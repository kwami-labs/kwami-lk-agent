"""The config message `kwami-app` actually sends, driven through the real handler.

`tests/integration/test_config_wire_parsing.py` covers the shapes the handler
was *written* against. This file covers the shape the product *sends*, which
turned out not to be the same thing.

The frontend SDK's wire type is `VoicePipelineConfig`, whose pipeline field is
`type` with the values `'stt-llm-tts' | 'realtime' | 'hybrid'` (see
kwami/src/agent/voice/types.ts), and `useKwami.ts` puts `voiceStore.voiceConfig`
-- which is exactly that object -- straight onto the config message. The handler
read `pipelineType` with the values `'standard' | 'realtime'`: two spellings no
client has ever produced. Nothing warned, because the key was simply absent.
Selecting the realtime pipeline in the app therefore built a standard
STT+LLM+TTS agent every time, and parsed the user's realtime provider, model
and voice into fields that branch would never read.

So these payloads are copied from the app, not invented here. If the app
changes its wire format, this is where it should break.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.domain import KwamiConfig
from src.handlers.config_handler import handle_config_update, handle_full_config
from src.session import SessionState


class FakeSession:
    def __init__(self) -> None:
        self.agent: Any = None

    def update_agent(self, agent: Any) -> None:
        self.agent = agent


@pytest.fixture
def captured() -> dict:
    return {}


@pytest.fixture
def create_agent_fn(captured: dict):
    def _create(config, vad, memory=None, skip_greeting=False):
        from src.agent import KwamiAgent

        captured["config"] = config
        captured["skip_greeting"] = skip_greeting
        return KwamiAgent(config=config)

    return _create


def _app_config_message(pipeline_type: str) -> dict[str, Any]:
    """The `type: "config"` message kwami-app sends on connect.

    Mirrors `voiceStore.voiceConfig` (src/stores/voice.ts): a `type` field plus
    the four provider sections, all of them present regardless of which
    pipeline is selected.
    """
    return {
        "type": "config",
        "kwamiId": "kwami_abc_1",
        "kwamiName": "Kwami",
        "voice": {
            "type": pipeline_type,
            "stt": {"provider": "deepgram", "model": "nova-3", "language": "en"},
            "llm": {"provider": "openai", "model": "gpt-4.1-mini", "temperature": 0.7},
            "tts": {"provider": "openai", "model": "tts-1", "voice": "nova", "speed": 1.0},
            "realtime": {
                "provider": "openai",
                "model": "gpt-realtime",
                "voice": "cedar",
            },
        },
        "soul": {"name": "Kwami", "personality": "helpful"},
    }


async def _connect(message: dict[str, Any], create_agent_fn) -> None:
    await handle_full_config(
        FakeSession(), SessionState(), message, vad=None, create_agent_fn=create_agent_fn
    )


# -- the bug this file exists for -------------------------------------------


async def test_selecting_realtime_in_the_app_builds_a_realtime_pipeline(
    captured, create_agent_fn
) -> None:
    """The app's own spelling of "realtime", end to end."""
    await _connect(_app_config_message("realtime"), create_agent_fn)

    voice = captured["config"].voice
    assert voice.pipeline_type == "realtime", (
        "the app selected the realtime pipeline and got a standard one"
    )
    assert voice.realtime_provider == "openai"
    assert voice.realtime_model == "gpt-realtime"
    assert voice.realtime_voice == "cedar"


async def test_the_apps_standard_spelling_is_understood(captured, create_agent_fn) -> None:
    """`stt-llm-tts`, not `standard`. Both must mean the same thing."""
    await _connect(_app_config_message("stt-llm-tts"), create_agent_fn)

    assert captured["config"].voice.pipeline_type == "standard"


async def test_realtime_sections_are_still_parsed_on_the_standard_pipeline(
    captured, create_agent_fn
) -> None:
    """The app always sends all four sections; the unused ones must survive.

    They are what a later mid-session switch to realtime starts from, so
    dropping them would make the first switch fall back to defaults.
    """
    await _connect(_app_config_message("stt-llm-tts"), create_agent_fn)

    voice = captured["config"].voice
    assert voice.pipeline_type == "standard"
    assert voice.realtime_voice == "cedar"
    assert voice.tts_voice == "nova"


async def test_an_unknown_pipeline_type_keeps_the_default(captured, create_agent_fn) -> None:
    message = _app_config_message("quantum")
    await _connect(message, create_agent_fn)

    assert captured["config"].voice.pipeline_type == "standard"


async def test_hybrid_degrades_to_standard(captured, create_agent_fn) -> None:
    """`hybrid` is in the SDK's union and has no implementation here.

    Resolving it to standard is a decision; leaving it unrecognised would have
    been the same outcome reached by accident.
    """
    await _connect(_app_config_message("hybrid"), create_agent_fn)

    assert captured["config"].voice.pipeline_type == "standard"


async def test_a_config_with_no_pipeline_field_keeps_the_default(captured, create_agent_fn):
    message = _app_config_message("realtime")
    del message["voice"]["type"]

    await _connect(message, create_agent_fn)

    assert captured["config"].voice.pipeline_type == "standard"


# -- mid-session, from the app's own tools ----------------------------------


async def test_the_apps_pipeline_mode_control_switches_live(create_agent_fn, captured) -> None:
    """`set_voice_control('pipelineMode', 'realtime')` must not need a reconnect.

    The app's own message for this control says "reconnect to apply", because
    until the handler understood the switch there was nothing else it could
    say. The backend can do it live now, so the same value sent as a config
    update has to land.
    """
    from src.agent import KwamiAgent

    agent = KwamiAgent(config=KwamiConfig())
    state = SessionState(current_agent=agent)

    await handle_config_update(
        FakeSession(),
        state,
        {"updateType": "pipeline", "config": {"pipelineMode": "realtime", "type": "realtime"}},
        None,
        create_agent_fn,
    )

    assert captured["config"].voice.pipeline_type == "realtime"
    assert captured["skip_greeting"] is True, "a mid-session switch must not re-greet"


# -- the memory block's own branches ----------------------------------------


async def test_a_memory_block_without_an_enabled_flag_tunes_the_live_client(
    env_setting, create_agent_fn, captured
) -> None:
    """The app sends retrieval knobs without restating `enabled`.

    Two branches meet here and both are easy to get wrong. `enabled` absent must
    leave the setting alone rather than reading as False, and a config that
    names no kwami must not overwrite the user id of the memory it is about to
    reuse -- doing so scatters one conversation across several Zep threads, so
    recall finds nothing.
    """
    from src.agent import KwamiAgent

    env_setting("ZEP_API_KEY", "zep-test-key")

    class LiveMemory:
        """Stands in for an initialised `KwamiMemory`.

        Only the surface `_reuse_existing_memory` reads: reuse is what keeps
        this test off the network, and it is also the production path -- a fresh
        client per config message churned one unclosable client each time.
        """

        is_initialized = True

        def __init__(self) -> None:
            self.config = KwamiConfig().memory
            self.config.user_id = ""

        def set_usage_tracker(self, tracker: Any) -> None: ...

    memory = LiveMemory()
    state = SessionState(current_agent=KwamiAgent(config=KwamiConfig(), memory=memory))

    await handle_full_config(
        FakeSession(),
        state,
        {"voice": {"type": "stt-llm-tts"}, "memory": {"maxContextMessages": 20}},
        vad=None,
        create_agent_fn=create_agent_fn,
    )

    assert captured["config"].memory.enabled is True, "an absent `enabled` read as False"
    assert captured["config"].memory.max_context_messages == 20
    assert memory.config.max_context_messages == 20, "the knob never reached the live client"
