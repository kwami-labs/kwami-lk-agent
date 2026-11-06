"""A live voice change between Cartesia's own voices and the ones designed in LiveKit.

Both travel as `tts_provider="cartesia"`, but they speak on different engines: a
UUID on Cartesia's plugin, a "v_" voice on LiveKit Inference. A same-provider
voice change only called `update_options` on the engine already built, so moving
from Marcos to Jarvis's LiveKit voice handed a "v_" id to a plugin that only
takes UUIDs -- and the old voice went on speaking. Moving between them rebuilds.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.domain import KwamiConfig
from src.handlers.config_handler import handle_config_update
from src.session import SessionState

JARVIS_LIVEKIT_VOICE = "v_ZMPWZvfoRpsC"
OTHER_LIVEKIT_VOICE = "v_Other12345"
MARCOS = "13ff5deb-2591-42ad-a356-63a04e524411"
GONZALO = "58e531e3-b212-49df-adee-c335a19c2429"


class FakeSession:
    def __init__(self) -> None:
        self.agent: Any = None

    def update_agent(self, agent: Any) -> None:
        self.agent = agent


def _make_agent(config: KwamiConfig) -> Any:
    """A real `KwamiAgent`: `handle_config_update` only acts on one."""
    from src.agent import KwamiAgent

    return KwamiAgent(config=config)


@pytest.fixture
def rebuilds() -> list[KwamiConfig]:
    return []


@pytest.fixture
def create_agent_fn(rebuilds: list[KwamiConfig]):
    def _create(config, vad, memory=None, skip_greeting=False):
        rebuilds.append(config)
        return _make_agent(config)

    return _create


def _cartesia(voice: str) -> KwamiConfig:
    config = KwamiConfig()
    config.voice.pipeline_type = "stt-llm-tts"
    config.voice.tts_provider = "cartesia"
    config.voice.tts_model = "sonic-3.5"
    config.voice.tts_voice = voice
    return config


async def _voice(agent: Any, voice: str, create_agent_fn) -> None:
    message = {
        "updateType": "voice",
        "config": {"tts_provider": "cartesia", "tts_model": "sonic-3.5", "tts_voice": voice},
    }
    await handle_config_update(
        FakeSession(), SessionState(current_agent=agent), message, None, create_agent_fn
    )


async def test_a_cartesia_voice_to_a_livekit_one_rebuilds(create_agent_fn, rebuilds) -> None:
    await _voice(_make_agent(_cartesia(MARCOS)), JARVIS_LIVEKIT_VOICE, create_agent_fn)

    assert len(rebuilds) == 1
    assert rebuilds[0].voice.tts_provider == "cartesia"
    assert rebuilds[0].voice.tts_voice == JARVIS_LIVEKIT_VOICE


async def test_a_livekit_voice_back_to_a_cartesia_one_rebuilds(create_agent_fn, rebuilds) -> None:
    await _voice(_make_agent(_cartesia(JARVIS_LIVEKIT_VOICE)), MARCOS, create_agent_fn)

    assert len(rebuilds) == 1
    assert rebuilds[0].voice.tts_voice == MARCOS


@pytest.mark.parametrize(
    ("current", "new"),
    [(MARCOS, GONZALO), (JARVIS_LIVEKIT_VOICE, OTHER_LIVEKIT_VOICE)],
)
async def test_a_voice_on_the_same_engine_is_changed_in_place(
    current: str, new: str, create_agent_fn, rebuilds
) -> None:
    """No rebuild: it would drop the session's browser and pending tool calls."""
    await _voice(_make_agent(_cartesia(current)), new, create_agent_fn)

    assert rebuilds == []
