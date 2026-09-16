"""A config message from the frontend, through the real handler.

These are the values a user is most likely to set deliberately, and every one
of them was silently discarded: a truthiness guard cannot tell `0` from a
missing key, and `"voice": null` raised AttributeError and dropped the whole
message -- leaving the placeholder agent live with its default persona.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.handlers.config_handler import handle_full_config
from src.session import SessionState


class FakeSession:
    def __init__(self) -> None:
        self.agent: Any = None

    def update_agent(self, agent: Any) -> None:
        self.agent = agent


class FakeAgent:
    def __init__(self, config: Any) -> None:
        self.kwami_config = config
        self._memory = None
        self._browser_session = None
        self.client_tools = type("T", (), {"pending_calls": {}})()
        self.room = None


@pytest.fixture
def captured() -> dict:
    return {}


@pytest.fixture
def create_agent_fn(captured: dict):
    def _create(config, vad, memory=None, skip_greeting=False):
        captured["config"] = config
        captured["skip_greeting"] = skip_greeting
        return FakeAgent(config)

    return _create


async def _run(message: dict, create_agent_fn) -> None:
    await handle_full_config(
        FakeSession(), SessionState(), message, vad=None, create_agent_fn=create_agent_fn
    )


async def test_zero_temperature_is_honoured(captured, create_agent_fn) -> None:
    """`temperature: 0` is the deterministic setting, and it was ignored."""
    await _run({"voice": {"llm": {"temperature": 0}}}, create_agent_fn)

    assert "config" in captured, "the config message was dropped"
    assert captured["config"].voice.llm_temperature == 0.0


async def test_zero_tts_speed_is_honoured(captured, create_agent_fn) -> None:
    await _run({"voice": {"tts": {"speed": 0}}}, create_agent_fn)
    assert captured["config"].voice.tts_speed == 0.0


async def test_zero_max_tokens_is_honoured(captured, create_agent_fn) -> None:
    await _run({"voice": {"llm": {"maxTokens": 0}}}, create_agent_fn)
    assert captured["config"].voice.llm_max_tokens == 0


async def test_a_null_voice_section_does_not_drop_the_config(captured, create_agent_fn) -> None:
    """This used to raise AttributeError and abandon the whole message."""
    await _run({"voice": None, "soul": {"name": "Ada"}}, create_agent_fn)

    assert "config" in captured, "a null voice section dropped the entire config"
    assert captured["config"].soul.name == "Ada"


async def test_a_null_nested_section_is_survivable(captured, create_agent_fn) -> None:
    await _run({"voice": {"tts": None, "llm": {"temperature": 0.3}}}, create_agent_fn)
    assert captured["config"].voice.llm_temperature == 0.3


async def test_a_wrong_typed_section_is_survivable(captured, create_agent_fn) -> None:
    await _run({"voice": "loud"}, create_agent_fn)
    assert "config" in captured


async def test_normal_values_still_apply(captured, create_agent_fn) -> None:
    await _run(
        {
            "voice": {
                "tts": {
                    "provider": "openai",
                    "model": "openai/tts-1",
                    "voice": "nova",
                    "speed": 1.2,
                },
                "llm": {"provider": "openai", "temperature": 0.7, "maxTokens": 512},
                "stt": {"provider": "deepgram", "language": "es"},
            }
        },
        create_agent_fn,
    )

    voice = captured["config"].voice
    assert voice.tts_provider == "openai"
    assert voice.tts_model == "tts-1", "provider prefix should be stripped"
    assert voice.tts_voice == "nova"
    assert voice.tts_speed == 1.2
    assert voice.llm_temperature == 0.7
    assert voice.llm_max_tokens == 512
    assert voice.stt_language == "es"


async def test_blank_strings_do_not_overwrite_defaults(captured, create_agent_fn) -> None:
    """An empty provider name is a client bug, not a request for "" as provider."""
    await _run({"voice": {"tts": {"provider": "   "}}}, create_agent_fn)
    assert captured["config"].voice.tts_provider.strip() != ""
