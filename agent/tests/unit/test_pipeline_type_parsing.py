"""`pipelineType` must reach the config, or the realtime pipeline is unreachable.

runtime/pipeline.py has branched on `voice.pipeline_type` all along, but the
config handler never read `pipelineType` or any `realtime*` key off the wire.
The realtime path therefore existed only in a test-only preset.
"""

from __future__ import annotations

import pytest

from src.domain import KwamiConfig
from src.handlers.config_handler import handle_full_config


class FakeSession:
    def update_agent(self, agent) -> None:
        self.agent = agent


class Recorder:
    """Captures the config the agent factory is handed."""

    def __init__(self) -> None:
        self.configs: list[KwamiConfig] = []

    def __call__(self, config, vad, memory=None, skip_greeting=False):
        self.configs.append(config)
        return object()


@pytest.fixture
def state():
    from src.session import SessionState

    return SessionState(current_agent=None)


async def _apply(state, message: dict) -> KwamiConfig:
    recorder = Recorder()
    await handle_full_config(FakeSession(), state, message, None, recorder)
    assert recorder.configs, "the agent factory was never called"
    return recorder.configs[-1]


async def test_pipeline_type_realtime_is_honoured(state) -> None:
    config = await _apply(state, {"voice": {"pipelineType": "realtime"}})
    assert config.voice.pipeline_type == "realtime"


async def test_snake_case_is_accepted_too(state) -> None:
    config = await _apply(state, {"voice": {"pipeline_type": "realtime"}})
    assert config.voice.pipeline_type == "realtime"


async def test_an_unknown_pipeline_type_is_ignored_not_applied(state) -> None:
    """A typo must not produce a pipeline that cannot be built."""
    config = await _apply(state, {"voice": {"pipelineType": "quantum"}})
    assert config.voice.pipeline_type == "standard"


async def test_realtime_provider_model_and_voice_are_parsed(state) -> None:
    config = await _apply(
        state,
        {
            "voice": {
                "pipelineType": "realtime",
                "realtime": {
                    "provider": "openai",
                    "model": "openai/gpt-realtime",
                    "voice": "marin",
                },
            }
        },
    )

    assert config.voice.realtime_provider == "openai"
    # The provider prefix is stripped, as it is for every other model field.
    assert config.voice.realtime_model == "gpt-realtime"
    assert config.voice.realtime_voice == "marin"


async def test_flat_realtime_keys_are_accepted(state) -> None:
    """The frontend sends these flat on `voice` in some versions."""
    config = await _apply(
        state,
        {"voice": {"realtimeProvider": "openai", "realtimeVoice": "cedar"}},
    )

    assert config.voice.realtime_provider == "openai"
    assert config.voice.realtime_voice == "cedar"


async def test_absent_pipeline_keys_leave_the_defaults_alone(state) -> None:
    config = await _apply(state, {"voice": {"tts": {"provider": "openai"}}})

    assert config.voice.pipeline_type == "standard"
