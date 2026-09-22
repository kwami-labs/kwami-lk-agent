"""Changing the realtime voice, model and pipeline mid-conversation.

The frontend SDK's `updateRealtimeLive()` sends `realtime_provider`,
`realtime_model` and `realtime_voice` under `updateType: "voice"` -- the same
message type a TTS change uses. `update_voice` read only `tts_*` and `stt_*`,
so every one of those keys was dropped: asking for a different realtime voice
mid-session did nothing at all, silently, and there was no way to move between
the standard and realtime pipelines without ending the session.

These tests drive the real handler with the payloads the SDK actually emits.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.domain import KwamiConfig
from src.handlers.config_handler import handle_config_update
from src.session import SessionState


class FakeRealtimeModel:
    """Stands in for `openai.realtime.RealtimeModel`.

    Carries the two attributes the production code keys on -- `update_options`
    and `session` -- so that "is this a realtime model?" is decided the same way
    here as it is against the real SDK.
    """

    def __init__(self, fail: bool = False) -> None:
        self.updates: list[dict[str, Any]] = []
        self._fail = fail

    def session(self) -> Any:  # pragma: no cover - presence is what matters
        raise NotImplementedError

    def update_options(self, **kwargs: Any) -> None:
        if self._fail:
            raise RuntimeError("socket closed")
        self.updates.append(kwargs)


class FakeSession:
    def __init__(self) -> None:
        self.agent: Any = None

    def update_agent(self, agent: Any) -> None:
        self.agent = agent


def _make_agent(config: KwamiConfig, llm: Any = None) -> Any:
    """A real `KwamiAgent`, because the handler's own guards depend on one.

    `handle_config_update` is gated on `isinstance(current_agent, KwamiAgent)`,
    and `realtime_model_of` reads `agent.llm` -- which the framework exposes as
    a property, not an instance attribute. A hand-rolled double satisfies
    neither, so the thing under test is constructed for real and only the model
    handed to it is a stand-in.
    """
    from src.agent import KwamiAgent

    return KwamiAgent(config=config, llm=llm)


@pytest.fixture
def rebuilds() -> list[dict[str, Any]]:
    return []


@pytest.fixture
def create_agent_fn(rebuilds: list[dict[str, Any]]):
    def _create(config, vad, memory=None, skip_greeting=False):
        rebuilds.append({"config": config, "skip_greeting": skip_greeting})
        return _make_agent(config)

    return _create


def _realtime_config(**voice: Any) -> KwamiConfig:
    config = KwamiConfig()
    config.voice.pipeline_type = "realtime"
    config.voice.realtime_provider = "openai"
    config.voice.realtime_model = "gpt-realtime"
    config.voice.realtime_voice = "marin"
    for key, value in voice.items():
        setattr(config.voice, key, value)
    return config


async def _update(agent: Any, message: dict[str, Any], create_agent_fn) -> SessionState:
    state = SessionState(current_agent=agent)
    await handle_config_update(FakeSession(), state, message, None, create_agent_fn)
    return state


# -- realtime voice ---------------------------------------------------------


async def test_realtime_voice_change_reaches_the_live_model(create_agent_fn, rebuilds) -> None:
    """The payload `updateRealtimeLive({voice})` actually sends."""
    model = FakeRealtimeModel()
    agent = _make_agent(_realtime_config(), llm=model)

    await _update(
        agent,
        {"updateType": "voice", "config": {"realtime_voice": "cedar"}},
        create_agent_fn,
    )

    assert model.updates == [{"voice": "cedar"}], "realtime voice never reached the model"
    assert agent.kwami_config.voice.realtime_voice == "cedar"
    assert rebuilds == [], "a live voice change must not tear down the realtime socket"


async def test_realtime_voice_change_falls_back_to_a_rebuild(create_agent_fn, rebuilds) -> None:
    """A failed live push must not be reported as a change the user will hear."""
    agent = _make_agent(_realtime_config(), llm=FakeRealtimeModel(fail=True))

    await _update(
        agent,
        {"updateType": "voice", "config": {"realtime_voice": "cedar"}},
        create_agent_fn,
    )

    assert len(rebuilds) == 1
    assert rebuilds[0]["config"].voice.realtime_voice == "cedar"
    assert rebuilds[0]["skip_greeting"] is True


async def test_realtime_model_change_rebuilds(create_agent_fn, rebuilds) -> None:
    """Model choice is a constructor argument, so it cannot be pushed live."""
    model = FakeRealtimeModel()
    agent = _make_agent(_realtime_config(), llm=model)

    await _update(
        agent,
        {
            "updateType": "voice",
            "config": {"realtime_provider": "google", "realtime_model": "gemini-2.0-flash-exp"},
        },
        create_agent_fn,
    )

    assert model.updates == []
    assert len(rebuilds) == 1
    rebuilt = rebuilds[0]["config"].voice
    assert rebuilt.realtime_provider == "google"
    assert rebuilt.realtime_model == "gemini-2.0-flash-exp"
    assert rebuilt.pipeline_type == "realtime"


async def test_realtime_model_prefix_is_stripped(create_agent_fn, rebuilds) -> None:
    """The models panel sends `provider/model`; the plugin wants the bare name."""
    agent = _make_agent(_realtime_config(), llm=FakeRealtimeModel())

    await _update(
        agent,
        {"updateType": "voice", "config": {"realtime_model": "openai/gpt-realtime-mini"}},
        create_agent_fn,
    )

    assert rebuilds[0]["config"].voice.realtime_model == "gpt-realtime-mini"


async def test_an_unchanged_realtime_voice_does_nothing(create_agent_fn, rebuilds) -> None:
    """Re-sending the current value must not churn the socket or the agent."""
    model = FakeRealtimeModel()
    agent = _make_agent(_realtime_config(realtime_voice="marin"), llm=model)

    await _update(
        agent,
        {"updateType": "voice", "config": {"realtime_voice": "marin"}},
        create_agent_fn,
    )

    assert model.updates == []
    assert rebuilds == []


async def test_tts_keys_do_not_rebuild_a_realtime_session(create_agent_fn, rebuilds) -> None:
    """A stale TTS payload must not silently drop the user onto a TTS pipeline."""
    agent = _make_agent(_realtime_config(), llm=FakeRealtimeModel())

    await _update(
        agent,
        {"updateType": "voice", "config": {"tts_voice": "nova", "tts_provider": "openai"}},
        create_agent_fn,
    )

    assert rebuilds == []
    assert agent.kwami_config.voice.pipeline_type == "realtime"


# -- pipeline switching -----------------------------------------------------


async def test_switch_from_standard_to_realtime(create_agent_fn, rebuilds) -> None:
    agent = _make_agent(KwamiConfig())

    await _update(
        agent,
        {
            "updateType": "voice",
            "config": {"pipelineType": "realtime", "realtime_voice": "cedar"},
        },
        create_agent_fn,
    )

    assert len(rebuilds) == 1
    rebuilt = rebuilds[0]["config"].voice
    assert rebuilt.pipeline_type == "realtime"
    assert rebuilt.realtime_voice == "cedar", "fields riding along in the switch were dropped"


async def test_switch_from_realtime_back_to_standard(create_agent_fn, rebuilds) -> None:
    agent = _make_agent(_realtime_config(), llm=FakeRealtimeModel())

    await _update(
        agent,
        {"updateType": "pipeline", "config": {"pipelineType": "standard"}},
        create_agent_fn,
    )

    assert rebuilds[0]["config"].voice.pipeline_type == "standard"


async def test_an_unknown_pipeline_type_is_refused(create_agent_fn, rebuilds) -> None:
    """A typo must not quietly mean "standard" and swap the user's pipeline."""
    agent = _make_agent(_realtime_config(), llm=FakeRealtimeModel())

    await _update(
        agent,
        {"updateType": "voice", "config": {"pipelineType": "reealtime"}},
        create_agent_fn,
    )

    assert rebuilds == []
    assert agent.kwami_config.voice.pipeline_type == "realtime"


async def test_switching_to_the_current_pipeline_is_not_a_rebuild(
    create_agent_fn, rebuilds
) -> None:
    agent = _make_agent(_realtime_config(), llm=FakeRealtimeModel())

    await _update(
        agent,
        {"updateType": "voice", "config": {"pipelineType": "realtime"}},
        create_agent_fn,
    )

    assert rebuilds == []


# -- llm updates on a realtime session --------------------------------------


async def test_llm_update_on_realtime_retargets_the_realtime_model(create_agent_fn, rebuilds):
    """The models panel sends `updateType: "llm"` whatever the pipeline is.

    Handled as a standard-pipeline LLM change, this built an STT+LLM+TTS agent
    underneath a realtime session -- a silent downgrade off the realtime model
    the user had chosen.
    """
    agent = _make_agent(_realtime_config(), llm=FakeRealtimeModel())

    await _update(
        agent,
        {"updateType": "llm", "config": {"provider": "openai", "model": "gpt-realtime-mini"}},
        create_agent_fn,
    )

    assert len(rebuilds) == 1
    rebuilt = rebuilds[0]["config"].voice
    assert rebuilt.pipeline_type == "realtime"
    assert rebuilt.realtime_model == "gpt-realtime-mini"


async def test_llm_temperature_on_realtime_goes_live(create_agent_fn, rebuilds) -> None:
    model = FakeRealtimeModel()
    agent = _make_agent(_realtime_config(llm_temperature=0.7), llm=model)

    await _update(
        agent,
        {"updateType": "llm", "config": {"temperature": 0.2}},
        create_agent_fn,
    )

    assert model.updates == [{"temperature": 0.2}]
    assert rebuilds == []


async def test_llm_update_on_standard_still_rebuilds_the_llm(create_agent_fn, rebuilds) -> None:
    """The realtime branch must not capture the standard pipeline's own path."""
    agent = _make_agent(KwamiConfig())

    await _update(
        agent,
        {"updateType": "llm", "config": {"provider": "anthropic", "model": "claude-opus-4"}},
        create_agent_fn,
    )

    rebuilt = rebuilds[0]["config"].voice
    assert rebuilt.llm_provider == "anthropic"
    assert rebuilt.llm_model == "claude-opus-4"
    assert rebuilt.pipeline_type == "standard"
