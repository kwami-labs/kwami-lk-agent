"""The app's listening settings reach the running session (handlers/listening.py).

They used to reach nothing: the panel's `{enhancements, vad}` update fell
through `update_voice` as a TTS no-op, so tuning listening for a noisy room --
from the panel or from the agent's own `set_enhancement_control` tool --
changed nothing. Checked against a real `AgentSession`, because a key misspelt
in a TypedDict is silently ignored by the framework.
"""

from __future__ import annotations

import logging
from typing import Any

from livekit.agents import AgentSession, inference

from src.agent import KwamiAgent
from src.domain import KwamiConfig, KwamiVoiceConfig
from src.handlers import listening as listening_module
from src.handlers.config_handler import handle_config_update, handle_full_config
from src.handlers.listening import (
    apply_listening,
    apply_to_session,
    copy_listening,
    is_listening_only,
    parse_listening,
)
from src.runtime.container import AgentDeps
from src.runtime.turn_taking import session_turn_handling
from src.session import SessionState

#: What the app's `syncAllConfigToBackend` sends on every connect.
PANEL = {
    "enhancements": {
        "turnDetection": {
            "enabled": True,
            "mode": "model",
            "model": "multilingual",
            "minEndpointingDelay": 0.6,
            "maxEndpointingDelay": 3.0,
            "allowInterruptions": True,
            "minInterruptionDuration": 0.7,
            "minInterruptionWords": 2,
        },
        "noiseCancellation": {"enabled": True, "mode": "krisp"},
        "echoCancellation": True,
        "autoGainControl": False,
        "preemptiveGeneration": False,
    },
    "vad": {
        "provider": "silero",
        "threshold": 0.6,
        "minSpeechDuration": 0.2,
        "minSilenceDuration": 0.4,
    },
}


class LiveSession(AgentSession):
    """A real session, whose agent switches are recorded rather than started."""

    def update_agent(self, agent: Any) -> None:  # type: ignore[override]
        self.switched_to = agent


def real_session() -> LiveSession:
    return LiveSession(vad=None, userdata=AgentDeps(), turn_handling=session_turn_handling())


class RecordingAgent:
    def __init__(self, config: KwamiConfig | None = None) -> None:
        self.kwami_config = config or KwamiConfig()
        self.vad_updates: list[Any] = []
        self._memory = None
        self._browser_session = None
        self.client_tools = type("T", (), {"pending_calls": {}})()
        self.room = None
        self.tts = None
        self.stt = None

    def update_options(self, *, vad: Any) -> None:
        self.vad_updates.append(vad)


# =============================================================================
# Parsing
# =============================================================================


def test_the_panels_nested_shape_is_read() -> None:
    assert parse_listening(PANEL) == {
        "turn_detector": "model",
        "min_endpointing_delay": 0.6,
        "max_endpointing_delay": 3.0,
        "min_interruption_duration": 0.7,
        "min_interruption_words": 2,
        "allow_interruptions": True,
        "noise_suppression": True,
        "noise_cancellation_mode": "krisp",
        "echo_cancellation": True,
        "auto_gain_control": False,
        "preemptive_generation": False,
        "vad_threshold": 0.6,
        "vad_min_speech_duration": 0.2,
        "vad_min_silence_duration": 0.4,
    }


def test_the_sdks_flat_shape_is_read() -> None:
    """`VoiceSession.toLiveKitConfig()` puts the same fields on the voice object."""
    flat = {
        "turnDetection": "stt",
        "allowInterruptions": False,
        "minInterruptionDuration": 0.3,
        "minEndpointingDelay": 0.5,
        "noiseCancellation": {"enabled": True, "mode": "bvc"},
        "preemptiveGeneration": True,
    }
    assert parse_listening(flat) == {
        "turn_detector": "stt",
        "allow_interruptions": False,
        "min_interruption_duration": 0.3,
        "min_endpointing_delay": 0.5,
        "noise_suppression": True,
        "noise_cancellation_mode": "bvc",
        "preemptive_generation": True,
    }


def test_turn_detection_off_means_plain_vad() -> None:
    payload = {"enhancements": {"turnDetection": {"enabled": False, "mode": "model"}}}
    assert parse_listening(payload)["turn_detector"] == "vad"


def test_manual_turns_are_refused(caplog) -> None:
    """Nothing commits the user's turns: honoured, the agent would never answer."""
    with caplog.at_level(logging.WARNING):
        values = parse_listening({"enhancements": {"turnDetection": {"mode": "manual"}}})
    assert "turn_detector" not in values
    assert "manual" in caplog.text


def test_the_panels_default_noise_mode_is_bvc() -> None:
    payload = {"enhancements": {"noiseCancellation": {"enabled": False, "mode": "default"}}}
    assert parse_listening(payload) == {
        "noise_suppression": False,
        "noise_cancellation_mode": "bvc",
    }


def test_values_out_of_range_are_clamped() -> None:
    payload = {
        "enhancements": {"turnDetection": {"minEndpointingDelay": -1, "minInterruptionWords": -3}},
        "vad": {"threshold": 1.5, "minSilenceDuration": 0},
    }
    assert parse_listening(payload) == {
        "min_endpointing_delay": 0.0,
        "min_interruption_words": 0,
        "vad_threshold": 0.95,
        "vad_min_silence_duration": 0.05,
    }


def test_nothing_to_read_is_nothing() -> None:
    assert parse_listening(None) == {}
    assert parse_listening({"tts_voice": "nova"}) == {}
    assert parse_listening({"enhancements": {"noiseCancellation": {"mode": "loud"}}}) == {}


def test_only_listening_keys_make_a_listening_only_message() -> None:
    assert is_listening_only(PANEL)
    assert not is_listening_only({**PANEL, "tts_voice": "nova"})
    assert not is_listening_only({})
    assert not is_listening_only(None)


# =============================================================================
# Config
# =============================================================================


def test_apply_reports_only_what_changed() -> None:
    voice = KwamiVoiceConfig()
    changed = apply_listening(voice, {"min_interruption_duration": 0.4, "vad_threshold": 0.7})
    assert changed == {"vad_threshold"}
    assert voice.vad_threshold == 0.7


def test_a_rebuilt_config_keeps_the_listening_settings() -> None:
    source = KwamiVoiceConfig(noise_suppression=False, min_endpointing_delay=0.9, llm_model="x")
    target = KwamiVoiceConfig()
    copy_listening(source, target)
    assert target.noise_suppression is False
    assert target.min_endpointing_delay == 0.9
    assert target.llm_model != "x", "only listening fields are carried"

    copy_listening(object(), target)  # a stand-in agent's config is ignored
    assert target.min_endpointing_delay == 0.9


# =============================================================================
# The running session
# =============================================================================


async def test_the_session_listens_as_configured(monkeypatch) -> None:
    monkeypatch.setattr(
        listening_module, "create_vad", lambda voice, prewarmed=None: ("vad", voice)
    )
    session = real_session()
    agent = RecordingAgent()
    voice = agent.kwami_config.voice
    changed = apply_listening(voice, parse_listening(PANEL))

    apply_to_session(session, agent, voice, changed, prewarmed_vad="prewarmed")

    options = session.options
    assert options.endpointing["min_delay"] == 0.6
    assert options.endpointing["max_delay"] == 3.0
    assert options.endpointing["mode"] == "dynamic"
    assert options.interruption["min_duration"] == 0.7
    assert options.interruption["min_words"] == 2
    assert options.interruption["enabled"] is True
    assert options.interruption["resume_false_interruption"] is True
    assert options.preemptive_generation["enabled"] is False
    assert agent.vad_updates == [("vad", voice)]


async def test_switching_the_turn_detector_reaches_the_session() -> None:
    session = real_session()
    voice = KwamiVoiceConfig(turn_detector="vad")

    apply_to_session(session, None, voice, {"turn_detector"})
    assert session.turn_detection == "vad"

    voice.turn_detector = "model"
    apply_to_session(session, None, voice, {"turn_detector"})
    assert isinstance(session.turn_detection, inference.TurnDetector)


def test_no_change_touches_nothing() -> None:
    class Untouchable:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"touched {name}")

    apply_to_session(Untouchable(), None, KwamiVoiceConfig(), set())  # type: ignore[arg-type]


def test_a_failure_is_logged_not_raised(caplog) -> None:
    """These run inside config handlers that swallow errors and skip the rest."""

    class Broken:
        def update_options(self, **kwargs: Any) -> None:
            raise RuntimeError("no")

    with caplog.at_level(logging.ERROR):
        apply_to_session(Broken(), None, KwamiVoiceConfig(), {"turn_detector"})  # type: ignore[arg-type]
    assert "Could not apply listening settings" in caplog.text


async def test_a_noise_change_says_when_it_applies(caplog) -> None:
    session = real_session()
    with caplog.at_level(logging.INFO):
        apply_to_session(session, None, KwamiVoiceConfig(), {"noise_suppression"})
    assert "next microphone track" in caplog.text


# =============================================================================
# Through the config handlers
# =============================================================================


class Swaps:
    """Records agent switches; a listening-only update must make none."""

    def __init__(self) -> None:
        self.switched: list[Any] = []

    def __call__(self, config, vad, memory=None, skip_greeting=False):
        agent = RecordingAgent(config)
        self.switched.append(agent)
        return agent


def state_with(agent: Any) -> SessionState:
    state = SessionState()
    state.current_agent = agent
    return state


async def test_the_panels_update_applies_without_a_rebuild(monkeypatch) -> None:
    monkeypatch.setattr(listening_module, "create_vad", lambda voice, prewarmed=None: "vad")
    session = real_session()
    agent = KwamiAgent(KwamiConfig())
    swaps = Swaps()
    message = {"type": "config_update", "updateType": "voice", "config": PANEL}

    await handle_config_update(session, state_with(agent), message, None, swaps)

    assert swaps.switched == [], "listening needs no new pipeline"
    assert agent.kwami_config.voice.noise_cancellation_mode == "krisp"
    assert session.options.interruption["min_duration"] == 0.7
    assert agent.vad == "vad"


async def test_a_realtime_session_takes_the_panels_update_too() -> None:
    session = real_session()
    config = KwamiConfig()
    config.voice.pipeline_type = "realtime"
    agent = KwamiAgent(config)
    swaps = Swaps()
    message = {
        "type": "config_update",
        "updateType": "voice",
        "config": {"enhancements": {"turnDetection": {"minInterruptionDuration": 1.0}}},
    }

    await handle_config_update(session, state_with(agent), message, None, swaps)

    assert session.options.interruption["min_duration"] == 1.0
    assert swaps.switched == []


async def test_a_full_config_keeps_what_the_session_listens_with() -> None:
    """A kwami switch resends `config` without the panel; the panel's values,
    already live in the session, must not silently revert in the config."""
    session = real_session()
    previous = RecordingAgent()
    previous.kwami_config.voice.min_interruption_duration = 0.9
    swaps = Swaps()
    message = {
        "type": "config",
        "voice": {"enhancements": {"turnDetection": {"maxEndpointingDelay": 4.0}}},
    }

    await handle_full_config(session, state_with(previous), message, None, swaps)

    voice = swaps.switched[-1].kwami_config.voice
    assert voice.min_interruption_duration == 0.9
    assert voice.max_endpointing_delay == 4.0
    assert session.options.endpointing["max_delay"] == 4.0
