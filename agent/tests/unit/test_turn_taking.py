"""Turn-taking for a Spanish conversation (runtime/turn_taking.py).

Checked against the framework's own resolution, not against our dicts: a key
misspelt in a TypedDict is silently ignored, and the session would then run on
library defaults while this module claimed otherwise.
"""

from __future__ import annotations

from typing import Any

from livekit.agents import AgentSession, inference
from livekit.agents.types import NOT_GIVEN

from src.agent import KwamiAgent
from src.domain import KwamiConfig
from src.runtime.turn_taking import (
    ENDPOINTING_MAX_DELAY,
    ENDPOINTING_MIN_DELAY,
    agent_turn_handling,
    session_turn_handling,
)


async def test_the_session_takes_the_spanish_tuning() -> None:
    session = AgentSession(vad=None, turn_handling=session_turn_handling())
    options = session.options

    assert isinstance(session.turn_detection, inference.TurnDetector)
    endpointing = options.turn_handling["endpointing"]
    assert endpointing["mode"] == "dynamic"
    assert endpointing["min_delay"] == ENDPOINTING_MIN_DELAY
    assert endpointing["max_delay"] == ENDPOINTING_MAX_DELAY
    preemptive = options.preemptive_generation
    assert preemptive["enabled"] is True
    assert preemptive["preemptive_tts"] is False, "a discarded guess must not cost audio"
    assert options.interruption["resume_false_interruption"] is True


def test_adaptive_interruptions_on_the_standard_pipeline_only() -> None:
    standard = KwamiAgent(KwamiConfig(), turn_handling=agent_turn_handling("standard"))
    assert standard.interruption_detection == "adaptive"

    assert agent_turn_handling("realtime") is NOT_GIVEN
    realtime = KwamiAgent(KwamiConfig(), turn_handling=agent_turn_handling("realtime"))
    assert realtime.interruption_detection is NOT_GIVEN


# =============================================================================
# Built from the config
# =============================================================================


async def test_the_session_takes_the_configured_values() -> None:
    from src.domain import KwamiVoiceConfig

    voice = KwamiVoiceConfig(
        turn_detector="stt",
        min_endpointing_delay=0.8,
        max_endpointing_delay=0.5,
        allow_interruptions=False,
        min_interruption_duration=1.0,
        min_interruption_words=3,
        preemptive_generation=False,
    )
    session = AgentSession(vad=None, turn_handling=session_turn_handling(voice))
    options = session.options

    assert session.turn_detection == "stt"
    assert options.endpointing["min_delay"] == 0.8
    assert options.endpointing["max_delay"] == 0.8, "a ceiling under the floor is the floor"
    assert options.interruption["enabled"] is False
    assert options.interruption["min_duration"] == 1.0
    assert options.interruption["min_words"] == 3
    assert options.preemptive_generation["enabled"] is False


# =============================================================================
# Noise cancellation
# =============================================================================


class Filters:
    """Stands in for livekit-plugins-noise-cancellation, naming what was built."""

    def BVC(self) -> str:  # noqa: N802
        return "BVC"

    def NC(self) -> str:  # noqa: N802
        return "NC"

    def BVCTelephony(self) -> str:  # noqa: N802
        return "BVCTelephony"


class Participant:
    def __init__(self, kind: Any = None, identity: str = "user-1") -> None:
        self.kind = kind
        self.identity = identity


def test_noise_cancellation_follows_the_config(monkeypatch) -> None:
    from livekit import rtc

    from src.domain import KwamiVoiceConfig
    from src.runtime import turn_taking

    monkeypatch.setattr(turn_taking, "noise_cancellation", Filters())
    pick = turn_taking.noise_cancellation_for
    user = Participant()

    assert pick(None, user) == "BVC", "on before the app's config arrives"
    assert pick(KwamiVoiceConfig(), user) == "BVC"
    assert pick(KwamiVoiceConfig(noise_cancellation_mode="krisp"), user) == "NC"
    assert pick(KwamiVoiceConfig(noise_suppression=False), user) is None
    caller = Participant(kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP)
    assert pick(KwamiVoiceConfig(noise_cancellation_mode="krisp"), caller) == "BVCTelephony"


def test_without_the_plugin_there_is_no_filter(monkeypatch) -> None:
    from src.runtime import turn_taking

    monkeypatch.setattr(turn_taking, "noise_cancellation", None)
    assert turn_taking.noise_cancellation_for(None, Participant()) is None


def test_the_selector_reads_the_config_live(monkeypatch, caplog) -> None:
    """The panel's choice has to be read when the track arrives, not at start."""
    import logging

    from livekit.agents.voice.room_io.types import NoiseCancellationParams

    from src.domain import KwamiVoiceConfig
    from src.runtime import turn_taking

    monkeypatch.setattr(turn_taking, "noise_cancellation", Filters())
    voice = KwamiVoiceConfig()
    select = turn_taking.noise_cancellation_selector(lambda: voice)
    params = NoiseCancellationParams(Participant(), object())  # type: ignore[arg-type]

    with caplog.at_level(logging.INFO):
        assert select(params) == "BVC"
        voice.noise_suppression = False
        assert select(params) is None
    assert "user-1: off" in caplog.text
