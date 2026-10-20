"""Turn-taking for a Spanish conversation (runtime/turn_taking.py).

Checked against the framework's own resolution, not against our dicts: a key
misspelt in a TypedDict is silently ignored, and the session would then run on
library defaults while this module claimed otherwise.
"""

from __future__ import annotations

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
