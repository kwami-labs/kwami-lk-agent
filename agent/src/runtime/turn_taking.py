"""When the user has finished talking, and when they are cutting in.

livekit-agents 1.8.8 already builds `inference.TurnDetector()` when nothing is
passed: the multilingual audio end-of-turn model, run on LiveKit Inference on a
hosted worker (`turn-detector-v1`) and locally elsewhere (`v1-mini`, which has
its own Spanish threshold). It is passed here explicitly so a library default
changing under a minor bump cannot silently change how Jarvis takes turns.

The rest is tuned for Spanish conversation:

* **Endpointing, dynamic.** The floor is learned from the user's own pauses, so
  a speaker who stops mid-sentence ("pues... a ver...") is not cut off, and one
  who answers in two words is not kept waiting. It starts at 0.4 s -- Spanish
  turns hand over quickly -- and never waits past 2.5 s.
* **Preemptive generation.** The LLM starts on the interim transcript while the
  turn detector decides; only text is generated early. TTS waits for the
  confirmed turn, so a discarded guess costs tokens, never synthesized audio.
* **Interruptions, adaptive.** LiveKit's barge-in model tells "sí, sí" and a
  cough from a real interruption. It is on by default only on hosted workers,
  so it is asked for explicitly -- per agent, and only on the standard
  pipeline: a realtime model detects turns on its own server and the framework
  would only warn and switch it off.
"""

from __future__ import annotations

from typing import Any

from livekit.agents import inference
from livekit.agents.types import NOT_GIVEN
from livekit.agents.voice.turn import TurnHandlingOptions

REALTIME_PIPELINE = "realtime"

#: Seconds of silence before the turn may end; learned upwards from here.
ENDPOINTING_MIN_DELAY = 0.4
#: The longest the agent waits for a user who has gone quiet.
ENDPOINTING_MAX_DELAY = 2.5
#: Speech shorter than this does not interrupt the agent.
INTERRUPTION_MIN_DURATION = 0.4
#: Silence after an interruption before it counts as a false one and the agent
#: picks up where it was.
FALSE_INTERRUPTION_TIMEOUT = 1.5
#: Longer turns are likely to change before they end; no early generation.
PREEMPTIVE_MAX_SPEECH = 8.0


def session_turn_handling() -> TurnHandlingOptions:
    """The session-wide turn handling: detector, endpointing, preemption."""
    return TurnHandlingOptions(
        turn_detection=inference.TurnDetector(),
        endpointing={
            "mode": "dynamic",
            "min_delay": ENDPOINTING_MIN_DELAY,
            "max_delay": ENDPOINTING_MAX_DELAY,
        },
        interruption={
            "min_duration": INTERRUPTION_MIN_DURATION,
            "resume_false_interruption": True,
            "false_interruption_timeout": FALSE_INTERRUPTION_TIMEOUT,
        },
        preemptive_generation={
            "enabled": True,
            "preemptive_tts": False,
            "max_speech_duration": PREEMPTIVE_MAX_SPEECH,
        },
    )


def agent_turn_handling(pipeline_type: str) -> Any:
    """Per-agent overrides: adaptive interruptions on the standard pipeline."""
    if pipeline_type == REALTIME_PIPELINE:
        return NOT_GIVEN
    return TurnHandlingOptions(interruption={"mode": "adaptive"})
