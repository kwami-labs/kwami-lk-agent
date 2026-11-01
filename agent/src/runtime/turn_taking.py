"""When the user has finished talking, when they are cutting in, and what noise
is allowed to count as either.

livekit-agents 1.8.8 already builds `inference.TurnDetector()` when nothing is
passed: the multilingual audio end-of-turn model, run on LiveKit Inference on a
hosted worker (`turn-detector-v1`) and locally elsewhere (`v1-mini`, which has
its own Spanish threshold). It is passed here explicitly so a library default
changing under a minor bump cannot silently change how Jarvis takes turns.

The defaults live on `KwamiVoiceConfig` and are tuned for Spanish
conversation; the app's Enhancements panel overrides them per session
(handlers/listening.py applies them to the running session):

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
* **Noise, removed before any of the above hears it.** Without noise
  cancellation a door, a keyboard or a TV crossed the VAD threshold, counted as
  the user talking, and cut the agent off mid-sentence; with nobody actually
  speaking afterwards it then sat in silence waiting for a turn that never
  came. BVC drops the noise and the background voices on the agent, before the
  VAD, the turn detector, the STT or a realtime model see the audio.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from livekit import rtc
from livekit.agents import inference
from livekit.agents.types import NOT_GIVEN
from livekit.agents.voice.room_io.types import NoiseCancellationParams
from livekit.agents.voice.turn import TurnHandlingOptions

from ..domain import KwamiVoiceConfig
from ..utils.logging import get_logger

try:
    from livekit.plugins import noise_cancellation
except Exception:  # pragma: no cover - a core dependency; guarded so a broken
    # native build degrades to no cancellation instead of no agent at all.
    noise_cancellation = None  # type: ignore[assignment]

logger = get_logger("turn_taking")

REALTIME_PIPELINE = "realtime"

_DEFAULTS = KwamiVoiceConfig()

#: Seconds of silence before the turn may end; learned upwards from here.
ENDPOINTING_MIN_DELAY = _DEFAULTS.min_endpointing_delay
#: The longest the agent waits for a user who has gone quiet.
ENDPOINTING_MAX_DELAY = _DEFAULTS.max_endpointing_delay
#: Speech shorter than this does not interrupt the agent.
INTERRUPTION_MIN_DURATION = _DEFAULTS.min_interruption_duration
#: Silence after an interruption before it counts as a false one and the agent
#: picks up where it was.
FALSE_INTERRUPTION_TIMEOUT = 1.5
#: Longer turns are likely to change before they end; no early generation.
PREEMPTIVE_MAX_SPEECH = 8.0


def turn_detection_for(voice: KwamiVoiceConfig) -> Any:
    """The session's turn detection mode for `voice.turn_detector`."""
    if voice.turn_detector in ("vad", "stt"):
        return voice.turn_detector
    return inference.TurnDetector()


def endpointing_for(voice: KwamiVoiceConfig) -> dict[str, Any]:
    """Dynamic endpointing between the configured floor and ceiling."""
    low = max(0.0, voice.min_endpointing_delay)
    # A ceiling under the floor would make the framework wait the floor anyway
    # and log nothing; clamp so the stored config says what actually happens.
    return {"mode": "dynamic", "min_delay": low, "max_delay": max(low, voice.max_endpointing_delay)}


def session_turn_handling(voice: KwamiVoiceConfig | None = None) -> TurnHandlingOptions:
    """The session-wide turn handling: detector, endpointing, preemption."""
    voice = voice or _DEFAULTS
    return TurnHandlingOptions(
        turn_detection=turn_detection_for(voice),
        endpointing=endpointing_for(voice),  # type: ignore[typeddict-item]
        interruption={
            "enabled": voice.allow_interruptions,
            "min_duration": voice.min_interruption_duration,
            "min_words": voice.min_interruption_words,
            "resume_false_interruption": True,
            "false_interruption_timeout": FALSE_INTERRUPTION_TIMEOUT,
        },
        preemptive_generation={
            "enabled": voice.preemptive_generation,
            "preemptive_tts": False,
            "max_speech_duration": PREEMPTIVE_MAX_SPEECH,
        },
    )


def agent_turn_handling(pipeline_type: str) -> Any:
    """Per-agent overrides: adaptive interruptions on the standard pipeline."""
    if pipeline_type == REALTIME_PIPELINE:
        return NOT_GIVEN
    return TurnHandlingOptions(interruption={"mode": "adaptive"})


def noise_cancellation_for(voice: KwamiVoiceConfig | None, participant: Any) -> Any:
    """The noise filter for one participant's microphone, or None for none."""
    if noise_cancellation is None or (voice is not None and not voice.noise_suppression):
        return None
    if getattr(participant, "kind", None) == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        # Phone audio is narrowband; the telephony model is trained on it.
        return noise_cancellation.BVCTelephony()
    mode = voice.noise_cancellation_mode if voice is not None else _DEFAULTS.noise_cancellation_mode
    return noise_cancellation.NC() if mode == "krisp" else noise_cancellation.BVC()


def noise_cancellation_selector(
    current_voice: Callable[[], KwamiVoiceConfig | None],
) -> Callable[[NoiseCancellationParams], Any]:
    """A room-input selector that reads the live config each time a track arrives.

    The filter is fixed for the life of an audio stream, so a change made in
    the panel mid-call applies from the next microphone track (a reconnect, or
    a new device), not to the one already flowing.
    """

    def select(params: NoiseCancellationParams) -> Any:
        voice = current_voice()
        selected = noise_cancellation_for(voice, params.participant)
        logger.info(
            "Noise cancellation for %s: %s",
            getattr(params.participant, "identity", "?"),
            type(selected).__name__ if selected is not None else "off",
        )
        return selected

    return select
