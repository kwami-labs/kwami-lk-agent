"""The app's listening settings: noise, turn-taking, interruptions and VAD.

The Enhancements panel sends these on every connect and on every change, and so
does `set_enhancement_control`, the tool the agent is told to use "if you
interrupt too much or the room is noisy". Nothing on this side read them: the
`config_update` carrying `{enhancements, vad}` fell through the TTS branch of
`update_voice` as a no-op, so the agent kept its built-in values whatever the
panel showed, and tuning listening for a noisy room changed nothing.

Two wire shapes reach here. The app sends the panel's state nested
(`enhancements.turnDetection.minEndpointingDelay`, `vad.minSilenceDuration`);
the SDK's `VoiceSession.toLiveKitConfig()` flattens the same fields onto the
voice object (`minEndpointingDelay`, `turnDetection: "model"`). Both are read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..domain import KwamiVoiceConfig, boolean, integer, number, section, text
from ..factories.vad import create_vad
from ..runtime.turn_taking import endpointing_for, turn_detection_for
from ..utils.logging import get_logger

if TYPE_CHECKING:
    from livekit.agents import AgentSession

logger = get_logger("listening")

#: Every config field this module owns.
LISTENING_FIELDS = (
    "echo_cancellation",
    "auto_gain_control",
    "noise_suppression",
    "noise_cancellation_mode",
    "turn_detector",
    "min_endpointing_delay",
    "max_endpointing_delay",
    "allow_interruptions",
    "min_interruption_duration",
    "min_interruption_words",
    "preemptive_generation",
    "vad_threshold",
    "vad_min_speech_duration",
    "vad_min_silence_duration",
)

#: Wire keys that carry nothing but listening settings. A message made only of
#: these has nothing left for the TTS/STT/realtime branches.
LISTENING_KEYS = frozenset(
    {
        "enhancements",
        "vad",
        "turnDetection",
        "allowInterruptions",
        "minInterruptionDuration",
        "minInterruptionWords",
        "minEndpointingDelay",
        "maxEndpointingDelay",
        "noiseCancellation",
        "echoCancellation",
        "autoGainControl",
        "preemptiveGeneration",
        "userAwayTimeout",
        "minConsecutiveSpeechDelay",
    }
)

_ENDPOINTING = {"min_endpointing_delay", "max_endpointing_delay"}
_VAD = {"vad_threshold", "vad_min_speech_duration", "vad_min_silence_duration"}
_NOISE = {"noise_suppression", "noise_cancellation_mode"}

#: The panel's "Por defecto" is this agent's default, BVC.
_NOISE_MODES = {"bvc": "bvc", "default": "bvc", "krisp": "krisp"}


def _clamped(value: float | None, low: float, high: float) -> float | None:
    return None if value is None else min(max(value, low), high)


def parse_listening(voice_data: Any) -> dict[str, Any]:
    """The listening fields `voice_data` sets, as config values. Never raises."""
    if not isinstance(voice_data, dict):
        return {}
    enhancements = section(voice_data, "enhancements")
    turn = section(enhancements, "turnDetection") or section(enhancements, "turn_detection")
    flat_turn = voice_data.get("turnDetection")
    values: dict[str, Any] = {}

    def first(read: Any, key: str) -> Any:
        """`key` from the nested block, then from the flat voice object."""
        found = read(turn, key)
        return found if found is not None else read(voice_data, key)

    mode = text(turn, "mode") or (flat_turn if isinstance(flat_turn, str) else None)
    if boolean(turn, "enabled") is False:
        # "Off" in the panel means no end-of-turn model: plain VAD silence.
        values["turn_detector"] = "vad"
    elif mode in ("model", "vad", "stt"):
        values["turn_detector"] = mode
    elif mode == "manual":
        # Manual turns wait for the client to commit each one, and no client
        # does: honoured, it is an agent that never answers.
        logger.warning("Ignoring manual turn detection: nothing commits the user's turns")

    for wire, field, low, high in (
        ("minEndpointingDelay", "min_endpointing_delay", 0.0, 5.0),
        ("maxEndpointingDelay", "max_endpointing_delay", 0.0, 10.0),
        ("minInterruptionDuration", "min_interruption_duration", 0.0, 5.0),
    ):
        value = _clamped(first(number, wire), low, high)
        if value is not None:
            values[field] = value

    words = first(integer, "minInterruptionWords")
    if words is not None:
        values["min_interruption_words"] = max(0, words)
    allow = first(boolean, "allowInterruptions")
    if allow is not None:
        values["allow_interruptions"] = allow

    noise = section(enhancements, "noiseCancellation") or section(voice_data, "noiseCancellation")
    noise_on = boolean(noise, "enabled")
    if noise_on is not None:
        values["noise_suppression"] = noise_on
    noise_mode = _NOISE_MODES.get((text(noise, "mode") or "").lower())
    if noise_mode is not None:
        values["noise_cancellation_mode"] = noise_mode

    for wire, field in (
        ("echoCancellation", "echo_cancellation"),
        ("autoGainControl", "auto_gain_control"),
        ("preemptiveGeneration", "preemptive_generation"),
    ):
        flag = boolean(enhancements, wire)
        flag = flag if flag is not None else boolean(voice_data, wire)
        if flag is not None:
            values[field] = flag

    vad = section(voice_data, "vad")
    for wire, field, low, high in (
        ("threshold", "vad_threshold", 0.05, 0.95),
        ("minSpeechDuration", "vad_min_speech_duration", 0.0, 2.0),
        ("minSilenceDuration", "vad_min_silence_duration", 0.05, 5.0),
    ):
        value = _clamped(number(vad, wire), low, high)
        if value is not None:
            values[field] = value
    return values


def is_listening_only(voice_data: Any) -> bool:
    """True when `voice_data` carries listening settings and nothing else."""
    return isinstance(voice_data, dict) and bool(voice_data) and set(voice_data) <= LISTENING_KEYS


def copy_listening(source: Any, target: KwamiVoiceConfig) -> None:
    """Carry the listening fields over, so a rebuilt config keeps them."""
    if not isinstance(source, KwamiVoiceConfig):
        return
    for field in LISTENING_FIELDS:
        setattr(target, field, getattr(source, field))


def apply_listening(voice: KwamiVoiceConfig, values: dict[str, Any]) -> set[str]:
    """Write `values` onto `voice`. Returns the fields that actually changed."""
    changed = set()
    for field, value in values.items():
        if getattr(voice, field) != value:
            setattr(voice, field, value)
            changed.add(field)
    return changed


def apply_to_session(
    session: AgentSession,
    agent: Any,
    voice: KwamiVoiceConfig,
    changed: set[str],
    prewarmed_vad: Any = None,
) -> None:
    """Make the running session listen the way `voice` says.

    Endpointing, turn detection and the interruption and preemption thresholds
    are session-wide and take effect on the next turn. The VAD belongs to the
    agent and is swapped in place. Noise cancellation is fixed per audio stream
    (runtime/turn_taking.py), so it applies from the next microphone track.
    """
    if not changed:
        return
    try:
        _apply_to_session(session, agent, voice, changed, prewarmed_vad)
    except Exception:
        # Listening is tuning: a failure here must not abort the config message
        # it arrived in, which would skip the greeting and the rest of it.
        logger.exception("Could not apply listening settings: %s", ", ".join(sorted(changed)))
        return
    logger.info("Listening updated: %s", ", ".join(sorted(changed)))


def _apply_to_session(
    session: AgentSession,
    agent: Any,
    voice: KwamiVoiceConfig,
    changed: set[str],
    prewarmed_vad: Any,
) -> None:
    if "turn_detector" in changed:
        session.update_options(turn_detection=turn_detection_for(voice))
    if changed & _ENDPOINTING:
        session.update_options(endpointing_opts=endpointing_for(voice))  # type: ignore[arg-type]

    # Read live from these dicts on every turn by the framework, and resolved
    # into fresh per-session copies, so writing to them is the update.
    interruption = session.options.interruption
    interruption["enabled"] = voice.allow_interruptions
    interruption["min_duration"] = voice.min_interruption_duration
    interruption["min_words"] = voice.min_interruption_words
    session.options.preemptive_generation["enabled"] = voice.preemptive_generation

    if changed & _VAD and agent is not None and hasattr(agent, "update_options"):
        agent.update_options(vad=create_vad(voice, prewarmed=prewarmed_vad))
    if changed & _NOISE:
        logger.info("Noise cancellation change applies from the next microphone track")
