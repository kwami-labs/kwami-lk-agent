"""Realtime model factory.

`openai.realtime.ServerVadOptions` never existed in livekit-plugins-openai
1.3.x, so every `pipeline_type="realtime"` session raised AttributeError before
it produced a single frame. The real kwarg type is `TurnDetection` from the
OpenAI SDK. This module also follows the fallback-and-log shape of
`factories/stt.py`: a provider that cannot be built logs an error and degrades
to a working default instead of taking the session down.
"""

from typing import Any, cast

from livekit.agents.types import NOT_GIVEN
from livekit.plugins import openai

from ..constants import resolve_realtime_voice
from ..domain import KwamiVoiceConfig
from ..utils.logging import get_logger
from ..utils.provider import strip_model_prefix
from .optional import optional_plugin

try:
    from openai.types.beta.realtime.session import TurnDetection
except ImportError:  # pragma: no cover - openai SDK is a hard dependency
    TurnDetection = None  # type: ignore

try:
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - arrives with livekit-plugins-google
    genai_types = None  # type: ignore

# Guarded like the other factories: typed `Any`, so mypy accepts the None branch
# whether or not the plugin happens to be installed where it runs.
google = optional_plugin("google")

logger = get_logger("factories")

DEFAULT_OPENAI_REALTIME_MODEL = "gpt-realtime"
DEFAULT_OPENAI_REALTIME_VOICE = "marin"
DEFAULT_GOOGLE_REALTIME_MODEL = "gemini-3.8-live"
DEFAULT_GOOGLE_REALTIME_VOICE = "Charon"


def _server_vad(config: KwamiVoiceConfig):
    """OpenAI's server-side turn detection, or NOT_GIVEN for the provider default.

    It was always `server_vad` with the Silero settings -- a 0.5 threshold and
    300 ms of silence -- which a door or a keyboard crosses: the model took the
    noise for the user starting to talk, cut its own answer, and with nothing
    said afterwards never replied. `semantic_vad` decides on what was said, not
    on loudness, and still interrupts for real speech.
    """
    if TurnDetection is None or config.turn_detection == "none":
        return NOT_GIVEN
    if config.turn_detection == "server_vad":
        return TurnDetection(
            type="server_vad",
            threshold=config.vad_threshold,
            prefix_padding_ms=300,
            silence_duration_ms=int(config.vad_min_silence_duration * 1000),
        )
    return TurnDetection(
        type="semantic_vad",
        eagerness="auto",
        create_response=True,
        interrupt_response=config.allow_interruptions,
    )


def _gemini_activity_detection(config: KwamiVoiceConfig) -> Any:
    """Gemini Live's server-side turn detection, or NOT_GIVEN for its default.

    Live's default start-of-speech sensitivity is high: background noise counts
    as the user barging in, and Live drops the answer it was speaking. Low
    sensitivity needs clearer speech to start a turn; the end of the turn is
    left to Live's default, which does not cut Spanish pauses short.
    """
    if config.turn_detection == "none" or genai_types is None:
        return NOT_GIVEN
    start = (
        genai_types.StartSensitivity.START_SENSITIVITY_HIGH
        if config.turn_detection == "server_vad"
        else genai_types.StartSensitivity.START_SENSITIVITY_LOW
    )
    return genai_types.RealtimeInputConfig(
        automatic_activity_detection=genai_types.AutomaticActivityDetection(
            start_of_speech_sensitivity=start,
            prefix_padding_ms=300,
        ),
        activity_handling=(
            genai_types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS
            if config.allow_interruptions
            else genai_types.ActivityHandling.NO_INTERRUPTION
        ),
    )


def _realtime_voice(provider: str, voice: str | None, default: str) -> str:
    """The provider's spelling of `voice`, or `default` if it doesn't offer it.

    A voice saved for another provider (the config default is OpenAI's
    "alloy") makes Gemini Live close the socket with 1007 at connect, which on
    the realtime pipeline means a silent agent rather than an error the user
    can see.
    """
    if not voice:
        return default
    resolved = resolve_realtime_voice(provider, voice)
    if resolved is None:
        logger.warning(
            "Realtime voice '%s' is not available for provider '%s'; using '%s'.",
            voice,
            provider,
            default,
        )
        return default
    return resolved


def _default_openai_realtime():
    return openai.realtime.RealtimeModel(
        model=DEFAULT_OPENAI_REALTIME_MODEL,
        voice=DEFAULT_OPENAI_REALTIME_VOICE,
    )


def create_realtime_model(config: KwamiVoiceConfig):
    """Create a Realtime model instance for ultra-low latency."""
    provider = (config.realtime_provider or "openai").lower()
    model = strip_model_prefix(config.realtime_model or "", provider)

    try:
        if provider == "openai":
            return openai.realtime.RealtimeModel(
                model=model or DEFAULT_OPENAI_REALTIME_MODEL,
                voice=_realtime_voice(
                    provider, config.realtime_voice, DEFAULT_OPENAI_REALTIME_VOICE
                ),
                temperature=config.llm_temperature,
                # Validated by the provider; the config carries plain strings.
                modalities=cast(Any, config.realtime_modalities or ["text", "audio"]),
                turn_detection=_server_vad(config),
            )

        if provider == "google":
            if google is None:
                logger.error(
                    "Realtime provider 'google' requested but livekit-plugins-google "
                    "is not installed; falling back to OpenAI Realtime."
                )
                return _default_openai_realtime()
            return google.beta.realtime.RealtimeModel(
                model=model or DEFAULT_GOOGLE_REALTIME_MODEL,
                voice=_realtime_voice(
                    provider, config.realtime_voice, DEFAULT_GOOGLE_REALTIME_VOICE
                ),
                temperature=config.llm_temperature,
                realtime_input_config=_gemini_activity_detection(config),
            )

        logger.warning(
            "Unknown realtime provider '%s'; falling back to OpenAI Realtime %s. "
            "The configured provider is NOT in use.",
            provider,
            DEFAULT_OPENAI_REALTIME_MODEL,
        )
        return _default_openai_realtime()

    except Exception:
        logger.exception(
            "Failed to create %s realtime model; falling back to OpenAI Realtime %s.",
            provider,
            DEFAULT_OPENAI_REALTIME_MODEL,
        )
        return _default_openai_realtime()
