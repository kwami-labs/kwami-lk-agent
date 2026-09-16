"""Realtime model factory.

`openai.realtime.ServerVadOptions` never existed in livekit-plugins-openai
1.3.x, so every `pipeline_type="realtime"` session raised AttributeError before
it produced a single frame. The real kwarg type is `TurnDetection` from the
OpenAI SDK. This module also follows the fallback-and-log shape of
`factories/stt.py`: a provider that cannot be built logs an error and degrades
to a working default instead of taking the session down.
"""

from livekit.plugins import openai

try:
    from livekit.plugins import google  # type: ignore[attr-defined]
except ImportError:
    google = None

try:
    from openai.types.beta.realtime.session import TurnDetection
except ImportError:  # pragma: no cover - openai SDK is a hard dependency
    TurnDetection = None  # type: ignore

from typing import Any, cast

from ..domain import KwamiVoiceConfig
from ..utils.logging import get_logger
from ..utils.provider import strip_model_prefix

logger = get_logger("factories")

DEFAULT_OPENAI_REALTIME_MODEL = "gpt-realtime"
DEFAULT_OPENAI_REALTIME_VOICE = "marin"
DEFAULT_GOOGLE_REALTIME_MODEL = "gemini-2.0-flash-exp"
DEFAULT_GOOGLE_REALTIME_VOICE = "Puck"


def _server_vad(config: KwamiVoiceConfig):
    """Build server-side VAD options, or None to let the provider default apply."""
    if TurnDetection is None:
        return None
    return TurnDetection(
        type="server_vad",
        threshold=config.vad_threshold,
        prefix_padding_ms=300,
        silence_duration_ms=int(config.vad_min_silence_duration * 1000),
    )


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
                voice=config.realtime_voice or DEFAULT_OPENAI_REALTIME_VOICE,
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
                voice=config.realtime_voice or DEFAULT_GOOGLE_REALTIME_VOICE,
                temperature=config.llm_temperature,
            )

        logger.warning(
            "Unknown realtime provider '%s'; falling back to OpenAI Realtime %s. "
            "The configured provider is NOT in use.",
            provider,
            DEFAULT_OPENAI_REALTIME_MODEL,
        )
        return _default_openai_realtime()

    except Exception as e:
        logger.error(
            "Failed to create %s realtime model (%s); falling back to OpenAI Realtime %s.",
            provider,
            e,
            DEFAULT_OPENAI_REALTIME_MODEL,
        )
        return _default_openai_realtime()
