"""Realtime model factory.

`openai.realtime.ServerVadOptions` never existed in livekit-plugins-openai
1.3.x, so every `pipeline_type="realtime"` session raised AttributeError before
it produced a single frame. The real kwarg type is `TurnDetection` from the
OpenAI SDK. This module also follows the fallback-and-log shape of
`factories/stt.py`: a provider that cannot be built logs an error and degrades
to a working default instead of taking the session down.
"""

from typing import Any, cast

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

# Guarded like the other factories: typed `Any`, so mypy accepts the None branch
# whether or not the plugin happens to be installed where it runs.
google = optional_plugin("google")

logger = get_logger("factories")

DEFAULT_OPENAI_REALTIME_MODEL = "gpt-realtime"
DEFAULT_OPENAI_REALTIME_VOICE = "marin"
DEFAULT_GOOGLE_REALTIME_MODEL = "gemini-3.8-live"
DEFAULT_GOOGLE_REALTIME_VOICE = "Charon"


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
