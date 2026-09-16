"""Voice-activity detection.

This module used to be dead: `create_vad` had no callers, and `prewarm` loaded
Silero with its own defaults. The consequence was subtle but real -- Silero's
defaults (min_silence_duration 0.55s) are not this project's defaults (0.3s),
so the snappier turn-taking the config documents was never actually in effect,
and `vad_threshold` / `vad_min_speech_duration` / `vad_min_silence_duration`
had no observable behaviour at all.

Prewarming exists to keep model loading off the per-job path, so the prewarmed
instance is reused whenever a session's settings match the ones it was loaded
with. Only a session that genuinely asks for something different pays to load
its own.
"""

from __future__ import annotations

from typing import Any

from livekit.plugins import silero

from ..domain import KwamiVoiceConfig
from ..utils.logging import get_logger

logger = get_logger("factories")


def _vad_settings(config: KwamiVoiceConfig) -> tuple[float, float, float]:
    return (
        config.vad_threshold,
        config.vad_min_speech_duration,
        config.vad_min_silence_duration,
    )


#: The settings `prewarm_vad` loads, i.e. the project defaults.
DEFAULT_VAD_SETTINGS = _vad_settings(KwamiVoiceConfig())


def prewarm_vad() -> Any:
    """Load the VAD model once per process, using this project's defaults.

    Loading Silero's defaults here instead would mean every session silently
    ran with different turn-taking than the config advertises.
    """
    threshold, min_speech, min_silence = DEFAULT_VAD_SETTINGS
    return silero.VAD.load(
        activation_threshold=threshold,
        min_speech_duration=min_speech,
        min_silence_duration=min_silence,
    )


def create_vad(config: KwamiVoiceConfig, prewarmed: Any = None) -> Any:
    """Return a VAD honouring `config`, reusing the prewarmed model when possible."""
    settings = _vad_settings(config)
    if prewarmed is not None and settings == DEFAULT_VAD_SETTINGS:
        return prewarmed

    if prewarmed is not None:
        logger.info(
            "Loading a session-specific VAD (threshold=%s, min_speech=%s, min_silence=%s)",
            *settings,
        )
    threshold, min_speech, min_silence = settings
    try:
        return silero.VAD.load(
            activation_threshold=threshold,
            min_speech_duration=min_speech,
            min_silence_duration=min_silence,
        )
    except Exception as e:
        # Never take a session down over turn-taking tuning.
        logger.error("Failed to load a configured VAD (%s); using the prewarmed model", e)
        return prewarmed if prewarmed is not None else silero.VAD.load()
