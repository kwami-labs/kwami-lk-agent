"""The voice a session has before -- or without -- the app's config.

The agent starts every job on a placeholder and replaces it when the app sends
its config. Without the app (the LiveKit Console, a phone call) the placeholder
is the agent the user talks to, and the built-in default is Deepgram nova-2,
gpt-4o-mini and OpenAI's "nova", in English: not Jarvis.

Two layers, each optional, over the built-in default:

1. **`KWAMI_DEFAULT_*` secrets** on the deployment: the voice every app-less
   session starts with.
2. **A `voice` object in the job metadata** (`{"voice": {"stt_provider": ...}}`,
   set in the Console under *Configure → Job metadata*): overrides the secrets
   for that one session, so a combination is tried without a redeploy.

Fields are `KwamiVoiceConfig`'s own names. An unknown field or a value of the
wrong type is skipped with a warning, never fatal: whatever arrives, the
session gets an agent.

A provider given without a model (or a TTS provider without a voice) clears the
old one, so the factory picks that provider's default instead of inheriting,
say, OpenAI's "gpt-4o-mini" for Groq.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import fields
from typing import Any

from ..domain import KwamiConfig, KwamiVoiceConfig
from ..utils.logging import get_logger

logger = get_logger("runtime.defaults")

#: Secret -> voice field. The values are KwamiVoiceConfig field names.
ENV_VOICE_FIELDS: dict[str, str] = {
    "KWAMI_DEFAULT_PIPELINE_TYPE": "pipeline_type",
    "KWAMI_DEFAULT_STT_PROVIDER": "stt_provider",
    "KWAMI_DEFAULT_STT_MODEL": "stt_model",
    "KWAMI_DEFAULT_STT_LANGUAGE": "stt_language",
    "KWAMI_DEFAULT_LLM_PROVIDER": "llm_provider",
    "KWAMI_DEFAULT_LLM_MODEL": "llm_model",
    "KWAMI_DEFAULT_TTS_PROVIDER": "tts_provider",
    "KWAMI_DEFAULT_TTS_MODEL": "tts_model",
    "KWAMI_DEFAULT_TTS_VOICE": "tts_voice",
    "KWAMI_DEFAULT_REALTIME_PROVIDER": "realtime_provider",
    "KWAMI_DEFAULT_REALTIME_MODEL": "realtime_model",
    "KWAMI_DEFAULT_REALTIME_VOICE": "realtime_voice",
}
#: The language the assistant speaks and the STT expects ("es").
ENV_LANGUAGE = "KWAMI_DEFAULT_LANGUAGE"
#: The assistant's name before the app's persona arrives ("Jarvis").
ENV_NAME = "KWAMI_DEFAULT_NAME"

#: Not a voice field, accepted in the metadata `voice` object all the same.
LANGUAGE_KEY = "language"

#: Fields whose value is one of a fixed set.
_CHOICES: dict[str, tuple[str, ...]] = {
    "pipeline_type": ("standard", "realtime"),
    "turn_detection": ("server_vad", "semantic", "none"),
    "turn_detector": ("model", "vad", "stt"),
    "noise_cancellation_mode": ("bvc", "krisp"),
}

#: Setting the key on the left without the ones on the right clears those.
_CLEARED_WITH: dict[str, tuple[str, ...]] = {
    "stt_provider": ("stt_model",),
    "llm_provider": ("llm_model",),
    "tts_provider": ("tts_model", "tts_voice"),
    "realtime_provider": ("realtime_model", "realtime_voice"),
}

_VOICE_DEFAULTS = KwamiVoiceConfig()
_VOICE_FIELDS = {f.name for f in fields(KwamiVoiceConfig)}


def _coerce(name: str, value: Any) -> Any:
    """`value` as the type of the field's default, or raise ValueError."""
    default = getattr(_VOICE_DEFAULTS, name)
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
    elif isinstance(default, int | float):
        if isinstance(value, int | float) and not isinstance(value, bool):
            return type(default)(value)
        if isinstance(value, str):
            return type(default)(float(value))
    elif isinstance(default, list):
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return list(value)
    elif isinstance(value, str):
        value = value.strip()
        choices = _CHOICES.get(name)
        if choices is not None and value not in choices:
            raise ValueError(f"must be one of {', '.join(choices)}")
        return value
    raise ValueError(f"expected {type(default).__name__}, got {type(value).__name__}")


def apply_voice_fields(config: KwamiConfig, values: Mapping[str, Any], source: str) -> list[str]:
    """Apply validated voice fields onto `config`. Returns the fields applied."""
    voice = config.voice
    applied: list[str] = []
    accepted: dict[str, Any] = {}
    for name, raw in values.items():
        if name == LANGUAGE_KEY:
            continue
        if name not in _VOICE_FIELDS:
            logger.warning("%s: unknown voice field %r ignored", source, name)
            continue
        try:
            accepted[name] = _coerce(name, raw)
        except (TypeError, ValueError) as e:
            logger.warning("%s: voice field %r ignored (%s)", source, name, e)

    for key, cleared in _CLEARED_WITH.items():
        if key in accepted:
            for name in cleared:
                if name not in accepted:
                    setattr(voice, name, "")
    for name, value in accepted.items():
        setattr(voice, name, value)
        applied.append(name)

    language = values.get(LANGUAGE_KEY)
    if isinstance(language, str) and language.strip():
        _set_language(config, language.strip(), keep_stt="stt_language" in accepted)
        applied.append(LANGUAGE_KEY)
    return applied


def _set_language(config: KwamiConfig, language: str, keep_stt: bool) -> None:
    config.soul.language = language
    if not keep_stt:
        config.voice.stt_language = language


def _metadata_voice(job_metadata: str | None) -> dict[str, Any]:
    """The `voice` object of the job metadata, or {} for anything else."""
    if not job_metadata:
        return {}
    try:
        parsed = json.loads(job_metadata)
    except ValueError:
        logger.warning("Job metadata is not JSON; using the default voice")
        return {}
    voice = parsed.get("voice") if isinstance(parsed, dict) else None
    if voice is None:
        return {}
    if not isinstance(voice, dict):
        logger.warning("Job metadata 'voice' is not an object; ignored")
        return {}
    return voice


def placeholder_config(
    env: Mapping[str, str] | None = None, job_metadata: str | None = None
) -> KwamiConfig:
    """The config of the agent that holds the room until the app's arrives."""
    env = os.environ if env is None else env
    config = KwamiConfig()
    try:
        from_env = {field: env[name] for name, field in ENV_VOICE_FIELDS.items() if env.get(name)}
        applied = apply_voice_fields(config, from_env, "KWAMI_DEFAULT_*")
        language = (env.get(ENV_LANGUAGE) or "").strip()
        if language:
            _set_language(config, language, keep_stt="stt_language" in applied)
        name = (env.get(ENV_NAME) or "").strip()
        if name:
            config.soul.name = name
            config.kwami_name = name

        overrides = _metadata_voice(job_metadata)
        if overrides:
            applied = apply_voice_fields(config, overrides, "job metadata")
            logger.info("Job metadata sets the voice for this session: %s", sorted(applied))
    except Exception:
        # The placeholder must always exist: a bad override costs the override,
        # never the session.
        logger.exception("Could not apply the default voice; using the built-in one")
        return KwamiConfig()
    return config
