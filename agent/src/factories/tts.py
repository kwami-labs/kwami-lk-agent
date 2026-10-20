"""TTS (Text-to-Speech) Factory Module.

Provides comprehensive TTS creation with:
- Voice ID validation per provider
- API key validation
- Detailed error handling
- Provider-specific voice constants
- Caching support
"""

import re

from livekit.agents import NOT_GIVEN, inference
from livekit.plugins import cartesia, deepgram, elevenlabs, openai

from ..constants import (
    CartesiaVoices,
    DeepgramVoices,
    ElevenLabsVoices,
    EnvVars,
    GoogleVoices,
    OpenAIModels,
    OpenAIVoices,
    TTSProviders,
    resolve_realtime_voice,
)
from ..domain import KwamiVoiceConfig
from ..settings import get_settings
from ..utils.logging import get_logger
from ..utils.provider import strip_model_prefix
from .optional import optional_plugin

logger = get_logger("tts")

google = optional_plugin("google")


# =============================================================================
# API Key Validation
# =============================================================================


def _check_api_key(provider: str) -> bool:
    """Check if the required API key is set for a provider."""
    key_map = {
        TTSProviders.OPENAI: EnvVars.OPENAI,
        TTSProviders.ELEVENLABS: EnvVars.ELEVENLABS,
        TTSProviders.CARTESIA: EnvVars.CARTESIA,
        TTSProviders.DEEPGRAM: EnvVars.DEEPGRAM,
        TTSProviders.GOOGLE: EnvVars.GOOGLE,
        TTSProviders.GEMINI: EnvVars.GOOGLE,
    }
    env_vars = key_map.get(provider, [])
    if not env_vars:
        return True  # Unknown provider, assume OK

    settings = get_settings()
    if any(settings.has_provider_key(env_var) for env_var in env_vars):
        return True

    logger.warning("⚠️ %s not set for %s TTS", " or ".join(env_vars), provider)
    return False


# =============================================================================
# TTS Factory
# =============================================================================


def create_tts(config: KwamiVoiceConfig):
    """Create TTS instance based on configuration.

    Args:
        config: Voice configuration with provider, model, voice, speed settings.

    Returns:
        TTS instance for the specified provider.

    Raises:
        VoiceProviderError: If creation fails and fallback is not possible/desired.
    """
    provider = config.tts_provider.lower()

    logger.info(
        "🔊 Creating TTS: provider=%s, model=%s, voice=%s",
        provider,
        config.tts_model,
        config.tts_voice,
    )

    # Check API key (warning only, don't block)
    _check_api_key(provider)

    try:
        if provider == TTSProviders.OPENAI:
            return _create_openai_tts(config)

        elif provider == TTSProviders.ELEVENLABS:
            return _create_elevenlabs_tts(config)

        elif provider == TTSProviders.CARTESIA:
            return _create_cartesia_tts(config)

        elif provider == TTSProviders.DEEPGRAM:
            return _create_deepgram_tts(config)

        elif provider == TTSProviders.GOOGLE:
            return _create_google_tts(config)

        elif provider == TTSProviders.RIME:
            return _create_rime_tts(config)

        elif provider == TTSProviders.GEMINI:
            return _create_gemini_tts(config)

        else:
            logger.warning("Unknown TTS provider '%s', falling back to OpenAI", provider)
            return _fallback_to_openai_tts(config, provider)

    except Exception:
        logger.exception("Failed to create %s TTS; falling back to OpenAI", provider)
        return _fallback_to_openai_tts(config, provider)


def _fallback_to_openai_tts(config: KwamiVoiceConfig, attempted_provider: str):
    """Serve OpenAI TTS and record that the configured provider is not in use.

    The config must be corrected, not just logged: `_update_tts_options` in the
    config handler re-detects the provider to validate incoming voice changes,
    so leaving `tts_provider` reading "elevenlabs" while an OpenAI client is
    actually running made every later voice update validate against the wrong
    provider's voice list.
    """
    if attempted_provider != TTSProviders.OPENAI:
        config.tts_provider = TTSProviders.OPENAI
        logger.warning(
            "TTS provider recorded as OpenAI after '%s' could not be created; "
            "voice validation now follows OpenAI's voice list.",
            attempted_provider,
        )
    return _create_openai_tts(config)


# =============================================================================
# Provider-Specific Factories
# =============================================================================


def _create_openai_tts(config: KwamiVoiceConfig):
    """Create OpenAI TTS with voice and model validation."""
    voice = config.tts_voice or OpenAIVoices.DEFAULT
    model = strip_model_prefix(config.tts_model or "", "openai") or OpenAIModels.TTS_1

    # Validate model
    if model not in OpenAIModels.ALL_TTS:
        logger.warning(
            "Model '%s' not supported by OpenAI TTS. Using '%s'. Valid: %s",
            model,
            OpenAIModels.TTS_1,
            ", ".join(sorted(OpenAIModels.ALL_TTS)),
        )
        model = OpenAIModels.TTS_1

    # Validate voice
    if voice not in OpenAIVoices.STANDARD:
        logger.warning(
            "Voice '%s' not supported by OpenAI TTS. Using '%s'. Valid: %s",
            voice,
            OpenAIVoices.DEFAULT,
            ", ".join(sorted(OpenAIVoices.STANDARD)),
        )
        voice = OpenAIVoices.DEFAULT

    return openai.TTS(
        model=model,
        voice=voice,
        speed=float(config.tts_speed or 1.0),
    )


def session_language(config: KwamiVoiceConfig) -> str:
    """The session's language as a bare code ("es-ES" -> "es"), or "" for multi."""
    language = (config.stt_language or "").split("-")[0].lower()
    return "" if language == "multi" else language


def elevenlabs_key() -> str:
    """The ElevenLabs key under either spelling, or "" when none is set."""
    keys = get_settings().provider_keys
    return next((keys[name] for name in EnvVars.ELEVENLABS if keys.get(name)), "")


def _create_elevenlabs_tts(config: KwamiVoiceConfig):
    """ElevenLabs TTS: the plugin on our own key, LiveKit Inference without one.

    Inference was the only path, so the agent's ELEVEN_API_KEY was never used,
    and on 2026-10-18 Inference answered "model not found" for
    elevenlabs/eleven_turbo_v2_5, which left the provider unusable.
    """
    voice_id = config.tts_voice or ElevenLabsVoices.DEFAULT

    # ElevenLabs uses 20-char alphanumeric voice IDs (e.g. "21m00Tcm4TlvDq8ikWAM").
    # Short names like "nova" or "alloy" are OpenAI voices that leak through when
    # the TTS provider is changed but the voice isn't updated. Fall back to default.
    if voice_id and voice_id not in ElevenLabsVoices.ALL and len(voice_id) < 15:
        logger.warning(
            "Voice '%s' is not a valid ElevenLabs voice ID. Using default: %s",
            voice_id,
            ElevenLabsVoices.DEFAULT,
        )
        voice_id = ElevenLabsVoices.DEFAULT

    model = strip_model_prefix(config.tts_model or "", "elevenlabs") or "eleven_turbo_v2_5"

    # Normalize model name: dashes to underscores, dots to underscores
    # Valid format: eleven_flash_v2_5, eleven_turbo_v2_5, etc.
    model = model.replace("-", "_").replace(".", "_")

    # Flash and Turbo v2.5 read every sentence as the language they guess
    # unless told; a short Spanish reply then came out with English phonetics.
    # Models without language enforcement ignore it.
    language = session_language(config) or NOT_GIVEN

    api_key = elevenlabs_key()
    if api_key:
        logger.info("🔊 Using the ElevenLabs plugin: %s:%s", model, voice_id)
        return elevenlabs.TTS(voice_id=voice_id, model=model, api_key=api_key, language=language)

    # Format: "elevenlabs/model:voice_id"
    model_string = f"elevenlabs/{model}"

    logger.info("🔊 Using LiveKit Inference for ElevenLabs: %s:%s", model_string, voice_id)

    return inference.TTS(
        model=model_string,
        voice=voice_id,
        language=language,
    )


def _create_rime_tts(config: KwamiVoiceConfig):
    """Create Rime TTS using LiveKit Inference.

    Supported models: rime/arcana, rime/mistv2
    Voices: astra, celeste, luna, ursa, orion, etc.
    """
    voice = config.tts_voice or "luna"
    model = strip_model_prefix(config.tts_model or "", "rime") or "arcana"

    # Use LiveKit Inference for Rime - format: "rime/model:voice"
    model_string = f"rime/{model}"

    logger.info("🔊 Using LiveKit Inference for Rime: %s:%s", model_string, voice)

    return inference.TTS(
        model=model_string,
        voice=voice,
    )


def _create_cartesia_tts(config: KwamiVoiceConfig):
    """Create Cartesia TTS."""
    voice = config.tts_voice or CartesiaVoices.DEFAULT

    # Check for friendly name mapping
    if voice.lower() in CartesiaVoices.NAME_MAP:
        voice = CartesiaVoices.NAME_MAP[voice.lower()]

    # Cartesia uses UUID format voice IDs
    if voice and len(voice) < 30 and "-" not in voice:
        logger.warning(
            "Voice '%s' may be invalid for Cartesia (expected UUID format). Using default: %s",
            voice,
            CartesiaVoices.DEFAULT,
        )
        voice = CartesiaVoices.DEFAULT

    model = strip_model_prefix(config.tts_model or "", "cartesia") or "sonic-2"

    # Without it the plugin speaks English: a Spanish voice then reads Spanish
    # text with English phonetics. Cartesia takes the bare code ("es").
    language = session_language(config)

    return cartesia.TTS(
        model=model,
        voice=voice,
        speed=float(config.tts_speed or 1.0),
        encoding="pcm_s16le",
        # The plugin's own default when there is no single language to pin.
        language=language or "en",
    )


#: A full Aura model id, voice and language included: aura-asteria-en, aura-2-thalia-en.
_DEEPGRAM_TTS_MODEL = re.compile(r"^aura-(2-)?[a-z]+-[a-z]{2}$")


def _create_deepgram_tts(config: KwamiVoiceConfig):
    """Create Deepgram Aura TTS with voice validation."""
    voice = config.tts_voice or DeepgramVoices.DEFAULT

    if voice not in DeepgramVoices.ALL:
        logger.warning(
            "Voice '%s' not in known Deepgram voices. Using '%s'. Valid: %s",
            voice,
            DeepgramVoices.DEFAULT,
            ", ".join(sorted(DeepgramVoices.ALL)),
        )
        voice = DeepgramVoices.DEFAULT

    # Deepgram model includes voice (e.g., "aura-asteria-en"). Anything else is
    # not a Deepgram model: the config default is OpenAI's "tts-1" and the app
    # sends the family "aura-1", and Deepgram answered 400 to both on LiveKit
    # Cloud (2026-10-18).
    model = strip_model_prefix(config.tts_model or "", "deepgram")
    if not _DEEPGRAM_TTS_MODEL.match(model):
        model = f"aura-{voice}-en"

    return deepgram.TTS(model=model)


def _create_google_tts(config: KwamiVoiceConfig):
    """Create Google Cloud TTS with fallback handling."""
    if google is None:
        logger.warning("Google TTS plugin not installed, falling back to OpenAI")
        return _create_openai_tts(config)

    # Cloud Text-to-Speech authenticates with a service account only. Without
    # one the plugin fails later with an opaque auth error; GOOGLE_API_KEY is
    # what Gemini TTS uses, and it does not count here.
    if not get_settings().has_provider_key("GOOGLE_APPLICATION_CREDENTIALS"):
        raise ValueError(
            "Google Cloud TTS needs GOOGLE_APPLICATION_CREDENTIALS (a service account); "
            "GOOGLE_API_KEY only serves the 'gemini' TTS provider"
        )

    voice = config.tts_voice or GoogleVoices.DEFAULT

    return google.TTS(
        voice_name=voice,
        speaking_rate=float(config.tts_speed or 1.0),
    )


DEFAULT_GEMINI_TTS_MODEL = "gemini-3.1-flash-tts-preview"
DEFAULT_GEMINI_TTS_VOICE = "Charon"


def gemini_tts_instructions(direction: str) -> str | None:
    """The direction in Google's prompt structure for Gemini TTS.

    The plugin sends `f'{instructions}:\\n"{text}"'` for every sentence, so this
    ends where the transcript begins. It is the structure the character
    samples the user chose were generated with (audio profile, scene,
    director's notes, transcript); a bare one-line style sounded flatter.
    None leaves the plugin's own neutral instruction in place.
    """
    direction = direction.strip()
    if not direction:
        return None
    return (
        "# AUDIO PROFILE\n"
        f"{direction}\n\n"
        "## THE SCENE\n"
        "La voz llega por el auricular del usuario, cercana y clara, mientras trabaja.\n\n"
        "### DIRECTOR'S NOTES\n"
        "Respeta exactamente el perfil, la entonación, el ritmo y la actitud descritos. "
        "Lee el texto tal cual, sin añadir ni omitir palabras.\n\n"
        "#### TRANSCRIPT"
    )


def _create_gemini_tts(config: KwamiVoiceConfig):
    """Gemini TTS: a prebuilt Gemini voice steered by the character's direction."""
    if google is None:
        logger.warning("Google plugin not installed, falling back to OpenAI TTS")
        return _create_openai_tts(config)

    instructions = gemini_tts_instructions(config.tts_style)
    model = strip_model_prefix(config.tts_model or "", "gemini")
    # Only a Gemini model: the config default is OpenAI's "tts-1", which would
    # otherwise ride along into a Gemini request and 404.
    if not model.startswith("gemini-"):
        model = DEFAULT_GEMINI_TTS_MODEL
    # Likewise only a Gemini prebuilt voice (Live and TTS share them): the
    # config default is OpenAI's "nova", which Gemini rejects.
    voice = resolve_realtime_voice("google", config.tts_voice) or DEFAULT_GEMINI_TTS_VOICE
    return google.beta.GeminiTTS(
        model=model,
        voice_name=voice,
        **({"instructions": instructions} if instructions else {}),
    )


# =============================================================================
# Utility Functions
# =============================================================================


def get_available_providers() -> list[str]:
    """Get list of available TTS providers based on installed plugins."""
    # ElevenLabs is served through LiveKit Inference and its plugin is a
    # mandatory dependency, so it is always available.
    providers = [
        TTSProviders.OPENAI,
        TTSProviders.DEEPGRAM,
        TTSProviders.CARTESIA,
        TTSProviders.RIME,
        TTSProviders.ELEVENLABS,
    ]

    if google is not None:
        providers.append(TTSProviders.GOOGLE)
        providers.append(TTSProviders.GEMINI)

    return providers


def get_voices_for_provider(provider: str) -> list[str]:
    """Get list of valid voice IDs for a provider."""
    provider = provider.lower()

    if provider == TTSProviders.OPENAI:
        return list(OpenAIVoices.STANDARD)
    elif provider == TTSProviders.ELEVENLABS:
        return list(ElevenLabsVoices.ALL)
    elif provider == TTSProviders.DEEPGRAM:
        return list(DeepgramVoices.ALL)
    elif provider == TTSProviders.CARTESIA:
        return list(CartesiaVoices.NAME_MAP.keys())  # Return friendly names
    elif provider == TTSProviders.GOOGLE:
        return [GoogleVoices.STUDIO_O, GoogleVoices.STUDIO_Q]

    return []


def get_default_voice(provider: str) -> str:
    """Get the default voice ID for a provider."""
    provider = provider.lower()

    defaults = {
        TTSProviders.OPENAI: OpenAIVoices.DEFAULT,
        TTSProviders.ELEVENLABS: ElevenLabsVoices.DEFAULT,
        TTSProviders.CARTESIA: CartesiaVoices.DEFAULT,
        TTSProviders.DEEPGRAM: DeepgramVoices.DEFAULT,
        TTSProviders.GOOGLE: GoogleVoices.DEFAULT,
    }

    return defaults.get(provider, "default")
