"""TTS (Text-to-Speech) Factory Module.

Provides comprehensive TTS creation with:
- Voice ID validation per provider
- API key validation
- Detailed error handling
- Provider-specific voice constants
- Caching support
"""

from livekit.agents import inference

# elevenlabs is a mandatory dependency (see pyproject [project.dependencies]),
# so guarding its import would be dead code. google is a genuine extra.
from livekit.plugins import cartesia, deepgram, elevenlabs, openai

try:
    from livekit.plugins import google
except ImportError:  # pragma: no cover - depends on an optional extra
    google = None  # type: ignore

from ..constants import (
    CartesiaVoices,
    DeepgramVoices,
    ElevenLabsVoices,
    EnvVars,
    GoogleVoices,
    OpenAIModels,
    OpenAIVoices,
    TTSProviders,
)
from ..domain import KwamiVoiceConfig
from ..settings import get_settings
from ..utils.logging import get_logger
from ..utils.provider import strip_model_prefix

logger = get_logger("tts")


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
    }
    env_vars = key_map.get(provider, [])
    if not env_vars:
        return True  # Unknown provider, assume OK

    settings = get_settings()
    if any(settings.has_provider_key(env_var) for env_var in env_vars):
        return True

    logger.warning(f"⚠️ {' or '.join(env_vars)} not set for {provider} TTS")
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
        f"🔊 Creating TTS: provider={provider}, model={config.tts_model}, voice={config.tts_voice}"
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

        else:
            logger.warning(f"Unknown TTS provider '{provider}', falling back to OpenAI")
            return _fallback_to_openai_tts(config, provider)

    except Exception as e:
        logger.error(f"Failed to create {provider} TTS: {e}, falling back to OpenAI")
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
            f"Model '{model}' not supported by OpenAI TTS. "
            f"Using '{OpenAIModels.TTS_1}'. Valid: {', '.join(sorted(OpenAIModels.ALL_TTS))}"
        )
        model = OpenAIModels.TTS_1

    # Validate voice
    if voice not in OpenAIVoices.STANDARD:
        logger.warning(
            f"Voice '{voice}' not supported by OpenAI TTS. "
            f"Using '{OpenAIVoices.DEFAULT}'. "
            f"Valid: {', '.join(sorted(OpenAIVoices.STANDARD))}"
        )
        voice = OpenAIVoices.DEFAULT

    return openai.TTS(
        model=model,
        voice=voice,
        speed=float(config.tts_speed or 1.0),
    )


def _create_elevenlabs_tts(config: KwamiVoiceConfig):
    """Create ElevenLabs TTS using LiveKit Inference (more reliable than direct plugin)."""
    voice_id = config.tts_voice or ElevenLabsVoices.DEFAULT

    # ElevenLabs uses 20-char alphanumeric voice IDs (e.g. "21m00Tcm4TlvDq8ikWAM").
    # Short names like "nova" or "alloy" are OpenAI voices that leak through when
    # the TTS provider is changed but the voice isn't updated. Fall back to default.
    if voice_id and voice_id not in ElevenLabsVoices.ALL and len(voice_id) < 15:
        logger.warning(
            f"Voice '{voice_id}' is not a valid ElevenLabs voice ID. "
            f"Using default: {ElevenLabsVoices.DEFAULT}"
        )
        voice_id = ElevenLabsVoices.DEFAULT

    model = strip_model_prefix(config.tts_model or "", "elevenlabs") or "eleven_turbo_v2_5"

    # Normalize model name: dashes to underscores, dots to underscores
    # Valid format: eleven_flash_v2_5, eleven_turbo_v2_5, etc.
    model = model.replace("-", "_").replace(".", "_")

    # Use LiveKit Inference for ElevenLabs - more reliable than direct plugin
    # Format: "elevenlabs/model:voice_id"
    model_string = f"elevenlabs/{model}"

    logger.info(f"🔊 Using LiveKit Inference for ElevenLabs: {model_string}:{voice_id}")

    return inference.TTS(
        model=model_string,
        voice=voice_id,
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

    logger.info(f"🔊 Using LiveKit Inference for Rime: {model_string}:{voice}")

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
            f"Voice '{voice}' may be invalid for Cartesia (expected UUID format). "
            f"Using default: {CartesiaVoices.DEFAULT}"
        )
        voice = CartesiaVoices.DEFAULT

    model = strip_model_prefix(config.tts_model or "", "cartesia") or "sonic-2"

    return cartesia.TTS(
        model=model,
        voice=voice,
        speed=float(config.tts_speed or 1.0),
        encoding="pcm_s16le",
    )


def _create_deepgram_tts(config: KwamiVoiceConfig):
    """Create Deepgram Aura TTS with voice validation."""
    voice = config.tts_voice or DeepgramVoices.DEFAULT

    if voice not in DeepgramVoices.ALL:
        logger.warning(
            f"Voice '{voice}' not in known Deepgram voices. "
            f"Using '{DeepgramVoices.DEFAULT}'. "
            f"Valid: {', '.join(sorted(DeepgramVoices.ALL))}"
        )
        voice = DeepgramVoices.DEFAULT

    # Deepgram model includes voice (e.g., "aura-asteria-en")
    model = strip_model_prefix(config.tts_model or "", "deepgram") or f"aura-{voice}-en"

    return deepgram.TTS(model=model)


def _create_google_tts(config: KwamiVoiceConfig):
    """Create Google Cloud TTS with fallback handling."""
    if google is None:
        logger.warning("Google TTS plugin not installed, falling back to OpenAI")
        return _create_openai_tts(config)

    voice = config.tts_voice or GoogleVoices.DEFAULT

    return google.TTS(
        voice=voice,
        speaking_rate=float(config.tts_speed or 1.0),
    )


# =============================================================================
# Utility Functions
# =============================================================================


def get_available_providers() -> list[str]:
    """Get list of available TTS providers based on installed plugins."""
    providers = [
        TTSProviders.OPENAI,
        TTSProviders.DEEPGRAM,
        TTSProviders.CARTESIA,
        TTSProviders.RIME,
    ]

    if elevenlabs is not None:
        providers.append(TTSProviders.ELEVENLABS)
    if google is not None:
        providers.append(TTSProviders.GOOGLE)

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
