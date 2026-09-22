from livekit.agents import inference

# cartesia, deepgram, elevenlabs and openai are mandatory dependencies (see
# pyproject [project.dependencies]), so guarding their import would be dead
# code. assemblyai and google are genuinely optional and are guarded.
from livekit.plugins import cartesia, deepgram, openai

from ..constants import (
    DeepgramModels,
    OpenAIModels,
    STTProviders,
)
from ..domain import KwamiVoiceConfig
from ..utils.logging import get_logger
from ..utils.provider import strip_model_prefix
from .optional import optional_plugin

logger = get_logger("stt")

# Genuinely optional extras: absent means the provider falls back, not that
# the session fails.
assemblyai = optional_plugin("assemblyai")
google = optional_plugin("google")


def create_stt(config: KwamiVoiceConfig):
    """Create STT instance based on configuration."""
    provider = config.stt_provider.lower()

    # Strip provider prefix from model name (e.g. "deepgram/nova-2" -> "nova-2")
    model = strip_model_prefix(config.stt_model or "", provider)

    logger.info("🎤 Creating STT: provider=%s, model=%s", provider, model or config.stt_model)

    try:
        if provider == STTProviders.DEEPGRAM:
            return deepgram.STT(
                model=model or DeepgramModels.DEFAULT_STT,
                language=config.stt_language,
                interim_results=True,
                smart_format=True,
                punctuate=True,
            )

        elif provider == STTProviders.OPENAI:
            return openai.STT(
                model=model or OpenAIModels.WHISPER_1,
                # The plugin annotates this `str` with a default of "en", but
                # Whisper auto-detects when the field is absent, and None is how
                # that reaches the request body. Omitting the argument would
                # pin every "multi" session to English instead, so the narrower
                # annotation is overridden deliberately rather than obeyed.
                language=config.stt_language  # type: ignore[arg-type]
                if config.stt_language != "multi"
                else None,
            )

        elif provider == STTProviders.ASSEMBLYAI and assemblyai is not None:
            return assemblyai.STT(
                word_boost=config.stt_word_boost or [],
            )

        elif provider == STTProviders.GOOGLE and google is not None:
            return google.STT(
                model=model or "chirp",
                languages=[config.stt_language or "en-US"],
            )

        elif provider == STTProviders.ELEVENLABS:
            # Use LiveKit Inference so no ELEVEN_API_KEY needed in the agent.
            # LiveKit Inference expects underscores: elevenlabs/scribe_v2_realtime
            stt_model = (model or "scribe-v2-realtime").replace("-", "_")
            if not stt_model.startswith("scribe"):
                stt_model = "scribe_v2_realtime"
            model_string = f"elevenlabs/{stt_model}"
            logger.info("🎤 Using LiveKit Inference for ElevenLabs STT: %s", model_string)
            return inference.STT(
                model=model_string,
                language=config.stt_language or "en",
            )

        elif provider == STTProviders.CARTESIA and cartesia is not None:
            return cartesia.STT(
                model=model or "ink-whisper",
                language=config.stt_language or "en",
            )

        else:
            logger.warning(
                "Unknown or unavailable STT provider '%s', falling back to Deepgram", provider
            )
            return deepgram.STT(
                model=DeepgramModels.DEFAULT_STT,
                language="en",
            )

    except Exception:
        logger.exception("Failed to create %s STT; falling back to Deepgram", provider)
        return deepgram.STT(
            model=DeepgramModels.DEFAULT_STT,
            language="en",
        )
