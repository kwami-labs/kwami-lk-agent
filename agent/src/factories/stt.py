from livekit.agents import NOT_GIVEN, inference

# cartesia, deepgram, elevenlabs and openai are mandatory dependencies (see
# pyproject [project.dependencies]), so guarding their import would be dead
# code. assemblyai and google are core dependencies too now, but stay guarded so
# a slimmed-down install degrades instead of failing to import.
from livekit.plugins import cartesia, deepgram, elevenlabs, openai
from livekit.plugins.elevenlabs.stt import VADOptions

from ..constants import (
    DeepgramModels,
    OpenAIModels,
    STTProviders,
)
from ..domain import KwamiVoiceConfig
from ..utils.logging import get_logger
from ..utils.provider import strip_model_prefix
from .optional import optional_plugin
from .tts import elevenlabs_key

logger = get_logger("stt")

# Genuinely optional extras: absent means the provider falls back, not that
# the session fails.
assemblyai = optional_plugin("assemblyai")
google = optional_plugin("google")

#: AssemblyAI models that take `language_codes` (the plugin raises otherwise).
ASSEMBLYAI_LANGUAGE_MODELS = frozenset(
    {"u3-rt-pro", "u3-rt-pro-beta-1", "universal-3-5-pro", "universal-3-6-pro"}
)


#: ElevenLabs' realtime model commits a transcript only on its own VAD, or on a
#: manual commit the framework never sends. Values from LiveKit's plugin docs.
ELEVENLABS_SERVER_VAD: VADOptions = {
    "vad_silence_threshold_secs": 0.5,
    "vad_threshold": 0.5,
    "min_speech_duration_ms": 100,
    "min_silence_duration_ms": 300,
}


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
            # `word_boost` is gone from the plugin (a TypeError that fell back
            # to Deepgram); key terms now travel as `keyterms_prompt`. Only an
            # AssemblyAI model id is passed on: the config default is Deepgram's.
            # `language_codes` is accepted by the Universal-3 family only (the
            # plugin's default model is one); universal-streaming-* detect the
            # language themselves and raise a ValueError if handed one.
            language = config.stt_language
            own_model = model if model.startswith(("universal", "u3")) else ""
            takes_language = not own_model or own_model in ASSEMBLYAI_LANGUAGE_MODELS
            return assemblyai.STT(
                **({"model": own_model} if own_model else {}),
                **({"keyterms_prompt": config.stt_word_boost} if config.stt_word_boost else {}),
                **(
                    {"language_codes": [language]}
                    if takes_language and language and language != "multi"
                    else {}
                ),
            )

        elif provider == STTProviders.GEMINI and google is not None:
            # Live transcription on the Gemini API key. A two-letter language is
            # widened to its usual region; "multi" lets the model detect it.
            language = config.stt_language or "en"
            regions = {
                "es": "es-ES",
                "en": "en-US",
                "fr": "fr-FR",
                "de": "de-DE",
                "it": "it-IT",
                "pt": "pt-PT",
            }
            return google.beta.GeminiSTT(
                # Only a Gemini model: the config default is Deepgram's "nova-2",
                # which would otherwise ride along and close the socket (1008).
                model=model if model.startswith("gemini-") else "gemini-3.5-transcribe-live",
                language=None if language == "multi" else regions.get(language, language),
            )

        elif provider == STTProviders.GOOGLE and google is not None:
            return google.STT(
                model=model or "chirp",
                languages=[config.stt_language or "en-US"],
            )

        elif provider == STTProviders.ELEVENLABS:
            # Both paths expect underscores: scribe_v2_realtime.
            stt_model = (model or "scribe-v2-realtime").replace("-", "_")
            if not stt_model.startswith("scribe"):
                stt_model = "scribe_v2_realtime"
            # Our own key first: Inference answered "model not found" for
            # elevenlabs/scribe_v2_realtime on 2026-10-18.
            api_key = elevenlabs_key()
            if api_key:
                logger.info("🎤 Using the ElevenLabs plugin for STT: %s", stt_model)
                language = config.stt_language
                return elevenlabs.STT(
                    api_key=api_key,
                    model=stt_model,
                    language_code=language if language and language != "multi" else NOT_GIVEN,
                    # Without server VAD the realtime model never commits, so
                    # only interim transcripts arrive and the user's turn never
                    # ends (seen on LiveKit Cloud, 2026-10-18).
                    server_vad=(
                        ELEVENLABS_SERVER_VAD if stt_model.endswith("realtime") else NOT_GIVEN
                    ),
                )
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
