from livekit.agents import inference
from livekit.plugins import deepgram, openai

try:
    from livekit.plugins import assemblyai
except ImportError:
    assemblyai = None  # type: ignore

try:
    from livekit.plugins import google
except ImportError:
    google = None  # type: ignore

try:
    from livekit.plugins import elevenlabs
except ImportError:
    elevenlabs = None  # type: ignore

try:
    from livekit.plugins import cartesia
except ImportError:
    cartesia = None  # type: ignore

from ..config import KwamiVoiceConfig
from ..constants import (
    STTProviders,
    OpenAIModels,
    DeepgramModels,
)
from ..utils.logging import get_logger
from ..utils.provider import strip_model_prefix

logger = get_logger("stt")


def create_stt(config: KwamiVoiceConfig):
    """Create STT instance based on configuration."""
    provider = config.stt_provider.lower()
    
    # Strip provider prefix from model name (e.g. "deepgram/nova-2" -> "nova-2")
    model = strip_model_prefix(config.stt_model or "", provider)
    
    logger.info(f"🎤 Creating STT: provider={provider}, model={model or config.stt_model}")
    
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
                language=config.stt_language if config.stt_language != "multi" else None,
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
            # Use LiveKit Inference so no ELEVEN_API_KEY needed in the agent
            stt_model = (model or "scribe-v2-realtime").replace("_", "-")
            if not stt_model.startswith("scribe"):
                stt_model = "scribe-v2-realtime"
            model_string = f"elevenlabs/{stt_model}"
            logger.info(f"🎤 Using LiveKit Inference for ElevenLabs STT: {model_string}")
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
            logger.warning(f"Unknown or unavailable STT provider '{provider}', falling back to Deepgram")
            return deepgram.STT(
                model=DeepgramModels.DEFAULT_STT,
                language="en",
            )
    
    except Exception as e:
        logger.error(f"Failed to create {provider} STT: {e}, falling back to Deepgram")
        return deepgram.STT(
            model=DeepgramModels.DEFAULT_STT,
            language="en",
        )
