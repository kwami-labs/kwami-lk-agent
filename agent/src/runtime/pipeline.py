"""Building a configured agent from a config object.

This decides the entire voice pipeline -- realtime model, or STT + LLM + TTS --
and it lived at module scope in `main.py` next to the `AgentServer` singleton
and a `load_dotenv()` side effect, so importing it meant starting a worker.
Nothing exercised it, including the realtime branch, which raised
AttributeError for every session it was asked to build.
"""

from __future__ import annotations

from typing import Any

from ..agent import KwamiAgent
from ..domain import KwamiConfig
from ..factories import create_llm, create_realtime_model, create_stt, create_tts
from ..factories.vad import create_vad
from ..utils.logging import get_logger

logger = get_logger("pipeline")

REALTIME_PIPELINE = "realtime"


def create_agent_from_config(
    config: KwamiConfig,
    vad: Any,
    memory: Any = None,
    skip_greeting: bool = False,
) -> KwamiAgent:
    """Create a KwamiAgent with the pipeline its config asks for.

    Args:
        config: The Kwami configuration.
        vad: Voice Activity Detection instance.
        memory: Optional memory instance.
        skip_greeting: If True, skip the initial greeting (for reconfigurations).

    Returns:
        Configured KwamiAgent instance.
    """
    voice_config = config.voice

    # Reuses the prewarmed model unless this session actually asks for
    # different turn-taking; the VAD settings were previously ignored entirely.
    session_vad = create_vad(voice_config, prewarmed=vad)

    if voice_config.pipeline_type == REALTIME_PIPELINE:
        logger.info(
            "Using realtime pipeline: %s/%s",
            voice_config.realtime_provider,
            voice_config.realtime_model,
        )
        return KwamiAgent(
            config,
            vad=session_vad,
            memory=memory,
            llm=create_realtime_model(voice_config),
            skip_greeting=skip_greeting,
        )

    logger.info(
        "Using standard pipeline: STT=%s/%s, LLM=%s/%s, TTS=%s/%s",
        voice_config.stt_provider,
        voice_config.stt_model,
        voice_config.llm_provider,
        voice_config.llm_model,
        voice_config.tts_provider,
        voice_config.tts_model,
    )
    return KwamiAgent(
        config,
        vad=session_vad,
        memory=memory,
        stt=create_stt(voice_config),
        llm=create_llm(voice_config),
        tts=create_tts(voice_config),
        skip_greeting=skip_greeting,
    )
