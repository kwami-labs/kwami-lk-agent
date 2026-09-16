from livekit.plugins import silero

from ..domain import KwamiVoiceConfig


def create_vad(config: KwamiVoiceConfig):
    """Create VAD instance based on configuration."""
    return silero.VAD.load(
        min_speech_duration=config.vad_min_speech_duration,
        min_silence_duration=config.vad_min_silence_duration,
    )
