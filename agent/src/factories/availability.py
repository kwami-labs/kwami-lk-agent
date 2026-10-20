"""Which voice providers this deployment can actually run.

The factories never refuse: a provider without its key is built anyway and
fails on first use, or is swapped for OpenAI or Deepgram by a fallback. Either
way the user hears a model they did not pick -- the "crossed models" the app's
Models panel is meant to prevent. So the agent tells the app, per pipeline
stage, which providers have their credential, and the app greys out the rest
(`brain_status.voiceProviders`).

The table mirrors the factories, not the providers' marketing: Google Cloud TTS
and STT authenticate with a service account only, Gemini TTS and STT with the
Gemini API key; ElevenLabs needs our own key, because LiveKit Inference answered
"model not found" for its models (2026-10-18); Rime only exists on LiveKit
Inference. A provider listed with no credential at all is one the factory
cannot serve on a cloud worker (Ollama needs a local server, and there is no
AWS realtime factory: it fell back to OpenAI).
"""

from __future__ import annotations

from collections.abc import Mapping

#: Each provider and the credentials it accepts; any one of them is enough. A
#: tuple of tuples means all of the inner names are needed together.
Requirement = tuple[str | tuple[str, ...], ...]

LIVEKIT_INFERENCE: tuple[str, ...] = ("LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")

STT_REQUIREMENTS: dict[str, Requirement] = {
    "deepgram": ("DEEPGRAM_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "assemblyai": ("ASSEMBLYAI_API_KEY",),
    "gemini": ("GOOGLE_API_KEY",),
    "google": ("GOOGLE_APPLICATION_CREDENTIALS",),
    "elevenlabs": ("ELEVEN_API_KEY", "ELEVENLABS_API_KEY"),
    "cartesia": ("CARTESIA_API_KEY",),
}

TTS_REQUIREMENTS: dict[str, Requirement] = {
    "openai": ("OPENAI_API_KEY",),
    "elevenlabs": ("ELEVEN_API_KEY", "ELEVENLABS_API_KEY"),
    "cartesia": ("CARTESIA_API_KEY",),
    "deepgram": ("DEEPGRAM_API_KEY",),
    "google": ("GOOGLE_APPLICATION_CREDENTIALS",),
    "gemini": ("GOOGLE_API_KEY",),
    "rime": (LIVEKIT_INFERENCE,),
}

LLM_REQUIREMENTS: dict[str, Requirement] = {
    "inference": (LIVEKIT_INFERENCE,),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GOOGLE_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "cerebras": ("CEREBRAS_API_KEY",),
    "ollama": (),
}

REALTIME_REQUIREMENTS: dict[str, Requirement] = {
    "openai": ("OPENAI_API_KEY",),
    "google": ("GOOGLE_API_KEY",),
    "aws": (),
}


def _met(requirement: Requirement, keys: Mapping[str, str]) -> bool:
    for option in requirement:
        names = option if isinstance(option, tuple) else (option,)
        if all(keys.get(name) for name in names):
            return True
    return False


def _stage(table: dict[str, Requirement], keys: Mapping[str, str]) -> dict[str, bool]:
    return {provider: _met(requirement, keys) for provider, requirement in table.items()}


def voice_provider_keys(keys: Mapping[str, str]) -> dict[str, dict[str, bool]]:
    """Every voice provider per stage, True when its credential is present."""
    return {
        "stt": _stage(STT_REQUIREMENTS, keys),
        "tts": _stage(TTS_REQUIREMENTS, keys),
        "llm": _stage(LLM_REQUIREMENTS, keys),
        "realtime": _stage(REALTIME_REQUIREMENTS, keys),
    }
