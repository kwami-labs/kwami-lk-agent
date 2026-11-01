"""Backup providers for the standard pipeline, so one dead provider is not a mute agent.

Measured on 2026-10-20: the deployment's Google key is on the free tier, which
allows Gemini 3.5 Flash 20 requests a day and Gemini TTS 10 -- one request per
reply and one per sentence. Past that every call is a 429 until the next day,
and nothing caught it: the LLM failed and the turn was dropped in silence, or
the TTS failed mid-reply and, after three such failures, the framework closed
the whole session. The user saw ElevenLabs "answer once": it was Gemini, the
thinking half of that pipeline, running out.

So the standard pipeline wraps each stage in the framework's FallbackAdapter:
the configured provider first, then the ones this deployment has a key for, in
order of measured latency (voice_lab, 2026-10-20). A provider that fails is
skipped until a background probe sees it recover, so a spent quota costs one
failed attempt, not one per turn.

The backups are only ever stand-ins. The configured provider stays first, and
the LLM chain is sticky so a recovered Gemini does not flip the voice's brain
back and forth mid-conversation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from livekit.agents import llm as lk_llm
from livekit.agents import stt as lk_stt
from livekit.agents import tts as lk_tts

from ..domain import KwamiVoiceConfig
from ..settings import get_settings
from ..utils.logging import get_logger
from .availability import LLM_REQUIREMENTS, STT_REQUIREMENTS, TTS_REQUIREMENTS, _met

logger = get_logger("fallback")

#: (provider, model, voice) per stage, fastest first. LLM: gpt-4.1-mini 0.74 s
#: and Claude Haiku 0.75 s to the first token, both solid with ~120 tools; Groq
#: is faster (0.3 s) but kept last, its models are weaker at tool choice.
LLM_BACKUPS: tuple[tuple[str, str], ...] = (
    ("openai", "gpt-4.1-mini"),
    ("anthropic", "claude-haiku-5-5"),
    ("groq", "openai/gpt-oss-120b"),
)

#: STT: AssemblyAI 0.13 s after end of speech, Deepgram streams and reads Spanish well.
STT_BACKUPS: tuple[tuple[str, str], ...] = (
    ("assemblyai", "universal-streaming-multilingual"),
    ("deepgram", "nova-3"),
)

#: TTS, all streaming: ElevenLabs Flash 0.24 s to first audio, Cartesia 0.44 s,
#: Deepgram Aura 0.74 s.
TTS_BACKUPS: tuple[tuple[str, str], ...] = (
    ("elevenlabs", "eleven_flash_v2_5"),
    ("cartesia", "sonic-3"),
    ("deepgram", "aura-2"),
)

#: Gemini's prebuilt voices that are female; every other prebuilt one is male.
GEMINI_FEMALE_VOICES = frozenset(
    {
        "zephyr",
        "kore",
        "leda",
        "aoede",
        "callirrhoe",
        "autonoe",
        "despina",
        "erinome",
        "laomedeia",
        "achernar",
        "gacrux",
        "pulcherrima",
        "vindemiatrix",
        "sulafat",
    }
)

#: ElevenLabs stock voices the free plan can speak with (library voices are a
#: 402): George for a male character, Jessica for a female one. They are the
#: app's Jarvis and Friday voices on that engine.
ELEVENLABS_MALE = "JBFqnCBsd6RMkjVDRZzb"
ELEVENLABS_FEMALE = "cgSgspJ2msm6clMCkdW9"
ELEVENLABS_FEMALE_IDS = frozenset(
    {ELEVENLABS_FEMALE, "Xb7hH8MSUJpSbSDYk0k2", "EXAVITQu4vr4xnSDxMaL", "pFZP5JQG7iQjIQuC4Bku"}
)


def _has_key(table: dict, provider: str) -> bool:
    requirement = table.get(provider)
    if requirement is None:
        return False
    return _met(requirement, get_settings().provider_keys)


def is_female_voice(voice: str | None) -> bool:
    """Best guess at the configured voice's register, to keep it across a fallback."""
    if not voice:
        return False
    return voice.lower() in GEMINI_FEMALE_VOICES or voice in ELEVENLABS_FEMALE_IDS


def _backup_voice(provider: str, primary_voice: str | None) -> str:
    if provider == "elevenlabs":
        return ELEVENLABS_FEMALE if is_female_voice(primary_voice) else ELEVENLABS_MALE
    # Cartesia and Deepgram: their factories pick a voice for the language.
    return ""


def _chain(
    stage: str,
    primary: Any,
    primary_provider: str,
    backups: tuple[tuple[str, str], ...],
    table: dict,
    build: Callable[[str, str], Any],
) -> list[Any]:
    members = [primary]
    for provider, model in backups:
        if provider == primary_provider or not _has_key(table, provider):
            continue
        try:
            members.append(build(provider, model))
        except Exception as e:
            logger.warning("Could not build %s backup %s/%s: %s", stage, provider, model, e)
    if len(members) > 1:
        logger.info(
            "%s fallback chain: %s",
            stage.upper(),
            " -> ".join(
                [
                    primary_provider,
                    *[p for p, _ in backups if p != primary_provider and _has_key(table, p)],
                ]
            ),
        )
    return members


#: Gemini through LiveKit Inference: the same family, billed to the LiveKit
#: project, so a spent free-tier Google quota does not silence it. First
#: backup when the voice thinks with Gemini (0.69 s to the first token).
GEMINI_VIA_INFERENCE = ("inference", "google/gemini-2.5-flash")


def with_llm_fallback(primary: Any, config: KwamiVoiceConfig, create_llm: Callable) -> Any:
    provider = (config.llm_provider or "").lower()

    def build(p: str, m: str) -> Any:
        return create_llm(replace(config, llm_provider=p, llm_model=m))

    backups = LLM_BACKUPS
    if provider == "google":
        backups = (GEMINI_VIA_INFERENCE, *LLM_BACKUPS)
    members = _chain("llm", primary, provider, backups, LLM_REQUIREMENTS, build)
    if len(members) == 1:
        return primary
    # sticky: once on a backup, stay there for the session rather than
    # bouncing back to a provider whose quota is spent for the day.
    # attempt_timeout reaches Gemini as the request deadline, and Gemini
    # answers 400 to anything under 10 s (the adapter default is 5).
    return lk_llm.FallbackAdapter(members, attempt_timeout=10.0, sticky=True)


def with_tts_fallback(primary: Any, config: KwamiVoiceConfig, create_tts: Callable) -> Any:
    provider = (config.tts_provider or "").lower()

    def build(p: str, m: str) -> Any:
        return create_tts(
            replace(
                config,
                tts_provider=p,
                tts_model=m,
                tts_voice=_backup_voice(p, config.tts_voice),
                # A backup never has the primary's direction: only Gemini reads one.
                tts_style="",
            )
        )

    members = _chain("tts", primary, provider, TTS_BACKUPS, TTS_REQUIREMENTS, build)
    if len(members) == 1:
        return primary
    # One retry on the primary: a blip should not change the voice, a spent
    # quota should, quickly.
    return lk_tts.FallbackAdapter(members, max_retry_per_tts=1)


def with_stt_fallback(
    primary: Any, config: KwamiVoiceConfig, create_stt: Callable, vad: Any
) -> Any:
    provider = (config.stt_provider or "").lower()

    def build(p: str, m: str) -> Any:
        return create_stt(replace(config, stt_provider=p, stt_model=m))

    members = _chain("stt", primary, provider, STT_BACKUPS, STT_REQUIREMENTS, build)
    if len(members) == 1:
        return primary
    try:
        return lk_stt.FallbackAdapter(members, vad=vad)
    except Exception as e:
        logger.warning("STT fallback unavailable (%s); using %s alone", e, provider)
        return primary


def primary_of(component: Any) -> Any:
    """The configured provider inside a fallback chain, or the component itself.

    Live option updates (a new voice, a new language) go to the provider the
    user picked; the adapters themselves take no options.
    """
    for attr in ("_tts_instances", "_llm_instances", "_stt_instances"):
        members = getattr(component, attr, None)
        if members:
            return members[0]
    return component
