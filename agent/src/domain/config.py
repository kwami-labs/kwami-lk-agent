"""Kwami agent configuration types.

Supports full configuration for voice AI pipelines including:
- STT (Speech-to-Text): Deepgram, OpenAI, AssemblyAI, Google, ElevenLabs
- LLM (Large Language Model): OpenAI, Google, Anthropic, Groq, DeepSeek, Mistral, Cerebras, Ollama
- TTS (Text-to-Speech): Cartesia, ElevenLabs, OpenAI, Deepgram, Google
- VAD (Voice Activity Detection): Silero
- Realtime: OpenAI GPT-4o Realtime, Google Gemini Live
- Memory: Zep Cloud for persistent agent memory
"""

from dataclasses import dataclass, field
from typing import Literal

from ..settings import get_settings


@dataclass
class KwamiMemoryConfig:
    """Memory configuration for Zep Cloud integration.

    Each Kwami maintains independent memory through unique user_id and session_id.
    Zep provides:
    - Persistent conversation history
    - Automatic fact extraction
    - Temporal knowledge graphs with custom entity types
    - Sub-200ms context retrieval
    """

    # Enable/disable memory (auto-enabled if ZEP_API_KEY is set)
    # Resolved through Settings rather than os.getenv directly, so a test can
    # construct a KwamiConfig without its result depending on what the developer
    # happens to have exported.
    enabled: bool = field(default_factory=lambda: get_settings().memory_enabled)

    # Zep API key (defaults to ZEP_API_KEY env var)
    api_key: str = field(default_factory=lambda: get_settings().zep_api_key)

    # Unique identifier for this Kwami's user in Zep
    # This should be the kwami_id for independent memory per Kwami
    user_id: str = ""

    # Session/thread ID for the current conversation
    # If empty, a new session is created each time
    session_id: str = ""

    # Whether to automatically inject memory context into system prompt
    auto_inject_context: bool = True

    # Maximum number of recent messages to include in context
    max_context_messages: int = 10

    # Whether to include extracted facts in context
    include_facts: bool = True

    # Whether to include entities in context
    include_entities: bool = True

    # Minimum relevance score for facts (0.0 - 1.0)
    min_fact_relevance: float = 0.5

    # Whether to configure custom ontology (entity/edge types)
    configure_ontology: bool = True


@dataclass
class KwamiCharterConfig:
    """The standing rules that turn a persona into a chief of staff.

    A soul says how the agent sounds; the charter says how it behaves: how it
    addresses the user, when it pushes back, how much initiative it takes and
    what it may learn about them. Off by default, so a kwami that never opts in
    keeps exactly the prompt it had. Option meanings live in `domain/charter.py`,
    which is the only reader.
    """

    enabled: bool = False
    #: How to address the user ("señor"). Empty means by name, as before.
    address_as: str = ""
    #: Formal second person (usted, vous, Sie) in languages that have one.
    formal: bool = False
    wit: Literal["none", "subtle", "marked"] = "subtle"
    #: off: never objects. once: objects once, then complies without argument.
    #: firm: waits for an explicit go-ahead after objecting.
    pushback: Literal["off", "once", "firm"] = "once"
    opinions: Literal["always", "on_request", "never"] = "always"
    #: act: researches straight away. propose: offers first. wait: only when asked.
    initiative: Literal["act", "propose", "wait"] = "act"
    #: Areas the agent watches for risk. See `charter.WATCH_TOPICS`.
    watch: list[str] = field(default_factory=list)
    #: announce: notes what it learns and says so. silent: notes quietly. off.
    learn: Literal["announce", "silent", "off"] = "announce"
    greeting: Literal["brief_pending", "brief", "briefing", "none"] = "brief_pending"
    #: Match answer length to the question instead of the soul's fixed length.
    adaptive_length: bool = True
    #: What the agent knows about the user, written by them and by the agent.
    user_profile: str = ""
    #: Things awaiting the user, mentioned in the greeting.
    pending: list[str] = field(default_factory=list)


@dataclass
class KwamiSoulConfig:
    """Soul figuration from the Kwami frontend."""

    name: str = "Kwami"
    personality: str = "A friendly and helpful AI companion"
    system_prompt: str = ""
    traits: list[str] = field(default_factory=list)
    language: str = "en"
    conversation_style: str = "friendly"
    response_length: Literal["short", "medium", "long"] = "medium"
    emotional_tone: Literal[
        "neutral",
        "warm",
        "enthusiastic",
        "calm",
        "playful",
        "confident",
        "serious",
        "compassionate",
    ] = "warm"
    emotional_traits: dict[str, float] = field(default_factory=dict)
    charter: KwamiCharterConfig = field(default_factory=KwamiCharterConfig)


@dataclass
class KwamiVoiceConfig:
    """Voice pipeline configuration from the Kwami frontend.

    This configuration supports:
    - Standard pipeline: STT → LLM → TTS
    - Realtime pipeline: Single model for ultra-low latency
    """

    # =========================================================================
    # Pipeline Type
    # =========================================================================

    pipeline_type: Literal["standard", "realtime"] = "standard"

    # =========================================================================
    # Speech-to-Text (STT) Configuration
    # =========================================================================

    # Provider: deepgram, openai, assemblyai, google, elevenlabs
    stt_provider: str = "deepgram"

    # Deepgram models: nova-3, nova-2, nova-2-conversationalai, nova-2-phonecall,
    #                  nova-2-meeting, enhanced, base
    # OpenAI models: whisper-1, whisper-large-v3, whisper-large-v3-turbo
    # AssemblyAI models: best, nano, conformer-2
    # Google models: chirp, chirp-2, telephony, command_and_search
    # ElevenLabs models: scribe_v1
    stt_model: str = "nova-2"

    # Language codes: en, en-US, en-GB, es, fr, de, it, pt, ja, ko, zh, hi, ar, ru, multi
    stt_language: str = "en"

    # Word boost for better recognition (provider-specific)
    stt_word_boost: list[str] = field(default_factory=list)

    # =========================================================================
    # Large Language Model (LLM) Configuration
    # =========================================================================

    # Provider: openai, google, anthropic, groq, deepseek, mistral, cerebras, ollama
    llm_provider: str = "openai"

    # OpenAI models: gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano,
    #                gpt-4-turbo, gpt-4, gpt-3.5-turbo, o1, o1-mini, o3-mini
    # Google models: gemini-2.0-flash, gemini-2.0-flash-lite, gemini-1.5-pro,
    #                gemini-1.5-flash, gemini-1.5-flash-8b
    # Anthropic models: claude-3-5-sonnet-latest, claude-3-5-haiku-latest,
    #                   claude-3-opus-latest, claude-sonnet-4-20250514
    # Groq models: llama-3.3-70b-versatile, llama-3.1-70b-versatile,
    #              llama-3.1-8b-instant, llama3-70b-8192, mixtral-8x7b-32768
    # DeepSeek models: deepseek-chat, deepseek-reasoner
    # Mistral models: mistral-large-latest, mistral-medium-latest, mistral-small-latest,
    #                 open-mixtral-8x22b, open-mixtral-8x7b, open-mistral-7b
    # Cerebras models: llama3.1-70b, llama3.1-8b
    # Ollama models: llama3.2, llama3.1, llama3, mistral, mixtral, phi3, gemma2, qwen2.5
    llm_model: str = "gpt-4o-mini"

    # Temperature for response creativity (0.0 - 1.0)
    llm_temperature: float = 0.7

    # Maximum tokens in response
    llm_max_tokens: int = 1024

    # =========================================================================
    # Text-to-Speech (TTS) Configuration
    # =========================================================================

    # Provider: openai, elevenlabs, cartesia, deepgram, google
    tts_provider: str = "openai"

    # OpenAI models: tts-1, tts-1-hd, gpt-4o-mini-tts
    # ElevenLabs models: eleven_turbo_v2_5, eleven_turbo_v2, eleven_multilingual_v2,
    #                    eleven_monolingual_v1, eleven_flash_v2_5, eleven_flash_v2
    # Cartesia models: sonic-2, sonic-english, sonic-multilingual
    # Deepgram models: aura-asteria-en, aura-luna-en, aura-stella-en, etc.
    # Google models: en-US-Studio-O, en-US-Studio-Q, en-US-Neural2-*
    tts_model: str = "tts-1"

    # Voice ID or name (provider-specific)
    # OpenAI TTS: alloy, ash, coral, echo, fable, nova, onyx, sage, shimmer
    #             (note: ballad, verse are Realtime API only - NOT for standard TTS)
    # ElevenLabs: Voice IDs (e.g., "21m00Tcm4TlvDq8ikWAM" for Rachel)
    # Cartesia: UUID voice IDs (e.g., "79a125e8-cd45-4c13-8a67-188112f4dd22")
    # Deepgram: asteria, luna, stella, athena, hera, orion, arcas, perseus, etc.
    # Google: en-US-Studio-O, en-US-Studio-Q, en-US-Neural2-A, etc.
    tts_voice: str = "nova"

    # Speech speed multiplier (0.5 - 2.0)
    tts_speed: float = 1.0

    # Spoken direction for TTS engines that take one (Gemini TTS): pace,
    # pitch, diction, attitude. The same text a character uses on Gemini Live
    # (`realtime_style`); here it goes to the TTS, not the LLM.
    tts_style: str = ""

    @property
    def active_style(self) -> str:
        """The voice direction of the pipeline in use."""
        return self.realtime_style if self.pipeline_type == "realtime" else self.tts_style

    @property
    def voice_locked(self) -> bool:
        """A character voice is set, so the voice changes only when the user picks
        another one in the Voice panel -- never by the agent's own tools (the
        user's rule, 2026-10-09: one voice on, the rest off)."""
        return bool(self.active_style.strip())

    # =========================================================================
    # Voice Activity Detection (VAD) Configuration
    # =========================================================================

    # Provider: silero (currently the only reliable option)
    vad_provider: str = "silero"

    # Speech detection threshold (0.0 - 1.0)
    vad_threshold: float = 0.5

    # Minimum speech duration to trigger detection (seconds)
    vad_min_speech_duration: float = 0.1

    # Minimum silence duration to end speech (seconds)
    vad_min_silence_duration: float = 0.3

    # =========================================================================
    # Realtime Model Configuration (for ultra-low latency)
    # =========================================================================

    # Provider: openai, google
    realtime_provider: str = "openai"

    # OpenAI models: gpt-4o-realtime-preview, gpt-4o-realtime-preview-2024-10-01,
    #                gpt-4o-realtime-preview-2024-12-17
    # Google models: gemini-3.8-live, gemini-3.1-flash-live-preview
    realtime_model: str = "gpt-4o-realtime-preview"

    # Realtime voice (provider-specific)
    # OpenAI: alloy, ash, ballad, coral, echo, sage, shimmer, verse
    # Google: Puck, Charon, Kore, Fenrir, Aoede
    realtime_voice: str = "alloy"

    # How the realtime voice should sound: pace, pitch, diction, accent. Gemini
    # Live only takes its 30 prebuilt voices, so a character's voice is a
    # prebuilt one plus this direction, which native-audio models follow. Goes
    # into the instructions (domain/prompt.py); empty means no direction.
    realtime_style: str = ""

    # Modalities: ["text", "audio"] or ["text"] or ["audio"]
    realtime_modalities: list[str] = field(default_factory=lambda: ["text", "audio"])

    # =========================================================================
    # Listening: noise, turn-taking and interruptions (runtime/turn_taking.py)
    # =========================================================================
    # The defaults are the Spanish tuning; the app's Enhancements panel, and the
    # `set_enhancement_control` tool the agent calls, override them per session.

    # Echo cancellation and gain control run in the browser, before the audio
    # leaves the user's device. Kept so the app's values round-trip.
    echo_cancellation: bool = True
    auto_gain_control: bool = True

    # Noise cancellation on the user's microphone, on the agent (LiveKit Cloud).
    noise_suppression: bool = True

    # "bvc" also removes voices in the background (a TV, people nearby), which
    # otherwise read as the user talking and cut the agent off; "krisp" is the
    # standard model and keeps them. Phone callers get BVC tuned for telephony.
    noise_cancellation_mode: Literal["bvc", "krisp"] = "bvc"

    # How the end of the user's turn is decided on the standard pipeline:
    # LiveKit's multilingual end-of-turn "model", plain "vad" silence, or "stt".
    turn_detector: Literal["model", "vad", "stt"] = "model"

    # Silence before the turn may end; learned upwards from the minimum.
    min_endpointing_delay: float = 0.4
    max_endpointing_delay: float = 2.5

    # Whether the user can cut the agent off, and how much speech it takes.
    allow_interruptions: bool = True
    min_interruption_duration: float = 0.4
    min_interruption_words: int = 0

    # The LLM starts on the interim transcript while the turn is still open.
    preemptive_generation: bool = True

    # Turn detection on a realtime model's own server. "semantic" waits for the
    # user to finish a thought (OpenAI's semantic VAD; Gemini Live with low
    # start-of-speech sensitivity); "server_vad" ends on silence alone and is
    # what noise trips; "none" leaves the provider's default.
    turn_detection: Literal["server_vad", "semantic", "none"] = "semantic"


@dataclass
class KwamiConfig:
    """Full Kwami configuration received from frontend.

    This configuration is sent when the user connects and can include:
    - Unique Kwami instance identifiers
    - Soul figuration (personality, traits, prompts)
    - Voice pipeline configuration (STT, LLM, TTS)
    - Memory configuration (Zep Cloud for persistent memory)
    - Tool definitions for function calling
    """

    kwami_id: str = ""
    kwami_name: str = "Kwami"
    #: IANA zone for the person on the other end ("Europe/Madrid"). Empty means
    #: unknown, and `get_current_time` then says UTC rather than implying the
    #: container's clock is theirs -- it answered in naive container-local time,
    #: which on both deploy targets is UTC presented as if it were the user's.
    timezone: str = ""
    #: BCP-47 tag ("en-GB", "es-ES") for date and number formatting. Distinct
    #: from `soul.language`, which decides what the model *writes*: someone can
    #: want Spanish replies with UK date formatting.
    locale: str = ""
    soul: KwamiSoulConfig = field(default_factory=KwamiSoulConfig)
    voice: KwamiVoiceConfig = field(default_factory=KwamiVoiceConfig)
    memory: KwamiMemoryConfig = field(default_factory=KwamiMemoryConfig)
    tools: list[dict] = field(default_factory=list)

    # Backward-compatible alias for clients/modules that still use "persona".
    @property
    def persona(self) -> KwamiSoulConfig:
        return self.soul

    @persona.setter
    def persona(self, value: KwamiSoulConfig) -> None:
        self.soul = value


# Backward-compatible type alias for old imports.
KwamiPersonaConfig = KwamiSoulConfig


# =============================================================================
# Preset Configurations
# =============================================================================


def get_preset_config(preset: str) -> KwamiVoiceConfig:
    """Get a preset voice configuration.

    Available presets:
    - fast: Optimized for lowest latency (Groq + Cartesia)
    - balanced: Good balance of speed and quality (OpenAI GPT-4o-mini + Cartesia)
    - quality: Best quality (OpenAI GPT-4o + ElevenLabs)
    - multilingual: Best for multiple languages
    - realtime: Ultra-low latency with OpenAI Realtime
    """
    presets = {
        "fast": KwamiVoiceConfig(
            stt_provider="deepgram",
            stt_model="nova-3",
            llm_provider="groq",
            llm_model="llama-3.1-8b-instant",
            llm_temperature=0.7,
            tts_provider="openai",
            tts_model="tts-1",
            tts_voice="nova",
        ),
        "balanced": KwamiVoiceConfig(
            stt_provider="deepgram",
            stt_model="nova-2",
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            llm_temperature=0.7,
            tts_provider="openai",
            tts_model="tts-1",
            tts_voice="alloy",
        ),
        "quality": KwamiVoiceConfig(
            stt_provider="deepgram",
            stt_model="nova-3",
            llm_provider="openai",
            llm_model="gpt-4o",
            llm_temperature=0.7,
            tts_provider="elevenlabs",
            tts_model="eleven_turbo_v2_5",
            tts_voice="JBFqnCBsd6RMkjVDRZzb",  # George; Rachel needs a paid plan
        ),
        "multilingual": KwamiVoiceConfig(
            stt_provider="deepgram",
            stt_model="nova-2",
            stt_language="multi",
            llm_provider="openai",
            llm_model="gpt-4o",
            llm_temperature=0.7,
            tts_provider="openai",
            tts_model="tts-1",
            tts_voice="alloy",
        ),
        "realtime": KwamiVoiceConfig(
            pipeline_type="realtime",
            realtime_provider="openai",
            realtime_model="gpt-4o-realtime-preview",
            realtime_voice="alloy",
        ),
    }

    return presets.get(preset, presets["balanced"])
