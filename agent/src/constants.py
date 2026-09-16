"""Centralized constants for Kwami Agent configuration.

This module contains constants for:
- Voice Providers (TTS, STT, LLM)
- Model Names and Types
- Voice IDs
- API Environment Variables
- Configuration Defaults
"""

# =============================================================================
# Providers
# =============================================================================


class TTSProviders:
    OPENAI = "openai"
    ELEVENLABS = "elevenlabs"
    CARTESIA = "cartesia"
    DEEPGRAM = "deepgram"
    GOOGLE = "google"
    RIME = "rime"

    ALL = {OPENAI, ELEVENLABS, CARTESIA, DEEPGRAM, GOOGLE, RIME}


class STTProviders:
    DEEPGRAM = "deepgram"
    OPENAI = "openai"
    ASSEMBLYAI = "assemblyai"
    GOOGLE = "google"
    ELEVENLABS = "elevenlabs"
    CARTESIA = "cartesia"

    ALL = {DEEPGRAM, OPENAI, ASSEMBLYAI, GOOGLE, ELEVENLABS, CARTESIA}


class LLMProviders:
    OPENAI = "openai"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"
    CEREBRAS = "cerebras"
    OLLAMA = "ollama"

    ALL = {OPENAI, GOOGLE, ANTHROPIC, GROQ, DEEPSEEK, MISTRAL, CEREBRAS, OLLAMA}


# =============================================================================
# Voice IDs
# =============================================================================


class OpenAIVoices:
    """OpenAI TTS voice IDs."""

    ALLOY = "alloy"  # Neutral
    ASH = "ash"  # Male
    CORAL = "coral"  # Female
    ECHO = "echo"  # Male
    FABLE = "fable"  # Neutral
    NOVA = "nova"  # Female
    ONYX = "onyx"  # Male
    SAGE = "sage"  # Female
    SHIMMER = "shimmer"  # Female

    # Realtime only
    BALLAD = "ballad"
    VERSE = "verse"

    ALL = {ALLOY, ASH, CORAL, ECHO, FABLE, NOVA, ONYX, SAGE, SHIMMER, BALLAD, VERSE}
    STANDARD = {ALLOY, ASH, CORAL, ECHO, FABLE, NOVA, ONYX, SAGE, SHIMMER}
    DEFAULT = NOVA


class ElevenLabsVoices:
    """ElevenLabs voice IDs (premade voices)."""

    RACHEL = "21m00Tcm4TlvDq8ikWAM"
    DOMI = "AZnzlk1XvdvUeBnXmlld"
    BELLA = "EXAVITQu4vr4xnSDxMaL"
    ELLI = "MF3mGyEYCl7XYWbV9V6O"
    JOSH = "TxGEqnHWrfWFTfGW9XjX"
    ARNOLD = "VR6AewLTigWG4xSOukaG"
    ADAM = "pNInz6obpgDQGcFmaJgB"
    SAM = "yoZ06aMxZJJ28mfd3POQ"
    DANIEL = "onwK4e9ZLuTAKqWW03F9"
    CHARLOTTE = "XB0fDUnXU5powFXDhCwa"
    LILY = "pFZP5JQG7iQjIQuC4Bku"
    CALLUM = "N2lVS1w4EtoT3dr4eOWO"
    CHARLIE = "IKne3meq5aSn9XLyUdCD"
    GEORGE = "JBFqnCBsd6RMkjVDRZzb"
    LIAM = "TX3LPaxmHKxFdv7VOQHJ"
    WILL = "bIHbv24MWmeRgasZH58o"
    JESSICA = "cgSgspJ2msm6clMCkdW9"
    ERIC = "cjVigY5qzO86Huf0OWal"
    CHRIS = "iP95p4xoKVk53GoZ742B"
    BRIAN = "nPczCjzI2devNBz1zQrb"

    ALL = {
        RACHEL,
        DOMI,
        BELLA,
        ELLI,
        JOSH,
        ARNOLD,
        ADAM,
        SAM,
        DANIEL,
        CHARLOTTE,
        LILY,
        CALLUM,
        CHARLIE,
        GEORGE,
        LIAM,
        WILL,
        JESSICA,
        ERIC,
        CHRIS,
        BRIAN,
    }
    DEFAULT = RACHEL


class CartesiaVoices:
    """Cartesia voice IDs (UUID format)."""

    # English - Female
    BRITISH_LADY = "79a125e8-cd45-4c13-8a67-188112f4dd22"
    JACQUELINE = "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
    CALIFORNIA_GIRL = "c2ac25f9-ecc4-4f56-9095-651354df60c0"
    READING_LADY = "b7d50908-b17c-442d-ad8d-810c63997ed9"
    SARAH = "00a77add-48d5-4ef6-8157-71e5437b282d"
    MIDWESTERN_WOMAN = "ed81fd13-2016-4a49-8fe3-c0d2761695fc"
    MARIA = "5619d38c-cf51-4d8e-9575-48f61a280413"
    COMMERCIAL_LADY = "f146dcec-e481-45be-8ad2-96e1e40e7f32"
    # English - Male
    NEWSMAN = "a167e0f3-df7e-4d52-a9c3-f949145efdab"
    COMMERCIAL_MAN = "63ff761f-c1e8-414b-b969-d1833d1c870c"
    FRIENDLY_SIDEKICK = "421b3369-f63f-4b03-8980-37a44df1d4e8"
    SOUTHERN_MAN = "638efaaa-4d0c-442e-b701-3fae16aad012"
    WISE_MAN = "fb26447f-308b-471e-8b00-8e9f04284eb5"
    BRITISH_NARRATOR = "2ee87190-8f84-4925-97da-e52547f9462c"

    # Friendly Name Mapping
    NAME_MAP = {
        "british lady": BRITISH_LADY,
        "sophia": BRITISH_LADY,
        "california girl": CALIFORNIA_GIRL,
        "reading lady": READING_LADY,
        "newsman": NEWSMAN,
        "blake": NEWSMAN,
        "commercial man": COMMERCIAL_MAN,
        "friendly sidekick": FRIENDLY_SIDEKICK,
    }

    DEFAULT = BRITISH_LADY


class DeepgramVoices:
    """Deepgram Aura voice IDs."""

    # Female
    ASTERIA = "asteria"
    LUNA = "luna"
    STELLA = "stella"
    ATHENA = "athena"
    HERA = "hera"
    # Male
    ORION = "orion"
    ARCAS = "arcas"
    PERSEUS = "perseus"
    ANGUS = "angus"
    ORPHEUS = "orpheus"
    HELIOS = "helios"
    ZEUS = "zeus"

    ALL = {ASTERIA, LUNA, STELLA, ATHENA, HERA, ORION, ARCAS, PERSEUS, ANGUS, ORPHEUS, HELIOS, ZEUS}
    DEFAULT = ASTERIA


class GoogleVoices:
    """Google Cloud TTS voice IDs."""

    STUDIO_O = "en-US-Studio-O"  # Female
    STUDIO_Q = "en-US-Studio-Q"  # Male
    NEURAL2_A = "en-US-Neural2-A"  # Male
    NEURAL2_C = "en-US-Neural2-C"  # Female
    NEURAL2_D = "en-US-Neural2-D"  # Male
    NEURAL2_E = "en-US-Neural2-E"  # Female
    NEURAL2_F = "en-US-Neural2-F"  # Female
    NEURAL2_G = "en-US-Neural2-G"  # Female
    NEURAL2_H = "en-US-Neural2-H"  # Female
    NEURAL2_I = "en-US-Neural2-I"  # Male
    NEURAL2_J = "en-US-Neural2-J"  # Male

    DEFAULT = STUDIO_O


# =============================================================================
# Models
# =============================================================================


class OpenAIModels:
    TTS_1 = "tts-1"
    TTS_1_HD = "tts-1-hd"
    GPT_4O_MINI_TTS = "gpt-4o-mini-tts"

    WHISPER_1 = "whisper-1"
    WHISPER_LARGE_V3 = "whisper-large-v3"

    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"

    ALL_TTS = {TTS_1, TTS_1_HD, GPT_4O_MINI_TTS}


class DeepgramModels:
    NOVA_2 = "nova-2"
    NOVA_3 = "nova-3"
    NOVA_2_CONVERSATIONAL = "nova-2-conversationalai"
    BASE = "base"
    ENHANCED = "enhanced"

    DEFAULT_STT = NOVA_2


# =============================================================================
# Environment Variables
# =============================================================================


class EnvVars:
    OPENAI = ["OPENAI_API_KEY"]
    ELEVENLABS = ["ELEVEN_API_KEY", "ELEVENLABS_API_KEY"]
    CARTESIA = ["CARTESIA_API_KEY"]
    DEEPGRAM = ["DEEPGRAM_API_KEY"]
    GOOGLE = ["GOOGLE_APPLICATION_CREDENTIALS"]
    ZEP = ["ZEP_API_KEY"]


# =============================================================================
# Misc Constants
# =============================================================================

LANGUAGE_GREETINGS = {
    "en": "Language changed to English. How can I help you?",
    "es": "Idioma cambiado a espanol. Como puedo ayudarte?",
    "fr": "Langue changee en francais. Comment puis-je vous aider?",
    "de": "Sprache auf Deutsch geändert. Wie kann ich Ihnen helfen?",
    "it": "Lingua cambiata in italiano. Come posso aiutarti?",
    "pt": "Idioma alterado para portugues. Como posso ajuda-lo?",
    "ja": "Language changed to Japanese. How can I help you?",
    "ko": "Language changed to Korean. How can I help you?",
    "zh": "Language changed to Chinese. How can I help you?",
}
