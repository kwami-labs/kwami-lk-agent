"""TTS construction and voice validation, provider by provider.

Real plugin constructors throughout -- a kwarg the SDK rejects is precisely the
bug class these factories exist to catch, and a mock would accept anything.

The behaviour most worth pinning is the fallback: when a provider cannot be
built, `config.tts_provider` must be *corrected* to openai, not merely logged.
Leaving it reading "elevenlabs" while an OpenAI client ran made every later
voice update validate against the wrong provider's voice list.
"""

from __future__ import annotations

import logging

import pytest

from src.constants import (
    CartesiaVoices,
    DeepgramVoices,
    ElevenLabsVoices,
    GoogleVoices,
    OpenAIModels,
    OpenAIVoices,
    TTSProviders,
)
from src.domain import KwamiVoiceConfig
from src.factories import tts as tts_module
from src.factories.tts import (
    _check_api_key,
    create_tts,
    get_available_providers,
    get_default_voice,
    get_voices_for_provider,
)


def voice_config(**overrides) -> KwamiVoiceConfig:
    return KwamiVoiceConfig(**overrides)


@pytest.fixture(autouse=True)
def tts_keys(fake_key):
    for name in (
        "OPENAI_API_KEY",
        "DEEPGRAM_API_KEY",
        "CARTESIA_API_KEY",
        "ELEVEN_API_KEY",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
    ):
        fake_key(name)


# =============================================================================
# _check_api_key
# =============================================================================


def test_a_configured_provider_passes_the_key_check() -> None:
    assert _check_api_key(TTSProviders.OPENAI) is True


def test_a_missing_key_warns_without_blocking(env_setting, caplog) -> None:
    """Warning only: a key can be injected by the platform at call time."""
    env_setting("CARTESIA_API_KEY", None)

    with caplog.at_level(logging.WARNING):
        assert _check_api_key(TTSProviders.CARTESIA) is False

    assert "CARTESIA_API_KEY" in caplog.text


def test_an_unknown_provider_is_assumed_configured() -> None:
    assert _check_api_key("nonesuch") is True


def test_either_elevenlabs_spelling_satisfies_the_check(env_setting) -> None:
    env_setting("ELEVEN_API_KEY", None)
    env_setting("ELEVENLABS_API_KEY", "alias-key")

    assert _check_api_key(TTSProviders.ELEVENLABS) is True


def test_either_google_spelling_satisfies_the_check(env_setting) -> None:
    """`.env.sample` documents GOOGLE_API_KEY; the plugin wants the credentials
    file. Either counts as configured."""
    env_setting("GOOGLE_APPLICATION_CREDENTIALS", None)
    env_setting("GOOGLE_API_KEY", "google-key")

    assert _check_api_key(TTSProviders.GOOGLE) is True


# =============================================================================
# OpenAI
# =============================================================================


def test_openai_is_constructed_with_a_valid_model_and_voice() -> None:
    instance = create_tts(voice_config(tts_provider="openai", tts_voice="nova"))

    assert type(instance).__module__.startswith("livekit.plugins.openai")


def test_an_unsupported_openai_model_falls_back(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        create_tts(voice_config(tts_provider="openai", tts_model="not-a-model"))

    assert "not supported by OpenAI TTS" in caplog.text


def test_an_unsupported_openai_voice_falls_back(caplog) -> None:
    """A Cartesia UUID left behind by a provider switch must not be sent."""
    with caplog.at_level(logging.WARNING):
        create_tts(voice_config(tts_provider="openai", tts_voice="79a125e8-cd45-4c13"))

    assert "not supported by OpenAI TTS" in caplog.text


def test_an_openai_model_prefix_is_stripped() -> None:
    assert create_tts(voice_config(tts_provider="openai", tts_model="openai/tts-1")) is not None


def test_openai_defaults_apply_when_nothing_is_configured() -> None:
    assert create_tts(voice_config(tts_provider="openai", tts_model="", tts_voice="")) is not None


@pytest.mark.parametrize("model", sorted(OpenAIModels.ALL_TTS))
def test_every_advertised_openai_model_constructs(model: str) -> None:
    assert create_tts(voice_config(tts_provider="openai", tts_model=model)) is not None


@pytest.mark.parametrize("voice", sorted(OpenAIVoices.STANDARD))
def test_every_advertised_openai_voice_constructs(voice: str) -> None:
    assert create_tts(voice_config(tts_provider="openai", tts_voice=voice)) is not None


def test_a_zero_speed_becomes_the_default() -> None:
    """`float(config.tts_speed or 1.0)` -- 0 is falsy and the plugin rejects it."""
    assert create_tts(voice_config(tts_provider="openai", tts_speed=0)) is not None


# =============================================================================
# ElevenLabs (via LiveKit Inference)
# =============================================================================


def test_elevenlabs_goes_through_livekit_inference() -> None:
    instance = create_tts(
        voice_config(tts_provider="elevenlabs", tts_voice=ElevenLabsVoices.DEFAULT)
    )

    assert type(instance).__module__.startswith("livekit.agents")


def test_a_short_voice_name_is_rejected_as_an_elevenlabs_id(caplog) -> None:
    """ "nova" is an OpenAI voice leaking through a provider switch; ElevenLabs
    IDs are 20-character alphanumerics."""
    with caplog.at_level(logging.WARNING):
        create_tts(voice_config(tts_provider="elevenlabs", tts_voice="nova"))

    assert "not a valid ElevenLabs voice ID" in caplog.text


def test_a_long_unknown_voice_id_is_accepted() -> None:
    """Only short names are rejected; an unfamiliar 20-char id may be a real
    custom voice on the user's account."""
    instance = create_tts(voice_config(tts_provider="elevenlabs", tts_voice="A" * 20))

    assert instance is not None


def test_an_elevenlabs_model_is_normalised(caplog) -> None:
    with caplog.at_level(logging.INFO):
        create_tts(voice_config(tts_provider="elevenlabs", tts_model="eleven-flash-v2.5"))

    assert "elevenlabs/eleven_flash_v2_5" in caplog.text


def test_an_elevenlabs_prefix_is_stripped(caplog) -> None:
    with caplog.at_level(logging.INFO):
        create_tts(
            voice_config(tts_provider="elevenlabs", tts_model="elevenlabs/eleven_turbo_v2_5")
        )

    assert "elevenlabs/eleven_turbo_v2_5" in caplog.text


# =============================================================================
# Rime, Cartesia, Deepgram
# =============================================================================


def test_rime_goes_through_livekit_inference(caplog) -> None:
    with caplog.at_level(logging.INFO):
        instance = create_tts(voice_config(tts_provider="rime", tts_model="", tts_voice="astra"))

    assert type(instance).__module__.startswith("livekit.agents")
    assert "rime/arcana:astra" in caplog.text


def test_rime_does_not_validate_the_model_it_is_given(caplog) -> None:
    """Pinning current behaviour, not endorsing it. Every other provider here
    validates the incoming model, so a stale `tts-1` left over from OpenAI is
    corrected. Rime passes it straight through as "rime/tts-1", which is not a
    Rime model -- the same leak the ElevenLabs and Cartesia voice guards exist
    to catch, on the model axis instead of the voice one.
    """
    with caplog.at_level(logging.INFO):
        create_tts(voice_config(tts_provider="rime", tts_model="tts-1", tts_voice="astra"))

    assert "rime/tts-1:astra" in caplog.text


def test_a_rime_prefix_is_stripped(caplog) -> None:
    with caplog.at_level(logging.INFO):
        create_tts(voice_config(tts_provider="rime", tts_model="rime/mistv2"))

    assert "rime/mistv2" in caplog.text


def test_cartesia_is_constructed_with_a_uuid_voice() -> None:
    instance = create_tts(voice_config(tts_provider="cartesia", tts_voice=CartesiaVoices.DEFAULT))

    assert type(instance).__module__.startswith("livekit.plugins.cartesia")


def test_a_friendly_cartesia_name_is_mapped_to_its_uuid() -> None:
    friendly = next(iter(CartesiaVoices.NAME_MAP))

    assert create_tts(voice_config(tts_provider="cartesia", tts_voice=friendly)) is not None


def test_a_short_non_uuid_cartesia_voice_falls_back(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        create_tts(voice_config(tts_provider="cartesia", tts_voice="zzz"))

    assert "expected UUID format" in caplog.text


def test_deepgram_is_constructed_with_a_known_voice() -> None:
    instance = create_tts(voice_config(tts_provider="deepgram", tts_voice=DeepgramVoices.DEFAULT))

    assert type(instance).__module__.startswith("livekit.plugins.deepgram")


def test_an_unknown_deepgram_voice_falls_back(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        create_tts(voice_config(tts_provider="deepgram", tts_voice="not-a-voice"))

    assert "not in known Deepgram voices" in caplog.text


def test_an_explicit_deepgram_model_wins_over_the_derived_one() -> None:
    assert (
        create_tts(voice_config(tts_provider="deepgram", tts_model="aura-2-thalia-en")) is not None
    )


# =============================================================================
# Google
# =============================================================================


def test_google_without_the_plugin_falls_back_to_openai(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    monkeypatch.setattr(tts_module, "google", None)

    with caplog.at_level(logging.WARNING):
        instance = create_tts(voice_config(tts_provider="google"))

    assert type(instance).__module__.startswith("livekit.plugins.openai")
    assert "Google TTS plugin not installed" in caplog.text


class StrictGoogle:
    """Strict stand-in: google is an extra and absent from the locked
    environment, so what is pinned is our call shape, not the SDK's."""

    class TTS:
        def __init__(self, *, voice: str, speaking_rate: float) -> None:
            self.voice = voice
            self.speaking_rate = speaking_rate


def test_google_receives_the_voice_and_speaking_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tts_module, "google", StrictGoogle)

    instance = create_tts(
        voice_config(tts_provider="google", tts_voice="en-US-Studio-O", tts_speed=1.25)
    )

    assert instance.voice == "en-US-Studio-O"
    assert instance.speaking_rate == 1.25


def test_google_defaults_its_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tts_module, "google", StrictGoogle)

    instance = create_tts(voice_config(tts_provider="google", tts_voice=""))

    assert instance.voice == GoogleVoices.DEFAULT


# =============================================================================
# Fallback
# =============================================================================


def test_an_unknown_provider_falls_back_to_openai(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        instance = create_tts(voice_config(tts_provider="nonesuch"))

    assert type(instance).__module__.startswith("livekit.plugins.openai")
    assert "falling back to OpenAI" in caplog.text


def test_the_fallback_rewrites_the_configured_provider(caplog) -> None:
    """The defect this exists for: leaving tts_provider reading the failed
    provider made every later voice update validate against the wrong list."""
    config = voice_config(tts_provider="nonesuch")

    with caplog.at_level(logging.WARNING):
        create_tts(config)

    assert config.tts_provider == TTSProviders.OPENAI
    assert "voice validation now follows OpenAI" in caplog.text


def test_a_failing_provider_falls_back_and_is_recorded(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    def explode(config):
        raise RuntimeError("cartesia is down")

    monkeypatch.setattr(tts_module, "_create_cartesia_tts", explode)
    config = voice_config(tts_provider="cartesia")

    with caplog.at_level(logging.ERROR):
        instance = create_tts(config)

    assert type(instance).__module__.startswith("livekit.plugins.openai")
    assert config.tts_provider == TTSProviders.OPENAI
    assert "Failed to create cartesia TTS" in caplog.text


def test_a_failing_openai_is_not_rewritten(monkeypatch: pytest.MonkeyPatch) -> None:
    """Already OpenAI: the guard stops a pointless rewrite and a confusing log."""
    calls: list[str] = []
    real = tts_module._create_openai_tts

    def once(config):
        calls.append("x")
        if len(calls) == 1:
            raise RuntimeError("transient")
        return real(config)

    monkeypatch.setattr(tts_module, "_create_openai_tts", once)
    config = voice_config(tts_provider="openai")

    assert create_tts(config) is not None
    assert config.tts_provider == TTSProviders.OPENAI


def test_the_provider_is_matched_case_insensitively() -> None:
    assert create_tts(voice_config(tts_provider="OPENAI")) is not None


# =============================================================================
# Catalogue helpers
# =============================================================================


def test_the_always_available_providers_are_listed() -> None:
    providers = get_available_providers()

    assert {
        TTSProviders.OPENAI,
        TTSProviders.DEEPGRAM,
        TTSProviders.CARTESIA,
        TTSProviders.RIME,
        TTSProviders.ELEVENLABS,
    } <= set(providers)


def test_google_is_listed_only_when_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tts_module, "google", None)
    assert TTSProviders.GOOGLE not in get_available_providers()

    monkeypatch.setattr(tts_module, "google", StrictGoogle)
    assert TTSProviders.GOOGLE in get_available_providers()


@pytest.mark.parametrize(
    "provider",
    [
        TTSProviders.OPENAI,
        TTSProviders.ELEVENLABS,
        TTSProviders.DEEPGRAM,
        TTSProviders.CARTESIA,
        TTSProviders.GOOGLE,
    ],
)
def test_every_provider_offers_voices(provider: str) -> None:
    assert get_voices_for_provider(provider)


def test_an_unknown_provider_offers_no_voices() -> None:
    assert get_voices_for_provider("nonesuch") == []


def test_voice_lookup_is_case_insensitive() -> None:
    assert get_voices_for_provider("OpenAI") == list(OpenAIVoices.STANDARD)


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        (TTSProviders.OPENAI, OpenAIVoices.DEFAULT),
        (TTSProviders.ELEVENLABS, ElevenLabsVoices.DEFAULT),
        (TTSProviders.CARTESIA, CartesiaVoices.DEFAULT),
        (TTSProviders.DEEPGRAM, DeepgramVoices.DEFAULT),
        (TTSProviders.GOOGLE, GoogleVoices.DEFAULT),
    ],
)
def test_each_provider_has_a_default_voice(provider: str, expected: str) -> None:
    assert get_default_voice(provider) == expected


def test_an_unknown_provider_has_a_generic_default() -> None:
    assert get_default_voice("nonesuch") == "default"


def test_the_default_lookup_is_case_insensitive() -> None:
    assert get_default_voice("OpenAI") == OpenAIVoices.DEFAULT
