"""STT construction, provider by provider.

The factories construct *real* plugin objects -- that is the point, since the
bug class they exist to catch is a kwarg the SDK rejects. Nothing here is
mocked; each assertion is that the real constructor accepted what we passed.
"""

from __future__ import annotations

import logging

import pytest

from src.constants import STTProviders
from src.domain import KwamiVoiceConfig
from src.factories import stt as stt_module
from src.factories.stt import ELEVENLABS_SERVER_VAD, create_stt


def voice(**overrides) -> KwamiVoiceConfig:
    return KwamiVoiceConfig(**overrides)


@pytest.fixture(autouse=True)
def stt_keys(fake_key):
    # LIVEKIT_API_KEY/SECRET are needed because the ElevenLabs path is routed
    # through LiveKit Inference rather than the plugin.
    for name in (
        "DEEPGRAM_API_KEY",
        "OPENAI_API_KEY",
        "CARTESIA_API_KEY",
        "ELEVEN_API_KEY",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
    ):
        fake_key(name)


def test_deepgram_is_constructed_with_the_configured_model() -> None:
    instance = create_stt(voice(stt_provider="deepgram", stt_model="nova-3"))

    assert type(instance).__module__.startswith("livekit.plugins.deepgram")


def test_a_prefixed_model_is_stripped_before_construction() -> None:
    """The frontend sends "deepgram/nova-3"; the plugin wants "nova-3"."""
    instance = create_stt(voice(stt_provider="deepgram", stt_model="deepgram/nova-3"))

    assert instance is not None


def test_an_empty_model_falls_back_to_the_default() -> None:
    assert create_stt(voice(stt_provider="deepgram", stt_model="")) is not None


def test_openai_is_constructed() -> None:
    instance = create_stt(voice(stt_provider="openai", stt_model="whisper-1"))

    assert type(instance).__module__.startswith("livekit.plugins.openai")


def test_openai_translates_multi_language_to_none() -> None:
    """ "multi" is a Deepgram concept; OpenAI rejects it as a language code."""
    assert create_stt(voice(stt_provider="openai", stt_language="multi")) is not None


def test_elevenlabs_uses_the_plugin_on_our_own_key() -> None:
    """Inference answered "model not found" for scribe_v2_realtime (2026-10-18)."""
    instance = create_stt(voice(stt_provider="elevenlabs", stt_language="es"))

    assert type(instance).__module__.startswith("livekit.plugins.elevenlabs")


def test_elevenlabs_realtime_commits_on_server_vad() -> None:
    """Without it only interim transcripts arrive and the turn never ends."""
    instance = create_stt(voice(stt_provider="elevenlabs", stt_model="scribe_v2_realtime"))

    assert instance._opts.server_vad == ELEVENLABS_SERVER_VAD


def test_elevenlabs_without_a_key_goes_through_livekit_inference(env_setting) -> None:
    env_setting("ELEVEN_API_KEY", None)
    env_setting("ELEVENLABS_API_KEY", None)

    instance = create_stt(voice(stt_provider="elevenlabs"))

    assert type(instance).__module__.startswith("livekit.agents")


def test_an_elevenlabs_model_is_normalised_to_underscores(caplog) -> None:
    """Both paths expect scribe_v2_realtime, not hyphens."""
    with caplog.at_level(logging.INFO):
        create_stt(voice(stt_provider="elevenlabs", stt_model="scribe-v2-realtime"))

    assert "ElevenLabs plugin for STT: scribe_v2_realtime" in caplog.text


def test_a_non_scribe_elevenlabs_model_is_replaced(caplog) -> None:
    with caplog.at_level(logging.INFO):
        create_stt(voice(stt_provider="elevenlabs", stt_model="nova-3"))

    assert "ElevenLabs plugin for STT: scribe_v2_realtime" in caplog.text


def test_cartesia_is_constructed() -> None:
    instance = create_stt(voice(stt_provider="cartesia", stt_model="ink-whisper"))

    assert type(instance).__module__.startswith("livekit.plugins.cartesia")


def test_an_unknown_provider_falls_back_to_deepgram(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        instance = create_stt(voice(stt_provider="nonesuch"))

    assert type(instance).__module__.startswith("livekit.plugins.deepgram")
    assert "falling back to Deepgram" in caplog.text


@pytest.mark.parametrize("provider", ["assemblyai", "google"])
def test_an_uninstalled_optional_provider_falls_back(
    monkeypatch: pytest.MonkeyPatch, provider: str, caplog
) -> None:
    """These two are genuine extras. Without the package the branch guard is
    False and the session must still get an STT rather than an exception."""
    monkeypatch.setattr(stt_module, provider, None)

    with caplog.at_level(logging.WARNING):
        instance = create_stt(voice(stt_provider=provider))

    assert type(instance).__module__.startswith("livekit.plugins.deepgram")
    assert "Unknown or unavailable STT provider" in caplog.text


def test_a_constructor_failure_falls_back_to_deepgram(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """A provider whose plugin raises must degrade, not kill the session."""

    class ExplodingSTT:
        def __init__(self, **kwargs: object) -> None:
            raise RuntimeError("bad credentials")

    monkeypatch.setattr(stt_module.openai, "STT", ExplodingSTT)

    with caplog.at_level(logging.ERROR):
        instance = create_stt(voice(stt_provider="openai"))

    assert type(instance).__module__.startswith("livekit.plugins.deepgram")
    assert "falling back to Deepgram" in caplog.text


def test_the_provider_is_matched_case_insensitively() -> None:
    assert create_stt(voice(stt_provider="DEEPGRAM")) is not None


@pytest.mark.parametrize("provider", sorted(STTProviders.ALL))
def test_every_advertised_provider_yields_an_stt(provider: str) -> None:
    """The catalogue is what the frontend offers; each entry must construct
    something, whether its own plugin or the documented fallback."""
    assert create_stt(voice(stt_provider=provider)) is not None


# =============================================================================
# AssemblyAI -- a core dependency now, so this is the real constructor. The
# stand-in it replaces pinned `word_boost`, a kwarg the plugin had dropped: the
# test passed while every real AssemblyAI session fell back to Deepgram.
# =============================================================================


@pytest.fixture
def assemblyai_key(fake_key):
    fake_key("ASSEMBLYAI_API_KEY")


@pytest.mark.usefixtures("assemblyai_key")
def test_assemblyai_receives_the_key_terms_and_language() -> None:
    instance = create_stt(
        voice(stt_provider="assemblyai", stt_word_boost=["Kwami"], stt_language="es")
    )

    assert type(instance).__module__.startswith("livekit.plugins.assemblyai")
    assert instance._opts.keyterms_prompt == ["Kwami"]
    assert instance._opts.language_codes == ["es"]


@pytest.mark.usefixtures("assemblyai_key")
def test_assemblyai_ignores_a_model_that_is_not_its_own() -> None:
    """The config default is Deepgram's nova-2; it must not reach AssemblyAI."""
    instance = create_stt(voice(stt_provider="assemblyai", stt_model="nova-2"))

    assert type(instance).__module__.startswith("livekit.plugins.assemblyai")
    assert instance._opts.speech_model.startswith(("universal", "u3"))


@pytest.mark.usefixtures("assemblyai_key")
def test_assemblyai_streaming_models_get_no_language() -> None:
    """universal-streaming-* detect the language and reject `language_codes`:
    passing it raised a ValueError and fell back to Deepgram (2026-10-18)."""
    instance = create_stt(
        voice(
            stt_provider="assemblyai",
            stt_model="universal-streaming-multilingual",
            stt_language="es",
        )
    )

    assert type(instance).__module__.startswith("livekit.plugins.assemblyai")
    assert instance._opts.speech_model == "universal-streaming-multilingual"


# =============================================================================
# Google Cloud STT. The stand-in has a strict keyword signature, so what is
# pinned is *our* call shape: drop a kwarg or rename one and these fail.
# =============================================================================


class StrictGoogle:
    class STT:
        def __init__(self, *, model: str, languages: list) -> None:
            self.model = model
            self.languages = languages


def test_google_receives_the_model_and_language(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stt_module, "google", StrictGoogle)

    instance = create_stt(voice(stt_provider="google", stt_model="chirp-2", stt_language="fr-FR"))

    assert instance.model == "chirp-2"
    assert instance.languages == ["fr-FR"]


def test_google_defaults_the_model_and_language(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stt_module, "google", StrictGoogle)

    instance = create_stt(voice(stt_provider="google", stt_model="", stt_language=""))

    assert instance.model == "chirp"
    assert instance.languages == ["en-US"]
