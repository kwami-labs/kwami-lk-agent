"""The voice of a session without the app: secrets first, then the job metadata.

The user tries combinations from the LiveKit Console with no redeploy:
AssemblyAI universal-streaming-multilingual + Groq qwen3.8-27b + ElevenLabs
flash v2.5 with George, then others. Whatever the metadata holds, the session
must still get an agent.
"""

from __future__ import annotations

import json
import logging

from src.domain import KwamiVoiceConfig
from src.runtime.defaults import placeholder_config

JARVIS_ENV = {
    "KWAMI_DEFAULT_PIPELINE_TYPE": "standard",
    "KWAMI_DEFAULT_LANGUAGE": "es",
    "KWAMI_DEFAULT_NAME": "Jarvis",
    "KWAMI_DEFAULT_STT_PROVIDER": "gemini",
    "KWAMI_DEFAULT_STT_MODEL": "gemini-3.5-transcribe-live",
    "KWAMI_DEFAULT_LLM_PROVIDER": "google",
    "KWAMI_DEFAULT_LLM_MODEL": "gemini-3.5-flash",
    "KWAMI_DEFAULT_TTS_PROVIDER": "gemini",
    "KWAMI_DEFAULT_TTS_MODEL": "gemini-3.1-flash-tts-preview",
    "KWAMI_DEFAULT_TTS_VOICE": "Charon",
}

CONSOLE_TRIAL = {
    "voice": {
        "stt_provider": "assemblyai",
        "stt_model": "universal-streaming-multilingual",
        "llm_provider": "groq",
        "llm_model": "qwen/qwen3.8-27b",
        "tts_provider": "elevenlabs",
        "tts_model": "eleven_flash_v2_5",
        "tts_voice": "JBFqnCBsd6RMkjVDRZzb",
    }
}


def test_nothing_set_is_the_built_in_default() -> None:
    assert placeholder_config(env={}).voice == KwamiVoiceConfig()


def test_the_secrets_make_it_jarvis_in_spanish() -> None:
    config = placeholder_config(env=JARVIS_ENV)
    voice = config.voice
    assert (voice.stt_provider, voice.llm_provider, voice.tts_provider) == (
        "gemini",
        "google",
        "gemini",
    )
    assert voice.tts_voice == "Charon"
    assert voice.stt_language == "es" and config.soul.language == "es"
    assert config.soul.name == "Jarvis" and config.kwami_name == "Jarvis"


def test_the_old_realtime_secrets_still_work() -> None:
    config = placeholder_config(
        env={
            "KWAMI_DEFAULT_PIPELINE_TYPE": "realtime",
            "KWAMI_DEFAULT_REALTIME_PROVIDER": "google",
            "KWAMI_DEFAULT_REALTIME_MODEL": "gemini-3.1-flash-live-preview",
            "KWAMI_DEFAULT_REALTIME_VOICE": "Charon",
        }
    )
    voice = config.voice
    assert voice.pipeline_type == "realtime"
    assert (voice.realtime_provider, voice.realtime_voice) == ("google", "Charon")


def test_job_metadata_overrides_the_secrets_for_one_session() -> None:
    config = placeholder_config(env=JARVIS_ENV, job_metadata=json.dumps(CONSOLE_TRIAL))
    voice = config.voice
    assert (voice.stt_provider, voice.stt_model) == (
        "assemblyai",
        "universal-streaming-multilingual",
    )
    assert (voice.llm_provider, voice.llm_model) == ("groq", "qwen/qwen3.8-27b")
    assert (voice.tts_provider, voice.tts_model, voice.tts_voice) == (
        "elevenlabs",
        "eleven_flash_v2_5",
        "JBFqnCBsd6RMkjVDRZzb",
    )
    # What the metadata does not name stays as the secrets set it.
    assert voice.stt_language == "es" and config.soul.name == "Jarvis"
    # The secrets themselves are untouched for the next session.
    assert placeholder_config(env=JARVIS_ENV).voice.llm_provider == "google"


def test_a_provider_without_a_model_does_not_inherit_the_old_one() -> None:
    config = placeholder_config(
        env=JARVIS_ENV, job_metadata=json.dumps({"voice": {"llm_provider": "groq"}})
    )
    assert config.voice.llm_provider == "groq"
    assert config.voice.llm_model == "", "gemini-3.5-flash would go to Groq"
    config = placeholder_config(
        env=JARVIS_ENV, job_metadata='{"voice": {"tts_provider": "cartesia"}}'
    )
    assert (config.voice.tts_model, config.voice.tts_voice) == ("", "")


def test_language_in_the_metadata_sets_both_sides() -> None:
    config = placeholder_config(env={}, job_metadata='{"voice": {"language": "es"}}')
    assert config.voice.stt_language == "es" and config.soul.language == "es"


def test_bad_fields_are_skipped_with_a_warning(caplog) -> None:
    metadata = json.dumps(
        {
            "kwami_id": "k1",
            "voice": {
                "stt_provider": "deepgram",
                "colour": "blue",
                "pipeline_type": "telepathy",
                "tts_speed": "fast",
                "llm_temperature": 0.3,
                "llm_model": 42,
            },
        }
    )
    with caplog.at_level(logging.WARNING):
        voice = placeholder_config(env={}, job_metadata=metadata).voice

    assert voice.stt_provider == "deepgram"
    assert voice.llm_temperature == 0.3
    assert voice.pipeline_type == "standard"
    assert voice.tts_speed == 1.0
    assert voice.llm_model == KwamiVoiceConfig().llm_model
    for name in ("colour", "pipeline_type", "tts_speed", "llm_model"):
        assert name in caplog.text


def test_any_metadata_still_gives_an_agent() -> None:
    for metadata in ("not json", "[1, 2]", '{"voice": "assemblyai"}', '{"voice": null}', ""):
        config = placeholder_config(env=JARVIS_ENV, job_metadata=metadata)
        assert config.voice.llm_provider == "google"


def test_the_placeholder_builds_that_pipeline(monkeypatch) -> None:
    """End to end through the real factories: no fallback to another provider."""
    from src.runtime.pipeline import create_agent_from_config
    from src.settings import Settings, set_settings

    for name in ("ASSEMBLYAI_API_KEY", "GROQ_API_KEY", "ELEVEN_API_KEY"):
        monkeypatch.setenv(name, "test-key-not-real")
    set_settings(Settings.from_env())
    try:
        config = placeholder_config(env={}, job_metadata=json.dumps(CONSOLE_TRIAL))
        agent = create_agent_from_config(config, vad=None, skip_greeting=True)
    finally:
        set_settings(None)

    assert agent.stt.provider == "AssemblyAI"
    assert agent.llm.model == "qwen/qwen3.8-27b"
    assert agent.tts.provider == "ElevenLabs"


def test_switches_and_lists_are_read_by_type(caplog) -> None:
    metadata = json.dumps(
        {
            "voice": {
                "echo_cancellation": False,
                "noise_suppression": "FALSE",
                "auto_gain_control": "maybe",
                "realtime_modalities": ["audio"],
                "stt_word_boost": [1, 2],
                "llm_temperature": [0.3],
                "stt_language": "es-ES",
                "language": "es",
            }
        }
    )
    with caplog.at_level(logging.WARNING):
        config = placeholder_config(env={}, job_metadata=metadata)

    voice = config.voice
    assert voice.echo_cancellation is False and voice.noise_suppression is False
    assert voice.auto_gain_control is True
    assert voice.realtime_modalities == ["audio"]
    assert voice.stt_word_boost == []
    assert voice.llm_temperature == KwamiVoiceConfig().llm_temperature
    # An explicit STT language wins over the general one.
    assert (voice.stt_language, config.soul.language) == ("es-ES", "es")
    for name in ("auto_gain_control", "stt_word_boost", "llm_temperature"):
        assert name in caplog.text


def test_an_override_that_breaks_the_build_costs_only_the_override(caplog) -> None:
    """A number too large for an int field overflows rather than failing to parse."""
    with caplog.at_level(logging.ERROR):
        config = placeholder_config(
            env=JARVIS_ENV, job_metadata='{"voice": {"llm_max_tokens": 1e400}}'
        )
    assert config.voice == KwamiVoiceConfig()
    assert "Could not apply the default voice" in caplog.text
