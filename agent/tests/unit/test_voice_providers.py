"""`brain_status.voiceProviders` reports every voice provider and its key.

The factories never refuse: a provider without its key is built anyway and
fails, or falls back to OpenAI or Deepgram, so the user hears a model they did
not pick. The app greys out the providers reported without a key, per stage.
"""

from __future__ import annotations

from pathlib import Path

from src.brain import Bridge
from src.brain.ledger import Ledger
from src.factories.availability import (
    LLM_REQUIREMENTS,
    REALTIME_REQUIREMENTS,
    STT_REQUIREMENTS,
    TTS_REQUIREMENTS,
    voice_provider_keys,
)
from src.settings import Settings


def test_every_stage_lists_every_provider() -> None:
    report = voice_provider_keys({})
    assert set(report) == {"stt", "tts", "llm", "realtime"}
    assert set(report["stt"]) == set(STT_REQUIREMENTS)
    assert set(report["tts"]) == set(TTS_REQUIREMENTS)
    assert set(report["llm"]) == set(LLM_REQUIREMENTS)
    assert set(report["realtime"]) == set(REALTIME_REQUIREMENTS)
    assert not any(v for stage in report.values() for v in stage.values())


def test_the_factories_providers_are_all_reported() -> None:
    from src.constants import STTProviders, TTSProviders

    assert set(STTProviders.ALL) <= set(STT_REQUIREMENTS)
    assert set(TTSProviders.ALL) <= set(TTS_REQUIREMENTS)


def test_one_key_lights_up_every_stage_it_serves() -> None:
    report = voice_provider_keys({"GOOGLE_API_KEY": "g"})
    assert report["stt"]["gemini"] and report["tts"]["gemini"]
    assert report["llm"]["google"] and report["realtime"]["google"]
    # Google Cloud TTS/STT need a service account, not the Gemini key.
    assert not report["stt"]["google"] and not report["tts"]["google"]


def test_rime_needs_both_livekit_credentials() -> None:
    assert not voice_provider_keys({"LIVEKIT_API_KEY": "k"})["tts"]["rime"]
    both = {"LIVEKIT_API_KEY": "k", "LIVEKIT_API_SECRET": "s"}
    assert voice_provider_keys(both)["tts"]["rime"]


def test_either_elevenlabs_spelling_counts() -> None:
    assert voice_provider_keys({"ELEVENLABS_API_KEY": "e"})["tts"]["elevenlabs"]
    assert voice_provider_keys({"ELEVEN_API_KEY": "e"})["stt"]["elevenlabs"]


def test_brain_status_keeps_the_flat_realtime_keys(tmp_path: Path) -> None:
    bridge = Bridge(keys={"GOOGLE_API_KEY": "g"}, ledger=Ledger(tmp_path))
    voice = bridge.status()["voiceProviders"]
    assert voice["google"] is True and voice["openai"] is False and voice["aws"] is False
    assert voice["realtime"] == {"openai": False, "google": True, "aws": False}
    assert voice["stt"]["gemini"] is True and voice["llm"]["anthropic"] is False


def test_settings_mirror_the_keys_the_report_reads(monkeypatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "m")
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")
    keys = Settings.from_env().provider_keys
    report = voice_provider_keys(keys)
    assert report["llm"]["mistral"] and report["tts"]["rime"]
