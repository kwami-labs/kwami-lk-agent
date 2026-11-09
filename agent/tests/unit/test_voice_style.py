"""A character's voice on Gemini Live: a prebuilt voice plus a spoken direction.

Gemini Live takes only its 30 prebuilt voices, so "Jarvis" and "Friday" are a
prebuilt voice and a description of how to speak. The description must reach
the instructions on a realtime pipeline, stay out of a TTS one (where it would
describe a voice the model does not produce), and changing it must rebuild the
session, because instructions cannot be swapped on an open Live socket.
"""

from __future__ import annotations

from src.domain.config import KwamiSoulConfig, KwamiVoiceConfig
from src.domain.prompt import build_system_prompt
from src.handlers.realtime import LIVE_UPDATABLE, apply_realtime_fields, has_realtime_keys

STYLE = "Calma de mayordomo, ritmo pausado, dicción nítida."


def test_the_style_is_in_the_instructions() -> None:
    prompt = build_system_prompt(KwamiSoulConfig(name="Jarvis"), voice_style=STYLE)
    assert f"How your voice sounds when you speak: {STYLE}" in prompt


def test_no_style_no_line() -> None:
    assert "How your voice sounds" not in build_system_prompt(KwamiSoulConfig())


def test_a_style_update_is_a_realtime_change_that_rebuilds() -> None:
    voice = KwamiVoiceConfig(pipeline_type="realtime")
    payload = {"realtimeStyle": STYLE, "realtimeVoice": "Charon"}
    assert has_realtime_keys(payload)
    changed = apply_realtime_fields(voice, payload)
    assert changed == {"realtime_style", "realtime_voice"}
    assert voice.realtime_style == STYLE
    assert "realtime_style" not in LIVE_UPDATABLE


def test_an_empty_style_clears_it() -> None:
    voice = KwamiVoiceConfig(realtime_style=STYLE)
    assert apply_realtime_fields(voice, {"realtimeStyle": ""}) == {"realtime_style"}
    assert voice.realtime_style == ""


# -- the standard pipeline on Gemini: STT, TTS and the direction ------------------


def test_gemini_tts_carries_the_direction_in_googles_structure(monkeypatch) -> None:
    from src.factories.tts import create_tts

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")  # the plugin reads it at construction

    tts = create_tts(KwamiVoiceConfig(tts_provider="gemini", tts_voice="Charon", tts_style=STYLE))
    assert type(tts).__module__.endswith("gemini_tts")
    assert tts._opts.voice_name == "Charon"
    # The OpenAI default "tts-1" must not leak into a Gemini request.
    assert tts._opts.model.startswith("gemini-")
    assert tts._opts.instructions.startswith("# AUDIO PROFILE\n" + STYLE)
    assert tts._opts.instructions.endswith("#### TRANSCRIPT")


def test_gemini_tts_without_a_direction_keeps_the_plugin_default() -> None:
    from src.factories.tts import gemini_tts_instructions

    assert gemini_tts_instructions("  ") is None


def test_gemini_stt_widens_the_language_to_spain(monkeypatch) -> None:
    from src.factories.stt import create_stt

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    stt = create_stt(KwamiVoiceConfig(stt_provider="gemini", stt_language="es"))
    assert type(stt).__module__.endswith("gemini_stt")


def test_a_gemini_voice_or_model_means_the_gemini_tts() -> None:
    from src.utils.provider import detect_provider_change

    assert detect_provider_change("openai", new_voice="Charon") == ("gemini", True)
    assert detect_provider_change("openai", new_model="gemini-3.1-flash-tts-preview")[0] == "gemini"
    assert detect_provider_change("gemini", new_voice="Aoede") == ("gemini", False)


def test_a_switch_to_standard_carries_its_own_engines() -> None:
    from src.handlers.realtime import apply_standard_fields

    voice = KwamiVoiceConfig(pipeline_type="realtime", stt_provider="deepgram")
    changed = apply_standard_fields(
        voice,
        {
            "sttProvider": "gemini",
            "stt_language": "es",
            "llmProvider": "google",
            "llmModel": "gemini-3.5-flash",
            "ttsProvider": "gemini",
            "ttsVoice": "Charon",
            "ttsStyle": STYLE,
        },
    )
    assert voice.stt_provider == "gemini" and voice.llm_model == "gemini-3.5-flash"
    assert (voice.tts_provider, voice.tts_voice, voice.tts_style) == ("gemini", "Charon", STYLE)
    assert "stt_model" not in changed  # absent fields are left alone


def test_gemini_stt_ignores_another_providers_default_model(monkeypatch) -> None:
    from src.factories.stt import create_stt

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    stt = create_stt(KwamiVoiceConfig(stt_provider="gemini", stt_model="nova-2"))
    assert stt._opts.model == "gemini-3.5-transcribe-live"


# -- one voice on: the agent cannot switch it --------------------------------------


def test_a_character_voice_locks_the_voice_switching_tools() -> None:
    from src.agent import KwamiAgent
    from src.domain.config import KwamiConfig
    from src.domain.tool_gating import VOICE_SWITCH_TOOLS

    def tool_names(config: KwamiConfig) -> set[str]:
        agent = KwamiAgent(config=config)
        return {getattr(getattr(t, "info", None), "name", "") for t in agent.tools}

    locked = KwamiConfig()
    locked.voice.pipeline_type = "standard"
    locked.voice.tts_style = STYLE
    assert locked.voice.voice_locked
    assert not (tool_names(locked) & VOICE_SWITCH_TOOLS)

    free = KwamiConfig()
    free.voice.pipeline_type = "standard"
    assert not free.voice.voice_locked
    assert tool_names(free) >= VOICE_SWITCH_TOOLS


def test_the_lock_follows_the_engine_in_use() -> None:
    voice = KwamiVoiceConfig(pipeline_type="realtime", tts_style=STYLE)
    assert not voice.voice_locked  # Live has no direction set
    voice.realtime_style = STYLE
    assert voice.voice_locked


def test_a_locked_voice_is_told_so() -> None:
    prompt = build_system_prompt(KwamiSoulConfig(), voice_locked=True)
    assert "Your voice is fixed by the user" in prompt
    assert "Your voice is fixed" not in build_system_prompt(KwamiSoulConfig())


# -- the standard pipeline must be able to describe its tools to Gemini --------------


def test_rediscovered_builtins_are_bound_and_keep_the_gating() -> None:
    """KeyError 'self' (2026-10-10): tools rediscovered on the class were unbound,
    and the Gemini LLM of the standard pipeline failed every turn building their
    schema. They also came back after being withheld."""
    from livekit.agents.llm.utils import build_legacy_openai_schema

    from src.agent import KwamiAgent
    from src.domain.config import KwamiConfig
    from src.domain.tool_gating import VOICE_SWITCH_TOOLS

    config = KwamiConfig()
    config.voice.pipeline_type = "standard"
    config.voice.tts_style = STYLE
    tools = KwamiAgent(config=config).available_builtin_tools()

    for tool in tools:
        build_legacy_openai_schema(tool, internally_tagged=True)  # raised KeyError: 'self'
    names = {t.info.name for t in tools}
    assert "delegate_task" in names
    assert not (names & VOICE_SWITCH_TOOLS)


def test_a_gemini_tts_model_is_kept_and_no_direction_sends_none(monkeypatch) -> None:
    from src.factories.tts import create_tts

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    tts = create_tts(
        KwamiVoiceConfig(tts_provider="gemini", tts_model="gemini/gemini-2.5-pro-preview-tts")
    )
    assert tts._opts.model == "gemini-2.5-pro-preview-tts"
    # No direction of ours: the plugin keeps its own default instructions.
    assert not (tts._opts.instructions or "").startswith("# AUDIO PROFILE")


def test_gemini_tts_takes_only_a_gemini_voice(monkeypatch) -> None:
    from src.factories.tts import DEFAULT_GEMINI_TTS_VOICE, create_tts

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    # The config default is OpenAI's "nova", which Gemini would reject.
    tts = create_tts(KwamiVoiceConfig(tts_provider="gemini"))
    assert KwamiVoiceConfig().tts_voice == "nova"
    assert tts._opts.voice_name == DEFAULT_GEMINI_TTS_VOICE
    # A Gemini voice is kept, in Gemini's own capitalisation.
    tts = create_tts(KwamiVoiceConfig(tts_provider="gemini", tts_voice="iapetus"))
    assert tts._opts.voice_name == "Iapetus"


def test_gemini_tts_without_the_google_plugin_falls_back_to_openai(
    monkeypatch, caplog, fake_key
) -> None:
    import logging

    import src.factories.tts as tts_module

    fake_key("OPENAI_API_KEY")
    monkeypatch.setattr(tts_module, "google", None)
    with caplog.at_level(logging.WARNING):
        tts = tts_module.create_tts(KwamiVoiceConfig(tts_provider="gemini"))
    assert type(tts).__module__.startswith("livekit.plugins.openai")
    assert "Google plugin not installed" in caplog.text


def test_resending_the_same_style_or_engine_changes_nothing() -> None:
    from src.handlers.realtime import apply_standard_fields

    voice = KwamiVoiceConfig(realtime_style=STYLE, tts_provider="gemini")
    assert apply_realtime_fields(voice, {"realtimeStyle": STYLE}) == set()
    assert apply_standard_fields(voice, {"ttsProvider": "gemini"}) == set()
