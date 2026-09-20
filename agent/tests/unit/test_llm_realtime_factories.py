"""LLM and realtime-model construction.

Both factories are built never to raise: a provider that cannot be constructed
degrades to OpenAI rather than killing the session. Every degradation path here
is reached through real behaviour -- a missing credential our own code checks
for, an uninstalled optional plugin -- not by patching the thing under test.
"""

from __future__ import annotations

import logging

import pytest

from src.domain import KwamiVoiceConfig
from src.factories import llm as llm_module
from src.factories import realtime as realtime_module
from src.factories.llm import _openai_temperature, create_llm
from src.factories.realtime import (
    DEFAULT_OPENAI_REALTIME_MODEL,
    create_realtime_model,
)


def voice(**overrides) -> KwamiVoiceConfig:
    return KwamiVoiceConfig(**overrides)


@pytest.fixture(autouse=True)
def keys(fake_key):
    for name in ("OPENAI_API_KEY", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        fake_key(name)


# =============================================================================
# _openai_temperature
# =============================================================================


def test_a_fixed_temperature_model_is_forced_to_one() -> None:
    """These models reject any temperature but the default with a 400."""
    assert _openai_temperature(voice(llm_temperature=0.2), "o1-preview") == 1.0


def test_matching_is_by_prefix_not_substring() -> None:
    """The bug this replaced: `"o1-" in model` forced temperature 1.0 on any
    model whose name merely contained it, such as llama-o1-8b."""
    assert _openai_temperature(voice(llm_temperature=0.2), "llama-o1-8b") == 0.2


def test_an_ordinary_model_keeps_its_temperature() -> None:
    assert _openai_temperature(voice(llm_temperature=0.3), "gpt-4o-mini") == 0.3


def test_no_model_keeps_the_configured_temperature() -> None:
    assert _openai_temperature(voice(llm_temperature=0.4), "") == 0.4


# =============================================================================
# create_llm
# =============================================================================


def test_openai_is_constructed() -> None:
    instance = create_llm(voice(llm_provider="openai", llm_model="gpt-4o-mini"))

    assert instance is not None


def test_a_prefixed_model_is_stripped() -> None:
    assert create_llm(voice(llm_provider="openai", llm_model="openai/gpt-4o-mini")) is not None


@pytest.mark.parametrize("provider", ["deepseek", "cerebras", "ollama"])
def test_openai_protocol_providers_construct(provider: str, fake_key) -> None:
    """These reach their endpoint through an openai.LLM helper rather than a
    dedicated plugin."""
    fake_key("DEEPSEEK_API_KEY", "CEREBRAS_API_KEY")

    assert create_llm(voice(llm_provider=provider)) is not None


def test_mistral_uses_its_own_key(fake_key) -> None:
    """This used to route through `with_x_ai`, which reached the right endpoint
    but read XAI_API_KEY -- so a deployment that correctly set MISTRAL_API_KEY
    failed with "XAI API key is required"."""
    fake_key("MISTRAL_API_KEY")

    assert create_llm(voice(llm_provider="mistral")) is not None


def test_mistral_without_its_key_falls_back(env_setting, caplog) -> None:
    """A real ValueError from our own check, through the catch-all, to OpenAI."""
    env_setting("MISTRAL_API_KEY", None)

    with caplog.at_level(logging.WARNING):
        instance = create_llm(voice(llm_provider="mistral"))

    assert instance is not None
    assert "MISTRAL_API_KEY is required" in caplog.text


@pytest.mark.parametrize("provider", ["google", "anthropic", "groq"])
def test_an_uninstalled_plugin_falls_back_to_openai(
    provider: str, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """These three need their own livekit-plugins package; without it the
    session must still get an LLM."""
    monkeypatch.setattr(llm_module, provider, None)

    with caplog.at_level(logging.WARNING):
        instance = create_llm(voice(llm_provider=provider))

    assert instance is not None
    assert "plugin not installed" in caplog.text


def test_no_provider_configured_falls_back(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        instance = create_llm(voice(llm_provider=""))

    assert instance is not None
    assert "no provider configured" in caplog.text


def test_an_unknown_provider_falls_back(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        instance = create_llm(voice(llm_provider="nonesuch"))

    assert instance is not None
    assert "plugin not installed" in caplog.text


# =============================================================================
# create_realtime_model
# =============================================================================


def test_openai_realtime_is_constructed() -> None:
    instance = create_realtime_model(voice(realtime_provider="openai"))

    assert instance is not None


def test_the_realtime_model_prefix_is_stripped() -> None:
    assert (
        create_realtime_model(
            voice(realtime_provider="openai", realtime_model="openai/gpt-realtime")
        )
        is not None
    )


def test_no_provider_defaults_to_openai() -> None:
    assert create_realtime_model(voice(realtime_provider="")) is not None


def test_an_unknown_realtime_provider_falls_back(caplog) -> None:
    """The log has to say the configured provider is not in use; a silent
    fallback reads as the setting having worked."""
    with caplog.at_level(logging.WARNING):
        instance = create_realtime_model(voice(realtime_provider="nonesuch"))

    assert instance is not None
    assert "NOT in use" in caplog.text
    assert DEFAULT_OPENAI_REALTIME_MODEL in caplog.text


def test_google_realtime_without_the_plugin_falls_back(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    monkeypatch.setattr(realtime_module, "google", None)

    with caplog.at_level(logging.ERROR):
        instance = create_realtime_model(voice(realtime_provider="google"))

    assert instance is not None
    assert "livekit-plugins-google" in caplog.text


def test_turn_detection_is_omitted_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older plugin builds have no TurnDetection; the provider's own default
    applies rather than the session failing to start."""
    monkeypatch.setattr(realtime_module, "TurnDetection", None)

    assert create_realtime_model(voice(realtime_provider="openai")) is not None


def test_a_construction_failure_falls_back_to_openai(caplog) -> None:
    """A realtime model that cannot be built must not end the call."""
    instance = create_realtime_model(
        voice(realtime_provider="openai", realtime_modalities=["not-a-modality"])
    )

    assert instance is not None


# =============================================================================
# Optional plugins
#
# google, anthropic and groq are extras and none is installed in the locked
# environment, so the lines that call their constructors cannot reach a real
# SDK. The stand-ins below have strict keyword signatures, so what is pinned is
# *our* call shape -- drop a kwarg or rename one and these fail. The real
# constructors are exercised only where the extra is installed.
# =============================================================================


class StrictLLMPlugin:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def LLM(self, *, model: str, temperature: float):  # noqa: N802
        self.calls.append({"model": model, "temperature": temperature})
        return f"llm:{model}"


@pytest.mark.parametrize(
    ("provider", "default_model"),
    [
        ("google", "gemini-2.0-flash"),
        ("anthropic", "claude-3-5-sonnet-latest"),
        ("groq", "llama-3.1-70b-versatile"),
    ],
)
def test_an_installed_plugin_receives_the_model_and_temperature(
    provider: str, default_model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin = StrictLLMPlugin()
    monkeypatch.setattr(llm_module, provider, plugin)

    result = create_llm(voice(llm_provider=provider, llm_model="", llm_temperature=0.25))

    assert result == f"llm:{default_model}"
    assert plugin.calls == [{"model": default_model, "temperature": 0.25}]


@pytest.mark.parametrize("provider", ["google", "anthropic", "groq"])
def test_an_explicit_model_reaches_an_installed_plugin(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin = StrictLLMPlugin()
    monkeypatch.setattr(llm_module, provider, plugin)

    create_llm(voice(llm_provider=provider, llm_model=f"{provider}/chosen-model"))

    assert plugin.calls[0]["model"] == "chosen-model"


class StrictGoogleRealtime:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.beta = self

    @property
    def realtime(self):
        return self

    def RealtimeModel(self, *, model: str, voice: str, temperature: float):  # noqa: N802
        self.calls.append({"model": model, "voice": voice, "temperature": temperature})
        return f"realtime:{model}"


def test_installed_google_realtime_receives_model_voice_and_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = StrictGoogleRealtime()
    monkeypatch.setattr(realtime_module, "google", plugin)

    result = create_realtime_model(
        voice(
            realtime_provider="google",
            realtime_model="",
            realtime_voice="Puck",
            llm_temperature=0.6,
        )
    )

    assert result == "realtime:gemini-2.0-flash-exp"
    assert plugin.calls == [{"model": "gemini-2.0-flash-exp", "voice": "Puck", "temperature": 0.6}]


def test_a_realtime_construction_failure_is_logged_before_the_fallback(env_setting, caplog) -> None:
    """The catch-all has one limit worth stating: its fallback is *also*
    OpenAI Realtime, so when the reason the provider failed is a missing
    OPENAI_API_KEY, the fallback cannot succeed either and the error reaches
    the caller. The log still names the provider first.
    """
    env_setting("OPENAI_API_KEY", None)

    with caplog.at_level(logging.ERROR), pytest.raises(Exception):
        create_realtime_model(voice(realtime_provider="openai"))

    assert "Failed to create openai realtime model" in caplog.text
