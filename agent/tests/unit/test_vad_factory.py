"""VAD settings must actually reach the model.

`create_vad` had no callers at all: prewarm loaded Silero with *Silero's*
defaults, which are not this project's, so the configured turn-taking was never
in effect -- not even at default settings.
"""

from __future__ import annotations

import pytest

from src.domain import KwamiVoiceConfig
from src.factories import vad as vad_factory


class FakeVAD:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


@pytest.fixture
def loads(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def _load(**kwargs):
        calls.append(kwargs)
        return FakeVAD(**kwargs)

    monkeypatch.setattr(vad_factory.silero.VAD, "load", staticmethod(_load))
    return calls


def test_project_defaults_differ_from_silero_defaults() -> None:
    """The reason this bug was invisible: both were called "defaults".

    Silero ships min_silence_duration=0.55; this project documents 0.3.
    """
    threshold, min_speech, min_silence = vad_factory.DEFAULT_VAD_SETTINGS
    assert (threshold, min_speech, min_silence) == (0.5, 0.1, 0.3)
    assert min_silence != 0.55


def test_prewarm_loads_with_project_defaults(loads: list[dict]) -> None:
    vad_factory.prewarm_vad()

    assert loads == [
        {"activation_threshold": 0.5, "min_speech_duration": 0.1, "min_silence_duration": 0.3}
    ]


def test_a_default_config_reuses_the_prewarmed_model(loads: list[dict]) -> None:
    """Prewarming exists to keep model loading off the per-job path."""
    prewarmed = object()

    result = vad_factory.create_vad(KwamiVoiceConfig(), prewarmed=prewarmed)

    assert result is prewarmed
    assert loads == [], "a default session should not pay to load its own VAD"


def test_a_custom_config_gets_its_own_model(loads: list[dict]) -> None:
    prewarmed = object()
    config = KwamiVoiceConfig(vad_min_silence_duration=0.9)

    result = vad_factory.create_vad(config, prewarmed=prewarmed)

    assert result is not prewarmed
    assert loads[0]["min_silence_duration"] == 0.9


@pytest.mark.parametrize(
    "field,value",
    [
        ("vad_threshold", 0.8),
        ("vad_min_speech_duration", 0.25),
        ("vad_min_silence_duration", 0.7),
    ],
)
def test_every_vad_setting_reaches_the_model(loads: list[dict], field: str, value: float) -> None:
    """Each of these was documented in config and read by nothing."""
    vad_factory.create_vad(KwamiVoiceConfig(**{field: value}), prewarmed=object())

    assert value in loads[0].values()


def test_without_a_prewarmed_model_one_is_always_loaded(loads: list[dict]) -> None:
    assert vad_factory.create_vad(KwamiVoiceConfig()) is not None
    assert len(loads) == 1


def test_a_load_failure_falls_back_rather_than_killing_the_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turn-taking tuning must never be able to take a call down."""

    def _boom(**kwargs):
        raise RuntimeError("model file missing")

    monkeypatch.setattr(vad_factory.silero.VAD, "load", staticmethod(_boom))
    prewarmed = object()

    assert (
        vad_factory.create_vad(KwamiVoiceConfig(vad_threshold=0.9), prewarmed=prewarmed)
        is prewarmed
    )
