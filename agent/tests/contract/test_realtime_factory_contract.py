"""What `factories/realtime.py` assumes about livekit-plugins-openai.

The realtime pipeline was dead on arrival: it passed
`openai.realtime.ServerVadOptions`, a name that has never existed in the 1.3.x
plugin, so every `pipeline_type="realtime"` session raised AttributeError
before producing a frame. Against a MagicMock'd `livekit` that looked fine.
"""

from __future__ import annotations

import inspect

import pytest
from livekit.plugins import openai

from src.config import KwamiVoiceConfig
from src.factories.realtime import create_realtime_model


def test_server_vad_options_does_not_exist() -> None:
    """Pin the absence, so a future plugin release that adds it is noticed."""
    assert not hasattr(openai.realtime, "ServerVadOptions"), (
        "openai.realtime.ServerVadOptions now exists -- revisit the turn_detection "
        "construction in src/factories/realtime.py"
    )


def test_realtime_model_takes_the_kwargs_we_pass() -> None:
    params = inspect.signature(openai.realtime.RealtimeModel.__init__).parameters
    for kwarg in ("model", "voice", "temperature", "modalities", "turn_detection"):
        assert kwarg in params, f"RealtimeModel no longer accepts {kwarg!r}"


def test_turn_detection_type_has_the_fields_we_set() -> None:
    from openai.types.beta.realtime.session import TurnDetection

    for field in ("type", "threshold", "prefix_padding_ms", "silence_duration_ms"):
        assert field in TurnDetection.model_fields, f"TurnDetection lost {field!r}"


@pytest.mark.parametrize("provider", ["openai", "google", "unknown-provider", ""])
def test_every_provider_branch_constructs(provider: str, fake_key) -> None:
    """No branch may raise: a failure here is a silent, voiceless session."""
    fake_key("OPENAI_API_KEY", "GOOGLE_API_KEY")
    model = create_realtime_model(KwamiVoiceConfig(realtime_provider=provider))
    assert model is not None


def test_openai_branch_actually_builds_turn_detection(fake_key) -> None:
    """The regression: this call raised AttributeError for every session."""
    fake_key("OPENAI_API_KEY")
    config = KwamiVoiceConfig(
        realtime_provider="openai",
        vad_threshold=0.6,
        vad_min_silence_duration=0.8,
    )
    model = create_realtime_model(config)
    assert type(model).__name__ == "RealtimeModel"


def test_unknown_provider_is_logged_not_silent(fake_key, caplog) -> None:
    """A substituted provider must be visible in the logs."""
    fake_key("OPENAI_API_KEY")
    with caplog.at_level("WARNING"):
        create_realtime_model(KwamiVoiceConfig(realtime_provider="totally-made-up"))
    assert any("totally-made-up" in r.getMessage() for r in caplog.records), (
        "silent provider substitution"
    )
