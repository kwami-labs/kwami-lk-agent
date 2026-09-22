"""Provider detection decides which plugin gets constructed on a voice swap.

Guessing wrong means the session rebuilds its TTS against a provider whose
credentials are absent, so each heuristic is pinned here -- including the
boundaries, which is where the original off-by-length bugs lived.
"""

from __future__ import annotations

import pytest

from src.utils.provider import (
    KNOWN_PROVIDERS,
    OPENAI_VOICES,
    detect_provider_change,
    detect_tts_provider_from_model,
    detect_tts_provider_from_voice,
    strip_model_prefix,
)

# =============================================================================
# strip_model_prefix
# =============================================================================


def test_a_matching_prefix_is_removed() -> None:
    assert strip_model_prefix("openai/gpt-4.1-mini", "openai") == "gpt-4.1-mini"


def test_a_bare_model_is_returned_unchanged() -> None:
    assert strip_model_prefix("gpt-4.1-mini", "openai") == "gpt-4.1-mini"


def test_a_prefix_for_a_different_provider_is_left_alone() -> None:
    """Stripping "openai/" off a Cartesia model would produce a model that
    exists for neither."""
    assert strip_model_prefix("cartesia/sonic-2", "openai") == "cartesia/sonic-2"


@pytest.mark.parametrize("model", ["", None])
def test_a_falsy_model_becomes_the_empty_string(model: str | None) -> None:
    assert strip_model_prefix(model, "openai") == ""  # type: ignore[arg-type]


def test_only_the_leading_prefix_is_stripped() -> None:
    """A slash later in the name is part of the model id, not a prefix."""
    assert strip_model_prefix("openai/some/nested", "openai") == "some/nested"


def test_a_prefix_appearing_mid_string_is_not_stripped() -> None:
    assert strip_model_prefix("x-openai/gpt", "openai") == "x-openai/gpt"


@pytest.mark.parametrize("provider", KNOWN_PROVIDERS)
def test_every_known_provider_prefix_round_trips(provider: str) -> None:
    assert strip_model_prefix(f"{provider}/some-model", provider) == "some-model"


# =============================================================================
# detect_tts_provider_from_model
# =============================================================================


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        pytest.param("elevenlabs/eleven-flash-v2.5", "elevenlabs", id="prefix elevenlabs"),
        pytest.param("openai/tts-1", "openai", id="prefix openai"),
        pytest.param("cartesia/sonic-2", "cartesia", id="prefix cartesia"),
        pytest.param("deepgram/aura-2", "deepgram", id="prefix deepgram"),
        pytest.param("google/chirp", "google", id="prefix google"),
        pytest.param("rime/arcana", "rime", id="prefix rime"),
    ],
)
def test_an_explicit_prefix_wins(model: str, expected: str) -> None:
    assert detect_tts_provider_from_model(model) == expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        pytest.param("eleven_flash_v2_5", "elevenlabs", id="eleven_ underscore"),
        pytest.param("eleven-flash", "elevenlabs", id="eleven- hyphen"),
        pytest.param("tts-1-hd", "openai", id="tts-"),
        pytest.param("gpt-4o-mini-tts", "openai", id="gpt-4o"),
        pytest.param("sonic-2", "cartesia", id="sonic"),
        pytest.param("aura-asteria-en", "deepgram", id="aura"),
        pytest.param("arcana", "rime", id="arcana"),
        pytest.param("mistv2", "rime", id="mistv"),
    ],
)
def test_bare_model_patterns_are_recognised(model: str, expected: str) -> None:
    assert detect_tts_provider_from_model(model) == expected


def test_detection_is_case_insensitive() -> None:
    assert detect_tts_provider_from_model("ELEVEN_FLASH") == "elevenlabs"
    assert detect_tts_provider_from_model("OpenAI/TTS-1") == "openai"


@pytest.mark.parametrize("model", ["", None])
def test_a_falsy_model_detects_nothing(model: str | None) -> None:
    assert detect_tts_provider_from_model(model) is None  # type: ignore[arg-type]


def test_an_unrecognised_model_detects_nothing() -> None:
    """Returning None means "keep the current provider", which is the safe
    default -- inventing one would rebuild the pipeline for no reason."""
    assert detect_tts_provider_from_model("some-unknown-model") is None


# =============================================================================
# detect_tts_provider_from_voice
# =============================================================================


def test_a_long_alphanumeric_id_is_elevenlabs() -> None:
    assert detect_tts_provider_from_voice("JBFqnCBsd6RMkjVDRZzb") == "elevenlabs"


def test_a_uuid_is_cartesia() -> None:
    assert detect_tts_provider_from_voice("79a125e8-cd45-4c13-8a67-188112f4dd22") == "cartesia"


@pytest.mark.parametrize("voice", sorted(OPENAI_VOICES))
def test_every_openai_voice_name_is_recognised(voice: str) -> None:
    assert detect_tts_provider_from_voice(voice) == "openai"


def test_an_openai_voice_is_matched_case_insensitively() -> None:
    assert detect_tts_provider_from_voice("Nova") == "openai"


def test_a_nineteen_character_id_is_not_elevenlabs() -> None:
    """The >=20 boundary: one short must not claim ElevenLabs."""
    assert len("a" * 19) == 19
    assert detect_tts_provider_from_voice("a" * 19) is None


def test_a_twenty_character_id_is_elevenlabs() -> None:
    assert detect_tts_provider_from_voice("a" * 20) == "elevenlabs"


def test_a_long_id_with_punctuation_is_not_elevenlabs() -> None:
    """`isalnum()` is the second half of the guard, and a UUID is exactly the
    long-but-not-alphanumeric case it exists to let through."""
    assert detect_tts_provider_from_voice("a" * 20 + "-b") is None


def test_a_hyphenated_string_of_the_wrong_length_is_not_cartesia() -> None:
    assert detect_tts_provider_from_voice("a-b-c-d") is None


def test_a_thirty_six_character_string_with_too_few_hyphens_is_not_cartesia() -> None:
    assert detect_tts_provider_from_voice("x" * 33 + "-y-") is None


@pytest.mark.parametrize("voice", ["", None])
def test_a_falsy_voice_detects_nothing(voice: str | None) -> None:
    assert detect_tts_provider_from_voice(voice) is None  # type: ignore[arg-type]


def test_an_unrecognised_voice_detects_nothing() -> None:
    assert detect_tts_provider_from_voice("bob") is None


# =============================================================================
# detect_provider_change
# =============================================================================


def test_no_model_and_no_voice_is_no_change() -> None:
    assert detect_provider_change("openai") == ("openai", False)


def test_a_model_from_another_provider_is_a_change() -> None:
    assert detect_provider_change("openai", new_model="sonic-2") == ("cartesia", True)


def test_a_model_from_the_same_provider_is_not_a_change() -> None:
    assert detect_provider_change("openai", new_model="tts-1") == ("openai", False)


def test_the_voice_is_consulted_when_the_model_says_nothing() -> None:
    assert detect_provider_change("openai", new_model="mystery", new_voice="a" * 20) == (
        "elevenlabs",
        True,
    )


def test_the_voice_is_consulted_when_no_model_is_given() -> None:
    assert detect_provider_change("openai", new_voice="a" * 20) == ("elevenlabs", True)


def test_the_model_wins_over_the_voice() -> None:
    """Model detection is higher confidence; once it has moved the provider the
    voice is not allowed to move it again."""
    detected, changed = detect_provider_change("openai", new_model="sonic-2", new_voice="a" * 20)

    assert (detected, changed) == ("cartesia", True)


def test_an_unrecognised_voice_leaves_the_provider_alone() -> None:
    assert detect_provider_change("openai", new_voice="bob") == ("openai", False)


def test_a_voice_matching_the_current_provider_is_not_a_change() -> None:
    assert detect_provider_change("openai", new_voice="nova") == ("openai", False)
