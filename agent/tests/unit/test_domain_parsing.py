"""Wire parsing must preserve deliberate zeros and survive null sections."""

from __future__ import annotations

import pytest

from src.domain.parsing import boolean, integer, number, section, text, value_from_keys

# -- value_from_keys --------------------------------------------------------


def test_the_first_present_key_wins() -> None:
    assert value_from_keys({"systemPrompt": "a", "system_prompt": "b"}, "systemPrompt") == "a"
    assert value_from_keys({"system_prompt": "b"}, "systemPrompt", "system_prompt") == "b"


def test_falsy_values_are_returned_not_skipped() -> None:
    """The whole point: 0 and False are values, not absences."""
    assert value_from_keys({"speed": 0}, "speed") == 0
    assert value_from_keys({"enabled": False}, "enabled") is False
    assert value_from_keys({"name": ""}, "name") == ""


def test_none_is_treated_as_absent_so_the_next_key_is_tried() -> None:
    assert value_from_keys({"a": None, "b": 3}, "a", "b") == 3


def test_a_non_dict_yields_none() -> None:
    for junk in (None, [], "text", 7):
        assert value_from_keys(junk, "anything") is None


# -- section ----------------------------------------------------------------


def test_a_missing_section_is_an_empty_dict() -> None:
    assert section({}, "voice") == {}


def test_a_null_section_is_an_empty_dict() -> None:
    """`"voice": null` used to raise AttributeError and drop the whole config."""
    assert section({"voice": None}, "voice") == {}


def test_a_wrong_typed_section_is_an_empty_dict() -> None:
    assert section({"voice": "loud"}, "voice") == {}
    assert section({"voice": []}, "voice") == {}


def test_sections_nest() -> None:
    assert section({"voice": {"tts": {"speed": 1}}}, "voice", "tts") == {"speed": 1}


def test_a_null_halfway_down_still_yields_a_dict() -> None:
    assert section({"voice": {"tts": None}}, "voice", "tts") == {}


# -- number -----------------------------------------------------------------


def test_zero_is_a_real_number() -> None:
    """`temperature=0.0` and `speed=0` were silently ignored."""
    assert number({"temperature": 0}, "temperature") == 0.0
    assert number({"speed": 0.0}, "speed") == 0.0


def test_numbers_are_floats() -> None:
    assert number({"t": 1}, "t") == 1.0
    assert number({"t": "0.7"}, "t") == 0.7


def test_unparseable_and_absent_numbers_are_none() -> None:
    assert number({}, "t") is None
    assert number({"t": "warm"}, "t") is None
    assert number({"t": None}, "t") is None


def test_a_bool_is_never_a_number() -> None:
    """`True` is an int in Python; it is not a temperature."""
    assert number({"t": True}, "t") is None


# -- integer ----------------------------------------------------------------


def test_zero_max_tokens_is_preserved() -> None:
    assert integer({"maxTokens": 0}, "maxTokens") == 0


def test_integers_truncate_floats() -> None:
    assert integer({"n": 4.9}, "n") == 4


def test_absent_integer_is_none() -> None:
    assert integer({}, "n") is None


# -- text -------------------------------------------------------------------


def test_text_returns_a_stripped_string() -> None:
    assert text({"provider": "  openai "}, "provider") == "openai"


def test_blank_text_counts_as_unset() -> None:
    """An empty provider name is never a deliberate choice."""
    assert text({"provider": ""}, "provider") is None
    assert text({"provider": "   "}, "provider") is None


def test_non_string_text_is_none() -> None:
    assert text({"provider": 7}, "provider") is None


# -- boolean ----------------------------------------------------------------


def test_false_is_distinguishable_from_absent() -> None:
    assert boolean({"enabled": False}, "enabled") is False
    assert boolean({}, "enabled") is None


@pytest.mark.parametrize("raw", ["true", "TRUE", " yes ", "on", "1"])
def test_truthy_spellings(raw: str) -> None:
    assert boolean({"enabled": raw}, "enabled") is True


@pytest.mark.parametrize("raw", ["false", "no", "off", "0"])
def test_falsy_spellings(raw: str) -> None:
    """A client sending the string "false" must not enable a feature."""
    assert boolean({"enabled": raw}, "enabled") is False


def test_unrecognised_boolean_is_none() -> None:
    assert boolean({"enabled": "maybe"}, "enabled") is None
