"""`validate_tool_definition` is the gate in front of the LLM's tool list.

A malformed client tool that gets through 400s every subsequent LLM request and
bricks the session, so every rejection branch here is load-bearing.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.utils.validation import (
    CAMEL_TO_SNAKE_MAP,
    normalize_config_keys,
    validate_tool_definition,
)


def test_a_flat_definition_with_a_name_is_valid() -> None:
    assert validate_tool_definition({"name": "set_theme"}) is True


def test_the_nested_function_format_is_unwrapped() -> None:
    """OpenAI-style `{"function": {...}}` is the other shape clients send."""
    assert validate_tool_definition({"function": {"name": "set_theme"}}) is True


def test_a_nested_definition_is_judged_by_its_inner_fields() -> None:
    assert validate_tool_definition({"function": {"description": "no name"}}) is False


@pytest.mark.parametrize(
    "tool_def",
    [
        pytest.param({}, id="no name key"),
        pytest.param({"name": ""}, id="empty name"),
        pytest.param({"name": None}, id="null name"),
    ],
)
def test_a_missing_or_empty_name_is_rejected(tool_def: dict[str, Any]) -> None:
    assert validate_tool_definition(tool_def) is False


def test_a_non_string_name_is_rejected() -> None:
    """Truthy but not a string: passes the `if not name` guard, fails the type one."""
    assert validate_tool_definition({"name": 123}) is False


def test_a_non_string_description_is_rejected() -> None:
    assert validate_tool_definition({"name": "ok", "description": {"not": "a string"}}) is False


def test_a_missing_description_is_fine() -> None:
    """Description is optional; only a present-and-wrong one is a defect."""
    assert validate_tool_definition({"name": "ok"}) is True


def test_a_falsy_description_skips_the_type_check() -> None:
    """`if description and ...` -- an empty string is allowed through."""
    assert validate_tool_definition({"name": "ok", "description": ""}) is True


def test_non_dict_parameters_are_rejected() -> None:
    assert validate_tool_definition({"name": "ok", "parameters": ["not", "a", "dict"]}) is False


def test_a_falsy_parameters_value_skips_the_type_check() -> None:
    assert validate_tool_definition({"name": "ok", "parameters": {}}) is True


def test_a_fully_populated_definition_is_valid() -> None:
    assert (
        validate_tool_definition(
            {
                "name": "set_theme",
                "description": "Switch the UI theme",
                "parameters": {"type": "object", "properties": {}},
            }
        )
        is True
    )


# =============================================================================
# normalize_config_keys
# =============================================================================


def test_known_camel_case_keys_become_snake_case() -> None:
    assert normalize_config_keys({"ttsProvider": "cartesia"}) == {"tts_provider": "cartesia"}


def test_unknown_keys_are_passed_through_untouched() -> None:
    """The map is a translation table, not an allow-list."""
    assert normalize_config_keys({"somethingElse": 1}) == {"somethingElse": 1}


def test_nested_dicts_are_normalized_recursively() -> None:
    normalized = normalize_config_keys({"voice": {"ttsVoice": "aura", "ttsSpeed": 1.1}})

    assert normalized == {"voice": {"tts_voice": "aura", "tts_speed": 1.1}}


def test_normalization_does_not_mutate_the_input() -> None:
    original = {"ttsProvider": "cartesia", "voice": {"ttsVoice": "aura"}}

    normalize_config_keys(original)

    assert original == {"ttsProvider": "cartesia", "voice": {"ttsVoice": "aura"}}


def test_an_empty_config_normalizes_to_an_empty_dict() -> None:
    assert normalize_config_keys({}) == {}


def test_non_dict_values_survive_normalization() -> None:
    """Lists are not recursed into, only dicts -- the loop's `isinstance` branch."""
    assert normalize_config_keys({"emotionalTraits": ["warm", "dry"]}) == {
        "emotional_traits": ["warm", "dry"]
    }


@pytest.mark.parametrize(("camel", "snake"), sorted(CAMEL_TO_SNAKE_MAP.items()))
def test_every_documented_alias_translates(camel: str, snake: str) -> None:
    """The map is what the frontend's wire format relies on; pin each entry."""
    assert normalize_config_keys({camel: "value"}) == {snake: "value"}
