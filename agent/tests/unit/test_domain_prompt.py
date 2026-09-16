"""The system prompt is pure logic over the soul config.

It used to be ~195 lines inside `KwamiAgent`, rebuilt from roughly thirty
appends and four literal dicts on every call -- including on every live config
update, on the voice path. Extracting it is what makes it testable at all.
"""

from __future__ import annotations

import pytest

from src.domain import KwamiSoulConfig
from src.domain.prompt import (
    MAX_SYSTEM_MEMORY_CONTEXT_CHARS,
    MEMORY_HEADER,
    STATIC_GUIDANCE,
    build_system_prompt,
    describe_emotional_traits,
)


def soul(**kwargs) -> KwamiSoulConfig:
    return KwamiSoulConfig(**kwargs)


# -- opening line -----------------------------------------------------------


def test_name_and_personality_open_the_prompt() -> None:
    prompt = build_system_prompt(soul(name="Ada", personality="a curious guide"))
    assert prompt.startswith("You are Ada, a curious guide.")


def test_an_explicit_system_prompt_replaces_the_generated_opening() -> None:
    prompt = build_system_prompt(soul(name="Ada", system_prompt="You are a pirate."))
    assert prompt.startswith("You are a pirate.")
    assert "You are Ada" not in prompt


# -- soul sections ----------------------------------------------------------


def test_traits_and_style_are_included() -> None:
    prompt = build_system_prompt(soul(traits=["warm", "precise"], conversation_style="socratic"))
    assert "Key traits: warm, precise" in prompt
    assert "Conversation style: socratic" in prompt


def test_empty_traits_add_no_section() -> None:
    assert "Key traits:" not in build_system_prompt(soul(traits=[]))


@pytest.mark.parametrize(
    ("length", "expected"),
    [("short", "brief and concise"), ("medium", "balanced responses"), ("long", "comprehensive")],
)
def test_each_response_length_has_guidance(length: str, expected: str) -> None:
    assert expected in build_system_prompt(soul(response_length=length))


def test_an_unknown_response_length_is_ignored() -> None:
    prompt = build_system_prompt(soul(response_length="epic"))
    for phrase in ("brief and concise", "balanced responses", "comprehensive"):
        assert phrase not in prompt


@pytest.mark.parametrize(
    ("tone", "expected"),
    [
        ("warm", "warmth and friendliness"),
        ("playful", "light, playful voice"),
        ("serious", "no-fluff voice"),
        ("compassionate", "emotionally supportive"),
    ],
)
def test_each_emotional_tone_has_guidance(tone: str, expected: str) -> None:
    assert expected in build_system_prompt(soul(emotional_tone=tone))


def test_an_unknown_tone_is_ignored() -> None:
    assert "tone." not in build_system_prompt(soul(emotional_tone="smug"))


# -- emotional sliders ------------------------------------------------------


def test_no_traits_yields_no_profile() -> None:
    assert describe_emotional_traits(None) is None
    assert describe_emotional_traits({}) is None


def test_a_weak_slider_is_not_worth_a_directive() -> None:
    """Below the magnitude floor the instruction is noise."""
    assert describe_emotional_traits({"energy": 5}) is None


def test_a_positive_slider_uses_the_high_label() -> None:
    assert "more energetic" in describe_emotional_traits({"energy": 70})


def test_a_negative_slider_uses_the_low_label() -> None:
    assert "more low-energy" in describe_emotional_traits({"energy": -70})


@pytest.mark.parametrize(
    ("value", "strength"),
    [(30, "slightly"), (50, "moderately"), (70, "strongly"), (95, "very strongly")],
)
def test_magnitude_maps_to_strength(value: int, strength: str) -> None:
    # energy has weight 1.0, so the slider value is the magnitude.
    assert describe_emotional_traits({"energy": value}).count(strength) == 1


def test_unknown_keys_and_junk_values_are_skipped() -> None:
    """These come straight off the wire, so they must never raise."""
    assert describe_emotional_traits({"bogus": 90}) is None
    assert describe_emotional_traits({"energy": "loud"}) is None
    assert describe_emotional_traits({"energy": None}) is None


def test_directives_are_ordered_by_magnitude() -> None:
    profile = describe_emotional_traits({"energy": 20, "confidence": 80})
    assert profile.index("more confident") < profile.index("more energetic")


def test_at_most_five_directives_are_emitted() -> None:
    profile = describe_emotional_traits(
        {k: 90 for k in ("happiness", "energy", "confidence", "calmness", "optimism", "empathy")}
    )
    assert profile.count(",") == 4, "expected exactly 5 directives"


def test_weighting_changes_the_ordering() -> None:
    """empathy is weighted 1.35 and socialness 0.9, so equal sliders differ."""
    profile = describe_emotional_traits({"empathy": 50, "socialness": 50})
    assert profile.index("more empathic") < profile.index("more social")


def test_the_profile_reaches_the_prompt() -> None:
    prompt = build_system_prompt(soul(emotional_traits={"empathy": 80}))
    assert "Voice emotion profile:" in prompt


# -- static guidance --------------------------------------------------------


def test_static_guidance_is_built_once() -> None:
    """Hoisted to a module constant instead of ~30 appends per call."""
    assert isinstance(STATIC_GUIDANCE, str)
    assert build_system_prompt(soul()).count(STATIC_GUIDANCE) == 1


def test_tool_guidance_survives_extraction() -> None:
    prompt = build_system_prompt(soul())
    for expected in ("navigate_to", "product_search", "dismiss_search_result", "set_ui_control"):
        assert expected in prompt


# -- memory context ---------------------------------------------------------


def test_without_memory_there_is_no_memory_section() -> None:
    assert MEMORY_HEADER not in build_system_prompt(soul())


def test_memory_context_is_appended_under_a_header() -> None:
    prompt = build_system_prompt(soul(), "FACTS: lives in Barcelona")
    assert MEMORY_HEADER in prompt
    assert prompt.endswith("FACTS: lives in Barcelona")


def test_memory_context_is_capped() -> None:
    """Zep decides this length, not us; an unbounded block crowds out the soul."""
    prompt = build_system_prompt(soul(), "x" * (MAX_SYSTEM_MEMORY_CONTEXT_CHARS * 3))
    injected = prompt.split(MEMORY_HEADER, 1)[1].lstrip("\n")
    assert len(injected) == MAX_SYSTEM_MEMORY_CONTEXT_CHARS


def test_empty_memory_context_is_treated_as_absent() -> None:
    assert MEMORY_HEADER not in build_system_prompt(soul(), "")
