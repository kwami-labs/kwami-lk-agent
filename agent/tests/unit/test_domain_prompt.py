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
    """Guidance for tools this repo actually defines is always present."""
    prompt = build_system_prompt(soul())
    for expected in ("navigate_to", "product_search", "dismiss_search_result"):
        assert expected in prompt


def test_ui_control_guidance_is_withheld_when_the_tools_are_not_registered() -> None:
    """set_ui_control and list_ui_controls are client-side tools.

    Nothing in this repo defines them; they arrive only if the frontend
    registers them. Naming them unconditionally told the model to call tools
    that did not exist on any deployment that never sent them.
    """
    prompt = build_system_prompt(soul())

    assert "set_ui_control" not in prompt
    assert "list_ui_controls" not in prompt


def test_ui_control_guidance_appears_once_the_client_registers_them() -> None:
    prompt = build_system_prompt(soul(), client_tool_names={"set_ui_control", "list_ui_controls"})

    assert "set_ui_control" in prompt
    assert "list_ui_controls" in prompt
    # A worked example is the point of the block: the model has to see how a
    # spoken request maps onto domain/control/value, not just be told it can.
    assert "theme/mode/dark" in prompt


def test_one_registered_ui_tool_is_enough_to_emit_the_guidance() -> None:
    assert "set_ui_control" in build_system_prompt(soul(), client_tool_names=["set_ui_control"])


def test_unrelated_client_tools_do_not_trigger_ui_guidance() -> None:
    """A client with only its own custom tools gets no app guidance at all.

    `send_email` used to stand in for "unrelated" here. It is not unrelated any
    more -- it is one of the app's own email tools -- so this now uses names
    that really are outside every capability block.
    """
    prompt = build_system_prompt(soul(), client_tool_names=["book_flight", "order_taxi"])

    assert "set_ui_control" not in prompt
    assert "Controlling the app" not in prompt


def test_registering_only_the_email_tools_describes_only_email() -> None:
    """Guidance follows the registration, not an all-or-nothing switch."""
    prompt = build_system_prompt(soul(), client_tool_names=["read_emails", "send_email"])

    assert "read_emails" in prompt
    assert "set_ui_control" not in prompt, "described a UI router the client never registered"


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


# -- Language -----------------------------------------------------------------
#
# `soul.language` was parsed, documented in docs/protocol.md and settable from
# the app, and then read by nothing: grep for `.language` outside `stt_language`
# returned no hits at all. Retuning STT and TTS leaves the model hearing and
# speaking the new language while still writing English, so the directive below
# is what actually makes "switch to Spanish" work.


def test_a_non_default_language_produces_a_directive() -> None:
    from src.domain.prompt import language_directive

    directive = language_directive("es")

    assert "Spanish" in directive
    assert "es" not in directive.split("Spanish")[0], "the name, not the bare code"


def test_english_produces_no_directive() -> None:
    """It is the model's default, so saying it spends prompt for nothing."""
    from src.domain.prompt import language_directive

    assert language_directive("en") == ""


def test_an_absent_language_produces_no_directive() -> None:
    from src.domain.prompt import language_directive

    assert language_directive(None) == ""
    assert language_directive("") == ""
    assert language_directive("   ") == ""


def test_an_unrecognised_language_produces_no_directive() -> None:
    """Better silent than naming a language the model cannot place."""
    from src.domain.prompt import language_directive

    assert language_directive("xx") == ""
    assert language_directive("klingon") == ""


def test_a_regional_variant_falls_back_to_its_base_language() -> None:
    """Clients send `pt-BR` and `en-GB`; neither is in the table by itself."""
    from src.domain.prompt import language_directive

    assert "Portuguese" in language_directive("pt-BR")
    assert language_directive("en-GB") == "", "still English, still the default"


def test_the_directive_reaches_the_system_prompt() -> None:
    from src.domain import KwamiConfig, build_system_prompt

    config = KwamiConfig()
    config.soul.language = "ja"

    prompt = build_system_prompt(config.soul)

    assert "Japanese" in prompt


def test_a_default_language_soul_gets_no_language_line() -> None:
    from src.domain import KwamiConfig, build_system_prompt

    prompt = build_system_prompt(KwamiConfig().soul)

    assert "Speak and write in" not in prompt
