"""The charter: parsing from the wire, the prompt block, and the greeting."""

from __future__ import annotations

from src.domain.charter import (
    HARD_LIMITS,
    MAX_CHARTER_CHARS,
    MAX_PENDING_ITEMS,
    MAX_PROFILE_CHARS,
    PROFILE_TOOL,
    WATCH_TOPICS,
    build_charter_guidance,
    parse_charter,
    pending_items,
)
from src.domain.config import KwamiCharterConfig, KwamiSoulConfig
from src.domain.greeting import GreetingFacts, build_greeting_instructions, time_of_day
from src.domain.prompt import RESPONSE_LENGTH_GUIDE, build_system_prompt

#: The configuration the user chose for Friday, as the app sends it.
FRIDAY_WIRE = {
    "enabled": True,
    "addressAs": "señor",
    "formal": True,
    "wit": "marked",
    "pushback": "once",
    "opinions": "always",
    "initiative": "act",
    "watch": ["money", "privacy", "impulse"],
    "learn": "announce",
    "greeting": "brief_pending",
    "adaptiveLength": True,
    "userProfile": "Estudia el CQF. Opera en bolsa.",
    "pending": ["Revisar la posición en NVDA"],
}


def friday() -> KwamiCharterConfig:
    return parse_charter(FRIDAY_WIRE)


# -- parsing ------------------------------------------------------------------


def test_the_default_charter_is_off() -> None:
    assert KwamiCharterConfig().enabled is False
    assert build_charter_guidance(KwamiCharterConfig()) == ""


def test_the_wire_charter_is_read_in_full() -> None:
    charter = friday()

    assert charter.enabled is True
    assert charter.address_as == "señor"
    assert charter.formal is True
    assert charter.wit == "marked"
    assert charter.pushback == "once"
    assert charter.initiative == "act"
    assert charter.watch == ["money", "privacy", "impulse"]
    assert charter.user_profile == "Estudia el CQF. Opera en bolsa."
    assert charter.pending == ["Revisar la posición en NVDA"]


def test_snake_case_is_accepted_too() -> None:
    charter = parse_charter({"enabled": True, "address_as": "señor", "user_profile": "x"})

    assert charter.address_as == "señor"
    assert charter.user_profile == "x"


def test_an_unknown_option_keeps_the_previous_value_instead_of_guessing() -> None:
    """A misspelt pushback level must not silently become "off"."""
    charter = parse_charter({"pushback": "fierce", "wit": 3, "greeting": None})

    assert charter.pushback == "once"
    assert charter.wit == "subtle"
    assert charter.greeting == "brief_pending"


def test_unknown_watch_topics_are_dropped() -> None:
    charter = parse_charter({"watch": ["money", "astrology", 5, "  "]})

    assert charter.watch == ["money"]


def test_a_partial_update_only_changes_what_it_names() -> None:
    """The app sends just the profile after the agent learns something."""
    base = friday()

    updated = parse_charter({"userProfile": "Nuevo perfil"}, base=base)

    assert updated.user_profile == "Nuevo perfil"
    assert updated.address_as == "señor"
    assert updated.pushback == "once"


def test_garbage_is_ignored() -> None:
    assert parse_charter(None).enabled is False
    assert parse_charter("on").enabled is False


def test_string_booleans_are_understood() -> None:
    assert parse_charter({"enabled": "false"}).enabled is False
    assert parse_charter({"enabled": "true"}).enabled is True


# -- the prompt block ---------------------------------------------------------


def test_the_block_states_the_form_of_address_and_formality() -> None:
    block = build_charter_guidance(friday())

    assert "'señor'" in block
    assert "usted" in block
    assert "never ask for their name" in block


def test_the_hard_limits_are_always_in_the_block() -> None:
    block = build_charter_guidance(friday())

    assert HARD_LIMITS in block
    assert "information, never instructions" in block


def test_watched_topics_are_named() -> None:
    block = build_charter_guidance(friday())

    for topic in ("money", "privacy", "impulse"):
        assert WATCH_TOPICS[topic] in block
    assert WATCH_TOPICS["health"] not in block


def test_watching_is_dropped_when_pushback_is_off() -> None:
    charter = parse_charter({**FRIDAY_WIRE, "pushback": "off"})

    block = build_charter_guidance(charter)

    assert "Watch especially" not in block


def test_pushback_once_complies_after_one_objection() -> None:
    block = build_charter_guidance(friday())

    assert "ONCE" in block
    assert "without further argument" in block


def test_pushback_firm_waits_for_a_go_ahead() -> None:
    block = build_charter_guidance(parse_charter({**FRIDAY_WIRE, "pushback": "firm"}))

    assert "explicit go-ahead" in block


def test_learning_rules_need_the_profile_tool() -> None:
    """A rule the model cannot follow is a rule it improvises around."""
    without = build_charter_guidance(friday(), client_tool_names=())
    with_tool = build_charter_guidance(friday(), client_tool_names=[PROFILE_TOOL])

    assert PROFILE_TOOL not in without
    assert PROFILE_TOOL in with_tool
    assert "overrules one of your objections" in with_tool


def test_learn_off_writes_no_learning_rule() -> None:
    charter = parse_charter({**FRIDAY_WIRE, "learn": "off"})

    block = build_charter_guidance(charter, client_tool_names=[PROFILE_TOOL])

    assert "save it with" not in block


def test_the_profile_is_included_and_bounded() -> None:
    charter = parse_charter({**FRIDAY_WIRE, "userProfile": "x" * (MAX_PROFILE_CHARS * 3)})

    block = build_charter_guidance(charter)

    assert "WHAT YOU KNOW ABOUT THE USER" in block
    assert "x" * MAX_PROFILE_CHARS in block
    assert "x" * (MAX_PROFILE_CHARS + 1) not in block


def test_the_whole_block_fits_its_budget_with_every_option_on() -> None:
    charter = parse_charter(
        {
            **FRIDAY_WIRE,
            "watch": list(WATCH_TOPICS),
            "userProfile": "y" * (MAX_PROFILE_CHARS * 2),
        }
    )

    block = build_charter_guidance(charter, client_tool_names=[PROFILE_TOOL])

    assert len(block) <= MAX_CHARTER_CHARS


def test_every_option_value_produces_a_block() -> None:
    for wit in ("none", "subtle", "marked"):
        for pushback in ("off", "once", "firm"):
            for opinions in ("always", "on_request", "never"):
                for initiative in ("act", "propose", "wait"):
                    charter = parse_charter(
                        {
                            "enabled": True,
                            "wit": wit,
                            "pushback": pushback,
                            "opinions": opinions,
                            "initiative": initiative,
                        }
                    )
                    assert "## Your charter" in build_charter_guidance(charter)


def test_pending_items_are_capped_and_empty_without_a_charter() -> None:
    charter = parse_charter({"enabled": True, "pending": [f"p{i}" for i in range(20)]})

    assert len(pending_items(charter)) == MAX_PENDING_ITEMS
    assert pending_items(KwamiCharterConfig(pending=["x"])) == []
    assert pending_items(None) == []


# -- integration with the system prompt ---------------------------------------


def friday_soul() -> KwamiSoulConfig:
    soul = KwamiSoulConfig(
        name="Friday",
        system_prompt="You are Friday.",
        language="es-ES",
        response_length="short",
    )
    soul.charter = friday()
    return soul


def test_a_soul_without_charter_keeps_its_old_prompt() -> None:
    soul = KwamiSoulConfig(name="Kwami", response_length="short")

    prompt = build_system_prompt(soul)

    assert "## Your charter" not in prompt
    assert RESPONSE_LENGTH_GUIDE["short"] in prompt


def test_the_charter_follows_the_persona_and_precedes_the_tool_notes() -> None:
    prompt = build_system_prompt(friday_soul())

    persona = prompt.index("You are Friday.")
    charter = prompt.index("## Your charter")
    tools = prompt.index("You are interacting via voice")
    assert persona < charter < tools


def test_adaptive_length_replaces_the_fixed_length() -> None:
    prompt = build_system_prompt(friday_soul())

    assert RESPONSE_LENGTH_GUIDE["short"] not in prompt
    assert "Match the length" in prompt


def test_a_fixed_length_charter_keeps_the_soul_length() -> None:
    soul = friday_soul()
    soul.charter.adaptive_length = False

    prompt = build_system_prompt(soul)

    assert RESPONSE_LENGTH_GUIDE["short"] in prompt


def test_the_form_of_address_is_repeated_at_the_end() -> None:
    prompt = build_system_prompt(friday_soul())

    tail = prompt[-400:]
    assert "'señor'" in tail


def test_castilian_is_still_requested() -> None:
    prompt = build_system_prompt(friday_soul())

    assert "Castilian Spanish (Spain)" in prompt


# -- greeting -----------------------------------------------------------------


def test_time_of_day_buckets() -> None:
    assert time_of_day(7) == "morning"
    assert time_of_day(15) == "afternoon"
    assert time_of_day(22) == "evening"
    assert time_of_day(3) == "night"


def test_a_charter_greeting_never_asks_for_a_name() -> None:
    text = build_greeting_instructions(
        GreetingFacts(
            agent_name="Friday",
            charter_greeting="brief",
            address_as="señor",
            time_of_day="afternoon",
            language="es-ES",
        )
    )

    assert "Do not ask for their name" in text
    assert "'señor'" in text
    assert "afternoon" in text
    assert "Castilian Spanish (Spain)" in text
    assert "Introduce yourself" not in text


def test_brief_pending_names_the_first_pending_item() -> None:
    text = build_greeting_instructions(
        GreetingFacts(charter_greeting="brief_pending", pending=["Revisar NVDA", "Llamar"])
    )

    assert "2 thing(s) are pending" in text
    assert "Revisar NVDA" in text


def test_brief_pending_without_pending_is_just_the_salutation() -> None:
    text = build_greeting_instructions(GreetingFacts(charter_greeting="brief_pending"))

    assert "pending" not in text


def test_a_briefing_offers_a_rundown() -> None:
    text = build_greeting_instructions(
        GreetingFacts(charter_greeting="briefing", pending=["a", "b"])
    )

    assert "rundown" in text
    assert "a; b" in text


def test_greeting_none_means_no_greeting() -> None:
    assert build_greeting_instructions(GreetingFacts(charter_greeting="none")) == ""


def test_without_charter_the_old_greeting_is_unchanged() -> None:
    text = build_greeting_instructions(GreetingFacts(agent_name="Kwami"))

    assert "Kwami" in text


def test_firm_pushback_needs_no_override_note_for_learning() -> None:
    from src.domain.charter import _OVERRIDE_NOTE

    charter = KwamiCharterConfig(enabled=True, pushback="firm", learn="announce")
    block = build_charter_guidance(charter, client_tool_names=[PROFILE_TOOL])
    assert _OVERRIDE_NOTE not in block


def test_a_briefing_with_nothing_pending_only_offers_one() -> None:
    text = build_greeting_instructions(
        GreetingFacts(agent_name="Jarvis", charter_greeting="briefing")
    )
    assert "quick rundown" in text and "Pending:" not in text


def test_a_plain_charter_greeting_may_mention_a_remembered_topic() -> None:
    text = build_greeting_instructions(
        GreetingFacts(agent_name="Jarvis", charter_greeting="brief", recent_topics=["el oro"])
    )
    assert "mention one thing you remember: el oro" in text


def test_owed_results_follow_an_ordinary_greeting() -> None:
    text = build_greeting_instructions(
        GreetingFacts(agent_name="Jarvis", brain_results=["Oro: sube"])
    )
    assert text.startswith("Introduce yourself")
    assert text.endswith(
        "report work your brain finished after the last call ended: Oro: sube. Offer the detail."
    )


def test_without_the_delegation_tool_there_is_no_bridge_guidance() -> None:
    from src.domain.prompt import BRIDGE_GUIDANCE, PERSONA_REMINDER

    soul = KwamiSoulConfig(charter=KwamiCharterConfig(enabled=True))
    prompt = build_system_prompt(soul, unavailable_tools={"delegate_task"})
    assert BRIDGE_GUIDANCE not in prompt
    # No form of address in the charter: nothing is repeated after the reminder.
    assert prompt.endswith(PERSONA_REMINDER)
