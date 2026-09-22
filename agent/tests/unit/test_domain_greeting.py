"""The opening turn: what the agent is told to say, given what it remembers.

This lived inside `KwamiAgent._build_greeting_instructions` as 110 lines mixing
memory I/O with wording, so the branch that decides how a returning user is
greeted could not be reached without standing up a memory double. It is pure
now, and these call it with plain data.

The greeting is the first thing every user hears, and it is the one place the
agent's memory is visible as a product feature rather than an implementation
detail -- so the distinctions below (named vs returning vs stranger, topics vs
summary) are the feature, not incidental branching.
"""

from __future__ import annotations

from src.domain.greeting import (
    MAX_GREETING_TOPICS,
    GreetingFacts,
    build_greeting_instructions,
    extract_name_from_facts,
    topics_from_facts,
)

# -- A stranger ---------------------------------------------------------------


def test_a_new_user_gets_an_introduction() -> None:
    instruction = build_greeting_instructions(GreetingFacts(agent_name="Ada"))

    assert "Introduce yourself" in instruction
    assert "Ada" in instruction


def test_an_agent_with_no_name_still_has_one() -> None:
    """`soul.name or kwami_name or "Kwami"` can still end up empty."""
    instruction = build_greeting_instructions(GreetingFacts(agent_name=""))

    assert "Kwami" in instruction


# -- Met before, name unknown -------------------------------------------------


def test_a_returning_user_without_a_name_is_asked_for_one() -> None:
    instruction = build_greeting_instructions(
        GreetingFacts(agent_name="Ada", is_returning_user=True)
    )

    assert "can't remember their name" in instruction
    assert "what's your name?" in instruction


# -- Named ---------------------------------------------------------------------


def test_a_named_user_is_greeted_by_name() -> None:
    instruction = build_greeting_instructions(GreetingFacts(user_name="Marie"))

    assert "Marie" in instruction
    assert "by name" in instruction


def test_remembered_topics_are_offered_to_the_model() -> None:
    instruction = build_greeting_instructions(
        GreetingFacts(user_name="Marie", recent_topics=["is learning to sail"])
    )

    assert "is learning to sail" in instruction
    assert "Pick ONE topic" in instruction


def test_only_a_few_topics_are_offered() -> None:
    """More than a handful invites the model to recite everything it knows,
    which reads as surveillance rather than memory."""
    topics = [f"topic number {i}" for i in range(10)]

    instruction = build_greeting_instructions(
        GreetingFacts(user_name="Marie", recent_topics=topics)
    )

    included = [t for t in topics if t in instruction]
    assert len(included) == MAX_GREETING_TOPICS


def test_topics_win_over_a_summary() -> None:
    """A specific thing to ask about beats a paragraph of context."""
    instruction = build_greeting_instructions(
        GreetingFacts(
            user_name="Marie",
            recent_topics=["is learning to sail"],
            recent_context_summary="a long summary",
        )
    )

    assert "is learning to sail" in instruction
    assert "a long summary" not in instruction


def test_a_summary_is_used_when_there_are_no_topics() -> None:
    instruction = build_greeting_instructions(
        GreetingFacts(user_name="Marie", recent_context_summary="they moved to Lisbon")
    )

    assert "they moved to Lisbon" in instruction


def test_a_known_name_with_nothing_remembered_still_greets_warmly() -> None:
    instruction = build_greeting_instructions(GreetingFacts(user_name="Marie"))

    assert "great to see you" in instruction


# -- Language ------------------------------------------------------------------


def test_a_configured_language_reaches_the_greeting() -> None:
    """The instruction itself is English, so without this a Spanish-configured
    agent opened in English and only switched once the user replied."""
    instruction = build_greeting_instructions(GreetingFacts(user_name="Marie", language="es"))

    assert "Greet them in Spanish." in instruction


def test_english_adds_no_language_clause() -> None:
    for language in ("en", "en-GB", None, ""):
        instruction = build_greeting_instructions(GreetingFacts(language=language))
        assert "Greet them in" not in instruction


def test_an_unknown_language_adds_no_clause() -> None:
    assert "Greet them in" not in build_greeting_instructions(GreetingFacts(language="xx"))


def test_the_language_clause_reaches_every_branch() -> None:
    for facts in (
        GreetingFacts(language="ja"),
        GreetingFacts(is_returning_user=True, language="ja"),
        GreetingFacts(user_name="Marie", language="ja"),
        GreetingFacts(user_name="Marie", recent_context_summary="x", language="ja"),
        GreetingFacts(user_name="Marie", recent_topics=["x"], language="ja"),
    ):
        assert "Japanese" in build_greeting_instructions(facts)


# -- Facts → topics ------------------------------------------------------------


def test_name_facts_are_not_offered_as_topics() -> None:
    """ "your name is Marie" is a poor "how did that go?"."""
    topics = topics_from_facts(["their name is Marie", "is learning to sail"])

    assert topics == ["is learning to sail"]


def test_only_the_first_few_facts_are_scanned() -> None:
    topics = topics_from_facts([f"fact {i}" for i in range(20)])

    assert len(topics) <= 5


def test_no_facts_yields_no_topics() -> None:
    assert topics_from_facts([]) == []


# -- Facts → name --------------------------------------------------------------


def test_a_name_is_found_in_a_fact() -> None:
    assert extract_name_from_facts(["The user's name is Marie"]) == "Marie"


def test_several_phrasings_are_recognised() -> None:
    assert extract_name_from_facts(["I'm Ada"]) == "Ada"
    assert extract_name_from_facts(["they are called Bruno"]) == "Bruno"


def test_the_agents_own_name_is_not_mistaken_for_the_users() -> None:
    """A fact like "the assistant is called Ada" must not rename the user."""
    assert extract_name_from_facts(["the assistant is called Ada"], agent_name="Ada") is None


def test_obvious_non_names_are_rejected() -> None:
    assert extract_name_from_facts(["the user is called The"]) is None
    assert extract_name_from_facts(["I am a user"]) is None


def test_no_match_yields_none_rather_than_a_guess() -> None:
    """Greeting someone by the wrong name is worse than using none."""
    assert extract_name_from_facts(["likes sailing", "lives in Lisbon"]) is None
    assert extract_name_from_facts([]) is None


def test_the_first_usable_name_wins() -> None:
    assert extract_name_from_facts(["is called The", "name is Marie"]) == "Marie"
