"""What the agent is told to say when it joins.

Extracted from `KwamiAgent._build_greeting_instructions`, which mixed two
unrelated jobs in one 110-line method: fetching memory (I/O, needs a Zep client,
needs the agent) and choosing the wording (pure, needs neither). Only the first
belongs on the agent, and keeping the second there meant the branch that decides
how a returning user is greeted could not be exercised without standing up a
memory double.

`domain/__init__.py` states the layer rule -- "nothing here performs I/O,
imports a provider SDK, or reads the environment" -- so the split is along that
line. `GreetingFacts` is what the agent knows after it has done its I/O;
`build_greeting_instructions` turns that into a sentence.

The name extraction lives here too. It is a regex over strings the memory layer
already fetched, which is exactly the kind of thing that should be testable
without a network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .prompt import LANGUAGE_NAMES

#: Facts whose only content is the user's name are skipped when choosing a topic
#: to open with -- "your name is Ada" is not a conversation starter.
_NAME_FACT_MARKERS = ("name is", "called", "i am", "i'm")

#: Words a name-extraction regex will happily match but which are never names.
_NEVER_A_NAME = frozenset({"the", "a", "user", "assistant", "kwami"})

#: Matches "my name is Ada", "I'm Ada", "called Ada".
_NAME_PATTERN = re.compile(r"(?:name is|called|i'm|i am)\s+([A-Z][a-z]+)", re.IGNORECASE)

#: How many remembered topics to offer the model. More than a few invites it to
#: recite everything it knows, which reads as surveillance rather than memory.
MAX_GREETING_TOPICS = 3

#: Facts are scanned in order; past the first handful they are rarely the most
#: relevant thing to open with.
MAX_FACTS_SCANNED = 5

#: A context block goes verbatim into the greeting instruction, so it is capped.
MAX_CONTEXT_SUMMARY_CHARS = 500


@dataclass(frozen=True)
class GreetingFacts:
    """What the agent knows about the user at the moment it greets them.

    Everything here has already been fetched. This carries no client and does no
    I/O, which is what lets every greeting branch be tested directly.
    """

    agent_name: str = "Kwami"
    user_name: str | None = None
    is_returning_user: bool = False
    recent_context_summary: str | None = None
    recent_topics: list[str] = field(default_factory=list)
    language: str | None = None


def extract_name_from_facts(facts: list[str], *, agent_name: str = "") -> str | None:
    """Find the user's name in remembered facts, or None.

    A fallback for when the memory layer has no cached name. Returns None rather
    than a guess: greeting someone by the wrong name is worse than not using one.
    """
    excluded = _NEVER_A_NAME | ({agent_name.lower()} if agent_name else set())
    for fact in facts:
        match = _NAME_PATTERN.search(fact)
        if not match:
            continue
        candidate = match.group(1).capitalize()
        if candidate.lower() not in excluded:
            return candidate
    return None


def topics_from_facts(facts: list[str]) -> list[str]:
    """Facts worth opening a conversation with.

    Drops the ones that only restate the user's name, which the greeting already
    uses directly and which make a poor "how did that go?".
    """
    topics: list[str] = []
    for fact in facts[:MAX_FACTS_SCANNED]:
        lowered = fact.lower()
        if any(marker in lowered for marker in _NAME_FACT_MARKERS):
            continue
        topics.append(fact)
    return topics


def _language_clause(language: str | None) -> str:
    """Tell the model to greet in the configured language.

    The greeting is generated from an English instruction, so without this a
    Spanish-configured agent opened the conversation in English and only
    switched once the user replied.
    """
    code = (language or "").strip().lower()
    if code not in LANGUAGE_NAMES:
        code = code.split("-", 1)[0]
    name = LANGUAGE_NAMES.get(code)
    if not name or code == "en":
        return ""
    return f" Greet them in {name}."


def build_greeting_instructions(facts: GreetingFacts) -> str:
    """The instruction handed to `generate_reply` for the opening turn.

    Four cases, in decreasing order of what is known: a named user with things
    to talk about, a named user with a summary, a named user, and someone the
    agent has either met without learning a name or never met at all.
    """
    language = _language_clause(facts.language)
    agent_name = facts.agent_name or "Kwami"

    if facts.user_name:
        user_name = facts.user_name
        if facts.recent_topics:
            topics = "; ".join(facts.recent_topics[:MAX_GREETING_TOPICS])
            return (
                f"Greet {user_name} warmly by name, like you're happy to see them again. "
                "Reference something from your recent conversations naturally. "
                f"Here's what you remember about recent topics: {topics}. "
                "Ask a casual follow-up question about one of these topics, or just ask "
                "how something is going. "
                f"Examples: 'Hey {user_name}! How did that [project/thing] turn out?' or "
                f"'What's up {user_name}? Been thinking about [topic] lately?' "
                "Keep it short, friendly, and chill. Don't be formal or robotic. "
                "Pick ONE topic and ask about it naturally - don't list everything you "
                f"remember.{language}"
            )
        if facts.recent_context_summary:
            return (
                f"Greet {user_name} warmly by name, like you're happy to see them again. "
                f"Here's a summary of your past conversations: {facts.recent_context_summary}. "
                "Ask a casual follow-up about something relevant, or just check in on how "
                "things are going. "
                f"Example: 'Hey {user_name}! How's everything going?' or reference something "
                f"specific. Keep it short, friendly, and natural.{language}"
            )
        return (
            f"Greet {user_name} casually by name, like you're happy to see them again. "
            f"Something like 'Hey {user_name}, great to see you! What's on your mind today?' or "
            f"'What's up {user_name}? How've you been?' "
            "Keep it short, friendly, and chill. Don't be formal or robotic. "
            f"Don't repeat the same greeting every time - vary it naturally.{language}"
        )

    if facts.is_returning_user:
        return (
            "Greet the user casually like you've talked before but can't remember their name. "
            f"Something like 'Hey there! Good to hear from you again. By the way, I'm "
            f"{agent_name} - what's your name?' Keep it natural and chill.{language}"
        )

    return (
        "Introduce yourself casually to this new user. "
        f"Something like 'Hey there! I'm {agent_name}, what's your name?' "
        "Keep it short, friendly, and natural. Don't be overly formal or give a long "
        f"introduction.{language}"
    )
