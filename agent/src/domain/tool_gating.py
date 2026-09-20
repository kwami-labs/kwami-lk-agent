"""Which built-in tools this deployment can actually deliver.

The agent registers forty function tools unconditionally. Without
`SERPAPI_KEY`, `product_search` is still offered to the model, still described
in full, still chosen when someone asks to buy something -- and answers "Product
search is not configured". The user hears a refusal to a request the agent
advertised it could handle, and the turn is spent.

That is the visible cost. The invisible one is larger: every tool's name,
description and JSON schema is sent on **every** request of every session. Forty
definitions is a few thousand tokens the model re-reads every turn, and it also
consumes the 128-tool budget that `tools/limits.py` defends -- built-ins are
kept in preference to client tools there, so a dead built-in can evict a live
UI control.

`domain/capabilities.py` already solved exactly this shape for *client* tools:
guidance is assembled from what the frontend actually registered, because "a
tool the model is not told about is a tool the user cannot reach by asking".
This is the same rule pointed at our own half of the list.

Gating is on **credentials**, not on preference. A tool is dropped only when it
cannot possibly work -- never to make the list shorter, because a missing tool
the deployment could have served is the failure mode this module is trying to
avoid, not cause.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolRequirement:
    """One built-in tool and the credential it cannot work without.

    `settings_field` names an attribute on `Settings`; the tool is available
    when that attribute is truthy. Named rather than passed as a callable so the
    whole table stays readable as data.
    """

    tool: str
    settings_field: str
    reason: str


#: Tools whose entire body is "call this vendor". Each was verified against the
#: tool's own guard -- these all return a "not configured" string today rather
#: than doing anything useful.
REQUIREMENTS: tuple[ToolRequirement, ...] = (
    ToolRequirement(
        "product_search",
        "serpapi_key",
        "SerpAPI is the only source of product cards; without it the tool "
        "returns a refusal and points at web_search.",
    ),
    ToolRequirement(
        "web_search",
        "tavily_api_key",
        "Tavily is the only search backend wired up.",
    ),
    ToolRequirement(
        "deep_research",
        "tavily_api_key",
        "A research pass is several Tavily searches.",
    ),
)

#: Browsing is a family: one credential decides whether any of it works, and a
#: browser tool offered without one strands the model mid-task -- it navigates,
#: gets a refusal, and has no way to finish what it started.
BROWSER_TOOLS: frozenset[str] = frozenset(
    {
        "navigate_to",
        "go_back_in_browser",
        "go_forward_in_browser",
        "close_navigation",
        "click_in_navigation",
        "type_in_navigation",
        "press_key_in_navigation",
        "scroll_navigation",
        "run_js_in_navigation",
        "read_navigation_page",
        "play_media",
        "control_playback",
        "set_playback_volume",
        "get_now_playing",
    }
)

#: Memory tools need a Zep credential. `get_memory_status` is deliberately NOT
#: here: "do you remember me?" is a reasonable question to ask an agent with no
#: memory, and it is the one tool whose honest answer is "memory is not
#: configured".
MEMORY_TOOLS: frozenset[str] = frozenset({"remember_fact", "recall_memories"})


def unavailable_builtin_tools(settings: object) -> frozenset[str]:
    """Built-in tool names this deployment cannot serve.

    Args:
        settings: A `Settings`. Typed loosely so the domain layer keeps its rule
            of importing nothing from the outer layers.
    """
    missing: set[str] = set()

    for requirement in REQUIREMENTS:
        if not getattr(settings, requirement.settings_field, ""):
            missing.add(requirement.tool)

    browser_configured = bool(
        getattr(settings, "browserbase_api_key", "") or getattr(settings, "browser_use_api_key", "")
    )
    if not browser_configured:
        missing |= BROWSER_TOOLS

    if not getattr(settings, "zep_api_key", ""):
        missing |= MEMORY_TOOLS

    return frozenset(missing)


def describe_gating(dropped: Iterable[str]) -> str:
    """A one-line summary for the startup log.

    Worth logging loudly: "the agent cannot browse" is something an operator
    should learn from a log line at startup, not from a user reporting that it
    refuses to open pages.
    """
    names = sorted(dropped)
    if not names:
        return "all built-in tools are available"
    return f"{len(names)} built-in tool(s) withheld for missing credentials: {', '.join(names)}"
