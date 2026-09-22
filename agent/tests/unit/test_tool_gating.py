"""Built-in tools this deployment cannot serve are not offered to the model.

Forty function tools were registered unconditionally. Without `SERPAPI_KEY`,
`product_search` was still described in full, still chosen when someone asked to
buy something, and still answered "Product search is not configured" -- a
refusal to a request the agent had advertised it could handle, costing the
user's turn.

Two costs, one visible and one not. The invisible one is bigger at scale: every
tool's name, description and JSON schema goes out on *every* request of every
session, and dead built-ins also consume the 128-tool budget that
`tools/limits.py` defends -- where built-ins are kept in preference to client
tools, so a dead built-in can evict a live UI control.

The rule these pin: a tool is withheld only when it *cannot work*. Never to
shorten the list.
"""

from __future__ import annotations

from src.domain.tool_gating import (
    BROWSER_TOOLS,
    MEMORY_TOOLS,
    REQUIREMENTS,
    describe_gating,
    unavailable_builtin_tools,
)
from src.settings import Settings


def _fully_configured() -> Settings:
    return Settings(
        serpapi_key="k",
        tavily_api_key="k",
        zep_api_key="k",
        browserbase_api_key="k",
        browserbase_project_id="p",
    )


def test_a_fully_configured_deployment_withholds_nothing() -> None:
    """The rule is "withhold only what cannot work" -- so a complete
    deployment must lose nothing at all."""
    assert unavailable_builtin_tools(_fully_configured()) == frozenset()


def test_an_empty_deployment_withholds_every_gated_tool() -> None:
    missing = unavailable_builtin_tools(Settings())

    assert "product_search" in missing
    assert "web_search" in missing
    assert missing >= BROWSER_TOOLS
    assert missing >= MEMORY_TOOLS


def test_each_credential_gates_only_its_own_tools() -> None:
    """A missing SerpAPI key must not cost the user their browser."""
    settings = Settings(
        tavily_api_key="k",
        zep_api_key="k",
        browserbase_api_key="k",
    )

    assert unavailable_builtin_tools(settings) == frozenset({"product_search"})


def test_either_browser_vendor_enables_the_browser_tools() -> None:
    """The two vendors are alternatives, not both required."""
    via_browserbase = Settings(browserbase_api_key="k")
    via_browser_use = Settings(browser_use_api_key="k")

    assert not (BROWSER_TOOLS & unavailable_builtin_tools(via_browserbase))
    assert not (BROWSER_TOOLS & unavailable_builtin_tools(via_browser_use))


def test_memory_status_is_never_withheld() -> None:
    """ "Do you remember me?" is a fair question to ask an agent with no memory,
    and this is the one tool whose honest answer is "not configured"."""
    assert "get_memory_status" not in unavailable_builtin_tools(Settings())


def test_tools_needing_no_credential_are_never_withheld() -> None:
    """The market quote uses a keyless endpoint; the voice controls are local."""
    missing = unavailable_builtin_tools(Settings())

    for always_on in (
        "get_current_time",
        "get_kwami_info",
        "change_voice",
        "change_language",
        "get_market_quote",
        "list_available_models",
    ):
        assert always_on not in missing


def test_every_requirement_names_a_real_settings_field() -> None:
    """A typo would silently gate a tool forever -- `getattr` returns "" for a
    field that does not exist, which reads as "credential absent"."""
    fields = set(vars(Settings()))

    for requirement in REQUIREMENTS:
        assert requirement.settings_field in fields, (
            f"{requirement.tool} is gated on {requirement.settings_field!r}, "
            "which is not a Settings field"
        )


def test_every_requirement_carries_a_reason() -> None:
    for requirement in REQUIREMENTS:
        assert requirement.reason.strip(), f"{requirement.tool} is gated without a reason"


def test_the_gated_names_are_real_tools() -> None:
    """Guards against a rename leaving a gate pointed at nothing, which would
    silently stop withholding a tool that still cannot work."""
    from livekit.agents.voice.agent import find_function_tools

    from src.agent import KwamiAgent

    real = {t.info.name for t in find_function_tools(KwamiAgent)}
    gated = {r.tool for r in REQUIREMENTS} | set(BROWSER_TOOLS) | set(MEMORY_TOOLS)

    assert gated <= real, f"gated names that are not tools: {sorted(gated - real)}"


def test_the_summary_names_what_was_withheld() -> None:
    summary = describe_gating({"web_search", "product_search"})

    assert "2 built-in tool(s) withheld" in summary
    assert "product_search" in summary
    assert "web_search" in summary


def test_the_summary_says_so_when_nothing_was_withheld() -> None:
    assert describe_gating([]) == "all built-in tools are available"


# -- The agent actually applies it -------------------------------------------


def test_an_unconfigured_agent_carries_fewer_tools(all_tools_available) -> None:
    """End to end: the gate has to reach `Agent._tools`, not just compute a set.

    The framework sets `_tools = tools + find_function_tools(self)` *after* our
    `__init__` body, so withholding has to happen after `super().__init__` --
    which is the part a unit test of the domain function alone would not catch.
    """
    from src.agent import KwamiAgent
    from src.settings import set_settings

    full = len(KwamiAgent().tools)

    set_settings(Settings())
    bare = len(KwamiAgent().tools)

    assert bare < full
    assert bare > 0, "gating removed everything, which is never right"


def test_a_withheld_tool_is_absent_from_the_agents_tool_list() -> None:
    from src.agent import KwamiAgent
    from src.settings import set_settings

    set_settings(Settings(tavily_api_key="k", zep_api_key="k", browserbase_api_key="k"))

    names = {t.info.name for t in KwamiAgent().tools}

    assert "product_search" not in names
    assert "web_search" in names, "only the tool whose credential is missing goes"


def test_gating_does_not_disturb_client_tools() -> None:
    """Client tools are the frontend's, not ours, and are gated by their own
    registration -- `domain/capabilities.py`, not this module."""
    from src.agent import KwamiAgent
    from src.domain import KwamiConfig
    from src.settings import set_settings

    set_settings(Settings())
    config = KwamiConfig(
        tools=[{"name": "set_ui_control", "description": "change a control", "parameters": {}}]
    )

    names = {t.info.name for t in KwamiAgent(config=config).tools}

    assert "set_ui_control" in names
