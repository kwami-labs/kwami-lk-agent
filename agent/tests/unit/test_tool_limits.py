"""Staying under the provider's tool ceiling.

The agent hands the model its own built-ins plus every tool the frontend
registered, and neither number is bounded by anything in this repo. OpenAI caps
a request at 128 tools and rejects 129 outright, so crossing it does not
degrade one feature -- it fails *every turn of every session*, and it arrives on
a frontend deploy with nothing here having changed.

The count is real: 40 built-ins and 53 client tools today, and the app added
sixteen in one afternoon. These tests pin that the list is trimmed rather than
sent over, that the built-ins survive in preference to client tools, and that
nothing silently disappears without a log line saying what and why.
"""

from __future__ import annotations

import logging

import pytest

from src.agent import KwamiAgent
from src.domain import KwamiConfig
from src.tools.limits import (
    MAX_TOOLS_PER_REQUEST,
    TOOL_COUNT_WARN_THRESHOLD,
    enforce_tool_limit,
    trim_client_tools,
)


class FakeTool:
    """Carries the `info.name` shape the real tools expose, and nothing else."""

    def __init__(self, name: str) -> None:
        self.info = type("Info", (), {"name": name})()


def _tools(prefix: str, count: int) -> list[FakeTool]:
    return [FakeTool(f"{prefix}{i}") for i in range(count)]


# -- under the limit --------------------------------------------------------


def test_everything_fits_when_it_fits() -> None:
    combined = enforce_tool_limit(_tools("b", 40), _tools("c", 53))
    assert len(combined) == 93


def test_exactly_at_the_limit_is_kept_whole() -> None:
    """128 succeeds against the real API; only 129 fails."""
    combined = enforce_tool_limit(_tools("b", 28), _tools("c", 100))
    assert len(combined) == MAX_TOOLS_PER_REQUEST


def test_no_client_tools_is_fine() -> None:
    """Telephony: no app on the other end, so nothing is registered."""
    assert len(enforce_tool_limit(_tools("b", 40), [])) == 40


# -- over the limit ---------------------------------------------------------


def test_one_over_is_trimmed_rather_than_sent() -> None:
    """A session with 128 tools that works beats 129 that cannot answer."""
    combined = enforce_tool_limit(_tools("b", 40), _tools("c", 89))

    assert len(combined) == MAX_TOOLS_PER_REQUEST


def test_the_builtins_survive_a_trim() -> None:
    """Losing web_search silently is worse than losing the 129th UI control."""
    builtins = _tools("b", 40)
    combined = enforce_tool_limit(builtins, _tools("c", 200))

    kept = {tool.info.name for tool in combined}
    for tool in builtins:
        assert tool.info.name in kept, "a built-in was dropped in favour of a client tool"


def test_a_trim_says_what_it_dropped(caplog: pytest.LogCaptureFixture) -> None:
    """A capability vanishing with no log line is the failure to avoid."""
    with caplog.at_level(logging.ERROR):
        enforce_tool_limit(_tools("b", 40), _tools("c", 100))

    assert any("Tool limit exceeded" in record.message for record in caplog.records)
    combined_text = " ".join(record.getMessage() for record in caplog.records)
    assert "c99" in combined_text, "the dropped tool names were not logged"
    assert "NOT available" in combined_text


def test_approaching_the_limit_warns_before_it_bites(caplog: pytest.LogCaptureFixture) -> None:
    """The ceiling should appear in logs before it appears as a dead session."""
    with caplog.at_level(logging.WARNING):
        enforce_tool_limit(_tools("b", 40), _tools("c", TOOL_COUNT_WARN_THRESHOLD - 40))

    assert any("of a 128 limit" in record.getMessage() for record in caplog.records)


def test_a_comfortable_count_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        enforce_tool_limit(_tools("b", 40), _tools("c", 10))

    assert not caplog.records


def test_builtins_alone_overflowing_still_returns_a_usable_list(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Not reachable today at 40 built-ins, but it must not produce a 400."""
    with caplog.at_level(logging.CRITICAL):
        combined = enforce_tool_limit(_tools("b", 200), _tools("c", 5))

    assert len(combined) == MAX_TOOLS_PER_REQUEST
    assert any(record.levelno == logging.CRITICAL for record in caplog.records)


# -- the construction-time half ---------------------------------------------


def test_trim_client_tools_returns_only_the_client_half() -> None:
    """The framework appends the built-ins itself, so it must not get them twice."""
    kept = trim_client_tools(40, _tools("c", 53))

    assert len(kept) == 53
    assert all(tool.info.name.startswith("c") for tool in kept)


def test_trim_client_tools_leaves_room_for_the_builtins() -> None:
    kept = trim_client_tools(40, _tools("c", 200))
    assert len(kept) == MAX_TOOLS_PER_REQUEST - 40


@pytest.mark.parametrize("builtin_count", [0, 1, 40, MAX_TOOLS_PER_REQUEST])
def test_the_total_is_never_over_the_limit(builtin_count: int) -> None:
    kept = trim_client_tools(builtin_count, _tools("c", 500))
    assert builtin_count + len(kept) <= MAX_TOOLS_PER_REQUEST


# -- the real agent ---------------------------------------------------------


def _agent_with(client_tool_count: int) -> KwamiAgent:
    config = KwamiConfig()
    config.tools = [
        {
            "name": f"client_tool_{i}",
            "description": "a registered client tool",
            "parameters": {"type": "object", "properties": {}},
        }
        for i in range(client_tool_count)
    ]
    return KwamiAgent(config=config)


def test_todays_real_tool_count_is_under_the_limit() -> None:
    """53 client tools plus the built-ins. The headroom is worth knowing."""
    agent = _agent_with(53)

    assert len(agent.tools) <= MAX_TOOLS_PER_REQUEST
    assert len(agent.tools) == 93, (
        "the tool count moved; if it is climbing, the 128 ceiling is the thing to watch"
    )


def test_an_agent_never_exceeds_the_limit_however_many_tools_arrive() -> None:
    """The frontend decides this number, and it is not a trusted bound."""
    agent = _agent_with(300)

    assert len(agent.tools) <= MAX_TOOLS_PER_REQUEST


def test_an_over_subscribed_agent_keeps_its_own_tools() -> None:
    agent = _agent_with(300)

    names = {tool.info.name for tool in agent.tools}
    for essential in ("web_search", "navigate_to", "change_ai_model", "remember_fact"):
        assert essential in names, f"{essential} was crowded out by client tools"
