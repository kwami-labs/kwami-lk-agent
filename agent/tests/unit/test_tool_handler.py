"""`handle_tool_result` routes a client's answer back to whatever is waiting.

Every branch is a way the answer can be dropped, and a dropped result leaves the
LLM's tool call pending until the session ends.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from src.handlers.tool_handler import handle_tool_result


class RecordingClientTools:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, str | None]] = []

    def handle_tool_result(
        self, tool_call_id: str, result: str | None, error: str | None
    ) -> None:
        self.calls.append((tool_call_id, result, error))


class AgentWithClientTools:
    def __init__(self, client_tools: Any) -> None:
        self.client_tools = client_tools


class AgentWithDirectMethod:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, str | None]] = []

    def handle_tool_result(
        self, tool_call_id: str, result: str | None, error: str | None
    ) -> None:
        self.calls.append((tool_call_id, result, error))


class AgentThatCannotHandleResults:
    """No `client_tools`, no `handle_tool_result`."""


def test_a_result_reaches_the_client_tools_manager() -> None:
    tools = RecordingClientTools()

    handle_tool_result(AgentWithClientTools(tools), "call-1", "the answer")

    assert tools.calls == [("call-1", "the answer", None)]


def test_an_error_is_forwarded_alongside_the_result() -> None:
    tools = RecordingClientTools()

    handle_tool_result(AgentWithClientTools(tools), "call-1", None, "it failed")

    assert tools.calls == [("call-1", None, "it failed")]


def test_an_agent_without_client_tools_falls_back_to_its_own_method() -> None:
    agent = AgentWithDirectMethod()

    handle_tool_result(agent, "call-2", "answer")

    assert agent.calls == [("call-2", "answer", None)]


def test_a_present_but_falsy_client_tools_still_falls_back() -> None:
    """`hasattr(...) and agent.client_tools` -- None must not be called."""
    agent = AgentWithDirectMethod()
    agent.client_tools = None  # type: ignore[attr-defined]

    handle_tool_result(agent, "call-3", "answer")

    assert agent.calls == [("call-3", "answer", None)]


@pytest.mark.parametrize(
    "tool_call_id",
    [pytest.param(None, id="none"), pytest.param("", id="empty")],
)
def test_a_result_with_no_call_id_is_dropped_with_a_warning(
    tool_call_id: str | None, caplog
) -> None:
    tools = RecordingClientTools()

    with caplog.at_level(logging.WARNING):
        handle_tool_result(AgentWithClientTools(tools), tool_call_id, "answer")

    assert tools.calls == []
    assert "no tool_call_id" in caplog.text


def test_a_result_with_no_agent_is_dropped_with_a_warning(caplog) -> None:
    """An agent swap can land between the tool call and its answer."""
    with caplog.at_level(logging.WARNING):
        handle_tool_result(None, "call-4", "answer")

    assert "No agent available" in caplog.text
    assert "call-4" in caplog.text


def test_an_agent_that_can_handle_nothing_warns_rather_than_raising(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        handle_tool_result(AgentThatCannotHandleResults(), "call-5", "answer")

    assert "cannot handle tool results" in caplog.text
    assert "call-5" in caplog.text
