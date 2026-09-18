"""The client-tool bridge: an LLM tool call published to the frontend, and the
answer routed back.

Every failure branch here ends a turn. A tool call that cannot be published, or
whose answer never arrives, leaves the model waiting -- so each one has to
return a string the model can read rather than raise or hang.
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from src.tools.client import ClientToolManager


class RecordingParticipant:
    def __init__(self, fail: bool = False) -> None:
        self.published: list[dict[str, Any]] = []
        self.fail = fail

    async def publish_data(self, payload: bytes, *args: Any, **kwargs: Any) -> None:
        if self.fail:
            raise RuntimeError("data channel is closed")
        self.published.append(json.loads(payload.decode()))


class RecordingRoom:
    def __init__(self, fail: bool = False) -> None:
        self.local_participant = RecordingParticipant(fail)


def manager_with_tool(room: Any = None, name: str = "set_theme") -> ClientToolManager:
    agent = SimpleNamespace(room=room)
    manager = ClientToolManager(kwami_agent=agent)
    manager.register_client_tools(
        [{"name": name, "description": "change the theme", "parameters": {}}]
    )
    return manager


def handler_of(manager: ClientToolManager):
    """The raw coroutine behind the registered FunctionTool."""
    return manager.create_client_tools()[0]._func


# =============================================================================
# Registration
# =============================================================================


@pytest.mark.parametrize("definitions", [[], None])
def test_an_empty_registration_is_a_no_op(definitions: Any) -> None:
    manager = ClientToolManager(kwami_agent=None)

    manager.register_client_tools(definitions)

    assert manager.tool_count == 0


def test_an_invalid_definition_is_skipped() -> None:
    """A malformed name 400s every subsequent LLM request and bricks the
    session, so it must never reach the tool list."""
    manager = ClientToolManager(kwami_agent=None)

    manager.register_client_tools([{"description": "no name"}, {"name": "ok"}])

    assert manager.tool_count == 1


def test_a_tool_without_parameters_gets_an_empty_object_schema() -> None:
    """Providers reject a function whose parameters are absent."""
    manager = ClientToolManager(kwami_agent=None)

    manager.register_client_tools([{"name": "ping", "description": "d"}])

    schema = manager.create_client_tools()[0].info.raw_schema
    assert schema["parameters"] == {"type": "object", "properties": {}, "required": []}


def test_declared_parameters_are_preserved() -> None:
    params = {"type": "object", "properties": {"mode": {"type": "string"}}}
    manager = ClientToolManager(kwami_agent=None)

    manager.register_client_tools([{"name": "set_theme", "description": "d", "parameters": params}])

    assert manager.create_client_tools()[0].info.raw_schema["parameters"] == params


def test_the_tool_count_reflects_registrations() -> None:
    manager = ClientToolManager(kwami_agent=None)

    manager.register_client_tools(
        [{"name": "a", "description": "d"}, {"name": "b", "description": "d"}]
    )

    assert manager.tool_count == 2


# =============================================================================
# Invocation
# =============================================================================


async def test_a_tool_call_is_published_to_the_frontend() -> None:
    room = RecordingRoom()
    manager = manager_with_tool(room)

    async def answer() -> None:
        await asyncio.sleep(0)
        call_id = room.local_participant.published[0]["toolCallId"]
        manager.handle_tool_result(call_id, "theme set to dark")

    task = asyncio.create_task(answer())
    result = await handler_of(manager)({"mode": "dark"}, None)
    await task

    published = room.local_participant.published[0]
    assert published["type"] == "tool_call"
    assert published["function"]["name"] == "set_theme"
    assert json.loads(published["function"]["arguments"]) == {"mode": "dark"}
    assert result == "theme set to dark"


async def test_no_room_returns_an_error_the_model_can_read(caplog) -> None:
    """Returning a string keeps the turn alive; raising would end it."""
    manager = manager_with_tool(room=None)

    with caplog.at_level(logging.ERROR):
        result = await handler_of(manager)({}, None)

    assert result == "Error: Agent not connected to room"
    assert "No room connection" in caplog.text


async def test_the_room_can_come_from_the_run_context() -> None:
    """Tools registered before the room was wired up still work."""
    room = RecordingRoom()
    manager = manager_with_tool(room=None)
    context = SimpleNamespace(room=room)

    async def answer() -> None:
        await asyncio.sleep(0)
        call_id = room.local_participant.published[0]["toolCallId"]
        manager.handle_tool_result(call_id, "done")

    task = asyncio.create_task(answer())
    result = await handler_of(manager)({}, context)
    await task

    assert result == "done"


async def test_a_publish_failure_returns_an_error(caplog) -> None:
    manager = manager_with_tool(RecordingRoom(fail=True))

    with caplog.at_level(logging.ERROR):
        result = await handler_of(manager)({}, None)

    assert result.startswith("Error executing tool:")
    assert "Error executing client tool" in caplog.text


async def test_a_failed_call_leaves_no_pending_entry() -> None:
    """The `finally` cleanup: pending futures that are never resolved are a
    leak across an agent swap."""
    manager = manager_with_tool(RecordingRoom(fail=True))

    await handler_of(manager)({}, None)

    assert manager.pending_calls == {}


async def test_a_call_that_is_never_answered_times_out(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """Without the bound the model waits forever on a frontend that has gone
    away mid-turn."""
    real_wait_for = asyncio.wait_for

    async def impatient(awaitable, timeout):
        return await real_wait_for(awaitable, timeout=0.01)

    monkeypatch.setattr("src.tools.client.asyncio.wait_for", impatient)
    manager = manager_with_tool(RecordingRoom())

    with caplog.at_level(logging.WARNING):
        result = await handler_of(manager)({}, None)

    assert result == "Error: Tool execution timed out"
    assert "timed out" in caplog.text
    assert manager.pending_calls == {}


# =============================================================================
# Results
# =============================================================================


def test_a_result_for_an_unknown_call_is_dropped(caplog) -> None:
    """An agent swap can land between the call and its answer."""
    manager = ClientToolManager(kwami_agent=None)

    with caplog.at_level(logging.WARNING):
        manager.handle_tool_result("never-seen", "value")

    assert "unknown tool call" in caplog.text


async def test_a_second_result_for_the_same_call_is_dropped(caplog) -> None:
    """A client that answers twice must not raise InvalidStateError into the
    data-channel handler."""
    manager = ClientToolManager(kwami_agent=None)
    future: asyncio.Future = asyncio.get_running_loop().create_future()
    manager.pending_calls["call-1"] = future

    manager.handle_tool_result("call-1", "first")
    with caplog.at_level(logging.WARNING):
        manager.handle_tool_result("call-1", "second")

    assert await future == "first"
    assert "already completed" in caplog.text
