"""Client tool results go straight into the LLM context, so they need a bound."""

from __future__ import annotations

import asyncio

from src.tools.client import MAX_CLIENT_TOOL_RESULT_CHARS, ClientToolManager


def _manager_with_pending(call_id: str) -> tuple[ClientToolManager, asyncio.Future]:
    manager = ClientToolManager(kwami_agent=None)
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    manager.pending_calls[call_id] = future
    return manager, future


async def test_an_oversized_result_is_truncated() -> None:
    manager, future = _manager_with_pending("call-1")

    manager.handle_tool_result("call-1", "x" * (MAX_CLIENT_TOOL_RESULT_CHARS * 4))

    result = await future
    assert len(result) < MAX_CLIENT_TOOL_RESULT_CHARS + 200
    assert "truncated" in result


async def test_a_normal_result_is_passed_through_untouched() -> None:
    manager, future = _manager_with_pending("call-2")

    manager.handle_tool_result("call-2", "theme set to dark")

    assert await future == "theme set to dark"


async def test_an_oversized_error_is_also_capped() -> None:
    manager, future = _manager_with_pending("call-3")

    manager.handle_tool_result("call-3", None, error="e" * (MAX_CLIENT_TOOL_RESULT_CHARS * 2))

    result = await future
    assert len(result) < MAX_CLIENT_TOOL_RESULT_CHARS + 200
    assert result.startswith("Error from client:")


async def test_a_missing_result_becomes_an_empty_string() -> None:
    manager, future = _manager_with_pending("call-4")

    manager.handle_tool_result("call-4", None)

    assert await future == ""
