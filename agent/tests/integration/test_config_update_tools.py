"""A `tools` config update must not cost the agent its built-in tools."""

from __future__ import annotations

from typing import Any

import pytest

from src.agent import KwamiAgent
from src.handlers.config_handler import update_tools

CLIENT_TOOL = {
    "name": "set_theme",
    "description": "Change the interface theme",
    "parameters": {
        "type": "object",
        "properties": {"mode": {"type": "string"}},
        "required": ["mode"],
    },
}


def _tool_names(tools: Any) -> set[str]:
    return {tool.info.name for tool in tools}


@pytest.fixture
def agent() -> KwamiAgent:
    return KwamiAgent()


async def test_registering_client_tools_keeps_every_builtin(agent: KwamiAgent) -> None:
    """The regression that silently disabled web_search and all navigation tools.

    The handler used to assign `agent._tools = <client tools only>`, but the
    framework stores built-ins in that same list, so one `{"updateType":"tools"}`
    message from the frontend deleted all 22 of them for the rest of the session.
    """
    before = _tool_names(agent.tools)
    assert "web_search" in before, "precondition: built-ins present at construction"

    await update_tools(agent, [CLIENT_TOOL])

    after = _tool_names(agent.tools)
    assert before <= after, f"built-in tools were dropped: {sorted(before - after)}"
    assert "set_theme" in after, "the client tool was not registered"
    assert len(after) == len(before) + 1


async def test_repeated_updates_do_not_accumulate_or_drop_tools(agent: KwamiAgent) -> None:
    """Re-registering replaces client tools without disturbing the built-ins."""
    await update_tools(agent, [CLIENT_TOOL])
    first = _tool_names(agent.tools)

    other = {**CLIENT_TOOL, "name": "set_volume"}
    await update_tools(agent, [other])
    second = _tool_names(agent.tools)

    assert "set_volume" in second
    assert "set_theme" not in second, "stale client tool survived re-registration"
    assert "web_search" in second
    assert len(first) == len(second)


async def test_chat_context_is_retooled(agent: KwamiAgent) -> None:
    """Going through Agent.update_tools is what keeps the chat context in step.

    Assigning `_tools` directly left `_chat_ctx` holding the old tool set, so
    the LLM could be offered a schema the agent no longer had.
    """
    await update_tools(agent, [CLIENT_TOOL])

    ctx_tools = _tool_names(agent.chat_ctx.tools) if hasattr(agent.chat_ctx, "tools") else None
    if ctx_tools is None:
        pytest.skip("this livekit-agents build does not expose chat_ctx.tools")
    assert "set_theme" in ctx_tools
    assert "web_search" in ctx_tools


async def test_empty_payload_is_ignored(agent: KwamiAgent) -> None:
    """A malformed payload must not be able to clear the toolset."""
    before = _tool_names(agent.tools)

    await update_tools(agent, [])
    await update_tools(agent, None)
    await update_tools(agent, "not-a-list")

    assert _tool_names(agent.tools) == before
