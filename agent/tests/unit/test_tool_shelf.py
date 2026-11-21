"""The tool shelf: client tools past the ceiling, still within the model's reach.

Before it, the trim to 128 tools dropped the tail for the whole session: the
model could neither call those tools nor know they existed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.agent import KwamiAgent
from src.domain import KwamiConfig
from src.handlers import config_handler
from src.tools.limits import KEEP_LONGEST, MAX_TOOLS_PER_REQUEST
from src.tools.shelf import SHELF_TOOL, ToolShelfMixin, rank, swap, tool_name


def _defs(count: int, prefix: str = "client_tool_") -> list[dict[str, Any]]:
    return [
        {
            "name": f"{prefix}{i}",
            "description": f"does thing number {i}",
            "parameters": {"type": "object", "properties": {}},
        }
        for i in range(count)
    ]


def _agent(definitions: list[dict[str, Any]]) -> KwamiAgent:
    config = KwamiConfig()
    config.tools = definitions
    return KwamiAgent(config=config)


def _names(tools: list[Any]) -> list[str]:
    return [tool_name(tool) for tool in tools]


class _Tool:
    def __init__(self, name: str, description: str = "", raw: bool = True) -> None:
        schema = {"description": description} if raw else None
        self.info = SimpleNamespace(name=name, raw_schema=schema, description=description)


# -- what goes on the shelf ------------------------------------------------------------


def test_with_room_for_everything_there_is_no_shelf(all_tools_available) -> None:
    agent = _agent(_defs(10))
    assert agent._shelf == []
    assert SHELF_TOOL not in _names(agent.tools)


def test_what_does_not_fit_is_shelved_and_reachable(all_tools_available) -> None:
    agent = _agent(_defs(200))
    names = _names(agent.tools)

    assert len(agent.tools) == MAX_TOOLS_PER_REQUEST
    assert SHELF_TOOL in names
    offered = sum(1 for n in names if n.startswith("client_tool_"))
    assert offered + len(agent._shelf) == 200
    assert _names(agent._shelf)[-1] == "client_tool_199"


def test_a_tools_update_reshelves(all_tools_available) -> None:
    agent = _agent(_defs(10))
    agent.update_tools = AsyncMock()  # type: ignore[method-assign]
    agent.publish_tool_inventory = AsyncMock(return_value=True)  # type: ignore[method-assign]

    import asyncio

    asyncio.run(config_handler.update_tools(agent, _defs(200)))

    offered = agent.update_tools.call_args.args[0]
    assert len(offered) <= MAX_TOOLS_PER_REQUEST
    assert SHELF_TOOL in _names(offered)
    assert agent._shelf


# -- finding --------------------------------------------------------------------------


def test_a_name_counts_more_than_a_description() -> None:
    by_text = _Tool("set_theme", "colour of the calendar view")
    by_name = _Tool("create_calendar_event", "adds an entry")
    plain = _Tool("export_wallet", "nothing to do with it", raw=False)
    assert rank([by_text, by_name, plain], "calendar") == [by_name, by_text]
    assert rank([plain], "wallet") == [plain]
    assert rank([plain], "  ") == []


def test_a_swap_never_sends_out_a_builtin_or_a_kept_tool() -> None:
    builtin = _Tool("web_search")
    kept = _Tool(sorted(KEEP_LONGEST)[0])
    a, b = _Tool("a"), _Tool("b")
    wanted = _Tool("wanted")

    offered, brought, shelf = swap([builtin, a, b, kept], [a, b, kept], [wanted], [wanted])

    assert _names(offered) == ["web_search", "a", sorted(KEEP_LONGEST)[0], "wanted"]
    assert brought == [wanted] and shelf == [b]


def test_a_swap_with_nothing_to_send_out_brings_nothing() -> None:
    builtin, wanted = _Tool("web_search"), _Tool("wanted")
    offered, brought, shelf = swap([builtin], [], [wanted], [wanted])
    assert offered == [builtin] and brought == [] and shelf == [wanted]


# -- the tool -------------------------------------------------------------------------


def _plain(tool: Any) -> Any:
    return getattr(tool, "__wrapped__", tool)


find = _plain(ToolShelfMixin.find_more_tools)


@pytest.mark.asyncio
async def test_the_model_brings_a_shelved_tool_in(all_tools_available) -> None:
    agent = _agent([*_defs(140), {"name": "export_theme", "description": "save the theme"}])
    agent.update_tools = AsyncMock()  # type: ignore[method-assign]
    agent.publish_tool_inventory = AsyncMock(return_value=True)  # type: ignore[method-assign]
    assert "export_theme" in _names(agent._shelf)

    said = await find(agent, SimpleNamespace(), "export the theme")

    assert [f["name"] for f in said["found"]] == ["export_theme"]
    offered = agent.update_tools.call_args.args[0]
    assert "export_theme" in _names(offered)
    assert len(offered) == MAX_TOOLS_PER_REQUEST
    # One went out in its place, onto the shelf.
    assert "export_theme" not in _names(agent._shelf)
    agent.publish_tool_inventory.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_match_lists_what_is_shelved(all_tools_available) -> None:
    agent = _agent(_defs(200))
    said = await find(agent, SimpleNamespace(), "telepathy")
    assert said["found"] == [] and said["shelved"][0].startswith("client_tool_")


@pytest.mark.asyncio
async def test_with_nothing_shelved_it_says_so() -> None:
    agent = _agent(_defs(1))
    said = await find(agent, SimpleNamespace(), "anything")
    assert said["found"] == [] and "already available" in said["note"]


@pytest.mark.asyncio
async def test_with_no_room_to_make_nothing_changes() -> None:
    agent = _agent(_defs(1))
    agent._shelf = [_Tool("thing")]
    agent.update_tools = AsyncMock()  # type: ignore[method-assign]
    # The only client tool is kept longest, so nothing can be sent out.
    agent.client_tools._tools = []

    said = await find(agent, SimpleNamespace(), "thing")

    assert said["found"] == [] and "No room" in said["note"]
    agent.update_tools.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_agent_without_an_inventory_still_swaps() -> None:
    class Bare:
        tools: list[Any]

    bare = Bare()
    a, wanted = _Tool("a"), _Tool("wanted")
    bare.tools = [a]
    bare._shelf = [wanted]  # type: ignore[attr-defined]
    bare.client_tools = SimpleNamespace(create_client_tools=lambda: [a])  # type: ignore[attr-defined]
    bare.update_tools = AsyncMock()  # type: ignore[attr-defined]

    said = await find(bare, SimpleNamespace(), "wanted")

    assert said["found"][0]["name"] == "wanted"
    assert bare._shelf == [a]  # type: ignore[attr-defined]
