"""The tool inventory: what the model is offered, told to the app.

The 128-tool trim used to be silent from the user's side. A tool they had just
added -- registered after the app's hundred, so at the tail the trim cuts --
stopped existing for the model with nothing on screen to say so. These pin the
report the Tools panel reads, and the count that decides how much room client
tools get in the first place.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent import KwamiAgent
from src.domain import KwamiConfig
from src.handlers import config_handler
from src.tools.limits import MAX_TOOLS_PER_REQUEST, tool_inventory


class FakeTool:
    def __init__(self, name: str) -> None:
        self.info = type("Info", (), {"name": name})()


def _client(count: int, prefix: str = "client_tool_") -> list[dict]:
    return [
        {
            "name": f"{prefix}{i}",
            "description": "a registered client tool",
            "parameters": {"type": "object", "properties": {}},
        }
        for i in range(count)
    ]


def _agent_with(client_tool_count: int) -> KwamiAgent:
    config = KwamiConfig()
    config.tools = _client(client_tool_count)
    return KwamiAgent(config=config)


# -- the payload --------------------------------------------------------------


def test_the_inventory_names_the_builtins_and_the_dropped_client_tools() -> None:
    offered = [FakeTool("web_search"), FakeTool("send_email"), FakeTool("set_theme")]

    payload = tool_inventory(offered, ["send_email", "set_theme", "get_weather"])

    assert payload == {
        "type": "tool_inventory",
        "limit": MAX_TOOLS_PER_REQUEST,
        "total": 3,
        "builtin": ["web_search"],
        "client": 2,
        "dropped": ["get_weather"],
    }


def test_the_inventory_fits_the_data_channel_with_every_tool_named() -> None:
    offered = [FakeTool(f"builtin_tool_number_{i}") for i in range(60)]
    names = [f"client_tool_with_a_long_name_{i}" for i in range(200)]

    encoded = json.dumps(tool_inventory(offered, names)).encode()

    assert len(encoded) < 14_000


# -- the room client tools get ------------------------------------------------


def test_withheld_builtins_no_longer_take_client_tools_places() -> None:
    """The offline suite strips credentials, so most gated built-ins are withheld.

    They were counted anyway, and a client tool lost its place to each one. The
    app registers about a hundred tools: all of them fit next to the built-ins
    that are actually on.
    """
    agent = _agent_with(100)

    names = {tool.info.name for tool in agent.tools}
    client_kept = [f"client_tool_{i}" for i in range(100) if f"client_tool_{i}" in names]
    assert len(client_kept) == 100
    assert len(agent.tools) <= MAX_TOOLS_PER_REQUEST


def test_the_limit_still_holds_with_every_builtin_on(all_tools_available) -> None:
    agent = _agent_with(300)

    assert len(agent.tools) <= MAX_TOOLS_PER_REQUEST


# -- publishing ---------------------------------------------------------------


def _room() -> MagicMock:
    room = MagicMock()
    room.local_participant.publish_data = AsyncMock()
    return room


@pytest.mark.asyncio
async def test_the_agent_publishes_what_it_offers(all_tools_available) -> None:
    agent = _agent_with(150)
    room = _room()
    agent.room = room

    assert await agent.publish_tool_inventory() is True

    payload = json.loads(room.local_participant.publish_data.call_args.args[0])
    assert payload["type"] == "tool_inventory"
    assert payload["total"] == len(agent.tools) <= MAX_TOOLS_PER_REQUEST
    assert "web_search" in payload["builtin"]
    assert payload["client"] + len(payload["dropped"]) == 150
    # The tail goes first: the last ones the app registered are the dropped ones.
    assert payload["dropped"][-1] == "client_tool_149"


@pytest.mark.asyncio
async def test_publishing_never_raises() -> None:
    agent = _agent_with(1)
    room = _room()
    room.local_participant.publish_data.side_effect = RuntimeError("channel closed")
    agent.room = room

    assert await agent.publish_tool_inventory() is False


@pytest.mark.asyncio
async def test_publishing_without_a_room_is_a_quiet_no() -> None:
    agent = _agent_with(1)
    agent.room = None

    assert await agent.publish_tool_inventory() is False


@pytest.mark.asyncio
async def test_a_tools_update_reports_the_new_inventory() -> None:
    agent = _agent_with(2)
    agent.update_tools = AsyncMock()
    agent.publish_tool_inventory = AsyncMock(return_value=True)

    await config_handler.update_tools(agent, _client(3, prefix="fresh_"))

    agent.update_tools.assert_awaited_once()
    agent.publish_tool_inventory.assert_awaited_once()


def test_the_client_names_keep_the_order_the_app_sent() -> None:
    agent = _agent_with(0)
    agent.client_tools.registered_tools = [
        {"name": "second"},
        {"description": "no name"},
        {"name": "first"},
        {"name": ""},
    ]

    assert agent._client_tool_names_in_order() == ["second", "first"]
