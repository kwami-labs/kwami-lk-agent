"""A client tool whose name the model already has.

The framework builds the agent's tool list as client tools plus its own, and a
second tool of the same name is not ignored: `ToolContext` raises "duplicate
function name". The agent is built while a `config` is applied, so one user tool
called `get_current_time`, or a webhook and an MCP tool both called `weather`,
left the session on its placeholder persona with nothing on screen to say why.
"""

from __future__ import annotations

from src.agent import KwamiAgent
from src.domain import KwamiConfig
from src.tools.client import ClientToolManager


def _tool(name: str, description: str = "d") -> dict:
    return {"name": name, "description": description, "parameters": {}}


def test_a_client_tool_sent_twice_is_registered_once() -> None:
    manager = ClientToolManager(kwami_agent=None)

    manager.register_client_tools([_tool("weather", "first"), _tool("weather", "second")])

    assert manager.tool_count == 1
    assert manager.create_client_tools()[0].info.raw_schema["description"] == "first"


def test_a_later_registration_cannot_repeat_an_earlier_name() -> None:
    manager = ClientToolManager(kwami_agent=None)
    manager.register_client_tools([_tool("weather")])

    manager.register_client_tools([_tool("weather"), _tool("news")])

    assert [t["name"] for t in manager.registered_tools] == ["weather", "news"]


def test_the_nested_function_format_counts_as_the_same_name() -> None:
    manager = ClientToolManager(kwami_agent=None)

    manager.register_client_tools(
        [{"type": "function", "function": _tool("weather")}, _tool("weather")]
    )

    assert manager.tool_count == 1


def test_a_definition_that_is_not_an_object_is_skipped() -> None:
    manager = ClientToolManager(kwami_agent=None)

    manager.register_client_tools(["weather", None, _tool("news")])  # type: ignore[list-item]

    assert manager.tool_count == 1


def test_the_agent_builds_when_a_client_tool_takes_a_builtin_name() -> None:
    config = KwamiConfig()
    config.tools = [_tool("get_current_time"), _tool("set_theme"), _tool("set_theme")]

    agent = KwamiAgent(config=config)

    names = [tool.info.name for tool in agent.tools]
    assert names.count("get_current_time") == 1
    assert names.count("set_theme") == 1
    assert [t["name"] for t in agent.client_tools.registered_tools] == ["set_theme"]


def test_an_agent_stand_in_without_tools_reserves_nothing() -> None:
    class Bare:
        pass

    manager = ClientToolManager(kwami_agent=Bare())

    manager.register_client_tools([_tool("get_current_time")])

    assert manager.tool_count == 1


def test_a_stand_in_the_framework_cannot_inspect_reserves_nothing(monkeypatch) -> None:
    import src.tools.client as client

    def boom(_cls: type) -> list:
        raise TypeError("not an agent")

    monkeypatch.setattr(client, "find_function_tools", boom)
    manager = ClientToolManager(kwami_agent=object())

    manager.register_client_tools([_tool("weather")])

    assert manager.tool_count == 1
