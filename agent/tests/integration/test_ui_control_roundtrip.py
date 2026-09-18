"""Driving the app's UI from the agent, over the wire, with the app's real tools.

The capability tests prove the model is *told* about these tools, and the app's
own suite proves its handlers do the right thing once called. This is the join:
the agent registers the tool definitions `kwami-app` actually sends, the model
calls one, and the exact bytes that reach the data channel are inspected --
then a result is fed back the way the frontend replies.

Worth testing at this seam because the two halves are written in different
languages and neither side's suite can see the other. A rename of `toolCallId`,
a change to the `tool_call` envelope, or an argument that arrives JSON-encoded
when the client expects an object would pass both suites and break every UI
command in production.

The payloads here are the ones from `useWorkspaceAgentTools.ts` -- blob colours,
a renderer swap, a background preset, a profile switch -- so what is exercised
is the goal's "modify the blob, load a background, change profile", not a
stand-in shaped like it.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from src.agent import KwamiAgent
from src.domain import KwamiConfig
from tests.conftest import RecordingRoom

#: Lifted from useWorkspaceAgentTools.ts. These are the definitions the frontend
#: puts on the `config` message, in the shape `register_client_tools` parses.
APP_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "set_avatar_control",
        "description": "Control avatar UI settings across renderers.",
        "parameters": {"control": {"type": "string"}, "value": {}},
    },
    {
        "name": "set_workspace_renderer",
        "description": "Switch the visual renderer for the main Kwami avatar.",
        "parameters": {
            "renderer": {
                "type": "string",
                "enum": ["blob-xyz", "black-hole", "particles-face", "eye-iris"],
            }
        },
    },
    {
        "name": "apply_scene_preset",
        "description": "Set the scene background to one of the built-in presets.",
        "parameters": {"kind": {"type": "string"}, "name": {"type": "string"}},
    },
    {
        "name": "switch_kwami_profile",
        "description": "Switch to a different saved companion by name.",
        "parameters": {"name_or_id": {"type": "string"}, "confirm": {"type": "boolean"}},
    },
    {
        "name": "set_browser_panel",
        "description": "Move, resize, expand or dock the live browser panel.",
        "parameters": {"control": {"type": "string"}, "value": {}},
    },
    {
        "name": "set_ui_control",
        "description": "Primary tool for natural-language UI control.",
        "parameters": {
            "domain": {"type": "string"},
            "control": {"type": "string"},
            "value": {},
        },
    },
]


@pytest.fixture
def agent(room: RecordingRoom) -> KwamiAgent:
    config = KwamiConfig()
    config.tools = APP_TOOL_DEFINITIONS
    instance = KwamiAgent(config=config)
    instance.room = room
    return instance


def _tool(agent: KwamiAgent, name: str) -> Any:
    for tool in agent.client_tools.create_client_tools():
        if tool.info.name == name:
            return tool
    raise AssertionError(f"{name} was never registered")


async def _call(
    agent: KwamiAgent, room: RecordingRoom, tool_name: str, arguments: dict[str, Any]
) -> str:
    """Invoke a client tool and answer it the way the frontend does.

    Arguments go in a dict rather than as **kwargs: several of these tools take
    an argument literally called `name`, which would collide with this
    function's own parameter.
    """
    tool = _tool(agent, tool_name)
    before = len(room.published)
    task = asyncio.create_task(tool(raw_arguments=arguments, context=None))

    # The call is in flight until the client replies; wait for the publish.
    for _ in range(200):
        if len(room.published) > before:
            break
        await asyncio.sleep(0.005)
    assert len(room.published) > before, f"{tool_name} published nothing to the data channel"

    call_id = room.published[-1]["toolCallId"]
    agent.client_tools.handle_tool_result(call_id, json.dumps({"success": True}))
    return await task


# -- registration -----------------------------------------------------------


def test_the_apps_tools_become_callable_tools(agent: KwamiAgent) -> None:
    names = {tool.info.name for tool in agent.client_tools.create_client_tools()}
    for definition in APP_TOOL_DEFINITIONS:
        assert definition["name"] in names


def test_registering_ui_tools_does_not_displace_the_builtins(agent: KwamiAgent) -> None:
    """Assigning `_tools` directly used to delete every built-in for the session."""
    names = {tool.info.name for tool in agent.tools}

    assert "set_avatar_control" in names
    assert "web_search" in names
    assert "play_media" in names
    assert "change_ai_model" in names


# -- the wire envelope ------------------------------------------------------


async def test_modifying_the_blob_reaches_the_client(
    agent: KwamiAgent, room: RecordingRoom
) -> None:
    """ "Make your blob warmer" -- the goal's avatar case, over the wire."""
    result = await _call(
        agent,
        room,
        "set_avatar_control",
        {"control": "blobColors", "value": {"x": "#ff7a45", "y": "#ffb347", "z": "#ff3d71"}},
    )

    message = room.published[-1]
    assert message["type"] == "tool_call"
    assert message["function"]["name"] == "set_avatar_control"

    # Arguments travel JSON-encoded inside the envelope; the client parses them.
    arguments = json.loads(message["function"]["arguments"])
    assert arguments["control"] == "blobColors"
    assert arguments["value"] == {"x": "#ff7a45", "y": "#ffb347", "z": "#ff3d71"}
    assert json.loads(result)["success"] is True


async def test_switching_the_renderer_reaches_the_client(
    agent: KwamiAgent, room: RecordingRoom
) -> None:
    await _call(agent, room, "set_workspace_renderer", {"renderer": "eye-iris"})

    arguments = json.loads(room.published[-1]["function"]["arguments"])
    assert arguments["renderer"] == "eye-iris"


async def test_loading_a_background_reaches_the_client(
    agent: KwamiAgent, room: RecordingRoom
) -> None:
    """ "Put a waterfall behind you" -- by preset name, not a guessed URL."""
    await _call(agent, room, "apply_scene_preset", {"kind": "image", "name": "waterfall"})

    arguments = json.loads(room.published[-1]["function"]["arguments"])
    assert arguments == {"kind": "image", "name": "waterfall"}


async def test_changing_profile_reaches_the_client(agent: KwamiAgent, room: RecordingRoom) -> None:
    await _call(agent, room, "switch_kwami_profile", {"name_or_id": "Nova", "confirm": True})

    arguments = json.loads(room.published[-1]["function"]["arguments"])
    assert arguments["name_or_id"] == "Nova"
    assert arguments["confirm"] is True


async def test_moving_the_browser_panel_reaches_the_client(
    agent: KwamiAgent, room: RecordingRoom
) -> None:
    await _call(agent, room, "set_browser_panel", {"control": "layout", "value": "fullscreen"})

    arguments = json.loads(room.published[-1]["function"]["arguments"])
    assert arguments == {"control": "layout", "value": "fullscreen"}


async def test_the_generic_router_carries_a_nested_value(
    agent: KwamiAgent, room: RecordingRoom
) -> None:
    """set_ui_control's `value` is untyped; an object must survive the trip."""
    await _call(
        agent,
        room,
        "set_ui_control",
        {"domain": "avatar", "control": "blobSpikes", "value": {"x": 0.6, "y": 0.5, "z": 0.7}},
    )

    arguments = json.loads(room.published[-1]["function"]["arguments"])
    assert arguments["value"] == {"x": 0.6, "y": 0.5, "z": 0.7}


# -- the reply --------------------------------------------------------------


async def test_a_client_error_comes_back_as_an_error(
    agent: KwamiAgent, room: RecordingRoom
) -> None:
    tool = _tool(agent, "set_avatar_control")
    task = asyncio.create_task(tool(raw_arguments={"control": "nope"}, context=None))
    for _ in range(200):
        if room.published:
            break
        await asyncio.sleep(0.005)

    call_id = room.published[-1]["toolCallId"]
    agent.client_tools.handle_tool_result(call_id, None, error="Unknown avatar control")

    result = await task
    assert "Unknown avatar control" in result


async def test_each_call_gets_its_own_id(agent: KwamiAgent, room: RecordingRoom) -> None:
    """Two UI changes in one turn must not resolve against each other."""
    await _call(agent, room, "set_workspace_renderer", {"renderer": "blob-xyz"})
    first = room.published[-1]["toolCallId"]
    await _call(agent, room, "set_workspace_renderer", {"renderer": "black-hole"})
    second = room.published[-1]["toolCallId"]

    assert first != second


async def test_an_oversized_client_reply_is_capped(agent: KwamiAgent, room: RecordingRoom) -> None:
    """The reply lands verbatim in the LLM context; the client is not a size bound."""
    from src.tools.client import MAX_CLIENT_TOOL_RESULT_CHARS

    tool = _tool(agent, "set_ui_control")
    task = asyncio.create_task(tool(raw_arguments={"domain": "theme"}, context=None))
    for _ in range(200):
        if room.published:
            break
        await asyncio.sleep(0.005)

    call_id = room.published[-1]["toolCallId"]
    agent.client_tools.handle_tool_result(call_id, "x" * 100_000)

    result = await task
    assert len(result) < MAX_CLIENT_TOOL_RESULT_CHARS + 200


# -- the two halves agree ---------------------------------------------------


def test_the_registered_tools_are_the_ones_the_prompt_describes(agent: KwamiAgent) -> None:
    """Guidance is keyed on names; a mismatch here is a tool nobody can reach."""
    from src.domain.capabilities import available_capability_names

    registered = {tool.info.name for tool in agent.client_tools.create_client_tools()}
    enabled = available_capability_names(registered)

    for expected in ("avatar", "scene_presets", "profiles", "panel_layout", "ui_router"):
        assert expected in enabled, f"{expected} guidance is off despite its tool being live"

    prompt = agent._build_system_prompt()
    assert "eye-iris" in prompt
    assert "apply_scene_preset" in prompt
