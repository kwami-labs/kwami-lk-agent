"""Contract tests: what this codebase assumes about `livekit-agents`.

These are the tests that would have caught the production bugs. Every
assertion here pins an assumption `src/agent.py` makes about the installed
framework. When the SDK moves, these go red instead of the agent going quiet.
"""

from __future__ import annotations

import inspect

import pytest
from livekit.agents import Agent, llm
from livekit.agents.voice.agent import find_function_tools

from src.agent import KwamiAgent

# Hooks the framework actually dispatches. Verified against
# livekit/agents/voice/agent_activity.py, which calls on_enter()/on_exit() with
# no arguments and on_user_turn_completed(turn_ctx, new_message).
FRAMEWORK_DISPATCHED_HOOKS = {"on_enter", "on_exit", "on_user_turn_completed"}


def _overridden_hooks() -> set[str]:
    """Every `on_*` coroutine KwamiAgent defines in its own body."""
    return {
        name
        for name, value in vars(KwamiAgent).items()
        if name.startswith("on_") and inspect.iscoroutinefunction(value)
    }


def test_every_overridden_hook_exists_on_the_base_class() -> None:
    """An `on_*` method the framework never calls is dead code, not a hook.

    `on_agent_turn_completed` was defined here for months and is the only
    caller of the memory layer's `add_exchange`, so assistant turns were never
    persisted.
    """
    unknown = _overridden_hooks() - FRAMEWORK_DISPATCHED_HOOKS
    assert not unknown, (
        f"KwamiAgent defines {sorted(unknown)}, which livekit-agents never dispatches. "
        f"Dispatched hooks are {sorted(FRAMEWORK_DISPATCHED_HOOKS)}."
    )


def test_framework_hooks_still_exist_on_the_base_class() -> None:
    """Guards the other direction: the SDK removing a hook we rely on."""
    for hook in FRAMEWORK_DISPATCHED_HOOKS:
        assert hasattr(Agent, hook), f"livekit.agents.Agent no longer defines {hook}"


@pytest.mark.parametrize("hook", sorted(FRAMEWORK_DISPATCHED_HOOKS))
def test_overridden_hook_signatures_match_the_base_class(hook: str) -> None:
    """A hook with extra parameters is never called the way its body expects.

    `on_enter(self, room=None)` is dispatched as `on_enter()`, so `room` was
    always None: the duplicate-agent guard never ran and `self.room` was
    clobbered on every agent entry.
    """
    if hook not in vars(KwamiAgent):
        pytest.skip(f"KwamiAgent does not override {hook}")

    ours = inspect.signature(getattr(KwamiAgent, hook))
    theirs = inspect.signature(getattr(Agent, hook))
    assert list(ours.parameters) == list(theirs.parameters), (
        f"KwamiAgent.{hook}{ours} does not match Agent.{hook}{theirs}; "
        "the framework calls it with the base signature."
    )


def test_chat_message_content_is_a_list_and_text_content_is_the_accessor() -> None:
    """Pins the shape that broke memory: `content` is a list, not a str."""
    message = llm.ChatMessage(type="message", role="user", content=["I live in Barcelona"])
    assert isinstance(message.content, list)
    assert hasattr(message, "text_content")
    assert message.text_content == "I live in Barcelona"


def test_extract_message_content_returns_the_utterance() -> None:
    """The agent must store what the user said, not a pydantic repr.

    With `content` being a list, the str-only branch in
    `_extract_message_content` falls through to `str(message)` -- and the
    `startswith("<")` guard does not fire for pydantic v2 -- so Zep was being
    fed `id='item_...' type='message' role='user' content=[...]`.
    """
    spoken = "I live in Barcelona and I love hiking"
    message = llm.ChatMessage(type="message", role="user", content=[spoken])

    extracted = KwamiAgent._extract_message_content(None, message)

    assert extracted == spoken, (
        f"extracted a serialized object instead of the utterance: {extracted!r}"
    )


def _declared_tool_names() -> set[str]:
    """Every `@function_tool` method declared on the agent's own mixins.

    Derived rather than hard-coded. The count used to be written into the
    assertion as `22`, which made adding a tool -- the whole point of the
    mixins -- look like a contract violation, and told the next person to bump
    a number rather than to check that discovery still worked. What this test
    is actually defending is that nothing *drops out* of discovery, so the
    expectation is computed from the source of truth.
    """
    from livekit.agents import function_tool as _function_tool  # noqa: F401

    names: set[str] = set()
    for klass in KwamiAgent.__mro__:
        if klass.__module__.split(".")[0] != "src":
            continue
        for attr_name, attr in vars(klass).items():
            if hasattr(attr, "__livekit_tool_info") or hasattr(attr, "info"):
                names.add(attr_name)
    return names


def test_builtin_tools_are_discovered_by_the_framework() -> None:
    """Every `@function_tool` method on the agent's mixins must reach the LLM."""
    # Discover on the class: on an instance, inspect.getmembers evaluates
    # `realtime_llm_session`, which raises outside a running activity.
    discovered = {tool.info.name for tool in find_function_tools(KwamiAgent)}

    declared = _declared_tool_names()
    assert declared, "no @function_tool methods found; the detection above is stale"
    missing = declared - discovered
    assert not missing, f"declared but not discoverable: {sorted(missing)}"

    # Spot-check the ones a `tools` config update was silently deleting, plus
    # the self-service switches that make a mid-conversation model change
    # possible at all.
    for expected in (
        "web_search",
        "product_search",
        "navigate_to",
        "remember_fact",
        "change_ai_model",
        "switch_pipeline_mode",
        "change_realtime_voice",
    ):
        assert expected in discovered, f"{expected} is no longer discoverable"


def test_agent_tools_property_includes_the_builtins() -> None:
    """`Agent.tools` is what the running activity reads; built-ins must be in it.

    This is the invariant a `tools` config update violates by assigning
    `agent._tools = <client tools only>`.
    """
    agent = KwamiAgent()
    names = {tool.info.name for tool in agent.tools}
    assert "web_search" in names
    assert "navigate_to" in names


def test_update_tools_is_the_supported_way_to_change_tools() -> None:
    """Assigning `_tools` bypasses chat-context and realtime propagation.

    `Agent.update_tools` is a coroutine for a reason: it re-tools `_chat_ctx`
    and pushes the new set to a live realtime session.
    """
    assert inspect.iscoroutinefunction(Agent.update_tools), (
        "Agent.update_tools is no longer a coroutine; the reconfiguration path needs revisiting"
    )


def test_conversation_item_added_is_a_real_session_event() -> None:
    """The replacement hook for capturing assistant turns."""
    from livekit.agents.voice import events

    assert "conversation_item_added" in events.EventTypes.__args__


def test_remote_participant_has_no_is_connected_attribute() -> None:
    """The duplicate-agent guard used to read `participant.is_connected`.

    `rtc.RemoteParticipant` has never exposed it, so the guard raised
    AttributeError precisely when a second agent was in the room -- the one
    situation it existed to handle. Membership of `room.remote_participants`
    already means connected.
    """
    from livekit import rtc

    assert not hasattr(rtc.RemoteParticipant, "is_connected"), (
        "rtc.RemoteParticipant now has is_connected -- src/utils/room.py can use it directly"
    )
    assert hasattr(rtc.RemoteParticipant, "disconnect_reason")


def test_participant_kind_enum_uses_fully_qualified_names() -> None:
    """`ParticipantKind.AGENT` does not exist -- reading it raises.

    It is a protobuf enum wrapper, so only the fully-qualified member names
    resolve. Identity resolution and the duplicate-agent guard both compare
    against this enum, and both crash if the short name is used.
    """
    from livekit import rtc

    assert not hasattr(rtc.ParticipantKind, "AGENT")
    assert rtc.ParticipantKind.PARTICIPANT_KIND_AGENT is not None
    assert rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD is not None


async def test_duplicate_guard_survives_a_roomful_of_agents() -> None:
    """Regression: this path raised AttributeError instead of deciding."""
    from src.utils.room import should_disconnect_as_duplicate

    class FakeParticipant:
        def __init__(self, identity: str) -> None:
            self.identity = identity
            self.attributes: dict[str, str] = {}
            from livekit import rtc

            self.kind = rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
            self.metadata = ""
            self.disconnect_reason = None

    class FakeRoom:
        def __init__(self, participants: list[FakeParticipant]) -> None:
            self.remote_participants = {p.identity: p for p in participants}
            self.name = "test-room"

    room = FakeRoom([FakeParticipant("agent-AAA")])

    # Must return a decision, not raise.
    result = await should_disconnect_as_duplicate(room, "agent-ZZZ", check_delays=[0])
    assert isinstance(result, bool)
