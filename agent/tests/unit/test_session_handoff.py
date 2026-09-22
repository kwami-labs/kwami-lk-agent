"""What has to survive when one agent replaces another.

Every reconfiguration builds a *new* `KwamiAgent` -- a different voice, a
different model, a different pipeline. Four things live across that boundary
and each was, or would have been, lost:

* the conversation, which the framework does not copy;
* the paid cloud browser;
* in-flight client tool futures;
* the old agent's provider connections, which must be closed, but not while the
  framework is still draining it.

The fourth got worse with the tool-handoff path, where the session does not
swap until *after* the calling tool's output has been collected -- so the old
agent stays live for a while after its replacement has been prepared.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.agent import KwamiAgent
from src.domain import KwamiConfig
from src.session import AGENT_RELEASE_POLL_SECONDS, SessionState


class FakeSession:
    def __init__(self) -> None:
        self.agent: Any = None

    def update_agent(self, agent: Any) -> None:
        self.agent = agent


class ClosableProvider:
    """Stands in for an STT/LLM/TTS client, recording when it was closed."""

    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def _agent(**kwargs: Any) -> KwamiAgent:
    return KwamiAgent(config=KwamiConfig(), **kwargs)


async def _settle(state: SessionState) -> None:
    """Let the cleanup tasks prepare_handoff spawned actually run."""
    if state._cleanup_tasks:
        await asyncio.gather(*state._cleanup_tasks, return_exceptions=True)


# -- the conversation -------------------------------------------------------


async def test_the_conversation_survives_a_swap() -> None:
    old = _agent()
    old._chat_ctx.add_message(role="user", content="my name is Alex")
    old._chat_ctx.add_message(role="assistant", content="Hi Alex")
    state = SessionState(current_agent=old)

    new = _agent()
    state.prepare_handoff(new)

    assert [item.content for item in new.chat_ctx.items] == [
        ["my name is Alex"],
        ["Hi Alex"],
    ]
    await _settle(state)


async def test_an_empty_conversation_is_not_carried() -> None:
    state = SessionState(current_agent=_agent())
    new = _agent()

    state.prepare_handoff(new)

    assert list(new.chat_ctx.items) == []
    await _settle(state)


async def test_a_running_agents_context_is_not_overwritten() -> None:
    """The framework owns the context of a live activity."""
    old = _agent()
    old._chat_ctx.add_message(role="user", content="old")
    new = _agent()
    new._chat_ctx.add_message(role="user", content="already running")
    new._activity = object()  # pretend the framework started it

    state = SessionState(current_agent=old)
    state.prepare_handoff(new)

    assert [item.content for item in new.chat_ctx.items] == [["already running"]]
    await _settle(state)


# -- the browser and pending tool calls -------------------------------------


async def test_the_browser_moves_to_the_new_agent() -> None:
    """A browser left on a discarded agent keeps running, and keeps billing."""
    old = _agent()
    browser = object()
    old._browser_session = browser
    state = SessionState(current_agent=old)

    new = _agent()
    state.prepare_handoff(new)

    assert state.browser_session is browser
    assert new._browser_session is browser
    assert old._browser_session is None
    await _settle(state)


async def test_pending_tool_calls_move_to_the_new_agent() -> None:
    old = _agent()
    pending: asyncio.Future = asyncio.Future()
    old.client_tools.pending_calls["call-1"] = pending
    state = SessionState(current_agent=old)

    new = _agent()
    state.prepare_handoff(new)

    assert new.client_tools.pending_calls["call-1"] is pending
    assert old.client_tools.pending_calls == {}
    await _settle(state)


async def test_a_settled_tool_call_is_not_carried() -> None:
    old = _agent()
    done: asyncio.Future = asyncio.Future()
    done.set_result("already answered")
    old.client_tools.pending_calls["call-1"] = done
    state = SessionState(current_agent=old)

    new = _agent()
    state.prepare_handoff(new)

    assert "call-1" not in new.client_tools.pending_calls
    await _settle(state)


# -- provider cleanup ordering ----------------------------------------------


async def test_providers_are_not_closed_while_the_agent_is_still_live() -> None:
    """The tool-handoff path leaves the old agent running after the swap is prepared.

    `AgentActivity.aclose` sets `agent._activity = None` as its last act, so
    until that happens the framework may still be draining speech through the
    very TTS this cleanup is about to close.
    """
    tts = ClosableProvider()
    old = _agent(tts=tts)
    old._activity = object()  # the framework has not let go yet
    state = SessionState(current_agent=old)

    state.prepare_handoff(_agent())
    await asyncio.sleep(AGENT_RELEASE_POLL_SECONDS * 3)

    assert not tts.closed, "closed a provider out from under a running activity"

    old._activity = None  # the framework releases it
    await _settle(state)

    assert tts.closed, "the provider connection was leaked once the agent was released"


async def test_providers_are_closed_promptly_once_released() -> None:
    tts = ClosableProvider()
    old = _agent(tts=tts)
    state = SessionState(current_agent=old)

    state.prepare_handoff(_agent())
    await _settle(state)

    assert tts.closed


async def test_teardown_does_not_wait_for_a_release_that_is_not_coming() -> None:
    """At shutdown the same teardown is closing the activity.

    Waiting would spend the worker's ~10s budget on a release that only
    happens after this call returns, and the usage report -- which is revenue --
    runs last.
    """
    tts = ClosableProvider()
    agent = _agent(tts=tts)
    agent._activity = object()  # never released
    state = SessionState(current_agent=agent)

    await asyncio.wait_for(
        state._cleanup_agent_voice_pipeline(agent, wait_for_release=False),
        timeout=1.0,
    )

    assert tts.closed


# -- the session-level entry point ------------------------------------------


async def test_update_agent_still_installs_on_the_session() -> None:
    """`prepare_handoff` is a split, not a behaviour change, for this path."""
    state = SessionState(current_agent=_agent())
    session = FakeSession()
    new = _agent()

    state.update_agent(session, new)

    assert session.agent is new
    assert state.current_agent is new
    await _settle(state)


async def test_handing_off_to_the_same_agent_is_a_no_op() -> None:
    """Otherwise the agent's own providers would be scheduled for closing."""
    tts = ClosableProvider()
    agent = _agent(tts=tts)
    state = SessionState(current_agent=agent)

    state.prepare_handoff(agent)
    await _settle(state)

    assert not tts.closed
    assert state.current_agent is agent


@pytest.mark.parametrize("attribute", ["usage_tracker", "room"])
async def test_session_resources_are_wired_onto_the_new_agent(attribute: str) -> None:
    state = SessionState(current_agent=_agent())
    state.room = object()

    new = _agent()
    state.prepare_handoff(new)

    assert getattr(new, attribute) is not None
    await _settle(state)
