"""SessionState must not strand paid resources across an agent swap."""

from __future__ import annotations

import asyncio
from typing import Any

from src.session import SessionState


class FakeBrowser:
    def __init__(self, active: bool = True) -> None:
        self.is_active = active
        self.closed = False

    async def close(self) -> None:
        self.closed = True
        self.is_active = False


class FakeToolManager:
    def __init__(self) -> None:
        self.pending_calls: dict[str, asyncio.Future] = {}


class FakeAgent:
    def __init__(self) -> None:
        self._browser_session: Any = None
        self._memory = None
        self.client_tools = FakeToolManager()
        self.usage_tracker = None
        self.room = None
        self.kwami_config = type("Cfg", (), {"kwami_id": "kwami_abc"})()


class FakeSession:
    def __init__(self) -> None:
        self.agent = None

    def update_agent(self, agent: Any) -> None:
        self.agent = agent


async def test_browser_is_handed_to_the_new_agent_not_stranded() -> None:
    """A voice or LLM change must not orphan a running cloud browser.

    Agents are replaced wholesale on reconfiguration. The browser used to live
    only on the agent, so every swap left one running -- and billing per minute
    -- with nothing able to close it.
    """
    old_agent, new_agent = FakeAgent(), FakeAgent()
    browser = FakeBrowser()
    old_agent._browser_session = browser
    state = SessionState(current_agent=old_agent)

    state.update_agent(FakeSession(), new_agent)

    assert state.browser_session is browser
    assert new_agent._browser_session is browser
    assert old_agent._browser_session is None
    assert state.active_browser_session is browser


async def test_cleanup_closes_a_browser_opened_before_a_swap() -> None:
    """The leak this fixes: cleanup used to look only at the current agent."""
    old_agent, new_agent = FakeAgent(), FakeAgent()
    browser = FakeBrowser()
    old_agent._browser_session = browser
    state = SessionState(current_agent=old_agent, user_identity=None, room_name=None)

    state.update_agent(FakeSession(), new_agent)
    await state.cleanup()

    assert browser.closed, "cloud browser was left running after the session ended"


def test_active_browser_session_ignores_a_closed_browser() -> None:
    agent = FakeAgent()
    agent._browser_session = FakeBrowser(active=False)
    state = SessionState(current_agent=agent)

    assert state.active_browser_session is None


async def test_pending_tool_calls_move_to_the_new_agent() -> None:
    """An in-flight client tool call must survive a reconfiguration.

    Results are routed to `state.current_agent`; if the future stayed on the
    old manager the call was never resolved and the LLM blocked for its full
    30-second timeout.
    """
    loop = asyncio.get_running_loop()
    old_agent, new_agent = FakeAgent(), FakeAgent()
    unresolved = loop.create_future()
    resolved = loop.create_future()
    resolved.set_result("done")
    old_agent.client_tools.pending_calls = {"call-1": unresolved, "call-2": resolved}
    state = SessionState(current_agent=old_agent)

    state.update_agent(FakeSession(), new_agent)

    assert "call-1" in new_agent.client_tools.pending_calls
    assert "call-2" not in new_agent.client_tools.pending_calls, "already-done call moved"
    assert old_agent.client_tools.pending_calls == {}
    unresolved.cancel()


async def test_usage_report_timeout_does_not_take_down_cleanup() -> None:
    """A hung credits API must not cost us the rest of the shutdown."""

    class HangingReporter:
        async def report(self, **_: Any) -> bool:
            await asyncio.sleep(60)
            return True

    agent = FakeAgent()
    state = SessionState(
        current_agent=agent,
        user_identity="kwami_user123_abc",
        room_name="room-1",
    )
    state.usage_reporter = HangingReporter()  # type: ignore[assignment]
    state.usage_tracker.record_external_usage("search", "test/service", units_used=1.0)

    await asyncio.wait_for(state.cleanup(), timeout=15)
