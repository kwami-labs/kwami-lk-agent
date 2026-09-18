"""Session teardown: the paths that only run when something has gone wrong.

`test_session_handoff.py` covers the happy swap. This covers the rest of
`SessionState` -- cleanup, provider closing, and the usage report -- where
almost every branch is an error path, and where the cost of getting one wrong
is invisible at runtime:

* teardown runs inside a LiveKit shutdown callback with a ~10s process budget,
  so one exception that escapes takes the rest of the teardown with it. A
  browser left running keeps billing by the minute, and the usage report runs
  last, so anything that raises before it drops the session's revenue in
  silence.
* the reporter signals failure by *returning* falsy rather than raising, which
  is the kind of thing that reads as success unless something checks.

So the assertions here are mostly "the next step still happened", which is the
property that actually matters.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from src.agent import KwamiAgent
from src.domain import KwamiConfig
from src.session import AGENT_RELEASE_POLL_SECONDS, SessionState


class Recorder:
    """A closable that records, and optionally refuses."""

    def __init__(self, *, fail: bool = False, sync: bool = False) -> None:
        self.closed = False
        self._fail = fail
        self._sync = sync
        if sync:
            self.close = self._close_sync  # type: ignore[assignment]
        else:
            self.aclose = self._aclose  # type: ignore[assignment]

    async def _aclose(self) -> None:
        if self._fail:
            raise RuntimeError("provider refused to close")
        self.closed = True

    def _close_sync(self) -> None:
        if self._fail:
            raise RuntimeError("provider refused to close")
        self.closed = True


class FakeMemory:
    def __init__(self, *, fail: bool = False) -> None:
        self.closed = False
        self.tracker: Any = None
        self._fail = fail

    async def close(self) -> None:
        if self._fail:
            raise RuntimeError("zep refused to close")
        self.closed = True

    def set_usage_tracker(self, tracker: Any) -> None:
        self.tracker = tracker


class FakeBrowser:
    def __init__(self, *, fail: bool = False) -> None:
        self.is_active = True
        self.closed = False
        self._fail = fail

    async def close(self) -> None:
        if self._fail:
            raise RuntimeError("browser refused to close")
        self.closed = True
        self.is_active = False


class FakeReporter:
    """Stands in for `UsageReporter`, with the three ways it can end."""

    def __init__(self, *, result: Any = True, raises: Exception | None = None, hang: bool = False):
        self.calls: list[dict[str, Any]] = []
        self._result = result
        self._raises = raises
        self._hang = hang

    async def report(self, *, user_id: str, session_id: str, tracker: Any) -> Any:
        self.calls.append({"user_id": user_id, "session_id": session_id})
        if self._raises is not None:
            raise self._raises
        if self._hang:
            await asyncio.sleep(3600)
        return self._result


def _agent(**kwargs: Any) -> KwamiAgent:
    return KwamiAgent(config=KwamiConfig(), **kwargs)


def _billable(state: SessionState) -> None:
    """Give the tracker something to report, so `_report_usage` proceeds."""
    state.usage_tracker.record_external_usage("tool", "test/tool", units_used=1.0)


# -- carrying resources onto the new agent ----------------------------------


async def test_the_new_agents_memory_gets_the_usage_tracker() -> None:
    """Memory bills against the tracker; a fresh agent's must be wired up."""
    memory = FakeMemory()
    new_agent = _agent(memory=memory)
    state = SessionState(current_agent=_agent())

    state.prepare_handoff(new_agent)
    await asyncio.gather(*state._cleanup_tasks, return_exceptions=True)

    assert memory.tracker is state.usage_tracker


async def test_an_uncopyable_conversation_does_not_break_the_swap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A swap that raises here would leave the session with no agent at all."""

    class Exploding:
        items = ["something"]

        def copy(self, **_: Any) -> Any:
            raise RuntimeError("cannot copy")

    old = _agent()
    old._chat_ctx = Exploding()  # type: ignore[assignment]
    new_agent = _agent()
    state = SessionState(current_agent=old)

    with caplog.at_level(logging.WARNING):
        state.prepare_handoff(new_agent)

    assert state.current_agent is new_agent, "the swap was abandoned"
    assert any("Could not carry the conversation" in r.getMessage() for r in caplog.records)
    await asyncio.gather(*state._cleanup_tasks, return_exceptions=True)


# -- waiting for the framework to release an agent --------------------------


async def test_a_never_released_agent_is_closed_anyway(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wait is bounded: a stuck activity must not leak provider sockets forever."""
    monkeypatch.setattr("src.session.AGENT_RELEASE_TIMEOUT_SECONDS", 0.05)

    tts = Recorder()
    agent = _agent(tts=tts)
    agent._activity = object()  # the framework never lets go
    state = SessionState(current_agent=agent)

    with caplog.at_level(logging.WARNING):
        await state._cleanup_agent_voice_pipeline(agent)

    assert tts.closed, "the wait never gave up, so the connection leaked"
    assert any("closing its pipeline anyway" in r.getMessage() for r in caplog.records)


async def test_the_wait_ends_as_soon_as_the_agent_is_released() -> None:
    tts = Recorder()
    agent = _agent(tts=tts)
    agent._activity = object()
    state = SessionState(current_agent=agent)

    task = asyncio.create_task(state._cleanup_agent_voice_pipeline(agent))
    await asyncio.sleep(AGENT_RELEASE_POLL_SECONDS * 2)
    assert not tts.closed, "closed before the framework released the agent"

    agent._activity = None
    await asyncio.wait_for(task, timeout=2.0)

    assert tts.closed


# -- closing the pipeline ---------------------------------------------------


async def test_a_provider_with_only_a_sync_close_is_still_closed() -> None:
    """Not every provider offers `aclose`."""
    tts = Recorder(sync=True)
    agent = _agent(tts=tts)
    state = SessionState()

    await state._cleanup_agent_voice_pipeline(agent, wait_for_release=False)

    assert tts.closed


async def test_one_provider_refusing_to_close_does_not_strand_the_others() -> None:
    """Teardown has a budget; one bad socket must not cost the rest."""
    bad, good = Recorder(fail=True), Recorder()
    agent = _agent(stt=bad, tts=good)
    state = SessionState()

    await state._cleanup_agent_voice_pipeline(agent, wait_for_release=False)

    assert good.closed, "a failure on one provider skipped the next"


async def test_a_memory_that_refuses_to_close_is_logged_not_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`_cleanup_memory` runs in a task; an escape would be an unretrieved error."""
    state = SessionState()

    with caplog.at_level(logging.WARNING):
        await state._cleanup_memory(FakeMemory(fail=True))

    assert any("Failed to close memory" in r.getMessage() for r in caplog.records)


# -- cleanup ----------------------------------------------------------------


async def test_cleanup_closes_the_browser_and_then_reports() -> None:
    browser = FakeBrowser()
    memory = FakeMemory()
    reporter = FakeReporter()

    state = SessionState(current_agent=_agent(memory=memory))
    state.browser_session = browser
    state.user_identity = "user-1"
    state.room_name = "room-1"
    state.usage_reporter = reporter  # type: ignore[assignment]
    _billable(state)

    await state.cleanup()

    assert browser.closed, "the browser kept billing"
    assert memory.closed
    assert reporter.calls, "usage was never reported"


async def test_a_browser_that_refuses_to_close_still_lets_usage_report(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The report runs last, so anything raising before it drops the revenue."""
    reporter = FakeReporter()
    state = SessionState(current_agent=_agent())
    state.browser_session = FakeBrowser(fail=True)
    state.user_identity = "user-1"
    state.room_name = "room-1"
    state.usage_reporter = reporter  # type: ignore[assignment]
    _billable(state)

    with caplog.at_level(logging.WARNING):
        await state.cleanup()

    assert reporter.calls, "a failed browser close cost the session its billing"
    assert any("Failed to close cloud browser" in r.getMessage() for r in caplog.records)


async def test_cleanup_recovers_the_identity_from_the_agent() -> None:
    """Without an identity the session's usage is silently never billed."""
    config = KwamiConfig(kwami_id="kwami_abc_1")
    reporter = FakeReporter()
    state = SessionState(current_agent=KwamiAgent(config=config))
    state.room_name = "room-1"
    state.usage_reporter = reporter  # type: ignore[assignment]
    _billable(state)

    await state.cleanup()

    assert state.user_identity == "kwami_abc_1"
    # Credits are keyed on the Supabase id embedded in the memory id.
    assert reporter.calls[0]["user_id"] == "abc"


async def test_cleanup_awaits_the_pending_cleanup_tasks() -> None:
    state = SessionState(current_agent=_agent())
    finished: list[str] = []

    async def work() -> None:
        await asyncio.sleep(0)
        finished.append("done")

    state._cleanup_tasks.append(asyncio.create_task(work()))

    await state.cleanup()

    assert finished == ["done"]
    assert state._cleanup_tasks == []


# -- the usage report -------------------------------------------------------


async def test_a_reporter_that_raises_does_not_break_teardown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state = SessionState()
    state.user_identity = "user-1"
    state.room_name = "room-1"
    state.usage_reporter = FakeReporter(raises=RuntimeError("credits API down"))  # type: ignore[assignment]
    _billable(state)

    with caplog.at_level(logging.ERROR):
        await state._report_usage()

    assert any("Failed to report usage" in r.getMessage() for r in caplog.records)


async def test_a_rejected_report_is_not_treated_as_success(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The reporter signals failure by returning falsy, not by raising.

    Without this branch the session's revenue was dropped in silence.
    """
    state = SessionState()
    state.user_identity = "user-1"
    state.room_name = "room-1"
    state.usage_reporter = FakeReporter(result=False)  # type: ignore[assignment]
    _billable(state)

    with caplog.at_level(logging.ERROR):
        await state._report_usage()

    assert any("rejected or dropped" in r.getMessage() for r in caplog.records)


async def test_a_hanging_report_is_bounded(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Teardown shares a ~10s worker budget; a slow API must not eat it."""
    monkeypatch.setattr("src.session.USAGE_REPORT_TIMEOUT_SECONDS", 0.05)

    state = SessionState()
    state.user_identity = "user-1"
    state.room_name = "room-1"
    state.usage_reporter = FakeReporter(hang=True)  # type: ignore[assignment]
    _billable(state)

    with caplog.at_level(logging.ERROR):
        await asyncio.wait_for(state._report_usage(), timeout=2.0)

    assert any("timed out" in r.getMessage() for r in caplog.records)


async def test_nothing_is_reported_without_usage() -> None:
    reporter = FakeReporter()
    state = SessionState()
    state.user_identity = "user-1"
    state.room_name = "room-1"
    state.usage_reporter = reporter  # type: ignore[assignment]

    await state._report_usage()

    assert reporter.calls == []


# -- the small accessors ----------------------------------------------------


def test_has_agent_and_get_agent_or_none() -> None:
    empty = SessionState()
    assert empty.has_agent is False
    assert empty.get_agent_or_none() is None

    agent = _agent()
    filled = SessionState(current_agent=agent)
    assert filled.has_agent is True
    assert filled.get_agent_or_none() is agent


def test_an_inactive_browser_is_not_offered_as_active() -> None:
    """Cleanup is gated on this; a closed browser must not look live."""
    browser = FakeBrowser()
    browser.is_active = False
    state = SessionState(browser_session=browser)

    assert state.active_browser_session is None
