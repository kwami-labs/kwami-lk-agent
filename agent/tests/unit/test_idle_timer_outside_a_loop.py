"""`_cancel_idle_timer` when there is no running event loop.

The method is synchronous and guards `asyncio.current_task()` with a
`RuntimeError` handler, because that call raises when no loop is running. In
production it is only ever reached from async code, so the guard looked
unreachable and carried a `# pragma: no cover`. It is not unreachable -- it is
just a branch our own composition happens not to take, which is exactly the
case the coverage policy says to test rather than exclude.

Worth keeping beyond the coverage, for two reasons. It lets teardown run from a
synchronous context (an atexit hook, a shutdown callback whose loop has already
closed) without raising and abandoning a cloud browser that keeps billing. And
the `running is timer` comparison it feeds matters on the timer's own path:
`_idle_timeout` calls `close()`, `close()` cancels the idle timer, and the timer
*is* the task running it -- so without that check a CancelledError lands inside
`close()` and can skip `stop_browser`, leaving a metered browser alive.
"""

from __future__ import annotations

from src.browser.browser_session import CloudBrowserSession


class FakeTimer:
    """Only what `_cancel_idle_timer` touches."""

    def __init__(self, done: bool = False) -> None:
        self._done = done
        self.cancelled = False

    def done(self) -> bool:
        return self._done

    def cancel(self) -> None:
        self.cancelled = True


def _session_with(timer: object) -> CloudBrowserSession:
    """Bypass __init__: this method reads one attribute and nothing else."""
    session = object.__new__(CloudBrowserSession)
    session._idle_timer = timer  # type: ignore[attr-defined]
    return session


def test_a_pending_timer_is_cancelled_with_no_loop_running() -> None:
    """Sync test, so `asyncio.current_task()` raises RuntimeError and the guard
    has to treat it as "no task is running" rather than propagate."""
    timer = FakeTimer()
    session = _session_with(timer)

    session._cancel_idle_timer()

    assert timer.cancelled is True
    assert session._idle_timer is None


def test_an_already_finished_timer_is_just_dropped() -> None:
    timer = FakeTimer(done=True)
    session = _session_with(timer)

    session._cancel_idle_timer()

    assert timer.cancelled is False
    assert session._idle_timer is None


def test_no_timer_at_all_is_harmless() -> None:
    session = _session_with(None)

    session._cancel_idle_timer()

    assert session._idle_timer is None
