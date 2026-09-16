"""Cloud browser minutes are billable and were never metered.

Browser Use bills per minute, with a US proxy hard-coded on. `record_external_usage`
covered Tavily, Microlink, SerpApi and Zep but not the most expensive resource
the agent can hold, so the margin loss was invisible.
"""

from __future__ import annotations

import time

from src.browser.browser_session import CloudBrowserSession


class FakeTracker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float]] = []

    def record_external_usage(
        self, model_type: str, model_id: str, *, units_used: float = 1.0, request_count: int = 1
    ) -> None:
        self.calls.append((model_type, model_id, units_used))


def _session(tracker: FakeTracker | None = None) -> CloudBrowserSession:
    return CloudBrowserSession(room=None, usage_tracker=tracker)


def test_elapsed_minutes_are_billed_once_released() -> None:
    tracker = FakeTracker()
    session = _session(tracker)
    session._started_at = time.monotonic() - 90  # 1.5 minutes ago

    session._record_browser_minutes()

    assert len(tracker.calls) == 1
    model_type, model_id, units = tracker.calls[0]
    assert model_type == "browser"
    assert model_id == "browser_use/cloud"
    assert 1.4 < units < 1.6, f"expected ~1.5 minutes, got {units}"


def test_a_browser_that_never_started_is_not_billed() -> None:
    tracker = FakeTracker()
    _session(tracker)._record_browser_minutes()
    assert tracker.calls == []


def test_minutes_are_billed_only_once() -> None:
    """close() and the idle-release path can both run; the clock is consumed."""
    tracker = FakeTracker()
    session = _session(tracker)
    session._started_at = time.monotonic() - 60

    session._record_browser_minutes()
    session._record_browser_minutes()

    assert len(tracker.calls) == 1


def test_metering_without_a_tracker_is_harmless() -> None:
    session = _session(None)
    session._started_at = time.monotonic() - 60
    session._record_browser_minutes()  # must not raise
    assert session._started_at is None


def test_a_failing_tracker_never_breaks_teardown() -> None:
    class Exploding:
        def record_external_usage(self, *a, **kw):
            raise RuntimeError("tracker down")

    session = CloudBrowserSession(room=None, usage_tracker=Exploding())
    session._started_at = time.monotonic() - 60
    session._record_browser_minutes()  # must swallow
