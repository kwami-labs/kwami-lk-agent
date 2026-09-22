"""Cloud browser minutes are billable and were never metered.

Cloud browsers bill per minute. `record_external_usage` covered Tavily,
Microlink, SerpApi and Zep but not the most expensive resource the agent can
hold, so the margin loss was invisible.

The usage record names the vendor because the two of them price differently:
one bill line reading "browser/cloud" for both cannot be reconciled against
either invoice.
"""

from __future__ import annotations

import time

import pytest

from src.browser.browser_session import CloudBrowserSession
from src.browser.providers import BROWSER_USE, BROWSERBASE


class FakeTracker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float]] = []

    def record_external_usage(
        self, model_type: str, model_id: str, *, units_used: float = 1.0, request_count: int = 1
    ) -> None:
        self.calls.append((model_type, model_id, units_used))


def _session(tracker: FakeTracker | None = None, vendor: str = BROWSER_USE) -> CloudBrowserSession:
    session = CloudBrowserSession(room=None, usage_tracker=tracker)
    session._vendor = vendor
    return session


@pytest.mark.parametrize(
    ("vendor", "expected_id"),
    [(BROWSER_USE, "browser_use/cloud"), (BROWSERBASE, "browserbase/cloud")],
)
def test_elapsed_minutes_are_billed_to_the_vendor_that_ran_them(
    vendor: str, expected_id: str
) -> None:
    tracker = FakeTracker()
    session = _session(tracker, vendor=vendor)
    session._started_at = time.monotonic() - 90  # 1.5 minutes ago

    session._record_browser_minutes()

    assert len(tracker.calls) == 1
    model_type, model_id, units = tracker.calls[0]
    assert model_type == "browser"
    assert model_id == expected_id
    assert 1.4 < units < 1.6, f"expected ~1.5 minutes, got {units}"


def test_minutes_are_still_billed_when_the_vendor_is_unknown() -> None:
    """A browser released before `launch` returned still consumed wall clock."""
    tracker = FakeTracker()
    session = _session(tracker, vendor="")
    session._started_at = time.monotonic() - 60

    session._record_browser_minutes()

    assert [call[1] for call in tracker.calls] == ["browser/cloud"]


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
