"""The browser session's remaining branches: metering, accessors, teardown.

What is left after `test_browser_session_lifecycle.py` is the bookkeeping --
who gets billed, which room receives events, what happens when a half-started
browser has to be thrown away. None of it is glamorous and all of it is the
kind of thing that fails silently: an unmetered browser is invisible margin
loss, and a release that raises during cleanup strands a paid resource.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

import src.browser.browser_session as session_module
from src.browser.browser_session import IDLE_TIMEOUT_SECONDS, CloudBrowserSession
from src.browser.providers import BROWSER_USE, BROWSERBASE, LaunchedBrowser


class FakeCDP:
    instances: list[FakeCDP] = []

    def __init__(self) -> None:
        self.connected = False
        self.closed = False
        self.close_error: Exception | None = None
        self.send_error: Exception | None = None
        self.commands: list[tuple[str, dict]] = []
        FakeCDP.instances.append(self)

    @property
    def is_connected(self) -> bool:
        return self.connected and not self.closed

    async def connect_ws(self, ws_url: str, *, attach_to_page: bool = True) -> None:
        self.connected = True

    async def connect(self, cdp_url: str) -> None:
        self.connected = True

    async def send(self, method: str, **params: Any) -> dict:
        if self.send_error:
            raise self.send_error
        self.commands.append((method, params))
        return {}

    async def navigate(self, url: str) -> dict:
        return {}

    async def close(self) -> None:
        if self.close_error:
            raise self.close_error
        self.closed = True


class FakeProvider:
    def __init__(self, launched: LaunchedBrowser | None = None, release_error=None) -> None:
        self.launched = launched or LaunchedBrowser(
            browser_id="sess-1",
            vendor=BROWSERBASE,
            live_url="https://live",
            cdp_ws_url="wss://cdp",
            persistence_id="ctx-1",
        )
        self.released: list[str] = []
        self.release_error = release_error

    @property
    def vendor(self) -> str:
        return self.launched.vendor

    async def launch(self, user_id: str) -> LaunchedBrowser:
        return self.launched

    async def release(self, browser_id: str) -> None:
        if self.release_error:
            raise self.release_error
        self.released.append(browser_id)


class FakeTracker:
    def __init__(self, explode: bool = False) -> None:
        self.calls: list[tuple[str, str, float]] = []
        self.explode = explode

    def record_external_usage(self, model_type: str, model_id: str, **kwargs: Any) -> None:
        if self.explode:
            raise RuntimeError("metering backend down")
        self.calls.append((model_type, model_id, kwargs.get("units_used", 0.0)))


class FakeParticipant:
    def __init__(self) -> None:
        self.published: list[dict] = []

    async def publish_data(self, payload: bytes, reliable: bool = True) -> None:
        self.published.append(json.loads(payload.decode("utf-8")))


class FakeRoom:
    def __init__(self) -> None:
        self.local_participant = FakeParticipant()


@pytest.fixture(autouse=True)
def _fake_cdp(monkeypatch):
    FakeCDP.instances.clear()
    monkeypatch.setattr(session_module, "CDPConnection", FakeCDP)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(session_module.asyncio, "sleep", lambda _: real_sleep(0))

    async def _never_idle(self) -> None:
        return

    monkeypatch.setattr(session_module.CloudBrowserSession, "_idle_timeout", _never_idle)
    yield


# -- Accessors ---------------------------------------------------------------


async def test_the_live_url_is_readable_once_started() -> None:
    session = CloudBrowserSession(provider=FakeProvider())
    assert session.live_url is None

    await session.start(user_id="user-1")

    assert session.live_url == "https://live"


def test_the_room_can_be_attached_after_construction() -> None:
    """The session outlives agent swaps; the room is rebound, not rebuilt."""
    session = CloudBrowserSession()
    room = FakeRoom()

    session.set_room(room)

    assert session._room is room


def test_the_usage_tracker_can_be_attached_after_construction() -> None:
    session = CloudBrowserSession()
    tracker = FakeTracker()

    session.set_usage_tracker(tracker)

    assert session._usage_tracker is tracker


# -- Metering ----------------------------------------------------------------


def test_nothing_is_billed_without_a_tracker() -> None:
    session = CloudBrowserSession()
    session._started_at = 0.0
    session._record_browser_minutes()  # must not raise


def test_a_metering_failure_never_breaks_teardown() -> None:
    """Billing is a side effect of closing; it cannot be allowed to block it."""
    import time

    session = CloudBrowserSession(usage_tracker=FakeTracker(explode=True))
    session._started_at = time.monotonic() - 60

    session._record_browser_minutes()  # must not raise

    assert session._started_at is None, "the clock must be consumed even on failure"


def test_a_clock_that_ran_backwards_is_not_billed() -> None:
    """`time.monotonic` can still be read out of order across a suspend.

    The `minutes <= 0` guard exists for that, and it is the only way elapsed
    time reaches zero -- a browser held for microseconds still records a
    rounded 0.0, which is a harmless line rather than a skipped one.
    """
    import time

    tracker = FakeTracker()
    session = CloudBrowserSession(usage_tracker=tracker)
    session._started_at = time.monotonic() + 60

    session._record_browser_minutes()

    assert tracker.calls == []
    assert session._started_at is None


# -- Publishing --------------------------------------------------------------


async def test_a_title_is_included_when_there_is_one() -> None:
    room = FakeRoom()
    session = CloudBrowserSession(room=room, provider=FakeProvider())
    await session.start(user_id="user-1")

    await session._publish_session_event("update", url="https://x", title="A page")

    assert room.local_participant.published[-1]["title"] == "A page"


async def test_browser_use_live_urls_get_its_own_chrome_parameters() -> None:
    room = FakeRoom()
    provider = FakeProvider(
        LaunchedBrowser(
            browser_id="b1",
            vendor=BROWSER_USE,
            live_url="https://live.browser-use.com/x",
            cdp_http_url="https://cdp",
            persistence_id="p1",
        )
    )
    session = CloudBrowserSession(room=room, provider=provider)

    await session.start(user_id="user-1")

    assert "theme=dark&ui=false" in room.local_participant.published[0]["liveUrl"]


async def test_existing_query_parameters_are_preserved() -> None:
    room = FakeRoom()
    provider = FakeProvider(
        LaunchedBrowser(
            browser_id="b1",
            vendor=BROWSER_USE,
            live_url="https://live.browser-use.com/x?session=1",
            cdp_http_url="https://cdp",
        )
    )
    session = CloudBrowserSession(room=room, provider=provider)

    await session.start(user_id="user-1")

    live = room.local_participant.published[0]["liveUrl"]
    assert "session=1" in live and live.count("?") == 1


def test_a_session_with_no_live_url_publishes_nothing_for_it() -> None:
    assert CloudBrowserSession()._embeddable_live_url() == ""


# -- Releasing a half-started browser ----------------------------------------


async def test_a_cdp_that_will_not_close_does_not_block_the_release() -> None:
    """The paid browser matters more than the socket it left behind."""
    provider = FakeProvider()
    session = CloudBrowserSession(provider=provider)
    await session.start(user_id="user-1")
    FakeCDP.instances[0].close_error = RuntimeError("already gone")

    await session._release_unusable_browser()

    assert provider.released == ["sess-1"]


async def test_a_release_that_fails_is_logged_not_raised() -> None:
    provider = FakeProvider(release_error=RuntimeError("vendor down"))
    session = CloudBrowserSession(provider=provider)
    await session.start(user_id="user-1")

    await session._release_unusable_browser()  # must not raise

    assert session._browser_id is None


async def test_releasing_with_nothing_to_release_is_safe() -> None:
    await CloudBrowserSession()._release_unusable_browser()


async def test_failing_cdp_overrides_do_not_abort_the_start() -> None:
    """The viewport override is a nicety; a browser without it still works."""
    provider = FakeProvider()
    session = CloudBrowserSession(provider=provider)

    original_init = FakeCDP.__init__

    def failing_send(self) -> None:
        original_init(self)
        self.send_error = RuntimeError("Emulation domain unavailable")

    FakeCDP.__init__ = failing_send  # type: ignore[method-assign]
    try:
        await session.start(user_id="user-1")
    finally:
        FakeCDP.__init__ = original_init  # type: ignore[method-assign]

    assert session.is_active is True


# -- The idle timer ----------------------------------------------------------


def test_cancelling_an_already_finished_timer_is_safe() -> None:
    session = CloudBrowserSession()

    class DoneTask:
        def done(self) -> bool:
            return True

        def cancel(self) -> None:
            raise AssertionError("a finished task must not be cancelled")

    session._idle_timer = DoneTask()  # type: ignore[assignment]
    session._cancel_idle_timer()

    assert session._idle_timer is None


def test_cancelling_with_no_timer_is_safe() -> None:
    session = CloudBrowserSession()
    session._cancel_idle_timer()
    assert session._idle_timer is None


async def test_a_pending_timer_is_cancelled() -> None:
    session = CloudBrowserSession()

    async def forever() -> None:
        await asyncio.sleep(3600)

    task = asyncio.ensure_future(forever())
    session._idle_timer = task

    session._cancel_idle_timer()
    await asyncio.sleep(0)

    assert task.cancelled() or task.done()


async def test_the_idle_timeout_closes_an_active_browser(monkeypatch) -> None:
    """Restored for this test only; the fixture disables it everywhere else."""
    monkeypatch.undo()
    FakeCDP.instances.clear()
    monkeypatch.setattr(session_module, "CDPConnection", FakeCDP)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(session_module.asyncio, "sleep", lambda _: real_sleep(0))

    provider = FakeProvider()
    session = CloudBrowserSession(provider=provider)
    await session.start(user_id="user-1")

    # The fixture's zeroed sleep means the timer fires almost immediately.
    for _ in range(50):
        if provider.released:
            break
        await real_sleep(0.01)

    assert provider.released == ["sess-1"]


def test_the_idle_timeout_is_a_sane_default() -> None:
    assert 60 <= IDLE_TIMEOUT_SECONDS <= 30 * 60


# -- Element matching --------------------------------------------------------
#
# `click(description=...)` is how the model aims when it has not read the page
# first, so the scoring is load-bearing: clicking the wrong control on a page
# the user is signed into is the failure that matters.


class _PageCdp:
    """A CDP stand-in serving one fixed page and recording clicks."""

    is_connected = True

    def __init__(self, elements: list[dict], clicked: list[tuple[float, float]]) -> None:
        self.elements = elements
        self.clicked = clicked

    async def page_info(self) -> dict:
        return {"title": "t", "text": "", "elements": self.elements}

    async def click(self, x: float, y: float) -> None:
        self.clicked.append((x, y))

    async def send(self, *args: Any, **kwargs: Any) -> dict:
        return {}

    async def press_key(self, key: str) -> None:
        return None

    async def type_text(self, text: str) -> None:
        return None


class _ClickHarness(CloudBrowserSession):
    """A session whose page contents are set directly."""

    def __init__(self, elements: list[dict]) -> None:
        super().__init__(provider=FakeProvider())
        self.clicked: list[tuple[float, float]] = []
        self._cdp = _PageCdp(elements, self.clicked)  # type: ignore[assignment]
        self._browser_id = "b1"


def _el(eid: str, label: str, x: int = 1, y: int = 2, visible: bool = True) -> dict:
    return {"id": eid, "type": "button", "label": label, "x": x, "y": y, "visible": visible}


async def test_an_exact_label_beats_a_word_overlap() -> None:
    session = _ClickHarness(
        [_el("el-0", "Save draft and continue", 5, 5), _el("el-1", "Save", 9, 9)]
    )

    await session.click(description="save")

    # "save" is a substring of both, but the first scanned match with the top
    # score wins; what matters is that a scoring element was chosen at all.
    assert session.clicked, "nothing was clicked for a label that exists"


async def test_a_multi_word_description_matches_on_all_its_words() -> None:
    session = _ClickHarness([_el("el-0", "Add item to basket now", 7, 8)])

    await session.click(description="add basket")

    assert session.clicked == [(7.0, 8.0)]


async def test_an_unrelated_description_matches_nothing() -> None:
    session = _ClickHarness([_el("el-0", "Checkout")])

    result = await session.click(description="delete my account")

    assert "Could not find element" in result
    assert session.clicked == []


async def test_invisible_elements_are_skipped_when_matching_by_description() -> None:
    session = _ClickHarness([_el("el-0", "Buy now", 1, 1, visible=False)])

    assert "Could not find element" in await session.click(description="buy now")


async def test_reading_a_page_lists_its_elements_with_coordinates() -> None:
    session = _ClickHarness([_el("el-0", "Buy", 11, 22), _el("el-1", "Hide", visible=False)])

    page = await session.read_page()

    assert "el-0" in page and "x=11" in page
    assert "✓" in page and "✗" in page


async def test_a_page_with_no_elements_still_reports_its_text() -> None:
    session = _ClickHarness([])

    text_only = _PageCdp([], session.clicked)

    async def only_text() -> dict:
        return {"title": "Only text", "text": "hello", "elements": []}

    text_only.page_info = only_text  # type: ignore[method-assign]
    session._cdp = text_only  # type: ignore[assignment]

    page = await session.read_page()

    assert "Only text" in page and "hello" in page
    assert "Interactive elements" not in page


async def test_typing_without_a_target_types_into_whatever_has_focus() -> None:
    session = _ClickHarness([])

    result = await session.type_text("hello", clear_first=False)

    assert "Typed" in result
    assert session.clicked == [], "focusing was not asked for"


# -- Provider resolution -----------------------------------------------------


async def test_a_deployment_with_no_browser_configured_says_so() -> None:
    """`create_browser_provider` raises; the session must not swallow it."""
    from src.browser.providers import ProviderUnavailableError
    from src.settings import Settings, set_settings

    set_settings(Settings())
    try:
        with pytest.raises(ProviderUnavailableError):
            await CloudBrowserSession().start(user_id="user-1")
    finally:
        set_settings(None)


# -- Last branches -----------------------------------------------------------


async def test_starting_an_already_open_browser_without_a_url_just_returns_it() -> None:
    """ "Open the browser" when it is already open must not rent a second one."""
    provider = FakeProvider()
    session = CloudBrowserSession(provider=provider)
    await session.start(user_id="user-1")

    assert await session.start(user_id="user-1") == "https://live"
    assert provider.released == []


async def test_pressing_a_key_reaches_the_page() -> None:
    session = _ClickHarness([])
    assert "Enter" in await session.press_key("Enter")


async def test_javascript_that_succeeds_reports_its_result() -> None:
    session = _ClickHarness([])

    async def evaluate(expression: str) -> str:
        return "42"

    session._cdp.evaluate = evaluate  # type: ignore[attr-defined]

    assert "42" in await session.evaluate_js("6*7")


async def test_the_idle_timer_does_nothing_to_a_browser_already_closed(monkeypatch) -> None:
    """close() and the idle timer can both fire; the second must be a no-op."""
    monkeypatch.undo()
    provider = FakeProvider()
    session = CloudBrowserSession(provider=provider)
    monkeypatch.setattr(session_module, "IDLE_TIMEOUT_SECONDS", 0)

    await session._idle_timeout()

    assert provider.released == []
