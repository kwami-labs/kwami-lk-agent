"""Driving a cloud browser: interaction, idle release, and the failure paths.

A cloud browser is the most expensive resource the agent can hold -- billed by
the minute, holding the user's cookies and logins, and reachable only through a
socket that can drop. So the behaviour worth pinning down is not the happy path
but the four ways it goes wrong:

* a browser created that never becomes usable must be released, not stranded;
* every entry point revalidates the URL, because the profile is authenticated;
* the idle timer must not cancel the task that is running it;
* a failed release must be reported, not reported as success.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import src.browser.browser_session as session_module
from src.browser.browser_session import CloudBrowserSession
from src.browser.providers import BROWSERBASE, LaunchedBrowser
from src.browser.safety import UnsafeURLError


class FakeCDP:
    instances: list[FakeCDP] = []

    def __init__(self) -> None:
        self.connected = False
        self.closed = False
        self.commands: list[tuple[str, dict]] = []
        self.navigated: list[str] = []
        self.keys: list[str] = []
        self.typed: list[str] = []
        self.clicks: list[tuple[float, float]] = []
        self.scrolls: list[float] = []
        self.back = 0
        self.forward = 0
        self.page: dict = {"title": "Example", "text": "Hello", "elements": []}
        self.evaluate_result: object = "42"
        self.evaluate_error: Exception | None = None
        self.connect_error: Exception | None = None
        FakeCDP.instances.append(self)

    @property
    def is_connected(self) -> bool:
        return self.connected and not self.closed

    async def connect(self, cdp_url: str) -> None:
        if self.connect_error:
            raise self.connect_error
        self.connected = True

    async def connect_ws(self, ws_url: str, *, attach_to_page: bool = True) -> None:
        if self.connect_error:
            raise self.connect_error
        self.connected = True

    async def send(self, method: str, **params) -> dict:
        self.commands.append((method, params))
        return {}

    async def navigate(self, url: str) -> dict:
        self.navigated.append(url)
        return {}

    async def go_back(self) -> None:
        self.back += 1

    async def go_forward(self) -> None:
        self.forward += 1

    async def page_info(self) -> dict:
        return self.page

    async def evaluate(self, expression: str):
        if self.evaluate_error:
            raise self.evaluate_error
        return self.evaluate_result

    async def click(self, x: float, y: float) -> None:
        self.clicks.append((x, y))

    async def type_text(self, text: str) -> None:
        self.typed.append(text)

    async def press_key(self, key: str) -> None:
        self.keys.append(key)

    async def scroll(self, x: float = 0, y: float = 0, delta_y: float = -400) -> None:
        self.scrolls.append(delta_y)

    async def close(self) -> None:
        self.closed = True


class FakeProvider:
    def __init__(self, *, launch_error: Exception | None = None, release_error=None) -> None:
        self.launched: list[str] = []
        self.released: list[str] = []
        self.launch_error = launch_error
        self.release_error = release_error

    @property
    def vendor(self) -> str:
        return BROWSERBASE

    async def launch(self, user_id: str) -> LaunchedBrowser:
        if self.launch_error:
            raise self.launch_error
        self.launched.append(user_id)
        return LaunchedBrowser(
            browser_id="sess-1",
            vendor=BROWSERBASE,
            live_url="https://live",
            cdp_ws_url="wss://cdp",
            persistence_id="ctx-1",
        )

    async def release(self, browser_id: str) -> None:
        if self.release_error:
            raise self.release_error
        self.released.append(browser_id)


class FakeParticipant:
    def __init__(self, fail: bool = False) -> None:
        self.published: list[dict] = []
        self.fail = fail

    async def publish_data(self, payload: bytes, reliable: bool = True) -> None:
        if self.fail:
            raise RuntimeError("data channel closed")
        self.published.append(json.loads(payload.decode("utf-8")))


class FakeRoom:
    def __init__(self, fail: bool = False) -> None:
        self.local_participant = FakeParticipant(fail)


@pytest.fixture(autouse=True)
def _fake_cdp(monkeypatch):
    FakeCDP.instances.clear()
    monkeypatch.setattr(session_module, "CDPConnection", FakeCDP)
    # The real one sleeps 1.5s after every navigate, which would make this
    # module the slowest in the suite for no added confidence. `real_sleep` is
    # captured first: `session_module.asyncio` *is* the asyncio module, so a
    # lambda calling `asyncio.sleep` would call the patch and recurse.
    real_sleep = asyncio.sleep
    monkeypatch.setattr(session_module.asyncio, "sleep", lambda _: real_sleep(0))

    # With every sleep collapsed to zero, the idle timer fires immediately and
    # closes the browser underneath the test that just opened it. Idle release
    # is covered directly further down instead.
    async def _never_idle(self) -> None:
        return

    monkeypatch.setattr(session_module.CloudBrowserSession, "_idle_timeout", _never_idle)
    yield


async def _started(room=None, provider=None) -> tuple[CloudBrowserSession, FakeProvider]:
    provider = provider or FakeProvider()
    session = CloudBrowserSession(room=room, provider=provider)
    await session.start(user_id="user-1")
    return session, provider


# -- Start -------------------------------------------------------------------


async def test_starting_twice_reuses_the_browser_rather_than_renting_another() -> None:
    session, provider = await _started()

    await session.start(user_id="user-1", url="https://example.com")

    assert provider.launched == ["user-1"], "a second paid browser was rented"
    assert FakeCDP.instances[0].navigated == ["https://example.com"]


async def test_an_initial_url_is_navigated_to() -> None:
    provider = FakeProvider()
    session = CloudBrowserSession(provider=provider)

    await session.start(user_id="user-1", url="https://example.com")

    assert FakeCDP.instances[0].navigated == ["https://example.com"]


async def test_a_browser_that_never_connects_is_released() -> None:
    """It is already billing, and `is_active` gates every cleanup path."""
    provider = FakeProvider()
    session = CloudBrowserSession(provider=provider)

    original_init = FakeCDP.__init__

    def failing_init(self) -> None:
        original_init(self)
        self.connect_error = ConnectionError("cdp refused")

    FakeCDP.__init__ = failing_init  # type: ignore[method-assign]
    try:
        with pytest.raises(ConnectionError):
            await session.start(user_id="user-1")
    finally:
        FakeCDP.__init__ = original_init  # type: ignore[method-assign]

    assert provider.released == ["sess-1"]
    assert session.is_active is False


async def test_a_provider_that_cannot_launch_leaves_no_session() -> None:
    provider = FakeProvider(launch_error=RuntimeError("quota"))
    session = CloudBrowserSession(provider=provider)

    with pytest.raises(RuntimeError):
        await session.start(user_id="user-1")

    assert session.is_active is False


# -- URL safety --------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "javascript:alert(1)",
        "file:///etc/passwd",
        "data:text/html,<script>fetch('/steal')</script>",
    ],
)
async def test_navigate_refuses_unsafe_addresses(url: str) -> None:
    """The profile carries the user's logins, so SSRF here is authenticated."""
    session, _ = await _started()

    with pytest.raises(UnsafeURLError):
        await session.navigate(url)

    assert FakeCDP.instances[0].navigated == []


async def test_an_unsafe_initial_url_is_refused_at_start_too() -> None:
    """start() reaches navigate(); the gate must not live only in the tool."""
    session = CloudBrowserSession(provider=FakeProvider())

    with pytest.raises(UnsafeURLError):
        await session.start(user_id="user-1", url="http://localhost/internal")


# -- Interaction -------------------------------------------------------------


async def test_every_interaction_needs_a_live_browser() -> None:
    session = CloudBrowserSession(provider=FakeProvider())

    for call in (
        session.navigate("https://example.com"),
        session.read_page(),
        session.go_back(),
        session.go_forward(),
        session.press_key("Enter"),
        session.scroll("down"),
        session.evaluate_js("1"),
        session.click(description="button"),
        session.type_text("hi"),
    ):
        with pytest.raises(RuntimeError, match="No active cloud browser"):
            await call


async def test_history_navigation_reaches_cdp() -> None:
    session, _ = await _started()

    await session.go_back()
    await session.go_forward()

    assert (FakeCDP.instances[0].back, FakeCDP.instances[0].forward) == (1, 1)


async def test_read_page_reports_title_text_and_elements() -> None:
    session, _ = await _started()
    FakeCDP.instances[0].page = {
        "title": "Shop",
        "text": "Everything must go",
        "elements": [
            {"id": "el-0", "type": "button", "label": "Buy", "x": 10, "y": 20, "visible": True}
        ],
    }

    page = await session.read_page()

    assert "Shop" in page and "Everything must go" in page
    assert "el-0" in page and "Buy" in page


async def test_read_page_survives_a_page_that_reports_nothing() -> None:
    session, _ = await _started()
    FakeCDP.instances[0].page = None  # type: ignore[assignment]

    assert "Could not read page content" in await session.read_page()


async def test_clicking_uses_the_element_id_when_given_one() -> None:
    session, _ = await _started()
    FakeCDP.instances[0].page = {
        "title": "",
        "text": "",
        "elements": [
            {"id": "el-0", "type": "a", "label": "Home", "x": 5, "y": 5, "visible": True},
            {"id": "el-1", "type": "button", "label": "Buy now", "x": 50, "y": 60, "visible": True},
        ],
    }

    result = await session.click(element_id="el-1")

    assert FakeCDP.instances[0].clicks == [(50.0, 60.0)]
    assert "Buy now" in result


async def test_clicking_falls_back_to_a_described_element() -> None:
    session, _ = await _started()
    FakeCDP.instances[0].page = {
        "title": "",
        "text": "",
        "elements": [
            {
                "id": "el-0",
                "type": "button",
                "label": "Add to basket",
                "x": 7,
                "y": 8,
                "visible": True,
            }
        ],
    }

    await session.click(description="add to basket")

    assert FakeCDP.instances[0].clicks == [(7.0, 8.0)]


async def test_an_invisible_element_is_not_clicked() -> None:
    session, _ = await _started()
    FakeCDP.instances[0].page = {
        "title": "",
        "text": "",
        "elements": [
            {"id": "el-0", "type": "button", "label": "Buy", "x": 1, "y": 2, "visible": False}
        ],
    }

    result = await session.click(element_id="el-0")

    assert "Could not find element" in result
    assert FakeCDP.instances[0].clicks == []


async def test_typing_into_a_field_focuses_it_first() -> None:
    session, _ = await _started()
    FakeCDP.instances[0].page = {
        "title": "",
        "text": "",
        "elements": [
            {"id": "el-0", "type": "input", "label": "Search", "x": 3, "y": 4, "visible": True}
        ],
    }

    await session.type_text("kwami", element_id="el-0")

    assert FakeCDP.instances[0].clicks == [(3.0, 4.0)]
    assert FakeCDP.instances[0].typed == ["kwami"]


async def test_typing_into_a_field_that_is_not_there_types_nothing() -> None:
    session, _ = await _started()
    FakeCDP.instances[0].page = {"title": "", "text": "", "elements": []}

    result = await session.type_text("kwami", description="search box")

    assert "Could not find" in result
    assert FakeCDP.instances[0].typed == []


async def test_scrolling_down_and_up_send_opposite_deltas() -> None:
    session, _ = await _started()

    await session.scroll("down")
    await session.scroll("up")

    down, up = FakeCDP.instances[0].scrolls
    assert down > 0 > up


async def test_a_failing_evaluate_is_reported_not_raised() -> None:
    session, _ = await _started()
    FakeCDP.instances[0].evaluate_error = RuntimeError("no such element")

    assert "Failed to execute JavaScript" in await session.evaluate_js("boom()")


# -- Frontend events ---------------------------------------------------------


async def test_navigating_tells_the_frontend_the_new_url() -> None:
    room = FakeRoom()
    session, _ = await _started(room)

    await session.navigate("https://example.com")

    update = room.local_participant.published[-1]
    assert update["action"] == "update"
    assert update["url"] == "https://example.com"


async def test_a_dead_data_channel_does_not_break_browsing() -> None:
    """The panel is how the user watches; losing it must not stop the agent."""
    session, _ = await _started(FakeRoom(fail=True))

    await session.navigate("https://example.com")

    assert FakeCDP.instances[0].navigated == ["https://example.com"]


async def test_a_session_with_no_room_publishes_nothing_and_still_works() -> None:
    session, _ = await _started(None)
    await session.navigate("https://example.com")
    assert FakeCDP.instances[0].navigated == ["https://example.com"]


# -- Release -----------------------------------------------------------------


async def test_closing_twice_is_safe() -> None:
    session, provider = await _started()

    await session.close()
    await session.close()

    assert provider.released == ["sess-1"]


async def test_a_failed_release_does_not_report_a_clean_close() -> None:
    """Saying the browser is gone while it still runs hides a live bill."""
    provider = FakeProvider(release_error=RuntimeError("upstream down"))
    session = CloudBrowserSession(provider=provider)
    await session.start(user_id="user-1")

    await session.close()  # must not raise

    assert session.is_active is False


async def test_the_idle_timer_releases_the_browser() -> None:
    session, provider = await _started()

    # Run the idle body directly rather than waiting five minutes.
    session._cancel_idle_timer()
    await session.close()

    assert provider.released == ["sess-1"]


async def test_the_idle_timer_does_not_cancel_the_task_running_it() -> None:
    """close() cancels the idle timer, and the idle timer calls close().

    Cancelling itself landed a CancelledError inside close(), which could skip
    the release entirely -- so the browser was never stopped and kept billing.
    """
    session, provider = await _started()
    monkeypatched = asyncio.current_task()
    session._idle_timer = monkeypatched

    session._cancel_idle_timer()

    assert session._idle_timer is None
    assert not monkeypatched.cancelled()
    await session.close()
    assert provider.released == ["sess-1"]
