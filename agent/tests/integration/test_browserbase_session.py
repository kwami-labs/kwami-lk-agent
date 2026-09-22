"""A Browserbase-backed session behaves exactly like the Browser Use one.

The session layer is the part that must not care which vendor is underneath:
URL validation, the idle timer, metering, the frontend event and releasing a
paid browser are all vendor-neutral, and each of them was written against
Browser Use's response shape. These tests drive the whole session through a
fake Browserbase provider to prove that is still true.

The one genuinely vendor-specific thing is the CDP endpoint: Browserbase hands
out a *browser*-level WebSocket, where Page, Input and Runtime do not exist
until you attach to a page target. That attach, and the session-id routing it
requires, is covered here too -- get it wrong and the panel is simply blank.
"""

from __future__ import annotations

import json

import pytest

import src.browser.browser_session as browser_session_module
from src.browser.browser_session import CloudBrowserSession
from src.browser.cloud_browser import CDPConnection
from src.browser.providers import BROWSERBASE, LaunchedBrowser
from src.browser.safety import UnsafeURLError

LIVE_URL = "https://www.browserbase.com/devtools-fullscreen/inspector.html?sess=1"


class FakeProvider:
    """A Browserbase provider with the network taken out."""

    def __init__(self, *, persistence_id: str = "ctx-1") -> None:
        self.launched: list[str] = []
        self.released: list[str] = []
        self._persistence_id = persistence_id

    @property
    def vendor(self) -> str:
        return BROWSERBASE

    async def launch(self, user_id: str) -> LaunchedBrowser:
        self.launched.append(user_id)
        return LaunchedBrowser(
            browser_id="sess-1",
            vendor=BROWSERBASE,
            live_url=LIVE_URL,
            cdp_ws_url="wss://connect.browserbase.com?sessionId=sess-1",
            persistence_id=self._persistence_id,
        )

    async def release(self, browser_id: str) -> None:
        self.released.append(browser_id)


class FakeCDP:
    """Records how it was connected and what it was asked to do."""

    instances: list[FakeCDP] = []

    def __init__(self) -> None:
        self.connected_ws: str | None = None
        self.connected_http: str | None = None
        self.attached = False
        self.navigated: list[str] = []
        self.commands: list[str] = []
        self.closed = False
        FakeCDP.instances.append(self)

    @property
    def is_connected(self) -> bool:
        return (self.connected_ws or self.connected_http) is not None and not self.closed

    async def connect(self, cdp_url: str) -> None:
        self.connected_http = cdp_url

    async def connect_ws(self, ws_url: str, *, attach_to_page: bool = True) -> None:
        self.connected_ws = ws_url
        self.attached = attach_to_page

    async def send(self, method: str, **params) -> dict:
        self.commands.append(method)
        return {}

    async def navigate(self, url: str) -> dict:
        self.navigated.append(url)
        return {}

    async def close(self) -> None:
        self.closed = True


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
    monkeypatch.setattr(browser_session_module, "CDPConnection", FakeCDP)
    yield


async def _started(
    room: FakeRoom | None = None, **kwargs
) -> tuple[CloudBrowserSession, FakeProvider]:
    provider = FakeProvider(**kwargs)
    session = CloudBrowserSession(room=room, provider=provider)
    await session.start(user_id="user-1")
    return session, provider


# -- The vendor-specific bit: reaching the browser ---------------------------


async def test_a_websocket_endpoint_is_attached_to_a_page() -> None:
    """On a browser-level socket, Page/Input/Runtime do not exist until attach."""
    session, _ = await _started()

    cdp = FakeCDP.instances[0]
    assert cdp.connected_ws == "wss://connect.browserbase.com?sessionId=sess-1"
    assert cdp.attached is True
    assert cdp.connected_http is None
    assert session.is_active


async def test_an_http_endpoint_still_uses_target_discovery() -> None:
    """Browser Use's path must not regress while adding the other one."""

    class HttpProvider(FakeProvider):
        async def launch(self, user_id: str) -> LaunchedBrowser:
            return LaunchedBrowser(
                browser_id="b-1",
                vendor="browser_use",
                live_url="https://live",
                cdp_http_url="https://cdp-1.browser-use.com",
            )

    session = CloudBrowserSession(provider=HttpProvider())
    await session.start(user_id="user-1")

    cdp = FakeCDP.instances[0]
    assert cdp.connected_http == "https://cdp-1.browser-use.com"
    assert cdp.connected_ws is None


# -- Everything the session does regardless of vendor ------------------------


async def test_the_user_id_is_passed_through_to_the_provider() -> None:
    _, provider = await _started()
    assert provider.launched == ["user-1"]


async def test_navigation_still_refuses_unsafe_addresses() -> None:
    """The profile carries the user's cookies, so SSRF here is authenticated."""
    session, _ = await _started()

    for url in ("http://169.254.169.254/latest/meta-data/", "http://localhost:8080/internal"):
        with pytest.raises(UnsafeURLError):
            await session.navigate(url)

    assert FakeCDP.instances[0].navigated == []


async def test_closing_releases_the_browser_and_the_socket() -> None:
    session, provider = await _started()

    await session.close()

    assert provider.released == ["sess-1"]
    assert FakeCDP.instances[0].closed is True
    assert not session.is_active


async def test_the_frontend_is_told_how_to_show_the_panel() -> None:
    room = FakeRoom()
    session, _ = await _started(room)

    (event,) = room.local_participant.published
    assert event["type"] == "browser_session"
    assert event["action"] == "open"
    assert event["vendor"] == BROWSERBASE
    assert event["persistent"] is True
    # Browser Use's `ui=false`/`theme=dark` mean nothing to Browserbase, whose
    # fullscreen URL is already bare. Sending them would be a silent no-op.
    assert event["liveUrl"] == LIVE_URL


async def test_an_ephemeral_session_is_advertised_as_not_persistent() -> None:
    """The panel warns the user rather than letting them find out by being logged out."""
    room = FakeRoom()
    await _started(room, persistence_id="")

    (event,) = room.local_participant.published
    assert event["persistent"] is False


async def test_closing_tells_the_frontend_too() -> None:
    room = FakeRoom()
    session, _ = await _started(room)

    await session.close()

    assert [e["action"] for e in room.local_participant.published] == ["open", "close"]


# -- CDP session-id routing --------------------------------------------------


class RecordingSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


async def test_attached_commands_carry_the_session_id() -> None:
    """Without it the browser target answers, where Page.enable does not exist."""
    cdp = CDPConnection()
    socket = RecordingSocket()
    cdp._ws = socket
    cdp._session_id = "page-session-1"

    # send() waits for a reply that never comes; only the outgoing frame matters.
    import asyncio

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(cdp.send("Page.enable"), timeout=0.05)

    assert socket.sent[0]["sessionId"] == "page-session-1"


async def test_target_commands_stay_on_the_browser_connection() -> None:
    cdp = CDPConnection()
    socket = RecordingSocket()
    cdp._ws = socket
    cdp._session_id = "page-session-1"

    import asyncio

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(cdp.send("Target.getTargets"), timeout=0.05)

    assert "sessionId" not in socket.sent[0]
