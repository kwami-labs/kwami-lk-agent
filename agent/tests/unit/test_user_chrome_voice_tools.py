"""The voice's browser tools with the user's own Chrome connected (2026-11-17).

Their Chrome comes first when nothing is open; a page shown to the user is not a
page worked in; their tabs can be listed and read; and Jarvis never closes theirs.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.browser import chrome_bridge, user_chrome_session, window_reader
from src.browser.browser_session import CloudBrowserSession
from src.browser.chrome_bridge import BridgeError
from src.browser.user_chrome_session import UserChromeSession
from src.browser.window_reader import BrowserWindow
from src.domain import KwamiConfig
from src.domain.tool_result import was_refused
from src.tools.builtin import (
    DESKTOP_TABS_NOTE,
    NO_DESKTOP_TABS,
    USER_CHROME_SIGN_IN_HINT,
    AgentToolsMixin,
    _sign_in_hint,
)


class FakeBridge:
    def __init__(self, answers: dict[str, Any] | None = None) -> None:
        self.answers = answers or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(
        self, cmd: str, args: dict[str, Any] | None = None, timeout: float = 20.0
    ) -> Any:
        self.calls.append((cmd, args or {}))
        answer = self.answers.get(cmd)
        if isinstance(answer, BaseException):
            raise answer
        return answer(args or {}) if callable(answer) else answer

    async def ready(self) -> bool:
        return True


TABS = {
    "tabs": [
        {"id": 1, "title": "Gmail - Inbox", "url": "https://mail.google.com/", "focused": True},
        {"id": 2, "title": "Banco", "url": "https://banco.example/"},
        {"id": 3, "title": "Ajustes", "url": "chrome://settings/"},
    ]
}
PAGE = {"title": "Gmail", "url": "https://mail.google.com/", "text": "3 new", "elements": []}


class FakePublisher:
    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []

    async def publish(self, payload: dict[str, Any], *, topic: str | None = None) -> bool:
        self.published.append(payload)
        return True


class Tools(AgentToolsMixin):
    """The mixin with its real session choice: only the transport is faked."""

    def __init__(self, session: Any = None) -> None:
        self.kwami_config = KwamiConfig()
        self.kwami_config.kwami_id = "kwami-1"
        self._current_voice_config = self.kwami_config.voice
        self._memory = None
        self.session = None
        self.room = None
        self.usage_tracker = None
        self._browser_session = session
        self.publisher = FakePublisher()
        self.refused_sites: set[str] = set()

    def _publisher(self, context: Any = None):
        return self.publisher

    async def _site_refusal(self, context: Any, needed: str, address: str | None = None):
        if address and any(site in address for site in self.refused_sites):
            return "refused: the user's list"
        return None

    async def _remember_site(self, context: Any, url: str) -> None:
        self.remembered = url


def connected(monkeypatch: pytest.MonkeyPatch, ready: bool, bridge: FakeBridge | None = None):
    async def answer() -> bool:
        return ready

    monkeypatch.setattr(chrome_bridge, "user_chrome_ready", answer)
    shared = bridge or FakeBridge()
    monkeypatch.setattr(user_chrome_session, "chrome_bridge", lambda: shared)
    return shared


class Idle:
    """A browser that is not open, of either kind's shape."""

    is_active = False
    _room = None
    _usage_tracker = None

    async def close(self, **_: Any) -> None:
        self.closed = True


# -- which browser ------------------------------------------------------------------------


async def test_with_nothing_open_their_chrome_comes_first(monkeypatch: pytest.MonkeyPatch) -> None:
    connected(monkeypatch, True)
    tools = Tools()
    tools.room = object()
    session = await tools._get_browser_session()
    assert isinstance(session, UserChromeSession)
    assert session._room is tools.room
    assert await tools._get_browser_session() is session


async def test_without_their_chrome_jarvis_uses_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    connected(monkeypatch, False)
    tools = Tools(session=UserChromeSession(bridge=FakeBridge()))
    tools.usage_tracker = object()
    session = await tools._get_browser_session()
    assert isinstance(session, CloudBrowserSession)
    assert session._usage_tracker is tools.usage_tracker

    fresh = Tools()
    assert isinstance(await fresh._get_browser_session(), CloudBrowserSession)


async def test_a_browser_already_open_is_kept_whatever_connects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connected(monkeypatch, True)
    busy = Idle()
    busy.is_active = True
    tools = Tools(session=busy)
    assert await tools._get_browser_session() is busy

    idle = Idle()
    tools = Tools(session=idle)
    assert isinstance(await tools._get_browser_session(), UserChromeSession)


# -- working in a page ----------------------------------------------------------------------


async def test_navigate_to_opens_in_their_chrome_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = connected(
        monkeypatch,
        True,
        FakeBridge(
            {
                "open": {"id": 9, "url": "https://example.com/", "jarvis": True},
                "probe": {"address": "example.com/", "password": False},
            }
        ),
    )
    tools = Tools()
    said = await tools.navigate_to(None, "https://example.com/")
    assert "user's own Chrome" in said
    assert bridge.calls[0][0] == "open"


async def test_signing_in_happens_in_their_chrome_in_front_of_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = connected(
        monkeypatch,
        True,
        FakeBridge({"open": {"id": 9, "url": "https://github.com/login"}, "activate": {}}),
    )
    tools = Tools()
    said = await tools.navigate_to(None, "https://github.com/login", sign_in=True)
    assert USER_CHROME_SIGN_IN_HINT in said
    assert ("activate", {"tabId": 9, "focus": True}) in bridge.calls
    assert tools.remembered == "https://github.com/login"


async def test_a_sign_in_page_met_on_the_way_is_left_to_the_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connected(
        monkeypatch,
        True,
        FakeBridge(
            {
                "open": {"id": 9, "url": "https://github.com/login"},
                "probe": {"address": "github.com/login", "password": True},
                "activate": {},
            }
        ),
    )
    tools = Tools()
    said = await tools.navigate_to(None, "https://github.com/login")
    assert said.endswith(USER_CHROME_SIGN_IN_HINT)


def test_the_hint_for_their_chrome_is_its_own() -> None:
    session = UserChromeSession(bridge=FakeBridge())
    assert _sign_in_hint(session) == USER_CHROME_SIGN_IN_HINT


async def test_closing_closes_only_jarvis_tab(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = connected(monkeypatch, True, FakeBridge({"close": {}}))
    mine = UserChromeSession(bridge=bridge)
    mine._tab_id, mine._owned = 4, True
    assert await Tools(session=mine).close_navigation(None) == (
        "Closed Jarvis's tab in the user's Chrome."
    )
    assert ("close", {"tabId": 4}) in bridge.calls

    theirs = UserChromeSession(bridge=bridge)
    theirs._tab_id, theirs._owned = 5, False
    said = await Tools(session=theirs).close_navigation(None)
    assert "stays open" in said
    assert ("close", {"tabId": 5}) not in bridge.calls


# -- their tabs -------------------------------------------------------------------------------


DESKTOP = [
    BrowserWindow(
        "chrome.exe",
        "Recibidos - Gmail - Google Chrome",
        "mail.google.com/mail/u/0/",
        ["Recibidos - Gmail", "Factura"],
    ),
    BrowserWindow("brave.exe", "Banco - Brave", "banco.example/", ["Banco"]),
]


def on_windows(monkeypatch: pytest.MonkeyPatch, windows: list[BrowserWindow]) -> None:
    async def read() -> list[BrowserWindow]:
        return windows

    monkeypatch.setattr(window_reader, "browser_tabs", read)


async def test_without_an_extension_their_tabs_come_from_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connected(monkeypatch, False)
    on_windows(monkeypatch, DESKTOP)
    tools = Tools()
    tools.refused_sites = {"banco"}
    said = await tools.read_navigation_page(None, which="all")
    assert said.startswith(DESKTOP_TABS_NOTE)
    assert "3 tabs open" in said and "1. Recibidos - Gmail" in said and "3. Banco" in said
    assert "Brave window, the tab in front is on a site their list keeps you out of" in said
    assert "extension" not in said.removeprefix(DESKTOP_TABS_NOTE)
    one = await tools.read_navigation_page(None, which="gmail")
    assert "https://mail.google.com/mail/u/0/" in one and "navigate_to" in one


async def test_without_an_extension_or_windows_it_says_so_plainly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connected(monkeypatch, False)
    on_windows(monkeypatch, [])
    said = await Tools().read_navigation_page(None, which="all")
    assert was_refused(said)
    assert NO_DESKTOP_TABS in said and "chrome://extensions" not in said


async def test_it_lists_their_tabs_keeping_out_what_their_list_forbids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connected(monkeypatch, True, FakeBridge({"tabs": TABS}))
    tools = Tools()
    tools.refused_sites = {"banco.example"}
    said = await tools.read_navigation_page(None, which="todas")
    assert said.startswith("[UNTRUSTED")
    assert "1. Gmail - Inbox -- https://mail.google.com/ [in front]" in said
    assert "2. (a site the user's list keeps you out of)" in said
    assert "Banco" not in said
    assert "(and 1 of Chrome's own pages, not readable)" in said


async def test_a_list_with_only_web_tabs_has_no_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    connected(monkeypatch, True, FakeBridge({"tabs": {"tabs": TABS["tabs"][:1]}}))
    said = await Tools().read_navigation_page(None, which="all")
    assert "Chrome's own pages" not in said


async def test_it_reads_a_tab_of_theirs_and_works_there(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = connected(
        monkeypatch,
        True,
        FakeBridge(
            {"tabs": TABS, "info": {"id": 1, "url": "https://mail.google.com/"}, "read": PAGE}
        ),
    )
    own = Idle()
    own.is_active = True
    tools = Tools(session=own)
    said = await tools.read_navigation_page(None, which="gmail")
    assert "You now work in this tab of the user's" in said and "3 new" in said
    assert getattr(own, "closed", False), "Jarvis's own browser gave way to their tab"
    assert isinstance(tools._browser_session, UserChromeSession)
    assert tools._browser_session.tab_id == 1
    assert ("info", {"tabId": 1}) in bridge.calls

    again = await tools.read_navigation_page(None, which="1")
    assert "3 new" in again


async def test_reading_their_tab_respects_their_list_and_reports_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connected(monkeypatch, True, FakeBridge({"tabs": TABS, "info": BridgeError("No tab with id")}))
    tools = Tools()
    tools.refused_sites = {"banco.example"}
    assert await tools.read_navigation_page(None, which="banco") == "refused: the user's list"
    assert "No tab of the user's matches" in await tools.read_navigation_page(None, which="nada")
    failed = await tools.read_navigation_page(None, which="gmail")
    assert was_refused(failed) and "was closed" in failed


async def test_a_chrome_that_does_not_answer_is_said(monkeypatch: pytest.MonkeyPatch) -> None:
    connected(monkeypatch, True, FakeBridge({"tabs": BridgeError("paused")}))
    said = await Tools().read_navigation_page(None, which="all")
    assert was_refused(said) and "paused" in said


async def test_an_idle_browser_of_jarvis_is_not_closed_for_their_tab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connected(
        monkeypatch,
        True,
        FakeBridge(
            {"tabs": TABS, "info": {"id": 1, "url": "https://mail.google.com/"}, "read": PAGE}
        ),
    )
    idle = Idle()
    tools = Tools(session=idle)

    async def jarvis_own() -> Any:
        return idle

    # Their Chrome is reached for the tabs even when the session in hand is Jarvis's own.
    monkeypatch.setattr(tools, "_get_browser_session", jarvis_own)
    said = await tools.read_navigation_page(None, which="gmail")
    assert "3 new" in said
    assert not getattr(idle, "closed", False)
    assert isinstance(tools._browser_session, UserChromeSession)


async def test_signing_in_asks_the_users_list_for_the_account_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connected(monkeypatch, True)
    tools = Tools()
    asked: list[str] = []

    async def refusal(context: Any, needed: str, address: str | None = None) -> str | None:
        asked.append(needed)
        return "refused: the user's list" if needed == "account" else None

    monkeypatch.setattr(tools, "_site_refusal", refusal)
    said = await tools.navigate_to(None, "https://github.com/login", sign_in=True)
    assert said == "refused: the user's list"
    assert asked == ["account"]


async def test_through_claude_in_chrome_it_lists_only_its_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ClaudeWay(FakeBridge):
        sees_all_tabs = False

    claude = ClaudeWay({"tabs": {"tabs": TABS["tabs"][:1]}})
    connected(monkeypatch, True, claude)

    async def answering() -> ClaudeWay:
        return claude

    monkeypatch.setattr(user_chrome_session, "user_browser", answering)
    said = await Tools().read_navigation_page(None, which="all")
    assert "Jarvis's tab group in the user's Chrome has 1 web tabs open" in said
    assert "the user drags it into the group" in said
