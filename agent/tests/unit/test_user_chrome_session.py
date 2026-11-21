"""Jarvis working in the user's own Chrome: its tabs, their tabs, and what it never does."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import pytest

from src.browser import user_chrome_session
from src.browser.chrome_bridge import BridgeError, BridgeUnavailableError
from src.browser.safety import UnsafeURLError
from src.browser.user_chrome_session import (
    UserChromeSession,
    explain,
    find_tab,
    tab_line,
)


class FakeBridge:
    """Answers each command from a table: a value, a callable on the args, or an error."""

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
        if callable(answer):
            return answer(args or {})
        return answer

    async def ready(self) -> bool:
        return True

    def sent(self, cmd: str) -> list[dict[str, Any]]:
        return [args for name, args in self.calls if name == cmd]


class Writer:
    def __init__(self, room: FakeRoom) -> None:
        self.room = room

    async def write(self, data: bytes) -> None:
        self.room.frames.append(data)

    async def aclose(self) -> None:
        return None


class Participant:
    def __init__(self, room: FakeRoom) -> None:
        self.room = room

    async def publish_data(self, payload: bytes, reliable: bool = True) -> None:
        if self.room.broken:
            raise RuntimeError("room gone")
        self.room.events.append(json.loads(payload))

    async def stream_bytes(self, name: str, **kwargs: Any) -> Writer:
        self.room.topics.append(kwargs.get("topic"))
        return Writer(self.room)


class FakeRoom:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.frames: list[bytes] = []
        self.topics: list[Any] = []
        self.broken = False
        self.local_participant = Participant(self)


OPENED = {"id": 11, "url": "https://example.com/", "title": "Example", "jarvis": True}
PAGE = {
    "title": "Example",
    "url": "https://example.com/",
    "text": "Hello there",
    "elements": [
        {"id": "el-0", "type": "a", "label": "Below", "visible": True, "inView": False},
        {"id": "el-1", "type": "button", "label": "Send", "visible": True, "inView": True},
        {"id": "el-2", "type": "a", "label": "Hidden", "visible": False, "inView": False},
        "not an element",
    ],
}


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(user_chrome_session, "SETTLE_SECONDS", 0)
    monkeypatch.setattr(user_chrome_session, "FRAME_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(user_chrome_session, "FRAME_HIDDEN_SECONDS", 0)


def session_with(answers: dict[str, Any], room: Any = None) -> tuple[UserChromeSession, FakeBridge]:
    bridge = FakeBridge({"open": OPENED, "read": PAGE, **answers})
    return UserChromeSession(room=room, bridge=bridge), bridge


async def opened(answers: dict[str, Any] | None = None, room: Any = None):
    session, bridge = session_with(answers or {}, room)
    await session.set_frames_wanted(False)
    await session.start("u1", "https://example.com/")
    return session, bridge


# -- the words ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "words"),
    [
        ("secret-field", "password"),
        ("no-field", "No field"),
        ("not-a-field", "not a field"),
        ("not-found", "gone"),
        ("protected-page", "Web Store"),
        ("paused", "paused"),
        ("No tab with id: 4", "was closed"),
        ("something else", "something else"),
    ],
)
def test_the_extensions_errors_are_said_so_the_model_can_act(error: str, words: str) -> None:
    assert words in explain(BridgeError(error))


def test_a_tab_is_one_line_with_what_matters() -> None:
    tab = {"title": "Inbox\n(3)", "url": "https://mail.example/", "focused": True, "jarvis": True}
    assert tab_line(2, tab) == "2. Inbox (3) -- https://mail.example/ [in front, Jarvis's]"
    assert tab_line(1, {"url": "https://x.io/", "audible": True}) == (
        "1. (no title) -- https://x.io/ [playing sound]"
    )
    assert tab_line(3, {"title": "Plain"}) == "3. Plain -- "


def test_a_tab_is_found_by_number_or_by_words_in_front_first() -> None:
    tabs = [
        {"id": 1, "title": "Factura de luz", "url": "https://a.es/"},
        {"id": 2, "title": "Gmail - Inbox", "url": "https://mail.google.com/"},
        {"id": 3, "title": "Gmail - Other", "url": "https://mail.google.com/u/1", "focused": True},
    ]
    assert find_tab(tabs, "1")["id"] == 1
    assert find_tab(tabs, "9") is None
    assert find_tab(tabs, "0") is None
    assert find_tab(tabs, "gmail")["id"] == 3
    assert find_tab(tabs, "FACTURA luz")["id"] == 1
    assert find_tab(tabs, "nothing like it") is None
    assert find_tab(tabs, "   ") is None


# -- its face -------------------------------------------------------------------------


async def test_it_looks_like_any_browser_to_the_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    session, _bridge = session_with({})
    assert (session.is_active, session.runs_locally, session.is_builtin) == (False, True, False)
    assert session.uses_user_chrome is True
    assert session.vendor == "user_chrome"
    assert session.live_url is None
    assert session.tab_id is None and session.owns_tab is False
    session.set_usage_tracker(object())
    room = FakeRoom()
    session.set_room(room)
    assert session._room is room

    shared = FakeBridge()
    monkeypatch.setattr(user_chrome_session, "chrome_bridge", lambda: shared)
    assert UserChromeSession().bridge is shared


async def test_nothing_works_before_a_page_is_open() -> None:
    session, _bridge = session_with({})
    with pytest.raises(RuntimeError, match="navigate_to"):
        await session.read_page()
    assert await session.page_address() == ""
    await session.bring_to_front()
    assert await session.on_sign_in_page() is False
    assert await session.sign_in_refused() is False


# -- opening and moving -----------------------------------------------------------------


async def test_it_opens_a_page_in_its_own_window_and_tells_the_app() -> None:
    room = FakeRoom()
    session, bridge = session_with({}, room)
    await session.set_frames_wanted(False)
    assert await session.start("u1", "https://example.com/") == ""
    assert bridge.sent("open") == [{"url": "https://example.com/", "focus": False}]
    assert session.is_active and session.owns_tab and session.tab_id == 11
    assert room.events == [
        {
            "type": "browser_session",
            "action": "open",
            "liveUrl": "",
            "vendor": "user_chrome",
            "persistent": True,
            "url": "https://example.com/",
        }
    ]
    await session.start("u1", "https://example.org/")
    assert bridge.sent("navigate")[0]["url"] == "https://example.org/"


async def test_it_refuses_no_address_or_an_unsafe_one() -> None:
    session, _bridge = session_with({})
    with pytest.raises(ValueError, match="address"):
        await session.start("u1", "")
    with pytest.raises(UnsafeURLError):
        await session.start("u1", "http://127.0.0.1:8080/")


async def test_an_answer_without_a_tab_is_an_error() -> None:
    session, _bridge = session_with({"open": {"url": "https://example.com/"}})
    with pytest.raises(BridgeError, match="which tab"):
        await session.start("u1", "https://example.com/")


async def test_navigation_goes_where_asked_and_says_where_it_is() -> None:
    session, bridge = await opened(
        {"navigate": {"url": "https://example.org/x", "title": "X"}, "back": {}, "forward": {}}
    )
    said = await session.navigate("https://example.org/x")
    assert "user's own Chrome" in said
    assert session._title == "X"
    assert await session.go_back() == "Going back to the previous page."
    assert await session.go_forward() == "Going forward to the next page."
    bridge.answers["navigate"] = None
    await session.navigate("https://example.org/y")
    assert session._current_url == "https://example.org/y"


async def test_the_address_comes_from_the_tab_when_it_can() -> None:
    session, bridge = await opened({"info": {"url": "https://example.com/moved"}})
    assert await session.page_address() == "https://example.com/moved"
    bridge.answers["info"] = {"url": "chrome://settings"}
    assert await session.page_address() == "https://example.com/moved"
    bridge.answers["info"] = "odd"
    assert await session.page_address() == "https://example.com/moved"
    bridge.answers["info"] = BridgeError("slow")
    assert await session.page_address() == "https://example.com/moved"


async def test_it_brings_its_tab_to_the_front_and_shrugs_if_it_cannot() -> None:
    session, bridge = await opened({"activate": {}})
    await session.bring_to_front()
    assert bridge.sent("activate") == [{"tabId": 11, "focus": True}]
    bridge.answers["activate"] = BridgeError("gone")
    await session.bring_to_front()


# -- their tabs ---------------------------------------------------------------------------


async def test_it_lists_every_tab_and_skips_what_is_not_one() -> None:
    session, bridge = session_with({"tabs": {"tabs": [{"id": 1}, "junk", {"id": 2}]}})
    assert await session.list_tabs() == [{"id": 1}, {"id": 2}]
    bridge.answers["tabs"] = "odd"
    assert await session.list_tabs() == []


async def test_a_tab_of_the_users_is_worked_in_and_never_closed() -> None:
    room = FakeRoom()
    session, bridge = session_with(
        {"info": {"id": 5, "url": "https://mail.example/", "title": "Mail", "jarvis": False}},
        room,
    )
    await session.set_frames_wanted(False)
    await session.use_tab(5)
    assert session.tab_id == 5 and not session.owns_tab
    assert room.events[-1]["title"] == "Mail"
    await session.close(quit_browser=True)
    assert bridge.sent("close") == []
    assert room.events[-1] == {"type": "browser_session", "action": "close"}
    assert not session.is_active


async def test_a_tab_of_jarvis_is_its_own() -> None:
    session, _bridge = session_with({"info": {"id": 6, "url": "https://x.io/", "jarvis": True}})
    await session.set_frames_wanted(False)
    await session.use_tab(6)
    assert session.owns_tab


async def test_its_own_tab_closes_when_asked_and_stays_otherwise() -> None:
    session, bridge = await opened({"close": BridgeError("already gone")})
    await session.close()
    assert bridge.sent("close") == []
    await session.start("u1", "https://example.com/")
    await session.close(quit_browser=True)
    assert bridge.sent("close") == [{"tabId": 11}]
    await session.start("u1", "https://example.com/")
    await session.leave_page()
    assert len(bridge.sent("close")) == 2
    assert not session.is_active


async def test_another_tab_is_read_without_moving_there() -> None:
    session, bridge = session_with({})
    assert (await session.read_tab_text(3))["title"] == "Example"
    assert bridge.sent("read") == [{"tabId": 3, "maxText": 4000}]
    bridge.answers["read"] = None
    assert await session.read_tab_text(3) == {}


async def test_a_closed_tab_or_a_lost_chrome_is_forgotten() -> None:
    session, bridge = await opened({"read": BridgeError("No tab with id: 11")})
    with pytest.raises(BridgeError):
        await session.read_page()
    assert not session.is_active

    session, bridge = await opened({"read": BridgeUnavailableError("gone")})
    with pytest.raises(BridgeUnavailableError):
        await session.read_page()
    assert not session.is_active

    session, bridge = await opened({"read": BridgeError("slow")})
    with pytest.raises(BridgeError):
        await session.read_page()
    assert session.is_active


# -- the page -----------------------------------------------------------------------------


async def test_it_reads_the_page_with_what_is_on_screen_first() -> None:
    session, _bridge = await opened()
    page = await session.read_page()
    assert page.startswith("Page title: Example\nAddress: https://example.com/")
    assert "Hello there" in page
    lines = [line.strip() for line in page.splitlines() if line.strip().startswith("el-")]
    assert lines == ['el-1 [button] ✓ "Send"', 'el-0 [a] ✓ "Below"', 'el-2 [a] ✗ "Hidden"']


async def test_an_empty_page_reads_as_just_its_title() -> None:
    session, bridge = await opened({"read": {"title": "Blank"}})
    assert await session.read_page() == "Page title: Blank\nAddress: "
    bridge.answers["read"] = None
    assert await session.read_page() == "Page title: \nAddress: "


async def test_it_clicks_what_was_meant_and_never_a_final_button() -> None:
    session, bridge = await opened({"click": {"ok": True}})
    assert await session.click(element_id="el-1") == 'Clicked on "Send" (el-1).'
    assert bridge.sent("click") == [{"tabId": 11, "elementId": "el-1"}]
    assert "Could not find" in await session.click(description="nowhere")
    refused = await session.click(description="send", refuse=lambda label: f"no {label}")
    assert refused == "no Send"
    assert await session.click(description="below", refuse=lambda _label: None)
    bridge.answers["click"] = BridgeError("not-found")
    assert "gone" in await session.click(element_id="el-1")


async def test_it_types_where_asked_and_never_a_secret() -> None:
    session, bridge = await opened({"type": {"ok": True}})
    assert await session.type_text("hola") == "Typed 'hola' into the field."
    assert bridge.sent("type")[-1] == {"tabId": 11, "elementId": "", "text": "hola", "clear": True}
    await session.type_text("más", element_id="el-1", clear_first=False)
    assert bridge.sent("type")[-1]["elementId"] == "el-1"
    assert bridge.sent("type")[-1]["clear"] is False
    assert "Could not find" in await session.type_text("x", description="nowhere")
    bridge.answers["type"] = BridgeError("secret-field")
    assert "Nothing was typed" in await session.type_text("hunter2", element_id="el-1")


async def test_keys_and_scrolling() -> None:
    session, bridge = await opened({"key": {"ok": True}, "scroll": {"ok": True}})
    assert await session.press_key("Tab") == "Pressed 'Tab'."
    bridge.answers["key"] = {"ok": True, "submitted": True}
    assert "form was sent" in await session.press_key("Enter")
    bridge.answers["key"] = BridgeError("paused")
    assert "paused" in await session.press_key("Enter")
    assert await session.scroll("UP") == "Scrolled up."
    assert await session.scroll("sideways") == "Scrolled down."
    assert [a["direction"] for a in bridge.sent("scroll")] == ["up", "down"]


async def test_it_runs_no_script_only_its_own_named_actions() -> None:
    session, bridge = await opened({"media": {"result": "paused"}})
    assert "does not run scripts" in await session.evaluate_js("document.cookie")
    assert await session.run_named("pause") == "paused"
    assert bridge.sent("media") == [{"tabId": 11, "action": "pause", "level": 0.0}]
    bridge.answers["media"] = "plain"
    assert await session.run_named("volume", 0.5) == "plain"


async def test_it_knows_a_sign_in_page_and_a_refusal() -> None:
    probe = {"address": "github.com/login", "password": False, "head": ""}
    session, bridge = await opened({"probe": probe})
    assert await session.on_sign_in_page() is True
    bridge.answers["probe"] = {"address": "example.com/", "password": True}
    assert await session.on_sign_in_page() is True
    bridge.answers["probe"] = {"address": "example.com/", "password": False}
    assert await session.on_sign_in_page() is False
    bridge.answers["probe"] = {"head": "Este navegador o aplicación puede que no sea seguro"}
    assert await session.sign_in_refused() is True
    bridge.answers["probe"] = "odd"
    assert await session.sign_in_refused() is False
    bridge.answers["probe"] = BridgeError("slow")
    assert await session.on_sign_in_page() is False
    assert await session.sign_in_refused() is False


# -- the app's panel ------------------------------------------------------------------------


async def test_a_room_that_fails_does_not_stop_the_work() -> None:
    room = FakeRoom()
    room.broken = True
    session, _bridge = session_with({}, room)
    await session.set_frames_wanted(False)
    await session.start("u1", "https://example.com/")
    assert session.is_active


async def test_pictures_of_the_tab_reach_the_panel_when_they_change() -> None:
    room = FakeRoom()
    shots = iter(
        [
            {"jpeg": base64.b64encode(b"one").decode()},
            {"jpeg": base64.b64encode(b"one").decode()},
            RuntimeError("not-visible"),
            RuntimeError("mid-navigation"),
            "odd",
            {"jpeg": base64.b64encode(b"two").decode()},
            BridgeUnavailableError("gone"),
        ]
    )

    def capture(_args: dict[str, Any]) -> Any:
        shot = next(shots)
        if isinstance(shot, BaseException):
            raise shot
        return shot

    session, _bridge = session_with({"capture": capture}, room)
    await session.start("u1", "https://example.com/")
    task = session._frame_task
    assert task is not None
    await asyncio.wait_for(task, 2)
    assert room.frames == [b"one", b"two"]
    assert set(room.topics) == {"browser_frame"}


async def test_pictures_stop_with_the_tab_and_when_nobody_looks() -> None:
    room = FakeRoom()
    session, _bridge = session_with({}, room)
    calls = 0

    def capture(_args: dict[str, Any]) -> Any:
        nonlocal calls
        calls += 1
        session._tab_id = None
        return {}

    _bridge.answers["capture"] = capture
    await session.start("u1", "https://example.com/")
    assert session._frame_task is not None
    await asyncio.wait_for(session._frame_task, 2)
    assert calls == 1

    session._tab_id = 11
    await session.set_frames_wanted(True)  # Already wanted: nothing changes.
    await session.set_frames_wanted(False)
    assert session._frame_task is None
    await session.set_frames_wanted(True)
    assert session._frame_task is not None
    session._stop_frames()

    quiet = UserChromeSession(bridge=FakeBridge())
    await quiet._start_frames()
    assert quiet._frame_task is None
    await quiet._publish_frame(b"nobody")


# -- which way in ---------------------------------------------------------------------------


async def test_without_a_way_given_it_takes_the_one_that_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ClaudeWay(FakeBridge):
        sees_all_tabs = False
        place = "Jarvis's tab group of the user's own Chrome (through the Claude extension)"

    claude = ClaudeWay({"open": {"id": 4, "url": "https://example.com/"}})
    ways: list[Any] = [claude]

    async def answering() -> Any:
        return ways[0]

    monkeypatch.setattr(user_chrome_session, "user_browser", answering)
    session = UserChromeSession()
    assert session.sees_all_tabs is True  # nothing chosen yet
    await session.set_frames_wanted(False)
    await session.start("u1", "https://example.com/")
    assert claude.sent("open") and session.sees_all_tabs is False
    claude.answers["navigate"] = {"url": "https://example.org/"}
    assert "through the Claude extension" in await session.navigate("https://example.org/")

    # The way it took went away: the next call chooses again.
    claude.answers["read"] = BridgeUnavailableError("not paired")
    with pytest.raises(BridgeUnavailableError):
        await session.read_page()
    kwami = FakeBridge({"tabs": {"tabs": []}})
    ways[0] = kwami
    assert await session.list_tabs() == []
    assert kwami.sent("tabs") and session.place == "Jarvis's window of the user's own Chrome"


async def test_a_way_given_is_kept_even_when_it_goes_away() -> None:
    session, bridge = await opened({"read": BridgeUnavailableError("gone")})
    with pytest.raises(BridgeUnavailableError):
        await session.read_page()
    assert session.bridge is bridge


async def test_with_no_way_at_all_it_asks_kwamis_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    async def nothing() -> None:
        return None

    shared = FakeBridge({"tabs": {"tabs": [{"id": 1}]}})
    monkeypatch.setattr(user_chrome_session, "user_browser", nothing)
    monkeypatch.setattr(user_chrome_session, "chrome_bridge", lambda: shared)
    assert await UserChromeSession().list_tabs() == [{"id": 1}]
