"""Claude in Chrome, the way Claude Code reaches it: Kwami's commands on top of its tools.

The answers below are the server's own, captured from Claude in Chrome on 2026-11-18.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from src.browser import claude_chrome
from src.browser.chrome_bridge import BridgeError, BridgeUnavailableError
from src.browser.claude_chrome import (
    ClaudeChromePort,
    StdioTools,
    ToolResult,
    claude_cli,
    parse_elements,
    parse_tabs,
    split_page_text,
)

CONTEXT = (
    '{"availableTabs":[{"tabId":11,"title":"Example Domain","url":"https://example.org/"},'
    '{"tabId":12,"title":"New Tab","url":"chrome://newtab/"}],"tabGroupId":99}\n\n'
    'Tab Context:\n- Available tabs:\n  • tabId 11: "Example Domain" ("https://example.org/")'
)
NAVIGATED = (
    "Navigated to https://example.org/\n\nTab Context:\n- Executed on tabId: 11\n"
    '- Available tabs:\n  • tabId 11: "Example Domain" ("https://example.org/")\n'
    '  • tabId 12: "Nueva pestaña" ("chrome://newtab/")'
)
NO_GROUP = (
    "This session has no tab group of its own. Any tabs opened earlier in this conversation "
    "may still be open in Chrome. Use createIfEmpty: true to create this session's group."
)
TREE = (
    'link "Learn more" [ref_1] href="https://iana.org/help/example-domains"\n'
    'textbox "Search Wikipedia" [ref_58] type="search" placeholder="Search Wikipedia"\n'
    'textbox [ref_60] type="password" autocomplete="current-password"\n'
    'textbox [ref_61] placeholder="Card number" autocomplete="cc-number"\n'
    "button [ref_62]\n"
    "textbox [ref_63]\n"
    "\nViewport: 1920x919"
)
PAGE_TEXT = (
    "Title: Example Domain\nURL: https://example.org/\nSource element: <main>\n---\n"
    "Example Domain\nThis domain is for use in documentation examples."
)
NOT_PAIRED = (
    "Browser extension is not connected: the OAuth token Claude Code is using belongs to "
    "a different claude.ai account than the one Claude Code is logged in as."
)


class FakeTools:
    """Claude in Chrome's tools from a table: a text, a result, a callable or an error."""

    def __init__(self, answers: dict[str, Any] | None = None) -> None:
        self.answers = answers or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    async def call_tool(self, name: str, arguments: dict[str, Any], timeout: float) -> ToolResult:
        self.calls.append((name, arguments))
        answer = self.answers.get(name, "")
        if callable(answer):
            answer = answer(arguments)
        if isinstance(answer, BaseException):
            raise answer
        return answer if isinstance(answer, ToolResult) else ToolResult(str(answer))

    async def close(self) -> None:
        self.closed = True

    def sent(self, name: str) -> list[dict[str, Any]]:
        return [args for called, args in self.calls if called == name]


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def port_with(answers: dict[str, Any] | None = None, clock: Clock | None = None):
    tools = FakeTools(answers)

    async def connect() -> FakeTools:
        return tools

    return ClaudeChromePort(connect=connect, clock=clock or Clock()), tools


@pytest.fixture(autouse=True)
def no_window_moves(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    moves: list[Any] = []
    monkeypatch.setattr(claude_chrome, "foreground_window", lambda: 77)
    monkeypatch.setattr(claude_chrome, "return_focus", lambda before: moves.append(before))
    return moves


# -- reading its answers -------------------------------------------------------------------


def test_tabs_come_from_its_json_or_its_tab_context() -> None:
    tabs = parse_tabs(CONTEXT)
    assert [t["id"] for t in tabs] == [11, 12]
    assert tabs[0] == {
        "id": 11,
        "title": "Example Domain",
        "url": "https://example.org/",
        "jarvis": True,
        "focused": False,
        "audible": False,
    }
    assert [(t["id"], t["title"]) for t in parse_tabs(NAVIGATED)] == [
        (11, "Example Domain"),
        (12, "Nueva pestaña"),
    ]
    assert parse_tabs(NO_GROUP) == []
    assert parse_tabs('{"availableTabs": not json') == []
    assert parse_tabs('{"availableTabs":[{"tabId":"x"},{"tabId":3}]}') == [
        {"id": 3, "title": "", "url": "", "jarvis": True, "focused": False, "audible": False}
    ]


def test_elements_come_from_its_tree_and_secrets_are_marked() -> None:
    elements = {e["id"]: e for e in parse_elements(TREE)}
    assert set(elements) == {"ref_1", "ref_58", "ref_60", "ref_61", "ref_63"}
    assert elements["ref_1"]["label"] == "Learn more" and elements["ref_1"]["type"] == "link"
    assert elements["ref_58"]["type"] == "input:search"
    assert not elements["ref_58"]["secret"]
    assert elements["ref_60"] | {} == {
        "id": "ref_60",
        "type": "input:password",
        "label": "field",
        "visible": True,
        "inView": True,
        "secret": True,
    }
    assert elements["ref_61"]["secret"] and elements["ref_61"]["label"] == "Card number"
    assert elements["ref_63"]["label"] == "field" and elements["ref_63"]["type"] == "textbox"
    assert parse_elements('link [ref_9] href="/wiki/Main_Page"')[0]["label"] == "/wiki/Main_Page"


def test_page_text_is_split_from_its_header() -> None:
    assert split_page_text(PAGE_TEXT) == (
        "Example Domain",
        "https://example.org/",
        "Example Domain\nThis domain is for use in documentation examples.",
    )
    assert split_page_text("just text") == ("", "", "just text")
    assert split_page_text("Other: x\n---\nbody") == ("", "", "body")


# -- which claude, and whether at all ------------------------------------------------------


def test_it_is_on_unless_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(claude_chrome.SWITCH, "on")
    assert claude_chrome.claude_in_chrome_enabled()
    port = claude_chrome.claude_chrome()
    assert isinstance(port, ClaudeChromePort) and claude_chrome.claude_chrome() is port
    monkeypatch.setenv(claude_chrome.SWITCH, "off")
    assert claude_chrome.claude_chrome() is None


def test_the_claude_it_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    exe = tmp_path / "claude.exe"
    exe.write_text("")
    monkeypatch.setenv(claude_chrome.CLI_ENV, str(exe))
    assert claude_cli() == str(exe)
    monkeypatch.setenv(claude_chrome.CLI_ENV, str(tmp_path / "missing.exe"))
    assert claude_cli() is None

    monkeypatch.delenv(claude_chrome.CLI_ENV)
    monkeypatch.setattr(claude_chrome.shutil, "which", lambda name: "/usr/bin/claude")
    sdk = tmp_path / "sdk"
    (sdk / "_bundled").mkdir(parents=True)
    fake_sdk = type(sys)("claude_agent_sdk")
    fake_sdk.__file__ = str(sdk / "__init__.py")
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_sdk)
    assert claude_cli() == "/usr/bin/claude"  # the SDK carries no copy
    bundled = sdk / "_bundled" / ("claude.exe" if claude_chrome.os.name == "nt" else "claude")
    bundled.write_text("")
    assert claude_cli() == str(bundled)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    assert claude_cli() == "/usr/bin/claude"


async def test_without_a_claude_it_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claude_chrome, "claude_cli", lambda: None)
    with pytest.raises(BridgeUnavailableError, match="no Claude Code"):
        await claude_chrome._connect_claude_code()

    started: list[str] = []

    class Started(StdioTools):
        async def start(self, timeout: float = 0) -> StdioTools:
            started.append(self._command)
            return self

    monkeypatch.setattr(claude_chrome, "claude_cli", lambda: "claude")
    monkeypatch.setattr(claude_chrome, "StdioTools", Started)
    assert isinstance(await claude_chrome._connect_claude_code(), Started)
    assert started == ["claude"]


async def test_closing_it_closes_its_server(monkeypatch: pytest.MonkeyPatch) -> None:
    await claude_chrome.close_claude_chrome()  # nothing started: nothing to do
    port, tools = port_with({"list_connected_browsers": "[]"})
    claude_chrome._port = port
    await port.ready()
    await claude_chrome.close_claude_chrome()
    assert tools.closed and claude_chrome._port is None


# -- the server ----------------------------------------------------------------------------


async def test_it_speaks_mcp_with_a_real_server() -> None:
    server = Path(__file__).with_name("fake_claude_in_chrome.py")
    tools = await StdioTools(sys.executable, [str(server)]).start()
    try:
        listed = await tools.call_tool("list_connected_browsers", {}, 10)
        assert '"deviceId":"d1"' in listed.text and not listed.error
        shot = await tools.call_tool("computer", {"action": "screenshot", "tabId": 7}, 10)
        assert shot.text == "did screenshot on 7" and shot.images == ["anBlZw=="]
        assert (await tools.call_tool("nope", {}, 10)).error
    finally:
        await tools.close()
    with pytest.raises(BridgeUnavailableError, match="not running"):
        await tools.call_tool("list_connected_browsers", {}, 10)
    await tools.close()  # twice is fine


async def test_a_server_that_cannot_start_or_never_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(BridgeUnavailableError, match="stopped"):
        await StdioTools(str(tmp_path / "no-such-claude.exe")).start()
    # One that never answers is cancelled when it does not stop either.
    monkeypatch.setattr(claude_chrome, "STOP_SECONDS", 0.2)
    silent = StdioTools(sys.executable, ["-c", "import time; time.sleep(30)"])
    with pytest.raises(BridgeUnavailableError, match="in time"):
        await silent.start(timeout=1.0)
    assert silent._task is None


async def test_a_failed_start_waits_before_trying_again() -> None:
    clock = Clock()
    attempts: list[int] = []

    async def connect() -> FakeTools:
        attempts.append(1)
        raise OSError("no claude")

    port = ClaudeChromePort(connect=connect, clock=clock)
    assert not await port.ready()
    clock.now += claude_chrome.NOT_READY_TTL_SECONDS + 1
    assert not await port.ready()
    assert len(attempts) == 1
    clock.now += claude_chrome.RETRY_SECONDS
    with pytest.raises(BridgeUnavailableError, match="could not start"):
        await port.call("tabs")
    assert len(attempts) == 2


# -- is the Claude extension there ---------------------------------------------------------


async def test_ready_means_a_browser_is_paired_and_is_asked_now_and_then() -> None:
    clock = Clock()
    browsers = {"list_connected_browsers": '[{"deviceId":"d1","name":"Browser 1"}]'}
    port, tools = port_with(browsers, clock)
    assert await port.ready()
    assert await port.ready()
    assert len(tools.sent("list_connected_browsers")) == 1
    clock.now += claude_chrome.READY_TTL_SECONDS + 1
    tools.answers["list_connected_browsers"] = "[]"
    assert not await port.ready()
    clock.now += 1
    assert not await port.ready()
    assert len(tools.sent("list_connected_browsers")) == 2
    assert (await port.call("status")) == {"connected": False}


async def test_not_paired_is_unavailable_whatever_the_tool() -> None:
    clock = Clock()
    port, tools = port_with(
        {"list_connected_browsers": '[{"deviceId":"d1"}]', "tabs_context_mcp": NOT_PAIRED}, clock
    )
    assert await port.ready()
    with pytest.raises(BridgeUnavailableError, match="not connected"):
        await port.call("tabs")
    assert not await port.ready()  # it no longer counts as there
    tools.answers["tabs_context_mcp"] = "[Claude in Chrome] Claude in Chrome isn't paired (x)"
    with pytest.raises(BridgeUnavailableError):
        await port.call("tabs")
    # A page that only talks about it is a page.
    tools.answers["get_page_text"] = "Title: t\nURL: u\n---\n" + NOT_PAIRED
    page = await port.call("read", {"tabId": 11, "maxText": 9000})
    assert "not connected" in page["text"]


async def test_its_errors_and_silences_are_bridge_errors() -> None:
    port, tools = port_with(
        {"tabs_close_mcp": ToolResult("Tab 5 is not in this group", error=True)}
    )
    with pytest.raises(BridgeError, match="not in this group"):
        await port.call("close", {"tabId": 5})
    tools.answers["tabs_close_mcp"] = ToolResult("", error=True)
    with pytest.raises(BridgeError, match="refused 'tabs_close_mcp'"):
        await port.call("close", {"tabId": 5})
    tools.answers["tabs_context_mcp"] = RuntimeError("pipe broke")
    with pytest.raises(BridgeError, match="failed on 'tabs_context_mcp': pipe broke"):
        await port.call("tabs")
    tools.answers["tabs_context_mcp"] = TimeoutError()
    with pytest.raises(BridgeError, match="in time"):
        await port.call("tabs")
    with pytest.raises(BridgeError, match="no 'hello'"):
        await port.call("hello")


async def test_a_server_that_went_away_is_started_again() -> None:
    port, tools = port_with({"tabs_context_mcp": BridgeUnavailableError("server stopped")})
    with pytest.raises(BridgeUnavailableError):
        await port.call("tabs")
    assert port._tools is None
    tools.answers["tabs_context_mcp"] = CONTEXT
    assert [t["id"] for t in (await port.call("tabs"))["tabs"]] == [11, 12]


# -- tabs ----------------------------------------------------------------------------------


async def test_it_lists_only_its_group_and_finds_a_tab_there() -> None:
    port, _ = port_with({"tabs_context_mcp": CONTEXT})
    listed = await port.call("tabs")
    assert listed["scope"] == "group" and len(listed["tabs"]) == 2
    assert (await port.call("info", {"tabId": 11}))["title"] == "Example Domain"
    with pytest.raises(BridgeError, match="No tab with id: 5"):
        await port.call("info", {"tabId": 5})
    assert not port.sees_all_tabs and "Claude extension" in port.place


async def test_its_first_page_opens_with_its_group_and_the_focus_goes_back(
    no_window_moves: list[Any],
) -> None:
    created = '{"availableTabs":[{"tabId":21,"title":"New Tab","url":"chrome://newtab/"}]}'

    def context(args: dict[str, Any]) -> str:
        return created if args.get("createIfEmpty") else NO_GROUP

    port, tools = port_with(
        {"tabs_context_mcp": context, "navigate": NAVIGATED.replace("11", "21")}
    )
    info = await port.call("open", {"url": "https://example.org/", "focus": False})
    assert info["id"] == 21 and info["title"] == "Example Domain"
    assert tools.sent("navigate") == [{"url": "https://example.org/", "tabId": 21}]
    assert no_window_moves == [77]
    assert not tools.sent("tabs_create_mcp")


async def test_another_page_opens_in_a_new_tab_of_its_group() -> None:
    port, tools = port_with(
        {
            "tabs_context_mcp": CONTEXT,
            "tabs_create_mcp": "Created new tab. Tab ID: 13",
            "navigate": "Navigated to https://example.com/",
        }
    )
    info = await port.call("open", {"url": "https://example.com/"})
    assert info == {"id": 13, "url": "https://example.com/", "title": "", "jarvis": True}


async def test_a_tab_it_could_not_open_is_an_error(no_window_moves: list[Any]) -> None:
    port, tools = port_with({"tabs_context_mcp": CONTEXT, "tabs_create_mcp": "Created."})
    with pytest.raises(BridgeError, match="which tab"):
        await port.call("open", {"url": "https://example.com/"})
    tools.answers["tabs_context_mcp"] = NO_GROUP
    with pytest.raises(BridgeError, match="did not open"):
        await port.call("open", {"url": "https://example.com/"})
    assert no_window_moves == [77, 77]


async def test_navigation_back_forward_and_forgetting_the_page() -> None:
    port, tools = port_with({"navigate": NAVIGATED, "read_page": TREE})
    await port.call("read", {"tabId": 11, "maxText": 1})
    await port.call("click", {"tabId": 11, "elementId": "ref_58"})
    assert (await port.call("navigate", {"tabId": 11, "url": "https://example.org/"}))["id"] == 11
    assert 11 not in port._elements and 11 not in port._clicked
    await port.call("back", {"tabId": 11})
    await port.call("forward", {"tabId": 11})
    assert [a["url"] for a in tools.sent("navigate")][1:] == ["back", "forward"]
    assert await port.call("activate", {"tabId": 11, "focus": True}) == {"ok": True}


async def test_closing_a_tab_forgets_what_it_knew_of_it() -> None:
    port, tools = port_with({"read_page": TREE, "computer": ToolResult("shot", ["AAA"])})
    await port.call("read", {"tabId": 11, "maxText": 1})
    await port.call("capture", {"tabId": 11})
    assert await port.call("close", {"tabId": 11}) == {"ok": True}
    assert tools.sent("tabs_close_mcp") == [{"tabId": 11}]
    assert not (port._elements or port._viewport or port._frames)


# -- the page ------------------------------------------------------------------------------


async def test_it_reads_the_text_and_what_can_be_used() -> None:
    port, tools = port_with({"get_page_text": PAGE_TEXT, "read_page": TREE})
    page = await port.call("read", {"tabId": 11, "maxText": 20})
    assert page["title"] == "Example Domain" and page["url"] == "https://example.org/"
    assert page["text"] == "Example Domain\nThis "
    assert [e["id"] for e in page["elements"]][:2] == ["ref_1", "ref_58"]
    assert port._viewport[11] == (1920, 919)
    assert tools.sent("read_page") == [{"tabId": 11, "filter": "interactive", "max_chars": 20000}]
    # Only the elements, when that is all a click needs.
    tools.calls.clear()
    small = await port.call("read", {"tabId": 11, "maxText": 1})
    assert small["text"] == "" and not tools.sent("get_page_text")


async def test_clicks_go_to_the_element() -> None:
    port, tools = port_with()
    assert await port.call("click", {"tabId": 11, "elementId": "ref_1"}) == {"ok": True}
    assert tools.sent("computer") == [{"action": "left_click", "ref": "ref_1", "tabId": 11}]


async def test_typing_fills_the_field_and_never_a_secret() -> None:
    port, tools = port_with({"read_page": TREE})
    with pytest.raises(BridgeError, match="no-field"):
        await port.call("type", {"tabId": 11, "elementId": "", "text": "x"})
    for secret in ("ref_60", "ref_61"):
        with pytest.raises(BridgeError, match="secret-field"):
            await port.call("type", {"tabId": 11, "elementId": secret, "text": "hunter2"})
    with pytest.raises(BridgeError, match="not-found"):
        await port.call("type", {"tabId": 11, "elementId": "ref_404", "text": "x"})
    assert not tools.sent("form_input")

    await port.call("type", {"tabId": 11, "elementId": "ref_58", "text": "Ada", "clear": True})
    assert tools.sent("form_input") == [{"ref": "ref_58", "value": "Ada", "tabId": 11}]
    # Read once for the page, again for the element it did not know, never for one it did.
    assert len(tools.sent("read_page")) == 2


async def test_typing_where_it_clicked_and_where_filling_fails() -> None:
    port, tools = port_with({"read_page": TREE})
    await port.call("click", {"tabId": 11, "elementId": "ref_60"})
    with pytest.raises(BridgeError, match="secret-field"):
        await port.call("type", {"tabId": 11, "elementId": "", "text": "hunter2"})

    await port.call("click", {"tabId": 11, "elementId": "ref_58"})
    tools.calls.clear()
    await port.call("type", {"tabId": 11, "text": "Ada", "clear": False})
    assert tools.sent("computer") == [
        {"action": "left_click", "ref": "ref_58", "tabId": 11},
        {"action": "type", "text": "Ada", "tabId": 11},
    ]
    tools.answers["form_input"] = ToolResult("Element is not a form field", error=True)
    tools.calls.clear()
    await port.call("type", {"tabId": 11, "elementId": "ref_58", "text": "Ada"})
    assert len(tools.sent("form_input")) == 1 and len(tools.sent("computer")) == 2


async def test_keys_and_scrolling() -> None:
    port, tools = port_with({"read_page": TREE})
    await port.call("key", {"tabId": 11, "key": "Enter"})
    await port.call("key", {"tabId": 11, "key": "Space"})
    await port.call("scroll", {"tabId": 11, "direction": "down"})
    await port.call("read", {"tabId": 11, "maxText": 1})
    await port.call("scroll", {"tabId": 11, "direction": "up"})
    sent = tools.sent("computer")
    assert [s.get("text") for s in sent[:2]] == ["Enter", "space"]
    assert sent[2]["coordinate"] == [640, 400] and sent[2]["scroll_direction"] == "down"
    assert sent[3]["coordinate"] == [960, 459] and sent[3]["scroll_direction"] == "up"


async def test_a_sign_in_page_is_told_by_its_password_field() -> None:
    port, _ = port_with({"get_page_text": PAGE_TEXT, "read_page": TREE})
    probe = await port.call("probe", {"tabId": 11})
    assert probe == {
        "href": "https://example.org/",
        "address": "example.org/",
        "password": True,
        "head": "Example Domain\nThis domain is for use in documentation examples.",
    }
    port, _ = port_with(
        {
            "get_page_text": "no header here",
            "read_page": 'link "Home" [ref_1] href="/"',
            "tabs_context_mcp": '{"availableTabs":[{"tabId":11,"title":"t",'
            '"url":"https://accounts.example.com/signin?next=1#x"}]}',
        }
    )
    probe = await port.call("probe", {"tabId": 11})
    assert probe["address"] == "accounts.example.com/signin" and not probe["password"]


async def test_media_actions_are_kwamis_own_by_name() -> None:
    port, tools = port_with({"javascript_tool": "paused\n\nTab Context:\n- Executed on tabId: 11"})
    assert await port.call("media", {"tabId": 11, "action": "pause"}) == {"result": "paused"}
    sent = tools.sent("javascript_tool")[0]
    assert sent["action"] == "javascript_exec" and "m.pause()" in sent["text"]
    await port.call("media", {"tabId": 11, "action": "volume", "level": 0.3})
    assert "m.volume=0.3" in tools.sent("javascript_tool")[1]["text"]
    assert await port.call("media", {"tabId": 11, "action": "rm -rf"}) == {
        "result": "unknown-action"
    }
    assert len(tools.sent("javascript_tool")) == 2


async def test_pictures_are_taken_at_most_now_and_then() -> None:
    clock = Clock()
    port, tools = port_with({"computer": ToolResult("shot", ["AAA"])}, clock)
    assert await port.call("capture", {"tabId": 11, "quality": 60}) == {"jpeg": "AAA"}
    tools.answers["computer"] = ToolResult("shot", ["BBB"])
    assert await port.call("capture", {"tabId": 11}) == {"jpeg": "AAA"}
    clock.now += claude_chrome.CAPTURE_EVERY_SECONDS
    assert await port.call("capture", {"tabId": 11}) == {"jpeg": "BBB"}
    assert tools.sent("computer")[0] == {"action": "screenshot", "scale": 0.5, "tabId": 11}
    clock.now += claude_chrome.CAPTURE_EVERY_SECONDS
    tools.answers["computer"] = ToolResult("no picture")
    with pytest.raises(BridgeError, match="not-visible"):
        await port.call("capture", {"tabId": 11})


async def test_the_tab_it_went_to_is_found_among_the_others() -> None:
    port, _ = port_with({"navigate": NAVIGATED})
    went = await port.call("navigate", {"tabId": 12, "url": "https://example.net/"})
    assert went["id"] == 12 and went["title"] == "Nueva pestaña"


async def test_only_text_and_pictures_are_kept_from_an_answer() -> None:
    class Item:
        def __init__(self, kind: str, **values: str) -> None:
            self.type = kind
            self.__dict__.update(values)

    class Answer:
        content = [Item("text", text="a"), Item("resource", uri="x"), Item("image", data="QQ==")]
        is_error = False

    class Session:
        async def call_tool(self, name: str, arguments: dict[str, Any], **_: Any) -> Answer:
            return Answer()

    tools = StdioTools("claude")
    tools._session = Session()
    assert await tools.call_tool("computer", {}, 5) == ToolResult("a", ["QQ=="], False)


async def test_a_port_never_started_closes_quietly() -> None:
    await ClaudeChromePort().close()


def test_media_scripts_by_name() -> None:
    from src.browser.media_scripts import (
        FIRST_RESULT_JS,
        MEDIA_ACTION_JS,
        NOW_PLAYING_JS,
        media_script,
    )

    assert media_script("first_result") == FIRST_RESULT_JS
    assert media_script("now_playing") == NOW_PLAYING_JS
    assert media_script("mute") == MEDIA_ACTION_JS["mute"]
    assert "m.volume=1.0" in (media_script("volume", 7) or "")
    assert "m.volume=0.0" in (media_script("volume", -1) or "")
    assert media_script("format c:") is None
