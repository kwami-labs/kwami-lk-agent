"""The work thread in the user's own Chrome (2026-11-17): their tabs, their sign-ins."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.brain import browser_tool
from src.brain.browser_tool import BrowserTools
from src.browser.browser_session import CloudBrowserSession
from src.browser.chrome_bridge import BridgeError
from src.browser.user_chrome_session import UserChromeSession


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


PAGE = {"title": "Mail", "url": "https://mail.example/", "text": "2 new", "elements": []}
TABS = {
    "tabs": [
        {"id": 1, "title": "Mail", "url": "https://mail.example/", "focused": True},
        {"id": 2, "title": "Settings", "url": "chrome://settings/"},
    ]
}


@pytest.fixture(autouse=True)
def quick(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(browser_tool, "_WINDOW", asyncio.Lock())
    monkeypatch.setattr(browser_tool, "SIGN_IN_SETTLE_SECONDS", 0)
    monkeypatch.setattr(browser_tool, "SIGN_IN_POLL_SECONDS", 0)

    async def same(url: str) -> str:
        return url

    monkeypatch.setattr(browser_tool, "validate_url_async", same)


def theirs(answers: dict[str, Any]) -> tuple[BrowserTools, FakeBridge]:
    bridge = FakeBridge(
        {"open": {"id": 7, "url": "https://mail.example/"}, "read": PAGE, **answers}
    )
    session = UserChromeSession(bridge=bridge)
    session._frames_wanted = False
    return BrowserTools("user", session, their_chrome=True), bridge


async def test_a_task_takes_their_chrome_when_it_is_there(monkeypatch: pytest.MonkeyPatch) -> None:
    async def ready() -> bool:
        return True

    monkeypatch.setattr(browser_tool, "user_chrome_ready", ready)
    assert isinstance(await BrowserTools("u", their_chrome=True)._browser(), UserChromeSession)


async def test_otherwise_or_for_jarvis_own_sites_it_takes_jarvis_chrome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def ready() -> bool:
        return False

    monkeypatch.setattr(browser_tool, "user_chrome_ready", ready)
    assert isinstance(await BrowserTools("u", their_chrome=True)._browser(), CloudBrowserSession)

    async def never() -> bool:
        raise AssertionError("the daily look at Jarvis's sites never asks for theirs")

    monkeypatch.setattr(browser_tool, "user_chrome_ready", never)
    assert isinstance(await BrowserTools("u")._browser(), CloudBrowserSession)
    await BrowserTools("u").close()


async def test_signing_in_puts_the_page_in_front_and_waits_for_them() -> None:
    probes = iter([{"password": True}, {"password": True}, {"password": False}])
    tools, bridge = theirs({"activate": {}, "probe": lambda _args: next(probes)})
    said = await tools.call("sign_in", {"url": "https://mail.example/login"})
    assert said.startswith("The user is in") and "2 new" in said
    assert ("activate", {"tabId": 7, "focus": True}) in bridge.calls


async def test_a_sign_in_left_unfinished_ends_the_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(browser_tool, "SIGN_IN_WAIT_SECONDS", 0)
    tools, _bridge = theirs({"activate": {}, "probe": {"password": True}})
    said = await tools.call("sign_in", {"url": "https://mail.example/login"})
    assert "not finished signing in" in said


async def test_it_lists_their_tabs_and_works_in_one() -> None:
    tools, bridge = theirs({"tabs": TABS, "info": {"id": 1, "url": "https://mail.example/"}})
    listed = await tools.call("tabs", {})
    assert "1 web tabs open" in listed and "1. Mail -- https://mail.example/ [in front]" in listed
    assert "Settings" not in listed
    assert await tools.call("tabs", {"which": "todas"}) == listed
    read = await tools.call("tabs", {"which": "mail"})
    assert "You now work in this tab" in read and "2 new" in read
    assert "No tab matches" in await tools.call("tabs", {"which": "bank"})


async def test_their_tabs_are_out_of_reach_in_jarvis_own_chrome() -> None:
    class Own:
        is_active = False

        async def close(self, **_: Any) -> None:
            return None

    tools = BrowserTools("user", Own())  # type: ignore[arg-type]
    assert "not connected to Kwami" in await tools.call("tabs", {})


async def test_what_their_chrome_refuses_is_said_in_words() -> None:
    tools, _bridge = theirs({"tabs": BridgeError("paused")})
    said = await tools.safe_call("tabs", {})
    assert said.startswith("The user's Chrome did not do it") and "paused" in said


# -- the user's list of sites (Settings > Access), as the voice keeps it ----------------


class FakeAccess:
    """Levels by host; what is not listed is at `default`."""

    def __init__(self, levels: dict[str, str], default: str = "account") -> None:
        from src.browser.access import Decision, allows, host_of

        self.levels, self.default = levels, default
        self.enabled = True
        self.asked: list[tuple[str, str, str]] = []
        self._decision, self._allows, self._host = Decision, allows, host_of

    async def decide(self, user_id: str, address: str, needed: str) -> Any:
        self.asked.append((user_id, address, needed))
        host = self._host(address)
        level = self.levels.get(host, self.default)
        if level == "boom":
            raise RuntimeError("database down")
        return self._decision(host, level, host, self._allows(level, needed))


@pytest.fixture
def listed(monkeypatch: pytest.MonkeyPatch) -> FakeAccess:
    access = FakeAccess({"bank.example": "blocked", "news.example": "read"})
    monkeypatch.setattr(browser_tool, "_site_access", access)
    return access


def kept(answers: dict[str, Any], **kwargs: Any) -> tuple[BrowserTools, FakeBridge]:
    bridge = FakeBridge(
        {"open": {"id": 7, "url": "https://news.example/"}, "read": PAGE, **answers}
    )
    session = UserChromeSession(bridge=bridge)
    session._frames_wanted = False
    return BrowserTools("kwami", session, their_chrome=True, site_user_id="u-1", **kwargs), bridge


async def test_a_blocked_site_is_never_opened(listed: FakeAccess) -> None:
    tools, bridge = kept({})
    said = await tools.call("open", {"url": "https://bank.example/login"})
    assert "Settings > Access" in said and "not allowed to open it" in said
    assert bridge.calls == []
    assert listed.asked == [("u-1", "https://bank.example/login", "read")]


async def test_each_step_needs_its_level_on_the_page_in_front(listed: FakeAccess) -> None:
    tools, bridge = kept({"info": {"url": "https://news.example/a"}, "click": {"ok": True}})
    assert "2 new" in await tools.call("open", {"url": "https://news.example/"})
    for name, args in (("click", {"element_id": "el-0"}), ("type", {"text": "x"})):
        said = await tools.call(name, args)
        assert "keeps you from that on news.example" in said, name
    assert [needed for *_, needed in listed.asked][-2:] == ["browse", "write"]
    assert not any(cmd in ("click", "type") for cmd, _ in bridge.calls)


async def test_a_page_that_moved_somewhere_blocked_is_left(listed: FakeAccess) -> None:
    tools, bridge = kept({"info": {"url": "https://bank.example/"}, "close": {}})
    said = await tools.call("open", {"url": "https://news.example/"})
    assert "bank.example" in said, "a redirect is checked as it lands"
    assert "No page is open" in await tools.call("read", {})
    assert ("close", {"tabId": 7}) in bridge.calls, "Jarvis's tab leaves the refused page"


async def test_signing_in_is_the_account_level(listed: FakeAccess) -> None:
    tools, bridge = kept({})
    said = await tools.call("sign_in", {"url": "https://news.example/login"})
    assert "keeps you from that on news.example" in said
    assert listed.asked[-1][2] == "account"
    assert bridge.calls == []


async def test_their_tabs_on_blocked_sites_stay_hidden(listed: FakeAccess) -> None:
    tabs = {
        "tabs": [
            {"id": 1, "title": "Mi banco", "url": "https://bank.example/"},
            {"id": 2, "title": "Noticias", "url": "https://news.example/"},
        ]
    }
    tools, _bridge = kept({"tabs": tabs, "info": {"id": 1, "url": "https://bank.example/"}})
    listed_tabs = await tools.call("tabs", {})
    assert (
        "Mi banco" not in listed_tabs
        and "1. (a site the user's list keeps you out of)" in listed_tabs
    )
    assert "2. Noticias" in listed_tabs
    assert "not allowed to open it" in await tools.call("tabs", {"which": "1"})


async def test_without_a_list_or_a_user_nothing_is_refused(
    listed: FakeAccess, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools, _bridge = kept({})
    tools._site_user_id = ""
    assert "2 new" in await tools.call("open", {"url": "https://bank.example/"})
    assert listed.asked == []
    await tools.close()

    monkeypatch.setattr(browser_tool, "_site_access", None)
    tools, _bridge = kept({})
    assert "2 new" in await tools.call("open", {"url": "https://bank.example/"})
    await tools.close()

    listed.enabled = False
    monkeypatch.setattr(browser_tool, "_site_access", listed)
    tools, _bridge = kept({})
    assert "2 new" in await tools.call("open", {"url": "https://bank.example/"})
    await tools.close()


async def test_a_list_that_cannot_be_read_does_not_stop_the_task(listed: FakeAccess) -> None:
    listed.levels["news.example"] = "boom"
    tools, _bridge = kept({"info": "odd"})
    assert "2 new" in await tools.call("open", {"url": "https://news.example/"})
    assert "2 new" in await tools.call("read", {}), "no address to check: nothing refused"


async def test_an_address_that_cannot_be_asked_counts_as_none(listed: FakeAccess) -> None:
    tools, bridge = kept({"info": BridgeError("slow")})
    bridge.answers["info"] = BridgeError("slow")
    await tools.call("open", {"url": "https://news.example/"})
    session = tools._session
    assert session is not None

    async def broken() -> str:
        raise RuntimeError("gone")

    session.page_address = broken  # type: ignore[method-assign]
    assert await tools._here(session) == ""


def test_the_work_thread_hands_the_list_to_every_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(browser_tool, "_site_access", None)
    marker = FakeAccess({})
    browser_tool.use_site_access(marker)  # type: ignore[arg-type]
    assert browser_tool._site_access is marker


async def test_a_page_with_no_web_address_is_not_a_site(listed: FakeAccess) -> None:
    tools, _bridge = kept({})
    assert await tools._refusal("read", "") is None
    assert await tools._refusal("read", "about:blank") is None
    assert listed.asked == []


async def test_through_claude_in_chrome_it_says_it_sees_only_its_group() -> None:
    class ClaudeWay(FakeBridge):
        sees_all_tabs = False

    bridge = ClaudeWay({"tabs": TABS})
    session = UserChromeSession(bridge=bridge)
    listed = await BrowserTools("user", session, their_chrome=True).call("tabs", {})
    assert "1 web tabs open in your tab group" in listed and "drags a tab into it" in listed
