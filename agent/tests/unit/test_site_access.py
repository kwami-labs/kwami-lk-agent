"""The user's list of sites: what Jarvis may do where, and that asking grants nothing."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.browser import access as a
from src.browser.access import SiteAccess
from src.domain.tool_result import was_refused
from src.runtime.container import AgentDeps
from tests.unit.test_builtin_navigation_tools import _tools

USER = "11111111-1111-1111-1111-111111111111"


class FakeStore:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows if rows is not None else []
        self.reads = 0
        self.inserted: list[dict[str, Any]] = []
        self.fail_reads = False
        self.fail_writes = False

    async def select(self, table: str, params: dict[str, str], *, what: str = ""):
        assert table == a.TABLE and params["user_id"] == f"eq.{USER}"
        self.reads += 1
        return None if self.fail_reads else [dict(r) for r in self.rows]

    async def insert(self, table: str, rows: dict[str, Any], **_: Any) -> bool:
        if self.fail_writes:
            return False
        self.inserted.append(rows)
        self.rows.append(rows)
        return True


def _row(pattern: str, access: str, status: str = "active") -> dict[str, Any]:
    return {"pattern": pattern, "access": access, "status": status}


class TestHostOf:
    @pytest.mark.parametrize(
        ("address", "host"),
        [
            ("https://www.GitHub.com/x?y=1", "github.com"),
            ("mail.google.com/u/0", "mail.google.com"),
            ("*.amazon.es", "amazon.es"),
            ("https://bücher.de/", "xn--bcher-kva.de"),
            ("http://[::1", ""),
            ("", ""),
        ],
    )
    def test_an_address_comes_down_to_its_host(self, address: str, host: str) -> None:
        assert a.host_of(address) == host

    def test_a_name_idna_cannot_encode_is_kept_as_is(self) -> None:
        name = "é" * 70 + ".com"
        assert a.host_of(name) == name


class TestLevels:
    def test_each_level_includes_the_ones_before(self) -> None:
        assert a.allows("account", "read") and a.allows("write", "browse")
        assert not a.allows("read", "browse") and not a.allows("blocked", "read")

    def test_an_unknown_level_allows_nothing(self) -> None:
        assert not a.allows("root", "read") and not a.allows("account", "root")

    def test_the_longest_pattern_decides(self) -> None:
        rules = {"google.com": "read", "mail.google.com": "account"}
        assert a.level_for("mail.google.com", rules) == "account"
        assert a.level_for("maps.google.com", rules) == "read"
        assert a.level_for("notgoogle.com", rules, "browse") == "browse"


class TestDecide:
    async def test_without_a_database_every_site_is_open(self) -> None:
        access = SiteAccess(None)
        assert not access.enabled
        decision = await access.decide(USER, "https://bank.com", "account")
        assert decision.allowed and decision.level == "account"

    async def test_the_list_decides_and_pending_requests_grant_nothing(self) -> None:
        store = FakeStore(
            [
                _row("*", "read"),
                _row("github.com", "write"),
                _row("evil.com", "blocked"),
                _row("bank.com", "account", status="pending"),
                _row("odd.com", "superuser"),
            ]
        )
        access = SiteAccess(store)

        assert (await access.decide(USER, "https://api.github.com", "write")).allowed
        evil = await access.decide(USER, "https://www.evil.com/", "read")
        assert not evil.allowed and evil.pattern == "evil.com"
        bank = await access.decide(USER, "bank.com", "browse")
        assert not bank.allowed and bank.level == "read" and bank.pattern == "*"
        assert (await access.decide(USER, "odd.com", "read")).level == "read"

    async def test_the_list_is_read_once_for_a_few_seconds(self) -> None:
        store = FakeStore([_row("github.com", "read")])
        access = SiteAccess(store)
        await access.decide(USER, "github.com", "read")
        await access.decide(USER, "github.com", "browse")
        assert store.reads == 1
        access.forget(USER)
        await access.decide(USER, "github.com", "read")
        assert store.reads == 2

    async def test_a_failed_read_keeps_the_last_list(self, monkeypatch) -> None:
        store = FakeStore([_row("evil.com", "blocked")])
        access = SiteAccess(store)
        await access.decide(USER, "evil.com", "read")
        monkeypatch.setattr(a, "CACHE_SECONDS", 0.0)
        store.fail_reads = True
        assert not (await access.decide(USER, "evil.com", "read")).allowed

    async def test_a_failed_first_read_leaves_the_sites_open(self) -> None:
        store = FakeStore()
        store.fail_reads = True
        assert (await SiteAccess(store).decide(USER, "a.com", "account")).allowed

    async def test_no_user_means_no_list(self) -> None:
        store = FakeStore([_row("*", "blocked")])
        assert (await SiteAccess(store).decide("", "a.com", "read")).allowed
        assert store.reads == 0


class TestRequest:
    async def test_it_leaves_a_pending_row_and_grants_nothing(self) -> None:
        store = FakeStore([_row("*", "read")])
        access = SiteAccess(store)

        said = await access.request(USER, "https://www.amazon.es/x", "browse", " prices " * 100)

        assert "waiting" in said and "amazon.es" in said
        row = store.inserted[0]
        assert (row["pattern"], row["access"], row["status"]) == ("amazon.es", "browse", "pending")
        assert len(row["reason"]) == a.MAX_REASON_CHARS
        assert not (await access.decide(USER, "amazon.es", "browse")).allowed

    async def test_an_empty_reason_is_none(self) -> None:
        store = FakeStore([_row("*", "read")])
        await SiteAccess(store).request(USER, "a.com", "write", "  ")
        assert store.inserted[0]["reason"] is None

    @pytest.mark.parametrize(
        ("rows", "address", "level", "words"),
        [
            ([], "a.com", "account", "already may"),
            (
                [_row("*", "read"), _row("a.com", "browse", "pending")],
                "a.com",
                "write",
                "already asked",
            ),
            ([_row("*", "read"), _row("a.com", "blocked")], "a.com", "read", "themselves"),
        ],
    )
    async def test_nothing_is_asked_twice(self, rows, address, level, words) -> None:
        store = FakeStore(rows)
        assert words in await SiteAccess(store).request(USER, address, level, "x")
        assert store.inserted == []

    @pytest.mark.parametrize(
        ("address", "level"), [("a.com", "blocked"), ("a.com", "x"), ("nohost", "read")]
    )
    async def test_a_bad_request_is_refused(self, address, level) -> None:
        store = FakeStore([_row("*", "blocked")])
        said = await SiteAccess(store).request(USER, address, level, "x")
        assert "Ask for one of" in said or "not a site" in said
        assert store.inserted == []

    async def test_without_a_database_there_is_nothing_to_ask(self) -> None:
        assert "every site" in await SiteAccess(None).request(USER, "a.com", "read", "x")

    async def test_requests_have_a_ceiling(self, monkeypatch) -> None:
        monkeypatch.setattr(a, "MAX_PENDING", 1)
        store = FakeStore([_row("*", "read"), _row("b.com", "write", "pending")])
        assert "many requests" in await SiteAccess(store).request(USER, "a.com", "write", "x")

    async def test_a_failed_write_says_so(self) -> None:
        store = FakeStore([_row("*", "read")])
        store.fail_writes = True
        assert "could not" in await SiteAccess(store).request(USER, "a.com", "write", "x")


class TestSummary:
    async def test_it_lists_sites_default_and_requests(self) -> None:
        store = FakeStore(
            [_row("*", "read"), _row("github.com", "write"), _row("a.com", "browse", "pending")]
        )
        text = await SiteAccess(store).summary(USER)
        assert text.splitlines()[0] == f"Every other site: {a.LEVEL_WORDS['read']}."
        assert "- github.com:" in text and "Waiting for the user: a.com (browse)." in text

    async def test_without_a_database(self) -> None:
        assert "any public site" in await SiteAccess(None).summary(USER)


class TestRefusalWords:
    def test_a_blocked_site_is_not_to_be_worked_around(self) -> None:
        d = a.Decision(host="evil.com", level="blocked", pattern="evil.com", allowed=False)
        assert "Do not try another way" in a.refusal_words(d, "read")

    def test_a_lower_level_points_at_the_request(self) -> None:
        d = a.Decision(host="a.com", level="read", pattern="*", allowed=False)
        assert "request_site_access" in a.refusal_words(d, "write")


# -- the browser tools ask the list ------------------------------------------------


def _context(store: FakeStore | None, user_id: str = USER) -> Any:
    deps = AgentDeps.__new__(AgentDeps)
    deps.bridge = SimpleNamespace(site_access=SiteAccess(store), user_id=user_id)
    return SimpleNamespace(userdata=deps)


class Page:
    """Adds the page's own address to the navigation tests' fake browser."""

    def __init__(self, session: Any, address: str) -> None:
        self.address = address
        self.left = 0

        async def page_address() -> str:
            return self.address

        async def leave_page() -> None:
            self.left += 1

        session.page_address = page_address
        session.leave_page = leave_page


async def test_a_blocked_site_is_never_opened() -> None:
    tools, session = _tools(active=False)
    ctx = _context(FakeStore([_row("evil.com", "blocked")]))

    result = await tools.navigate_to(ctx, "https://www.evil.com/x")

    assert was_refused(result) and "blocked evil.com" in result
    assert session.started == []


async def test_a_redirect_to_a_blocked_site_leaves_the_page() -> None:
    tools, session = _tools(active=True)
    page = Page(session, "https://evil.com/landing")
    ctx = _context(FakeStore([_row("evil.com", "blocked")]))

    result = await tools.navigate_to(ctx, "https://good.com")

    assert was_refused(result) and page.left == 1
    assert session.navigated == ["https://good.com"]


async def test_each_step_needs_its_level() -> None:
    tools, session = _tools(active=True)
    Page(session, "https://shop.com/item")
    ctx = _context(FakeStore([_row("shop.com", "read")]))

    assert not was_refused(await tools.read_navigation_page(ctx))
    for refused in (
        await tools.click_in_navigation(ctx, element_id="el-1"),
        await tools.scroll_navigation(ctx),
        await tools.press_key_in_navigation(ctx, "Enter"),
        await tools.go_back_in_browser(ctx),
        await tools.go_forward_in_browser(ctx),
        await tools.type_in_navigation(ctx, "hello"),
    ):
        assert was_refused(refused) and "request_site_access" in refused
    assert session.actions == []


async def test_scripts_need_the_account_level(monkeypatch) -> None:
    from src.tools import builtin

    monkeypatch.setattr(builtin, "javascript_execution_enabled", lambda: True)
    tools, session = _tools(active=True)
    Page(session, "https://shop.com/")
    ctx = _context(FakeStore([_row("shop.com", "write")]))

    assert was_refused(await tools.run_js_in_navigation(ctx, "1+1"))
    assert not was_refused(await tools.type_in_navigation(ctx, "hello"))


async def test_a_blank_page_or_no_list_is_not_checked() -> None:
    tools, session = _tools(active=True)
    page = Page(session, "about:blank")
    ctx = _context(FakeStore([_row("*", "blocked")]))
    assert not was_refused(await tools.scroll_navigation(ctx))

    page.address = "https://a.com"
    assert not was_refused(
        await tools.scroll_navigation(_context(FakeStore([_row("*", "blocked")]), ""))
    )
    assert not was_refused(await tools.scroll_navigation(_context(None)))


async def test_a_session_without_its_address_is_not_checked() -> None:
    tools, _session = _tools(active=True)
    ctx = _context(FakeStore([_row("*", "blocked")]))
    assert not was_refused(await tools.scroll_navigation(ctx))


async def test_a_failing_check_does_not_stop_the_browser(monkeypatch) -> None:
    tools, session = _tools(active=True)
    Page(session, "https://a.com")
    ctx = _context(FakeStore([_row("*", "blocked")]))

    async def broken(*_: Any) -> Any:
        raise RuntimeError("bug")

    monkeypatch.setattr(ctx.userdata.bridge.site_access, "decide", broken)
    assert not was_refused(await tools.scroll_navigation(ctx))


async def test_jarvis_can_ask_and_list_but_not_grant() -> None:
    tools, _session = _tools(active=True)
    store = FakeStore([_row("*", "read")])
    ctx = _context(store)

    assert "Ask for one of" in await tools.request_site_access(ctx, "a.com", "admin", "x")
    said = await tools.request_site_access(ctx, "a.com", " Write ", "send the form")
    assert "waiting" in said and store.inserted[0]["access"] == "write"
    assert "Waiting for the user: a.com (write)." in await tools.list_site_access(ctx)


async def test_without_a_bridge_the_tools_say_every_site_is_open() -> None:
    tools, _session = _tools(active=True)
    assert "every public site" in await tools.request_site_access(None, "a.com", "read", "x")
    assert "every public site" in await tools.list_site_access(None)


async def test_a_summary_without_requests_has_no_waiting_line() -> None:
    text = await SiteAccess(FakeStore([_row("github.com", "read")])).summary(USER)
    assert "Waiting" not in text and "- github.com:" in text


# -- where the page in front is -------------------------------------------------------


class AddressCDP:
    is_connected = True

    def __init__(self, href: Any = None, fail: bool = False) -> None:
        self.href = href
        self.fail = fail
        self.navigated: list[str] = []

    async def evaluate(self, expression: str) -> Any:
        if self.fail:
            raise RuntimeError("cdp closed")
        return self.href

    async def navigate(self, url: str) -> None:
        self.navigated.append(url)

    async def send(self, method: str, **params: Any) -> dict[str, Any]:
        return {}


def _session(cdp: AddressCDP | None) -> Any:
    from src.browser.browser_session import CloudBrowserSession

    session = CloudBrowserSession()
    if cdp is not None:
        session._cdp = cdp  # type: ignore[assignment]
        session._browser_id = "b-1"
    session._current_url = "https://sent-to.com/"
    return session


async def test_the_page_says_where_it_is() -> None:
    assert (
        await _session(AddressCDP("https://landed.com/x")).page_address() == "https://landed.com/x"
    )


@pytest.mark.parametrize(
    "cdp", [AddressCDP("about:blank"), AddressCDP(None), AddressCDP(fail=True)]
)
async def test_otherwise_the_last_address_is_the_answer(cdp: AddressCDP) -> None:
    assert await _session(cdp).page_address() == "https://sent-to.com/"


async def test_no_browser_no_address() -> None:
    assert await _session(None).page_address() == ""


async def test_leaving_a_page_keeps_the_browser() -> None:
    cdp = AddressCDP("https://evil.com")
    session = _session(cdp)
    await session.leave_page()
    assert cdp.navigated == ["about:blank"] and session.is_active


async def test_a_blocked_page_is_not_read() -> None:
    tools, session = _tools(active=True)
    Page(session, "https://evil.com/")
    result = await tools.read_navigation_page(_context(FakeStore([_row("evil.com", "blocked")])))
    assert was_refused(result) and "Page content" not in result
