"""The work thread's Chrome: passwords stay with the user, and so do the final buttons."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from src.brain import browser_tool, executor
from src.brain.browser_tool import BrowserTools, refuse_final_action
from src.brain.executor import _Lanes
from src.browser.safety import UnsafeURLError


class FakeSession:
    def __init__(self) -> None:
        self.signing_in = False
        self.states: list[bool] = []
        self.refused = False
        self.is_active = False
        self.calls: list[tuple[str, Any]] = []
        self.fail: Exception | None = None

    async def start(self, user_id: str, url: str) -> str:
        if self.fail:
            raise self.fail
        self.is_active = True
        self.calls.append(("start", url))
        return ""

    async def read_page(self) -> str:
        return "Page title: x"

    async def on_sign_in_page(self) -> bool:
        return self.states.pop(0) if self.states else self.signing_in

    async def sign_in_refused(self) -> bool:
        return self.refused

    async def bring_to_front(self) -> None:
        self.calls.append(("front", None))

    async def click(self, element_id: str, description: str, refuse: Any = None) -> str:
        self.calls.append(("click", refuse(description) if refuse else None))
        return "clicked"

    async def type_text(self, text: str, element_id: str, description: str) -> str:
        self.calls.append(("type", text))
        return "typed"

    async def press_key(self, key: str) -> str:
        return f"pressed {key}"

    async def scroll(self, direction: str) -> str:
        return f"scrolled {direction}"

    async def close(self, **_: Any) -> None:
        self.calls.append(("close", None))


@pytest.fixture(autouse=True)
def quick_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(browser_tool, "_WINDOW", asyncio.Lock())
    monkeypatch.setattr(browser_tool, "SIGN_IN_POLL_SECONDS", 0.001)


@pytest.fixture
def session() -> FakeSession:
    return FakeSession()


@pytest.fixture
def tools(session: FakeSession) -> BrowserTools:
    return BrowserTools("user", session)  # type: ignore[arg-type]


async def test_opening_a_page_reads_it(tools, session) -> None:
    assert "Page title" in await tools.call("open", {"url": "https://example.com"})
    assert session.calls[0] == ("start", "https://example.com")


async def test_a_sign_in_page_is_named_when_read(tools, session) -> None:
    session.signing_in = True
    assert "`sign_in`" in await tools.call("open", {"url": "https://example.com"})


async def test_nothing_is_typed_on_a_sign_in_page(tools, session) -> None:
    await tools.call("open", {"url": "https://example.com"})
    session.signing_in = True
    assert "never type" in await tools.call("type", {"text": "secreto"})
    assert ("type", "secreto") not in session.calls


async def test_signing_in_shows_the_page_and_carries_on_once_they_are_in(tools, session) -> None:
    session.states = [True, True, False, False]
    said = await tools.call("sign_in", {"url": "https://example.com/login"})
    assert session.calls[0][1] == "https://example.com/login"
    assert ("front", None) in session.calls
    assert said.startswith("The user is in") and "Page title" in said
    assert session.states == [False], "it looked until they were through, and no further"


async def test_a_user_who_never_signs_in_ends_the_wait(tools, session, monkeypatch) -> None:
    monkeypatch.setattr(browser_tool, "SIGN_IN_WAIT_SECONDS", 0.01)
    session.signing_in = True
    assert "did not sign in" in await tools.call("sign_in", {"url": "https://example.com/login"})


async def test_a_site_that_refuses_this_browser_is_reported(tools, session) -> None:
    session.signing_in = session.refused = True
    said = await tools.call("sign_in", {"url": "https://accounts.google.com"})
    assert "refuses" in said and "another way" in said


async def test_only_one_run_is_in_the_window_at_a_time(session) -> None:
    first, second = BrowserTools("u", session), BrowserTools("u", session)  # type: ignore[arg-type]
    await first.call("read", {})
    waiting = asyncio.create_task(second.call("read", {}))
    await asyncio.sleep(0.01)
    assert not waiting.done()
    await first.close()
    await waiting
    await second.close()
    await second.close()
    assert not browser_tool._WINDOW.locked()


def test_the_progress_line_names_the_site() -> None:
    assert browser_tool.sign_in_progress("https://github.com/login").endswith(": github.com")
    block = SimpleNamespace(name=browser_tool.SIGN_IN_TOOL, input={"url": "https://a.es/login"})
    assert executor._progress_line(block).endswith(": a.es")


async def test_elsewhere_it_types_clicks_presses_and_scrolls(tools, session) -> None:
    await tools.call("open", {"url": "https://example.com"})
    assert await tools.call("type", {"text": "hola"}) == "typed"
    assert await tools.call("click", {"description": "Buscar"}) == "clicked"
    assert await tools.call("press", {"key": "Enter"}) == "pressed Enter"
    assert await tools.call("scroll", {"direction": "up"}) == "scrolled up"


async def test_nothing_runs_before_a_page_is_open(tools) -> None:
    assert "first" in await tools.call("read", {})


@pytest.mark.parametrize(
    "label", ["Pagar ahora", "Place order", "Enviar", "Send", "Publicar", "Comprar", "Transferir"]
)
def test_the_final_buttons_are_the_users(label: str) -> None:
    assert "for the user" in str(refuse_final_action(label))


@pytest.mark.parametrize("label", ["Buscar", "Añadir a la cesta", "Siguiente", "Reservar asiento"])
def test_the_other_buttons_are_not(label: str) -> None:
    assert refuse_final_action(label) is None


async def test_what_goes_wrong_comes_back_in_words(tools, session) -> None:
    session.fail = UnsafeURLError("not public")
    assert "Refused" in await tools.safe_call("open", {"url": "http://localhost"})
    session.fail = ValueError("x")
    assert "ValueError" in await tools.safe_call("open", {"url": "https://example.com"})
    assert await tools.safe_call("read", {}) != ""


async def test_closing_leaves_it_to_the_session(tools, session) -> None:
    await tools.close()
    assert session.calls == [("close", None)]


def test_only_on_the_pc_and_for_someone(monkeypatch) -> None:
    settings = type("S", (), {"browser_provider": "local"})()
    monkeypatch.setattr(browser_tool, "get_settings", lambda: settings)
    assert browser_tool.browser_available("user")
    assert not browser_tool.browser_available("")
    settings.browser_provider = "browser_use"
    assert not browser_tool.browser_available("user")


def test_the_server_offers_each_tool() -> None:
    made: list[str] = []

    def tool(name: str, description: str, schema: dict[str, Any]) -> Any:
        made.append(name)
        return lambda handler: handler

    server = browser_tool.browser_server(
        BrowserTools("u", FakeSession()),  # type: ignore[arg-type]
        tool,
        lambda name, tools: (name, tools),
    )
    assert set(made) == set(browser_tool.TOOL_NAMES) and server[0] == "browser"


async def test_a_tool_of_the_server_answers_the_model() -> None:
    handlers: dict[str, Any] = {}

    def tool(name: str, description: str, schema: dict[str, Any]) -> Any:
        def keep(handler: Any) -> Any:
            handlers[name] = handler
            return handler

        return keep

    browser_tool.browser_server(
        BrowserTools("u", FakeSession()),  # type: ignore[arg-type]
        tool,
        lambda name, tools: None,
    )
    answer = await handlers["read"]({})
    assert "first" in answer["content"][0]["text"]


BROWSER = "mcp__browser__open"


class TestLanes:
    def test_the_browser_closes_commands_files_and_google(self) -> None:
        lanes = _Lanes(resuming=False)
        assert lanes.refuse(BROWSER) is None
        for name in ("Bash", "Read", "Write", "Agent", executor.ALLOWED_TOOL):
            assert "closed" in str(lanes.refuse(name))
        assert lanes.closed_session

    def test_it_keeps_the_web(self) -> None:
        lanes = _Lanes(resuming=False)
        assert lanes.refuse(BROWSER) is None
        assert lanes.refuse("WebFetch") is None
        assert lanes.refuse("mcp__browser__read") is None

    @pytest.mark.parametrize("first", ["Bash", "Read", "Edit", executor.ALLOWED_TOOL])
    def test_commands_files_or_google_close_the_browser(self, first: str) -> None:
        lanes = _Lanes(resuming=False)
        assert lanes.refuse(first) is None
        assert "closed" in str(lanes.refuse(BROWSER))

    def test_a_task_that_carries_on_starts_without_it(self) -> None:
        assert "closed" in str(_Lanes(resuming=True).refuse(BROWSER))

    def test_the_web_alone_leaves_it_open(self) -> None:
        lanes = _Lanes(resuming=False)
        assert lanes.refuse("WebSearch") is None
        assert lanes.refuse(BROWSER) is None


async def test_a_click_can_be_refused_by_what_it_finds() -> None:
    from src.browser.browser_session import CloudBrowserSession

    clicked: list[Any] = []

    class Connection:
        async def page_info(self) -> dict[str, Any]:
            return {"elements": [{"id": "el-1", "label": "Pagar", "visible": True}]}

        async def click(self, x: float, y: float) -> None:
            clicked.append((x, y))

    session = CloudBrowserSession()
    session._cdp = Connection()  # type: ignore[assignment]
    session._ensure_active = lambda: None  # type: ignore[method-assign]
    session._reset_idle_timer = lambda: None  # type: ignore[method-assign]
    said = await session.click("el-1", refuse=lambda label: f"no: {label}")
    assert said == "no: Pagar" and clicked == []


class TestTheSessionsEye:
    def _session(self, address: str, password: bool = False, text: str = "") -> Any:
        from src.browser.browser_session import CloudBrowserSession

        class Connection:
            sent: list[str] = []

            async def evaluate(self, expression: str) -> Any:
                if "location" in expression:
                    return address
                return text if "innerText" in expression else password

            async def send(self, method: str, **params: Any) -> None:
                self.sent.append(method)

        session = CloudBrowserSession()
        session._cdp = Connection()  # type: ignore[assignment]
        session._ensure_active = lambda: None  # type: ignore[method-assign]
        session._reset_idle_timer = lambda: None  # type: ignore[method-assign]
        return session

    @pytest.mark.parametrize(
        "address",
        [
            "github.com/login",
            "github.com/sessions/two-factor",
            "accounts.google.com/v3/signin/identifier",
            "accounts.google.com/v3/signin/challenge/pwd",
            "www.amazon.es/ap/signin",
            "login.microsoftonline.com/common",
        ],
    )
    async def test_a_sign_in_is_known_by_its_address(self, address: str) -> None:
        assert await self._session(address).on_sign_in_page() is True

    async def test_or_by_its_password_field(self) -> None:
        assert await self._session("example.com/x", password=True).on_sign_in_page() is True

    @pytest.mark.parametrize(
        "address", ["github.com/settings/profile", "mail.google.com/mail/u/0", "es.blog/authors"]
    )
    async def test_other_pages_are_not_sign_ins(self, address: str) -> None:
        assert await self._session(address).on_sign_in_page() is False

    async def test_unsure_is_no(self) -> None:
        session = self._session("x")
        session._cdp = None
        assert await session.on_sign_in_page() is False
        assert await session.sign_in_refused() is False
        await session.bring_to_front()

    async def test_a_browser_refused_by_the_site_is_seen(self) -> None:
        session = self._session("x", text="This browser or app may not be secure.")
        assert await session.sign_in_refused() is True
        spanish = "Es posible que el navegador o la aplicación no sean seguros."
        assert await self._session("x", text=spanish).sign_in_refused() is True
        assert await self._session("x", text="Hola").sign_in_refused() is False

    async def test_it_shows_itself(self) -> None:
        session = self._session("x")
        session._cdp.sent = []
        await session.bring_to_front()
        assert session._cdp.sent == ["Page.bringToFront"]
