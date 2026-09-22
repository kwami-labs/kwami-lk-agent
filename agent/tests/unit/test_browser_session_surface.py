"""Driving a live cloud browser: the accessors, the page reader, and the failures.

`test_browser_session_lifecycle.py` covers starting and releasing one. This
covers what happens in between — reading the page, clicking, typing, scrolling
— against a scripted CDP layer.

Two things here are worth more than the coverage. The element picker decides
what a spoken "click the login button" actually clicks, and getting it wrong is
not a crash but a *wrong action in the user's logged-in browser*. And every
interaction resets the idle timer, which is what stops a browser the user is
still using from being released underneath them while it bills by the minute.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.browser.browser_session import CloudBrowserSession


class ScriptedCDP:
    """A CDP connection that answers `page_info` from a script."""

    is_connected = True

    #: Distinguishes "use the default page" from "the page really returned None",
    #: which is a case the reader has to handle and `None` alone cannot express.
    _DEFAULT = object()

    def __init__(self, info: Any = _DEFAULT, fail: bool = False) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._info = (
            {"title": "", "text": "", "elements": []} if info is ScriptedCDP._DEFAULT else info
        )
        self._fail = fail
        self.navigated: list[str] = []
        self.clicked: list[tuple[float, float]] = []
        self.typed: list[str] = []
        self.keys: list[str] = []
        self.scrolled: list[int] = []

    async def send(self, method: str, **params: Any) -> dict[str, Any]:
        self.calls.append((method, params))
        return {}

    async def page_info(self) -> Any:
        if self._fail:
            raise RuntimeError("cdp closed")
        return self._info

    async def evaluate(self, expression: str) -> Any:
        if self._fail:
            raise RuntimeError("cdp closed")
        return "evaluated"

    async def navigate(self, url: str) -> None:
        self.navigated.append(url)

    async def click(self, x: float, y: float) -> None:
        self.clicked.append((x, y))

    async def type_text(self, text: str) -> None:
        self.typed.append(text)

    async def press_key(self, key: str) -> None:
        self.keys.append(key)

    async def scroll(self, x: int, y: int, delta_y: int) -> None:
        self.scrolled.append(delta_y)

    async def close(self) -> None: ...


def _live(cdp: ScriptedCDP) -> CloudBrowserSession:
    """A session that believes it has a running browser."""
    session = CloudBrowserSession()
    session._cdp = cdp  # type: ignore[assignment]
    session._browser_id = "b-1"
    return session


def _element(**overrides: Any) -> dict[str, Any]:
    element = {"id": "el-0", "type": "button", "label": "Log in", "x": 10, "y": 20, "visible": True}
    element.update(overrides)
    return element


# -- accessors --------------------------------------------------------------


def test_the_live_url_and_the_room_and_tracker_can_be_set() -> None:
    """All three are re-attached when a browser is handed to a new agent."""
    session = CloudBrowserSession()
    session._live_url = "https://live.example/view"
    room, tracker = object(), object()

    session.set_room(room)
    session.set_usage_tracker(tracker)

    assert session.live_url == "https://live.example/view"
    assert session._room is room
    assert session._usage_tracker is tracker


def test_minutes_are_not_billed_without_a_tracker_or_a_start() -> None:
    """Both guards matter: a handed-over session can be missing either."""
    session = CloudBrowserSession()

    session._started_at = None
    session._record_browser_minutes()  # no start time

    session._started_at = 1.0
    session._usage_tracker = None
    session._record_browser_minutes()  # no tracker

    assert session._started_at is None, "the start time was not consumed"


# -- reading the page -------------------------------------------------------


async def test_reading_a_page_lists_its_interactive_elements() -> None:
    cdp = ScriptedCDP(
        {
            "title": "Example",
            "text": "Some page text",
            "elements": [_element(), _element(id="el-1", label="Cancel", visible=False)],
        }
    )

    result = await _live(cdp).read_page()

    assert "Page title: Example" in result
    assert "Some page text" in result
    assert "el-0" in result and "Log in" in result
    assert "✓" in result and "✗" in result, "visibility is not shown, so the model cannot tell"


async def test_reading_a_page_that_returns_nothing_usable() -> None:
    """A CDP evaluate can come back as None on a page still loading."""
    session = _live(ScriptedCDP(info=None))  # really None, not the default

    assert "Could not read page content" in await session.read_page()


async def test_reading_a_bare_page_still_reports_its_title() -> None:
    result = await _live(ScriptedCDP({"title": "Empty", "text": "", "elements": []})).read_page()

    assert result == "Page title: Empty"


# -- choosing what to click -------------------------------------------------


async def test_clicking_by_element_id_is_exact() -> None:
    cdp = ScriptedCDP({"elements": [_element(), _element(id="el-1", label="Cancel", x=99, y=99)]})

    result = await _live(cdp).click(element_id="el-1")

    assert cdp.clicked == [(99.0, 99.0)]
    assert "Cancel" in result


async def test_an_invisible_element_is_not_clicked_by_id() -> None:
    """Clicking something off-screen does not do what the user asked."""
    cdp = ScriptedCDP({"elements": [_element(visible=False)]})

    result = await _live(cdp).click(element_id="el-0")

    assert cdp.clicked == []
    assert "Could not find element" in result


async def test_a_description_matches_on_a_substring() -> None:
    cdp = ScriptedCDP({"elements": [_element(label="Sign in with Google")]})

    await _live(cdp).click(description="sign in with google")

    assert cdp.clicked == [(10.0, 20.0)]


async def test_a_description_matches_on_scattered_words() -> None:
    """ "log in" should find "Log in to your account"."""
    cdp = ScriptedCDP({"elements": [_element(label="Log in to your account")]})

    await _live(cdp).click(description="log account")

    assert cdp.clicked == [(10.0, 20.0)]


async def test_the_best_match_wins_over_a_weaker_one() -> None:
    """A full-phrase match beats a scattered-word one, or the wrong thing is clicked."""
    cdp = ScriptedCDP(
        {
            "elements": [
                _element(id="el-0", label="Log out of account", x=1, y=1),
                _element(id="el-1", label="Log in", x=2, y=2),
            ]
        }
    )

    await _live(cdp).click(description="log in")

    assert cdp.clicked == [(2.0, 2.0)], "the weaker match was clicked"


async def test_an_invisible_element_is_skipped_when_matching_by_description() -> None:
    cdp = ScriptedCDP({"elements": [_element(label="Log in", visible=False)]})

    result = await _live(cdp).click(description="log in")

    assert cdp.clicked == []
    assert "Could not find element" in result


async def test_clicking_with_nothing_to_go_on() -> None:
    assert "Could not find element" in await _live(ScriptedCDP()).click()


# -- typing, keys and scrolling ---------------------------------------------


async def test_typing_focuses_the_field_first() -> None:
    """Text sent without focusing lands wherever the caret happened to be."""
    cdp = ScriptedCDP({"elements": [_element(type="input", label="Email")]})

    result = await _live(cdp).type_text("me@example.com", element_id="el-0")

    assert cdp.clicked == [(10.0, 20.0)], "the field was never focused"
    assert cdp.typed == ["me@example.com"]
    assert "me@example.com" in result


async def test_typing_into_a_field_that_cannot_be_found_types_nothing() -> None:
    """Otherwise the text goes into whatever had focus."""
    cdp = ScriptedCDP({"elements": []})

    result = await _live(cdp).type_text("secret", element_id="el-9")

    assert cdp.typed == [], "text was typed into an unknown element"
    assert "Could not find" in result


async def test_typing_can_skip_the_clear() -> None:
    cdp = ScriptedCDP()

    await _live(cdp).type_text("appended", clear_first=False)

    assert not any(method == "Input.dispatchKeyEvent" for method, _ in cdp.calls)
    assert cdp.typed == ["appended"]


async def test_pressing_a_key_and_scrolling_both_ways() -> None:
    cdp = ScriptedCDP()
    session = _live(cdp)

    assert "Enter" in await session.press_key("Enter")
    await session.scroll("down")
    await session.scroll("up")

    assert cdp.keys == ["Enter"]
    assert cdp.scrolled == [400, -400], "up and down scrolled the same way"


# -- javascript -------------------------------------------------------------


async def test_evaluating_javascript_returns_its_result() -> None:
    assert "evaluated" in await _live(ScriptedCDP()).evaluate_js("1 + 1")


async def test_javascript_that_throws_is_reported_not_raised() -> None:
    result = await _live(ScriptedCDP(fail=True)).evaluate_js("boom()")

    assert "Failed to execute JavaScript" in result


# -- the idle timer ---------------------------------------------------------


async def test_every_interaction_pushes_the_idle_timer_out() -> None:
    """A browser the user is still driving must not be released underneath them."""
    session = _live(ScriptedCDP())

    await session.press_key("Enter")
    first = session._idle_timer
    assert first is not None

    await session.scroll("down")
    second = session._idle_timer

    assert second is not first, "the idle timer was not reset, so the browser can expire in use"
    session._cancel_idle_timer()
    for timer in (first, second):
        if timer is not None:
            timer.cancel()
            await asyncio.gather(timer, return_exceptions=True)


async def test_interacting_without_a_live_browser_is_refused() -> None:
    session = CloudBrowserSession()

    with pytest.raises(RuntimeError, match="No active cloud browser"):
        await session.press_key("Enter")


async def test_a_browser_that_refuses_the_viewport_override_still_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The override is a nicety; failing it must not cost the whole session."""

    class Launch:
        browser_id = "b-1"
        live_url = "https://live.example/view"
        cdp_http_url = "https://cdp.example"
        cdp_ws_url = ""
        vendor = "browser_use"
        persistence_id = "p-1"

    class Provider:
        async def launch(self, user_id: str) -> Any:
            return Launch()

        async def release(self, browser_id: str) -> None: ...

    class PickyCDP:
        is_connected = True

        def __init__(self) -> None:
            self.calls: list[str] = []

        async def connect(self, url: str) -> None: ...

        async def send(self, method: str, **params: Any) -> dict[str, Any]:
            self.calls.append(method)
            if method == "Emulation.setDeviceMetricsOverride":
                raise RuntimeError("unsupported on this target")
            return {}

        async def close(self) -> None: ...

    cdp = PickyCDP()
    monkeypatch.setattr("src.browser.browser_session.CDPConnection", lambda: cdp)

    session = CloudBrowserSession(provider=Provider())
    live = await session.start("tenant-1")

    assert live, "a refused viewport override lost the whole session"
    assert "Page.enable" in cdp.calls
    session._cancel_idle_timer()


async def test_starting_an_already_live_session_just_navigates() -> None:
    """A second `navigate_to` must reuse the browser rather than rent another."""
    cdp = ScriptedCDP()
    session = _live(cdp)
    session._live_url = "https://live.example/view"

    result = await session.start("tenant-1", url="https://example.com/next")

    assert result == "https://live.example/view", "it rented a second browser"
    assert cdp.navigated == ["https://example.com/next"]
    session._cancel_idle_timer()
