"""Leaner and steadier for the app's HUD (2026-11-16).

* Task ids are unique across calls: a result can outlive its call.
* A finished task sends its report apart from what the voice says.
* The browser's frames come from a screencast (only when the page paints) and stop
  while nobody in the app is looking at them. The first is a screenshot, and so are
  they while the page is hidden (its window behind others), which paints nothing.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.brain.bridge import BrainTask, spoken_and_report
from src.browser import browser_session
from src.browser.browser_session import CloudBrowserSession
from src.browser.cloud_browser import CDPConnection
from src.browser.launched import HEADLESS, LOCAL

from .test_brain_bridge import FakePublisher, _bridge, _finish
from .test_runtime_dispatch import FakeState, _router

# -- task ids ----------------------------------------------------------------------


async def test_each_call_numbers_its_tasks_apart(tmp_path: Path) -> None:
    first, second = _bridge(tmp_path), _bridge(tmp_path / "b")
    one = first.start(task="a", title="A", kind="analysis", difficulty="normal")
    two = second.start(task="b", title="B", kind="analysis", difficulty="normal")
    assert isinstance(one, BrainTask) and isinstance(two, BrainTask)
    assert one.id.startswith("t1-") and two.id.startswith("t1-")
    assert one.id != two.id, "the next call's t1 is not this call's t1"
    await _finish(first)
    await _finish(second)


# -- the report, apart ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "spoken", "report"),
    [
        ("Hecho.\n\nDetalle:\n\n# Informe\nTexto", "Hecho.", "# Informe\nTexto"),
        ("Dicho.\n**Detalle:**\ncuerpo", "Dicho.", "cuerpo"),
        ("Detalle:\n# Solo", "", "# Solo"),
        ("Todo en una frase. Detalle: 3 archivos.", "Todo en una frase. Detalle: 3 archivos.", ""),
        ("  sin informe  ", "sin informe", ""),
    ],
)
def test_what_is_said_and_the_report_come_apart(result: str, spoken: str, report: str) -> None:
    assert spoken_and_report(result) == (spoken, report)


async def test_a_finished_task_sends_both(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    publisher = FakePublisher()
    bridge.attach(None, publisher)
    task = BrainTask(id="t1-ab", title="T", task="t", kind="research", difficulty="normal")
    task.status, task.result = "done", "Listo.\n\nDetalle:\n\n# Informe"
    await bridge.publish_result(task)
    sent = publisher.messages[-1]
    assert sent["spoken"] == "Listo." and sent["report"] == "# Informe"
    assert sent["result"] == task.result


# -- CDP events ------------------------------------------------------------------------


class _Socket:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = [json.dumps(m) for m in messages]

    def __aiter__(self) -> _Socket:
        return self

    async def __anext__(self) -> str:
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


async def test_an_event_reaches_its_handler_and_a_bad_handler_stops_nothing() -> None:
    conn = CDPConnection()
    heard: list[dict[str, Any]] = []
    conn.on("Page.screencastFrame", heard.append)

    def broken(_params: dict[str, Any]) -> None:
        raise RuntimeError("handler bug")

    conn.on("Page.loadEventFired", broken)
    conn._ws = _Socket(  # type: ignore[assignment]
        [
            {"method": "Page.loadEventFired", "params": {}},
            {"method": "Page.screencastFrame", "params": {"data": "x", "sessionId": 1}},
            {"method": "Page.screencastFrame"},
            {"method": "Network.requestWillBeSent", "params": {}},
        ]
    )
    await conn._reader_loop()
    assert heard == [{"data": "x", "sessionId": 1}, {}]
    conn.off("Page.screencastFrame")
    conn.off("never.registered")
    assert "Page.screencastFrame" not in conn._events


# -- frames by screencast -----------------------------------------------------------


def _session(vendor: str = LOCAL) -> tuple[CloudBrowserSession, MagicMock, AsyncMock]:
    writer = AsyncMock()
    room = MagicMock()
    room.local_participant.stream_bytes = AsyncMock(return_value=writer)
    session = CloudBrowserSession(room=room)
    session._vendor = vendor
    session._browser_id = "b-1"
    cdp = MagicMock(is_connected=True)
    cdp.send = AsyncMock(return_value={})
    cdp.evaluate = AsyncMock(return_value="https://example.com/")
    cdp.screenshot = AsyncMock(return_value="")
    session._cdp = cdp
    return session, cdp, writer


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


async def _until(condition: Any, seconds: float = 3.0) -> None:
    """Wait for `condition()`: each frame crosses a thread (the address check), and a busy
    machine takes longer than a fixed 10 ms to get the first one across."""
    for _ in range(int(seconds / 0.01)):
        if condition():
            return
        await asyncio.sleep(0.01)


def _page(visibility: str) -> AsyncMock:
    """A page at a public address, in sight or not."""

    async def evaluate(expression: str) -> str:
        return visibility if "visibilityState" in expression else "https://example.com/"

    return AsyncMock(side_effect=evaluate)


async def test_frames_come_from_a_screencast_and_only_the_latest_goes_out(monkeypatch) -> None:
    session, cdp, writer = _session()
    monkeypatch.setattr(browser_session, "FRAME_MIN_SECONDS", 0)
    cdp.screenshot = AsyncMock(return_value=_b64(b"first"))
    cdp.evaluate = _page("visible")
    await session._start_frames()
    assert session._screencasting
    assert cdp.send.await_args_list[0].args[0] == "Page.startScreencast"
    handler = cdp.on.call_args.args[1]
    await _until(lambda: writer.write.await_args is not None)
    assert writer.write.await_args.args[0] == b"first", "the panel never waits on a paint"

    jpeg = _b64(b"frame")
    handler({"data": _b64(b"old"), "sessionId": 7})
    handler({"data": jpeg, "sessionId": 8})
    await _until(lambda: writer.write.await_args.args[0] == b"frame")
    sent = [c.args for c in cdp.send.await_args_list]
    assert ("Page.screencastFrameAck",) in sent
    assert writer.write.await_args.args[0] == b"frame", "the newest frame, the older dropped"
    assert cdp.screenshot.await_count == 1, "a page in sight paints its own frames"

    cdp.is_connected = False
    handler({"data": jpeg})  # wakes the loop to see it is over
    await _until(lambda: session._frame_task is not None and session._frame_task.done())
    assert session._frame_task is not None and session._frame_task.done()


async def test_an_unshowable_page_or_a_failed_send_drops_the_frame(monkeypatch) -> None:
    session, cdp, writer = _session()
    monkeypatch.setattr(browser_session, "FRAME_MIN_SECONDS", 0)
    await session._start_frames()
    handler = cdp.on.call_args.args[1]
    cdp.evaluate = AsyncMock(return_value=None)
    handler({"data": base64.b64encode(b"x").decode()})
    await asyncio.sleep(0.01)
    writer.write.assert_not_awaited()
    cdp.evaluate = AsyncMock(side_effect=RuntimeError("navigating"))
    handler({"data": base64.b64encode(b"x").decode()})
    await asyncio.sleep(0.01)
    writer.write.assert_not_awaited()
    session._stop_frames()


async def test_a_hidden_page_is_shown_by_screenshots_sent_only_when_it_changed(
    monkeypatch,
) -> None:
    session, cdp, writer = _session()
    monkeypatch.setattr(browser_session, "FRAME_MIN_SECONDS", 0)
    monkeypatch.setattr(browser_session, "FRAME_STALE_SECONDS", 0.005)
    cdp.evaluate = _page("hidden")
    shots = iter([_b64(b"a"), _b64(b"a"), _b64(b"b")])
    cdp.screenshot = AsyncMock(side_effect=lambda **_: next(shots, _b64(b"b")))
    await session._start_frames()
    await asyncio.sleep(0.1)
    session._stop_frames()
    assert [c.args[0] for c in writer.write.await_args_list] == [b"a", b"b"]


async def test_a_still_page_in_sight_sends_its_first_frame_and_waits(monkeypatch) -> None:
    session, cdp, writer = _session()
    monkeypatch.setattr(browser_session, "FRAME_MIN_SECONDS", 0)
    monkeypatch.setattr(browser_session, "FRAME_STALE_SECONDS", 0.005)
    cdp.evaluate = _page("visible")
    cdp.screenshot = AsyncMock(return_value=_b64(b"still"))
    await session._start_frames()
    await asyncio.sleep(0.05)
    session._stop_frames()
    assert writer.write.await_count == 1
    assert cdp.screenshot.await_count == 1


async def test_a_frame_without_a_session_or_a_connection_is_kept_unacked() -> None:
    session, cdp, _ = _session()
    session._on_screencast_frame({"data": "x"})
    assert session._latest_frame == "x"
    session._cdp = None
    session._on_screencast_frame({"data": "y", "sessionId": 3})
    cdp.send.assert_not_awaited()


async def test_where_there_is_no_screencast_it_polls(monkeypatch) -> None:
    session, cdp, _ = _session(HEADLESS)
    cdp.send = AsyncMock(side_effect=RuntimeError("Page.startScreencast not found"))
    polled = AsyncMock()
    monkeypatch.setattr(session, "_stream_frames", polled)
    await session._start_frames()
    await asyncio.sleep(0)
    assert not session._screencasting
    polled.assert_awaited()


async def test_frames_stop_while_nobody_looks_and_come_back() -> None:
    session, cdp, _ = _session()
    await session._start_frames()
    await session.set_frames_wanted(True)  # already so: nothing to do
    await session.set_frames_wanted(False)
    assert session._frame_task is None and not session._screencasting
    assert cdp.send.await_args.args[0] == "Page.stopScreencast"
    cdp.off.assert_called_with("Page.screencastFrame")
    await session._start_frames()
    assert session._frame_task is None, "not wanted: nothing starts"

    await session.set_frames_wanted(True)
    assert session._screencasting
    session._stop_frames()


async def test_not_wanted_without_a_screencast_just_stops_the_polling(monkeypatch) -> None:
    session, cdp, _ = _session(HEADLESS)
    session._frame_task = asyncio.create_task(asyncio.sleep(10))
    await session.set_frames_wanted(False)
    assert session._frame_task is None
    cdp.send.assert_not_awaited()


async def test_a_browser_that_shows_no_frames_ignores_the_switch() -> None:
    session, cdp, _ = _session("browser_use")
    await session.set_frames_wanted(False)
    await session.set_frames_wanted(True)
    cdp.send.assert_not_awaited()
    idle = CloudBrowserSession()
    await idle.set_frames_wanted(False)


def test_the_app_switches_the_frames() -> None:
    browser = MagicMock()
    browser.set_frames_wanted = AsyncMock()
    state = FakeState(browser=browser)
    assert _router(state).handle({"type": "browser_frames", "on": False}) == "browser_frames"
    assert state.spawned == ["browser_frames"]
    _router(FakeState()).handle({"type": "browser_frames", "on": True})
