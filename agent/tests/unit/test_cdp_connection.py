"""The CDP client: framing, routing and the two endpoint shapes.

Everything the agent does to a page goes through this one socket, and its
failure modes are all quiet ones:

* **A dropped socket must fail every waiter.** A pending future that is never
  resolved is a tool call that hangs for its whole timeout, which on a voice
  turn is dead air.
* **Session routing decides whether commands arrive at all.** On a
  browser-level endpoint -- what Browserbase hands out -- `Page.enable` is
  answered by the browser target, where that domain does not exist.
* **A CDP error is in the response body, not the transport.** A 200-shaped
  frame carrying `{"error": ...}` has to raise, or the tool reports success
  over a click that never landed.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from src.browser.cloud_browser import CDPConnection


class FakeSocket:
    """A CDP peer that answers each command from a scripted queue."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.responses: dict[str, Any] = {}
        self.closed = False
        self.connection = None

    async def send(self, raw: str) -> None:
        message = json.loads(raw)
        self.sent.append(message)
        reply = self.responses.get(message["method"])
        if reply is None:
            return
        body = reply(message) if callable(reply) else reply
        frame = {"id": message["id"], **body}
        # The reader loop is what resolves the future, so feed it as the real
        # transport would rather than resolving directly.
        assert self.connection is not None
        self.connection._resolve(frame)

    async def close(self) -> None:
        self.closed = True


def _connected(socket: FakeSocket) -> CDPConnection:
    """A CDPConnection wired to a fake socket, with no reader task."""
    cdp = CDPConnection()
    cdp._ws = socket
    socket.connection = cdp

    def resolve(frame: dict) -> None:
        future = cdp._pending.get(frame.get("id"))
        if future is not None and not future.done():
            future.set_result(frame)

    cdp._resolve = resolve  # type: ignore[attr-defined]
    return cdp


# -- Framing and errors ------------------------------------------------------


async def test_a_command_carries_its_method_and_params() -> None:
    socket = FakeSocket()
    socket.responses["Page.navigate"] = {"result": {"frameId": "f1"}}
    cdp = _connected(socket)

    result = await cdp.navigate("https://example.com")

    assert socket.sent[0]["method"] == "Page.navigate"
    assert socket.sent[0]["params"] == {"url": "https://example.com"}
    assert result == {"frameId": "f1"}


async def test_message_ids_increment_so_replies_can_be_matched() -> None:
    socket = FakeSocket()
    socket.responses["Page.enable"] = {"result": {}}
    cdp = _connected(socket)

    await cdp.send("Page.enable")
    await cdp.send("Page.enable")

    assert [m["id"] for m in socket.sent] == [1, 2]


async def test_a_cdp_error_is_raised_rather_than_returned_as_a_result() -> None:
    """A tool that treats an error frame as success reports a click that never landed."""
    socket = FakeSocket()
    socket.responses["Input.dispatchMouseEvent"] = {
        "error": {"code": -32000, "message": "Target closed"}
    }
    cdp = _connected(socket)

    with pytest.raises(RuntimeError, match="Target closed"):
        await cdp.click(1, 2)


async def test_sending_without_a_socket_is_refused() -> None:
    with pytest.raises(ConnectionError):
        await CDPConnection().send("Page.enable")


async def test_a_resolved_command_leaves_nothing_pending() -> None:
    socket = FakeSocket()
    socket.responses["Page.enable"] = {"result": {}}
    cdp = _connected(socket)

    await cdp.send("Page.enable")

    assert cdp._pending == {}


async def test_a_timed_out_command_does_not_leak_its_future() -> None:
    socket = FakeSocket()  # answers nothing
    cdp = _connected(socket)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(cdp.send("Page.enable"), timeout=0.05)

    # The waiter was cancelled; the entry must not survive it, or a long
    # session accumulates one dead future per failed command.
    await asyncio.sleep(0)
    assert cdp._pending == {} or all(f.cancelled() for f in cdp._pending.values())


# -- Session routing ---------------------------------------------------------


async def test_commands_are_routed_to_the_attached_page() -> None:
    socket = FakeSocket()
    socket.responses["Page.enable"] = {"result": {}}
    cdp = _connected(socket)
    cdp._session_id = "page-1"

    await cdp.send("Page.enable")

    assert socket.sent[0]["sessionId"] == "page-1"


async def test_target_commands_are_answered_by_the_browser_itself() -> None:
    socket = FakeSocket()
    socket.responses["Target.getTargets"] = {"result": {"targetInfos": []}}
    cdp = _connected(socket)
    cdp._session_id = "page-1"

    await cdp.send("Target.getTargets")

    assert "sessionId" not in socket.sent[0]


async def test_without_an_attachment_nothing_carries_a_session_id() -> None:
    socket = FakeSocket()
    socket.responses["Page.enable"] = {"result": {}}
    cdp = _connected(socket)

    await cdp.send("Page.enable")

    assert "sessionId" not in socket.sent[0]


# -- Attaching to a page -----------------------------------------------------


async def test_attaching_picks_an_existing_page_target() -> None:
    socket = FakeSocket()
    socket.responses["Target.getTargets"] = {
        "result": {
            "targetInfos": [
                {"type": "browser", "targetId": "b1"},
                {"type": "page", "targetId": "p1"},
            ]
        }
    }
    socket.responses["Target.attachToTarget"] = {"result": {"sessionId": "s1"}}
    cdp = _connected(socket)

    await cdp._attach_to_page_target()

    assert cdp._session_id == "s1"
    attach = next(m for m in socket.sent if m["method"] == "Target.attachToTarget")
    assert attach["params"] == {"targetId": "p1", "flatten": True}


async def test_attaching_creates_a_page_when_the_browser_has_none() -> None:
    socket = FakeSocket()
    socket.responses["Target.getTargets"] = {"result": {"targetInfos": []}}
    socket.responses["Target.createTarget"] = {"result": {"targetId": "p2"}}
    socket.responses["Target.attachToTarget"] = {"result": {"sessionId": "s2"}}
    cdp = _connected(socket)

    await cdp._attach_to_page_target()

    assert cdp._session_id == "s2"
    created = next(m for m in socket.sent if m["method"] == "Target.createTarget")
    assert created["params"]["url"] == "about:blank"


async def test_a_browser_that_will_not_make_a_page_is_an_error() -> None:
    socket = FakeSocket()
    socket.responses["Target.getTargets"] = {"result": {"targetInfos": []}}
    socket.responses["Target.createTarget"] = {"result": {}}
    cdp = _connected(socket)

    with pytest.raises(ValueError, match="no page target"):
        await cdp._attach_to_page_target()


async def test_a_refused_attach_is_an_error_not_a_silent_no_op() -> None:
    """Without a session id every later command goes to the browser target."""
    socket = FakeSocket()
    socket.responses["Target.getTargets"] = {
        "result": {"targetInfos": [{"type": "page", "targetId": "p1"}]}
    }
    socket.responses["Target.attachToTarget"] = {"result": {}}
    cdp = _connected(socket)

    with pytest.raises(ValueError, match="attach"):
        await cdp._attach_to_page_target()


# -- Page operations ---------------------------------------------------------


async def test_going_back_moves_to_the_previous_history_entry() -> None:
    socket = FakeSocket()
    socket.responses["Page.getNavigationHistory"] = {
        "result": {"currentIndex": 1, "entries": [{"id": 10}, {"id": 11}]}
    }
    socket.responses["Page.navigateToHistoryEntry"] = {"result": {}}
    cdp = _connected(socket)

    await cdp.go_back()

    entry = next(m for m in socket.sent if m["method"] == "Page.navigateToHistoryEntry")
    assert entry["params"]["entryId"] == 10


async def test_going_back_at_the_start_of_history_does_nothing() -> None:
    socket = FakeSocket()
    socket.responses["Page.getNavigationHistory"] = {
        "result": {"currentIndex": 0, "entries": [{"id": 10}]}
    }
    cdp = _connected(socket)

    await cdp.go_back()

    assert not any(m["method"] == "Page.navigateToHistoryEntry" for m in socket.sent)


async def test_going_forward_at_the_end_of_history_does_nothing() -> None:
    socket = FakeSocket()
    socket.responses["Page.getNavigationHistory"] = {
        "result": {"currentIndex": 1, "entries": [{"id": 10}, {"id": 11}]}
    }
    cdp = _connected(socket)

    await cdp.go_forward()

    assert not any(m["method"] == "Page.navigateToHistoryEntry" for m in socket.sent)


async def test_evaluate_returns_the_value_by_value() -> None:
    socket = FakeSocket()
    socket.responses["Runtime.evaluate"] = {"result": {"result": {"type": "string", "value": "hi"}}}
    cdp = _connected(socket)

    assert await cdp.evaluate("'hi'") == "hi"
    assert socket.sent[0]["params"]["returnByValue"] is True
    assert socket.sent[0]["params"]["awaitPromise"] is True


async def test_evaluating_something_undefined_returns_none() -> None:
    socket = FakeSocket()
    socket.responses["Runtime.evaluate"] = {"result": {"result": {"type": "undefined"}}}
    cdp = _connected(socket)

    assert await cdp.evaluate("void 0") is None


async def test_a_click_presses_and_releases() -> None:
    socket = FakeSocket()
    socket.responses["Input.dispatchMouseEvent"] = {"result": {}}
    cdp = _connected(socket)

    await cdp.click(10, 20)

    assert [m["params"]["type"] for m in socket.sent] == ["mousePressed", "mouseReleased"]
    assert socket.sent[0]["params"]["x"] == 10


async def test_typing_uses_insert_text_so_it_works_in_any_field() -> None:
    socket = FakeSocket()
    socket.responses["Input.insertText"] = {"result": {}}
    cdp = _connected(socket)

    await cdp.type_text("kwami")

    assert socket.sent[0]["params"] == {"text": "kwami"}


@pytest.mark.parametrize(
    ("key", "code"),
    [("Enter", 13), ("Tab", 9), ("Escape", 27), ("Backspace", 8)],
)
async def test_known_keys_carry_their_virtual_key_code(key: str, code: int) -> None:
    socket = FakeSocket()
    socket.responses["Input.dispatchKeyEvent"] = {"result": {}}
    cdp = _connected(socket)

    await cdp.press_key(key)

    assert socket.sent[0]["params"]["windowsVirtualKeyCode"] == code
    assert [m["params"]["type"] for m in socket.sent] == ["keyDown", "keyUp"]


async def test_an_unknown_key_still_produces_a_usable_event() -> None:
    socket = FakeSocket()
    socket.responses["Input.dispatchKeyEvent"] = {"result": {}}
    cdp = _connected(socket)

    await cdp.press_key("a")

    assert socket.sent[0]["params"]["key"] == "a"


async def test_pressing_a_key_with_no_name_does_not_crash() -> None:
    socket = FakeSocket()
    socket.responses["Input.dispatchKeyEvent"] = {"result": {}}
    cdp = _connected(socket)

    await cdp.press_key("")

    assert socket.sent[0]["params"]["windowsVirtualKeyCode"] == 0


async def test_scrolling_sends_a_wheel_event() -> None:
    socket = FakeSocket()
    socket.responses["Input.dispatchMouseEvent"] = {"result": {}}
    cdp = _connected(socket)

    await cdp.scroll(x=5, y=6, delta_y=-400)

    assert socket.sent[0]["params"]["type"] == "mouseWheel"
    assert socket.sent[0]["params"]["deltaY"] == -400


async def test_a_screenshot_comes_back_as_base64_jpeg() -> None:
    socket = FakeSocket()
    socket.responses["Page.captureScreenshot"] = {"result": {"data": "abc"}}
    cdp = _connected(socket)

    assert await cdp.screenshot() == "abc"
    assert socket.sent[0]["params"]["format"] == "jpeg"


# -- Teardown ----------------------------------------------------------------


async def test_is_connected_is_false_before_and_after() -> None:
    cdp = CDPConnection()
    assert cdp.is_connected is False

    socket = FakeSocket()
    cdp._ws = socket
    await cdp.close()

    assert socket.closed is True
    assert cdp.is_connected is False


async def test_closing_clears_the_page_attachment() -> None:
    """A stale session id would route commands at a target that is gone."""
    socket = FakeSocket()
    cdp = _connected(socket)
    cdp._session_id = "s1"

    await cdp.close()

    assert cdp._session_id is None


async def test_closing_twice_is_safe() -> None:
    cdp = _connected(FakeSocket())
    await cdp.close()
    await cdp.close()


async def test_a_socket_that_refuses_to_close_does_not_raise() -> None:
    class StubbornSocket(FakeSocket):
        async def close(self) -> None:
            raise RuntimeError("already gone")

    cdp = _connected(StubbornSocket())
    await cdp.close()
    assert cdp._ws is None
