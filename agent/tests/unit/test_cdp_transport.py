"""Getting a CDP socket open, keeping it fed, and failing it cleanly.

The layer under `test_cdp_connection.py`: that file drives an already-open
socket, this one covers how the socket is obtained and what happens when it
dies. Both halves matter for the same reason -- the agent's whole browsing
ability hangs off one connection, and every failure here is silent from the
user's side.

Nothing can stand up a real Chrome in a unit test, so what is verified is our
*call shape*: which endpoints are asked, in which order, with which frames. A
kwarg dropped or renamed fails these, which is the point; the SDK's own
behaviour is not what is under test.
"""

from __future__ import annotations

import asyncio
import enum
import json
import sys
from typing import Any

import httpx
import pytest
import respx

import src.browser.cloud_browser as cloud_browser
from src.browser.cloud_browser import CDPConnection

CDP_HTTP = "https://cdp-1.browser-use.com"
PAGE_WS = "ws://cdp-1/devtools/page/p1"


class _State(enum.Enum):
    """Mirrors `websockets.protocol.State` closely enough for `is_connected`."""

    OPEN = 1
    CLOSED = 3


class _FakeProtocolModule:
    State = _State


class FakeWebSocket:
    """An async-iterable socket that yields the frames it was scripted with.

    Carries `state` because `is_connected` reads `websockets.protocol.State`
    on the modern client; the legacy `.open` path is exercised separately.
    """

    def __init__(self, frames: list[dict] | None = None, fail: Exception | None = None) -> None:
        self.frames = frames or []
        self.fail = fail
        self.sent: list[str] = []
        self.closed = False
        self.close_timeout: float | None = None
        self.state = _State.OPEN
        self.responses: dict[str, dict] = {}
        self._inbox: asyncio.Queue[str] = asyncio.Queue()
        self._released = asyncio.Event()

    def __aiter__(self) -> FakeWebSocket:
        return self

    async def __anext__(self) -> str:
        if self.fail:
            raise self.fail
        if self.frames:
            return json.dumps(self.frames.pop(0))
        # A reply only exists once a command has been sent, so wait for one
        # rather than draining a script: the reader starts at `_open`, before
        # any `send` has registered a pending future.
        inbound = asyncio.ensure_future(self._inbox.get())
        released = asyncio.ensure_future(self._released.wait())
        done, pending = await asyncio.wait({inbound, released}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if inbound in done:
            return inbound.result()
        if self.fail:
            raise self.fail
        raise StopAsyncIteration

    def break_connection(self, error: Exception) -> None:
        """Drop the socket the way a peer going away does."""
        self.fail = error
        self._released.set()

    async def send(self, raw: str) -> None:
        self.sent.append(raw)
        message = json.loads(raw)
        reply = self.responses.get(message.get("method"))
        if reply is not None:
            self._inbox.put_nowait(json.dumps({"id": message["id"], **reply}))

    async def recv(self) -> str:
        return json.dumps(self.frames.pop(0)) if self.frames else "{}"

    async def close(self) -> None:
        self.closed = True
        self.state = _State.CLOSED
        self._released.set()

    async def __aenter__(self) -> FakeWebSocket:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


class FakeWebsocketsModule:
    """Stands in for the `websockets` package inside `_open`/`_resolve_ws_url`."""

    def __init__(self, socket: FakeWebSocket) -> None:
        self.socket = socket
        self.calls: list[tuple[str, dict]] = []

    def connect(self, url: str, **kwargs: Any) -> Any:
        self.calls.append((url, kwargs))
        return _Connect(self.socket)


class _Connect:
    """`websockets.connect` is awaitable *and* an async context manager."""

    def __init__(self, socket: FakeWebSocket) -> None:
        self.socket = socket

    def __await__(self):
        async def _resolve() -> FakeWebSocket:
            return self.socket

        return _resolve().__await__()

    async def __aenter__(self) -> FakeWebSocket:
        return self.socket

    async def __aexit__(self, *exc: Any) -> None:
        return None


@pytest.fixture
def fake_websockets(monkeypatch):
    """Install a fake `websockets` module for the duration of a test."""

    def install(socket: FakeWebSocket, *, legacy: bool = False) -> FakeWebsocketsModule:
        module = FakeWebsocketsModule(socket)
        monkeypatch.setitem(sys.modules, "websockets", module)
        if legacy:
            # websockets < 13 has no `protocol.State`; the import fails and
            # `is_connected` falls back to `.open`.
            monkeypatch.delitem(sys.modules, "websockets.protocol", raising=False)
            monkeypatch.setitem(sys.modules, "websockets.protocol", None)
        else:
            monkeypatch.setitem(sys.modules, "websockets.protocol", _FakeProtocolModule)
        return module

    return install


async def _drain(cdp: CDPConnection) -> None:
    """Stop the reader task so it cannot outlive the test."""
    await cdp.close()


# -- Opening a socket --------------------------------------------------------


async def test_opening_starts_the_reader_and_bounds_the_frame_size(fake_websockets) -> None:
    """A screenshot frame is megabytes; the default limit drops it."""
    socket = FakeWebSocket()
    module = fake_websockets(socket)
    cdp = CDPConnection()

    await cdp._open(PAGE_WS)

    url, kwargs = module.calls[0]
    assert url == PAGE_WS
    assert kwargs["max_size"] == 10 * 1024 * 1024
    assert cdp.is_connected is True
    await _drain(cdp)


async def test_connect_ws_attaches_to_a_page_by_default(fake_websockets) -> None:
    socket = FakeWebSocket()
    fake_websockets(socket)
    cdp = CDPConnection()

    async def fake_attach() -> None:
        cdp._session_id = "s1"

    cdp._attach_to_page_target = fake_attach  # type: ignore[method-assign]
    await cdp.connect_ws(PAGE_WS)

    assert cdp._session_id == "s1"
    await _drain(cdp)


async def test_connect_ws_can_skip_attaching_for_a_page_endpoint(fake_websockets) -> None:
    socket = FakeWebSocket()
    fake_websockets(socket)
    cdp = CDPConnection()

    await cdp.connect_ws(PAGE_WS, attach_to_page=False)

    assert cdp._session_id is None
    await _drain(cdp)


# -- Discovering a page target over HTTP -------------------------------------


@respx.mock
async def test_connect_resolves_a_page_target_from_json_list(fake_websockets) -> None:
    respx.get(f"{CDP_HTTP}/json/list").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"type": "browser", "webSocketDebuggerUrl": "ws://browser"},
                {"type": "page", "webSocketDebuggerUrl": PAGE_WS},
            ],
        )
    )
    socket = FakeWebSocket()
    module = fake_websockets(socket)
    cdp = CDPConnection()

    await cdp.connect(CDP_HTTP)

    # The browser target is skipped: Page/Input/Runtime do not exist on it.
    assert module.calls[0][0] == PAGE_WS
    await _drain(cdp)


@respx.mock
async def test_a_browser_with_no_page_is_given_one(fake_websockets) -> None:
    """`/json/list` can come back with no page at all on a fresh browser."""
    respx.get(f"{CDP_HTTP}/json/list").mock(
        side_effect=[
            httpx.Response(200, json=[]),
            httpx.Response(200, json=[{"id": "p2", "webSocketDebuggerUrl": PAGE_WS}]),
        ]
    )
    respx.get(f"{CDP_HTTP}/json/version").mock(
        return_value=httpx.Response(200, json={"webSocketDebuggerUrl": "ws://browser"})
    )
    socket = FakeWebSocket(frames=[{"id": 1, "result": {"targetId": "p2"}}])
    module = fake_websockets(socket)

    resolved = await CDPConnection()._resolve_ws_url(CDP_HTTP)

    assert resolved == PAGE_WS
    created = json.loads(socket.sent[0])
    assert created["method"] == "Target.createTarget"
    assert created["params"]["url"] == "about:blank"
    assert module.calls[0][0] == "ws://browser"


@respx.mock
async def test_a_failing_json_list_falls_through_to_json_version(fake_websockets) -> None:
    respx.get(f"{CDP_HTTP}/json/list").mock(
        side_effect=[
            httpx.Response(500, text="boom"),
            httpx.Response(200, json=[{"id": "p2", "webSocketDebuggerUrl": PAGE_WS}]),
        ]
    )
    respx.get(f"{CDP_HTTP}/json/version").mock(
        return_value=httpx.Response(200, json={"webSocketDebuggerUrl": "ws://browser"})
    )
    fake_websockets(FakeWebSocket(frames=[{"id": 1, "result": {"targetId": "p2"}}]))

    assert await CDPConnection()._resolve_ws_url(CDP_HTTP) == PAGE_WS


@respx.mock
async def test_no_browser_websocket_url_is_an_error() -> None:
    respx.get(f"{CDP_HTTP}/json/list").mock(return_value=httpx.Response(200, json=[]))
    respx.get(f"{CDP_HTTP}/json/version").mock(return_value=httpx.Response(200, json={}))

    with pytest.raises(ValueError, match="webSocketDebuggerUrl"):
        await CDPConnection()._resolve_ws_url(CDP_HTTP)


@respx.mock
async def test_a_target_that_will_not_be_created_is_an_error(fake_websockets) -> None:
    respx.get(f"{CDP_HTTP}/json/list").mock(return_value=httpx.Response(200, json=[]))
    respx.get(f"{CDP_HTTP}/json/version").mock(
        return_value=httpx.Response(200, json={"webSocketDebuggerUrl": "ws://browser"})
    )
    fake_websockets(FakeWebSocket(frames=[{"id": 1, "result": {}}]))

    with pytest.raises(ValueError, match="Failed to create"):
        await CDPConnection()._resolve_ws_url(CDP_HTTP)


@respx.mock
async def test_a_created_target_missing_from_the_list_is_an_error(fake_websockets) -> None:
    respx.get(f"{CDP_HTTP}/json/list").mock(
        side_effect=[
            httpx.Response(200, json=[]),
            httpx.Response(200, json=[{"id": "someone-else"}]),
        ]
    )
    respx.get(f"{CDP_HTTP}/json/version").mock(
        return_value=httpx.Response(200, json={"webSocketDebuggerUrl": "ws://browser"})
    )
    fake_websockets(FakeWebSocket(frames=[{"id": 1, "result": {"targetId": "p2"}}]))

    with pytest.raises(ValueError, match="Could not find page target"):
        await CDPConnection()._resolve_ws_url(CDP_HTTP)


# -- The reader loop ---------------------------------------------------------


async def test_a_reply_resolves_the_command_that_asked_for_it(fake_websockets) -> None:
    socket = FakeWebSocket()
    socket.responses["Page.enable"] = {"result": {"ok": True}}
    fake_websockets(socket)
    cdp = CDPConnection()
    await cdp._open(PAGE_WS)

    assert await asyncio.wait_for(cdp.send("Page.enable"), timeout=2) == {"ok": True}
    await _drain(cdp)


async def test_a_frame_for_an_unknown_id_is_ignored(fake_websockets) -> None:
    """Events carry no id, and a late reply may arrive after a timeout."""
    socket = FakeWebSocket(frames=[{"method": "Page.loadEventFired"}, {"id": 999, "result": {}}])
    fake_websockets(socket)
    cdp = CDPConnection()

    await cdp._open(PAGE_WS)
    await asyncio.sleep(0)

    assert cdp.is_connected is True
    await _drain(cdp)


async def test_a_dropped_socket_fails_every_waiting_command(fake_websockets) -> None:
    """Otherwise each in-flight tool call hangs for its whole 30s timeout."""
    socket = FakeWebSocket()  # answers nothing
    fake_websockets(socket)
    cdp = CDPConnection()
    await cdp._open(PAGE_WS)

    pending = asyncio.ensure_future(cdp.send("Page.enable"))
    await asyncio.sleep(0)  # let the command register its future
    socket.break_connection(ConnectionResetError("peer went away"))

    with pytest.raises(ConnectionError, match="CDP connection lost"):
        await asyncio.wait_for(pending, timeout=2)

    await _drain(cdp)


# -- Connection state --------------------------------------------------------


async def test_is_connected_reads_an_older_websockets_client(fake_websockets) -> None:
    """websockets < 13 exposes `.open` rather than `.state`."""

    class LegacySocket(FakeWebSocket):
        open = True

    socket = LegacySocket()
    fake_websockets(socket, legacy=True)
    cdp = CDPConnection()
    await cdp._open(PAGE_WS)

    assert cdp.is_connected is True

    socket.open = False
    assert cdp.is_connected is False
    await _drain(cdp)


async def test_closing_cancels_the_reader_task(fake_websockets) -> None:
    socket = FakeWebSocket()
    fake_websockets(socket)
    cdp = CDPConnection()
    await cdp._open(PAGE_WS)
    reader = cdp._reader_task

    await cdp.close()

    assert socket.closed is True
    assert reader is not None and reader.done()


# -- Page inspection ---------------------------------------------------------


async def test_page_info_asks_for_the_shape_the_click_path_consumes() -> None:
    """`click` reads `elements[].x/y`, so the extraction script must return them."""
    cdp = CDPConnection()
    captured: list[str] = []

    async def fake_evaluate(expression: str) -> Any:
        captured.append(expression)
        return {"title": "T", "text": "body", "elements": []}

    cdp.evaluate = fake_evaluate  # type: ignore[method-assign]

    info = await cdp.page_info()

    assert info["title"] == "T"
    script = captured[0]
    for field in ("data-kwami-id", "getBoundingClientRect", "visible", "elements"):
        assert field in script


async def test_the_extraction_script_bounds_what_it_returns() -> None:
    """Page text and element counts reach the LLM context and the data channel."""
    cdp = CDPConnection()
    captured: list[str] = []

    async def fake_evaluate(expression: str) -> Any:
        captured.append(expression)
        return {}

    cdp.evaluate = fake_evaluate  # type: ignore[method-assign]
    await cdp.page_info()

    script = captured[0]
    assert "slice(0, 5000)" in script
    assert "i >= 80" in script


# -- Profile lookup edge ------------------------------------------------------


def test_the_module_points_at_the_documented_api_base() -> None:
    assert cloud_browser.BU_API_BASE.endswith("/api/v3")


# -- Remaining branches ------------------------------------------------------


@respx.mock
async def test_listing_profiles_without_a_query_omits_the_parameter() -> None:
    """An empty query would filter to nothing rather than list everything."""
    from src.browser.cloud_browser import BU_API_BASE, BrowserUseClient

    route = respx.get(f"{BU_API_BASE}/profiles").mock(
        return_value=httpx.Response(200, json={"items": []})
    )

    await BrowserUseClient(api_key="bu_key").list_profiles()

    assert "query" not in route.calls.last.request.url.params
    assert route.calls.last.request.url.params["pageSize"] == "20"


async def test_is_connected_is_false_for_a_socket_with_no_state_at_all(
    fake_websockets,
) -> None:
    """Neither the modern `.state` nor the legacy `.open`: assume closed."""

    class OpaqueSocket(FakeWebSocket):
        pass

    socket = OpaqueSocket()
    delattr(type(socket), "state") if hasattr(type(socket), "state") else None
    fake_websockets(socket, legacy=True)
    cdp = CDPConnection()
    await cdp._open(PAGE_WS)

    # `open` is absent on this class, so the fallback getattr default applies.
    assert cdp.is_connected is False
    await _drain(cdp)


async def test_the_reader_loop_ends_cleanly_when_the_socket_closes(
    fake_websockets,
) -> None:
    """A normal close must not resolve pending futures with an exception."""
    socket = FakeWebSocket()
    fake_websockets(socket)
    cdp = CDPConnection()
    await cdp._open(PAGE_WS)
    reader = cdp._reader_task

    await socket.close()
    await asyncio.wait_for(asyncio.shield(reader), timeout=2)

    assert reader.done() and reader.exception() is None
    await _drain(cdp)


async def test_going_forward_moves_to_the_next_history_entry() -> None:
    """The positive branch; the end-of-history no-op is covered separately."""
    cdp = CDPConnection()
    sent: list[tuple[str, dict]] = []

    async def fake_send(method: str, **params: Any) -> dict:
        sent.append((method, params))
        if method == "Page.getNavigationHistory":
            return {"currentIndex": 0, "entries": [{"id": 10}, {"id": 11}]}
        return {}

    cdp.send = fake_send  # type: ignore[method-assign]

    await cdp.go_forward()

    assert ("Page.navigateToHistoryEntry", {"entryId": 11}) in sent


async def test_a_dropped_socket_leaves_already_resolved_futures_alone(
    fake_websockets,
) -> None:
    """A reply that landed before the drop must not be overwritten with an error."""
    socket = FakeWebSocket()
    socket.responses["Page.enable"] = {"result": {"ok": True}}
    fake_websockets(socket)
    cdp = CDPConnection()
    await cdp._open(PAGE_WS)

    settled = await asyncio.wait_for(cdp.send("Page.enable"), timeout=2)
    # The future is resolved and popped; breaking the socket now walks a
    # `_pending` whose entries are all done.
    socket.break_connection(ConnectionResetError("peer went away"))
    await asyncio.sleep(0)

    assert settled == {"ok": True}
    await _drain(cdp)


async def test_a_dropped_socket_skips_futures_that_already_resolved() -> None:
    """Walked directly, because `send` pops its future before the loop sees it.

    The guard matters on the tool-handoff path, where a result can land and the
    socket drop in the same tick: setting an exception on a future that already
    has a result raises InvalidStateError inside the reader and kills it, so
    every *later* command then hangs for its full timeout instead of failing.
    """
    cdp = CDPConnection()
    settled: asyncio.Future = asyncio.get_running_loop().create_future()
    settled.set_result({"id": 1, "result": {}})
    pending: asyncio.Future = asyncio.get_running_loop().create_future()
    cdp._pending = {1: settled, 2: pending}

    class ExplodingSocket:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise ConnectionResetError("peer went away")

    cdp._ws = ExplodingSocket()
    await cdp._reader_loop()

    assert settled.result() == {"id": 1, "result": {}}, "a landed result was overwritten"
    assert isinstance(pending.exception(), ConnectionError)
