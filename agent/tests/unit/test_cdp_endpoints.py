"""Reaching a CDP endpoint, and the two shapes vendors hand out.

`test_cdp_connection.py` covers the protocol once a socket exists. This covers
getting to one, which is where the two vendors differ and where a wrong guess
is expensive:

* **Browser Use Cloud** gives an HTTP CDP base. `/json/list` names the page
  targets, and a *page* target is what we need — `Page.*`, `Runtime.*` and
  `Input.*` are not available on the browser target, so attaching to the wrong
  one fails later, at the first navigation, rather than here at connect.
* **Browserbase** gives a single `connectUrl`, the browser endpoint, with no
  HTTP discovery at all.

Everything here drives the real `CDPConnection` against a scripted socket and
a mocked HTTP layer. Nothing stands up a browser, so what is pinned is the
*call shape* — which endpoint is asked, in what order, and what is done with
each answer — and the docstrings say so where it matters.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
import respx

from src.browser.cloud_browser import CDPConnection

CDP_BASE = "https://cdp-test.browser-use.com"
PAGE_WS = "ws://cdp-test/devtools/page/ABC"
BROWSER_WS = "ws://cdp-test/devtools/browser/XYZ"


class ScriptedSocket:
    """A CDP socket that answers from a script, keyed by method.

    Strict on purpose: an unscripted method raises rather than returning a
    stub, so a test cannot pass because the code under test asked for
    something nobody thought about.
    """

    def __init__(self, replies: dict[str, dict[str, Any]] | None = None) -> None:
        self.sent: list[dict[str, Any]] = []
        self._replies = replies or {}
        self._outbox: asyncio.Queue[str] = asyncio.Queue()
        self.closed = False

    async def send(self, raw: str) -> None:
        message = json.loads(raw)
        self.sent.append(message)
        method = message["method"]
        if method not in self._replies:
            raise AssertionError(f"unscripted CDP method: {method}")
        reply = {"id": message["id"], **self._replies[method]}
        await self._outbox.put(json.dumps(reply))

    async def recv(self) -> str:
        return await self._outbox.get()

    def __aiter__(self) -> ScriptedSocket:
        return self

    async def __anext__(self) -> str:
        return await self._outbox.get()

    async def close(self) -> None:
        self.closed = True


def _targets(*entries: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=list(entries))


# -- resolving an HTTP CDP base to a page socket ----------------------------


@respx.mock
async def test_json_list_gives_up_a_page_target() -> None:
    """The happy path for the HTTP-base vendor."""
    respx.get(f"{CDP_BASE}/json/list").mock(
        return_value=_targets(
            {"type": "browser", "webSocketDebuggerUrl": BROWSER_WS},
            {"type": "page", "webSocketDebuggerUrl": PAGE_WS},
        )
    )

    assert await CDPConnection()._resolve_ws_url(CDP_BASE) == PAGE_WS


@respx.mock
async def test_a_browser_target_is_not_mistaken_for_a_page() -> None:
    """`Page.*` and `Input.*` do not exist on the browser target.

    Taking it would connect cleanly and then fail at the first navigation,
    which is a much worse place to find out.
    """
    respx.get(f"{CDP_BASE}/json/list").mock(
        return_value=_targets({"type": "browser", "webSocketDebuggerUrl": BROWSER_WS})
    )
    respx.get(f"{CDP_BASE}/json/version").mock(
        return_value=httpx.Response(200, json={"webSocketDebuggerUrl": BROWSER_WS})
    )

    with pytest.raises(Exception):
        # Falls through to creating one; with no socket factory that fails,
        # which is the point -- it did not settle for the browser target.
        await CDPConnection()._resolve_ws_url(CDP_BASE)


@respx.mock
async def test_a_page_target_without_a_socket_url_is_skipped() -> None:
    """A target can be listed before its debugger URL exists."""
    respx.get(f"{CDP_BASE}/json/list").mock(
        return_value=_targets(
            {"type": "page"},
            {"type": "page", "webSocketDebuggerUrl": PAGE_WS},
        )
    )

    assert await CDPConnection()._resolve_ws_url(CDP_BASE) == PAGE_WS


@respx.mock
async def test_a_page_is_created_when_the_browser_has_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh browser can list no pages at all; one is made.

    Two HTTP round trips and one browser-level command, in that order: list,
    open a browser socket, `Target.createTarget`, list again to find the new
    target's page socket.
    """
    listing = respx.get(f"{CDP_BASE}/json/list")
    listing.side_effect = [
        _targets(),  # nothing open yet
        _targets({"id": "T1", "webSocketDebuggerUrl": PAGE_WS}),
    ]
    respx.get(f"{CDP_BASE}/json/version").mock(
        return_value=httpx.Response(200, json={"webSocketDebuggerUrl": BROWSER_WS})
    )

    class BrowserSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send(self, raw: str) -> None:
            self.sent.append(raw)

        async def recv(self) -> str:
            return json.dumps({"id": 1, "result": {"targetId": "T1"}})

        async def close(self) -> None: ...

        async def __aenter__(self) -> BrowserSocket:
            return self

        async def __aexit__(self, *exc: Any) -> None: ...

    socket = BrowserSocket()
    import websockets

    monkeypatch.setattr(websockets, "connect", lambda *a, **k: socket)

    assert await CDPConnection()._resolve_ws_url(CDP_BASE) == PAGE_WS
    assert any("Target.createTarget" in raw for raw in socket.sent)


@respx.mock
async def test_a_created_target_that_never_appears_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Better a clear failure than a connection to nothing."""
    listing = respx.get(f"{CDP_BASE}/json/list")
    listing.side_effect = [_targets(), _targets({"id": "SOMETHING-ELSE"})]
    respx.get(f"{CDP_BASE}/json/version").mock(
        return_value=httpx.Response(200, json={"webSocketDebuggerUrl": BROWSER_WS})
    )

    class BrowserSocket:
        async def send(self, raw: str) -> None: ...

        async def recv(self) -> str:
            return json.dumps({"id": 1, "result": {"targetId": "T1"}})

        async def close(self) -> None: ...

        async def __aenter__(self) -> BrowserSocket:
            return self

        async def __aexit__(self, *exc: Any) -> None: ...

    import websockets

    monkeypatch.setattr(websockets, "connect", lambda *a, **k: BrowserSocket())

    with pytest.raises(ValueError, match="Could not find page target"):
        await CDPConnection()._resolve_ws_url(CDP_BASE)


@respx.mock
async def test_a_browser_that_will_not_open_a_target(monkeypatch: pytest.MonkeyPatch) -> None:
    respx.get(f"{CDP_BASE}/json/list").mock(return_value=_targets())
    respx.get(f"{CDP_BASE}/json/version").mock(
        return_value=httpx.Response(200, json={"webSocketDebuggerUrl": BROWSER_WS})
    )

    class Refusing:
        async def send(self, raw: str) -> None: ...

        async def recv(self) -> str:
            return json.dumps({"id": 1, "result": {}})  # no targetId

        async def close(self) -> None: ...

        async def __aenter__(self) -> Refusing:
            return self

        async def __aexit__(self, *exc: Any) -> None: ...

    import websockets

    monkeypatch.setattr(websockets, "connect", lambda *a, **k: Refusing())

    with pytest.raises(ValueError, match="Failed to create a page target"):
        await CDPConnection()._resolve_ws_url(CDP_BASE)


# -- opening a socket -------------------------------------------------------


async def test_connecting_by_http_base_resolves_then_opens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = ScriptedSocket()

    async def fake_resolve(self: Any, cdp_url: str) -> str:
        return PAGE_WS

    monkeypatch.setattr(CDPConnection, "_resolve_ws_url", fake_resolve)
    import websockets

    async def fake_connect(url: str, **kwargs: Any) -> ScriptedSocket:
        assert url == PAGE_WS
        return socket

    monkeypatch.setattr(websockets, "connect", fake_connect)

    conn = CDPConnection()
    await conn.connect(CDP_BASE)

    assert conn._ws is socket
    assert conn._reader_task is not None
    await conn.close()


async def test_connecting_by_websocket_can_skip_the_page_attach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`attach_to_page=False` is for a URL that is already a page endpoint."""
    socket = ScriptedSocket()
    import websockets

    async def fake_connect(url: str, **kwargs: Any) -> ScriptedSocket:
        return socket

    monkeypatch.setattr(websockets, "connect", fake_connect)

    conn = CDPConnection()
    await conn.connect_ws(PAGE_WS, attach_to_page=False)

    assert socket.sent == [], "it attached to a page on an endpoint that is already one"
    await conn.close()


async def test_a_browser_endpoint_attaches_to_a_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """Browserbase hands out the browser endpoint; `Page.enable` fails on it."""
    socket = ScriptedSocket(
        {
            "Target.getTargets": {"result": {"targetInfos": [{"type": "page", "targetId": "T1"}]}},
            "Target.attachToTarget": {"result": {"sessionId": "S1"}},
        }
    )
    import websockets

    async def fake_connect(url: str, **kwargs: Any) -> ScriptedSocket:
        return socket

    monkeypatch.setattr(websockets, "connect", fake_connect)

    conn = CDPConnection()
    await conn.connect_ws(BROWSER_WS)

    assert conn._session_id == "S1", "commands would go to the browser, not the page"
    await conn.close()


# -- teardown ---------------------------------------------------------------


async def test_closing_cancels_the_reader_and_drops_the_socket() -> None:
    conn = CDPConnection()
    socket = ScriptedSocket()
    conn._ws = socket  # type: ignore[assignment]
    conn._reader_task = asyncio.create_task(conn._reader_loop())
    conn._session_id = "S1"
    conn._pending[1] = asyncio.get_running_loop().create_future()

    await conn.close()

    assert socket.closed
    assert conn._ws is None
    assert conn._pending == {}
    assert conn._session_id is None


async def test_closing_a_socket_that_refuses_to_close() -> None:
    """Teardown is the last chance to release a metered browser."""

    class Stubborn(ScriptedSocket):
        async def close(self) -> None:
            raise RuntimeError("socket already gone")

    conn = CDPConnection()
    conn._ws = Stubborn()  # type: ignore[assignment]

    await conn.close()

    assert conn._ws is None


async def test_closing_twice_is_safe() -> None:
    conn = CDPConnection()
    await conn.close()
    await conn.close()


async def test_a_socket_with_no_state_attribute_is_read_by_the_old_flag() -> None:
    """websockets < 13 exposed `.open` instead of a `State` enum."""

    class Legacy:
        open = True

    conn = CDPConnection()
    conn._ws = Legacy()  # type: ignore[assignment]

    assert conn.is_connected is True


def test_no_socket_is_not_connected() -> None:
    assert CDPConnection().is_connected is False


# -- the discovery fallbacks ------------------------------------------------


@respx.mock
async def test_a_failing_json_list_falls_through_to_json_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HTTP base can be up while `/json/list` is not.

    Falling through rather than giving up is what lets a browser that has not
    finished opening its first page still be reached.
    """
    respx.get(f"{CDP_BASE}/json/list").side_effect = [
        httpx.Response(503),
        _targets({"id": "T1", "webSocketDebuggerUrl": PAGE_WS}),
    ]
    respx.get(f"{CDP_BASE}/json/version").mock(
        return_value=httpx.Response(200, json={"webSocketDebuggerUrl": BROWSER_WS})
    )

    class BrowserSocket:
        async def send(self, raw: str) -> None: ...

        async def recv(self) -> str:
            return json.dumps({"id": 1, "result": {"targetId": "T1"}})

        async def close(self) -> None: ...

        async def __aenter__(self) -> BrowserSocket:
            return self

        async def __aexit__(self, *exc: Any) -> None: ...

    import websockets

    monkeypatch.setattr(websockets, "connect", lambda *a, **k: BrowserSocket())

    assert await CDPConnection()._resolve_ws_url(CDP_BASE) == PAGE_WS


@respx.mock
async def test_a_version_response_with_no_socket_url_is_an_error() -> None:
    """Nothing to connect to; say so rather than opening `None`."""
    respx.get(f"{CDP_BASE}/json/list").mock(return_value=_targets())
    respx.get(f"{CDP_BASE}/json/version").mock(return_value=httpx.Response(200, json={}))

    with pytest.raises(ValueError, match="No webSocketDebuggerUrl"):
        await CDPConnection()._resolve_ws_url(CDP_BASE)


# -- the reader loop --------------------------------------------------------


async def test_a_dropped_socket_fails_everyone_still_waiting() -> None:
    """Otherwise each pending command blocks for its full timeout instead."""

    class Dropping:
        def __aiter__(self) -> Dropping:
            return self

        async def __anext__(self) -> str:
            raise ConnectionResetError("socket died")

    conn = CDPConnection()
    conn._ws = Dropping()  # type: ignore[assignment]
    waiting = asyncio.get_running_loop().create_future()
    settled = asyncio.get_running_loop().create_future()
    settled.set_result({"id": 2, "result": {}})
    conn._pending = {1: waiting, 2: settled}

    await conn._reader_loop()

    assert isinstance(waiting.exception(), ConnectionError)
    assert settled.result() == {"id": 2, "result": {}}, (
        "an answer that had already arrived was overwritten by the disconnect"
    )


async def test_the_reader_ignores_messages_that_match_nothing() -> None:
    """CDP events carry no id, and a late reply may have no pending future."""

    class Stream:
        def __init__(self) -> None:
            self._messages = [
                json.dumps({"method": "Page.loadEventFired", "params": {}}),
                json.dumps({"id": 99, "result": {}}),
                json.dumps({"id": 1, "result": {"ok": True}}),
            ]

        def __aiter__(self) -> Stream:
            return self

        async def __anext__(self) -> str:
            if not self._messages:
                raise StopAsyncIteration
            return self._messages.pop(0)

    conn = CDPConnection()
    conn._ws = Stream()  # type: ignore[assignment]
    waiting = asyncio.get_running_loop().create_future()
    conn._pending = {1: waiting}

    await conn._reader_loop()

    assert waiting.result() == {"id": 1, "result": {"ok": True}}


async def test_closing_cancels_a_reader_that_is_still_running() -> None:
    """The reader holds the socket; leaving it alive leaks the connection."""

    class Forever:
        def __aiter__(self) -> Forever:
            return self

        async def __anext__(self) -> str:
            await asyncio.sleep(3600)
            raise StopAsyncIteration  # pragma: no cover - never reached

        async def close(self) -> None: ...

    conn = CDPConnection()
    conn._ws = Forever()  # type: ignore[assignment]
    conn._reader_task = asyncio.create_task(conn._reader_loop())
    await asyncio.sleep(0)

    await conn.close()

    assert conn._reader_task.done()


# -- reading the page -------------------------------------------------------


async def test_page_info_asks_the_page_for_its_elements() -> None:
    """The stamped-element script is what every click and type depends on."""
    captured: list[str] = []

    class EvaluatingSocket(ScriptedSocket):
        async def send(self, raw: str) -> None:
            message = json.loads(raw)
            self.sent.append(message)
            captured.append(message["params"]["expression"])
            await self._outbox.put(
                json.dumps(
                    {
                        "id": message["id"],
                        "result": {
                            "result": {
                                "value": {"title": "T", "text": "", "elements": [], "html": ""}
                            }
                        },
                    }
                )
            )

    socket = EvaluatingSocket()
    conn = CDPConnection()
    conn._ws = socket  # type: ignore[assignment]
    conn._reader_task = asyncio.create_task(conn._reader_loop())

    info = await conn.page_info()

    assert info["title"] == "T"
    script = captured[0]
    assert "data-kwami-id" in script, "elements are not stamped, so clicks cannot address them"
    assert "getBoundingClientRect" in script, "no coordinates, so nothing can be clicked"
    await conn.close()


async def test_going_forward_moves_to_the_next_history_entry() -> None:
    """Forward is only meaningful when there is somewhere ahead to go."""
    replies = {
        "Page.getNavigationHistory": {
            "result": {
                "currentIndex": 0,
                "entries": [{"id": 1}, {"id": 2}],
            }
        },
        "Page.navigateToHistoryEntry": {"result": {}},
    }
    socket = ScriptedSocket(replies)
    conn = CDPConnection()
    conn._ws = socket  # type: ignore[assignment]
    conn._reader_task = asyncio.create_task(conn._reader_loop())

    await conn.go_forward()

    sent = [message["method"] for message in socket.sent]
    assert "Page.navigateToHistoryEntry" in sent
    entry = next(m for m in socket.sent if m["method"] == "Page.navigateToHistoryEntry")
    assert entry["params"]["entryId"] == 2, "it went back, or nowhere"
    await conn.close()
