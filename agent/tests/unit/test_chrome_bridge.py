"""Kwami's bridge to the user's own Chrome: who may connect, and what crosses it.

The hub is real here, on a free port of 127.0.0.1, and so are the sockets: the
extension's side is played by a websocket client sending the `Origin` Chrome would.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosedOK, InvalidStatus

from src.browser import chrome_bridge
from src.browser.chrome_bridge import (
    EXTENSION_ORIGIN,
    TOKEN_HEADER,
    BridgeClient,
    BridgeError,
    BridgeHub,
    BridgeUnavailableError,
    _Link,
)


@pytest.fixture
async def hub(tmp_path: Path):
    hub = BridgeHub(port=0, state_file=tmp_path / "chrome-bridge.json")
    assert await hub.start()
    yield hub
    await hub.stop()


class FakeExtension:
    """The extension's end of the socket, answering commands from a table.

    A value answers with it, a callable with what it returns for the args, an
    exception with its text as the error, and a missing command is never answered.
    """

    def __init__(self, hub: BridgeHub, answers: dict[str, Any] | None = None) -> None:
        self.hub = hub
        self.answers = answers or {}
        self.received: list[dict[str, Any]] = []
        self.pings = 0
        self.connection: Any = None
        self.task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> FakeExtension:
        self.connection = await connect(
            f"ws://127.0.0.1:{self.hub.port}/extension", origin=EXTENSION_ORIGIN
        )
        await self.connection.send(json.dumps({"type": "hello", "version": "1.0.0"}))
        self.task = asyncio.create_task(self._serve())
        for _ in range(200):
            if await self.hub.ready():
                break
            await asyncio.sleep(0.01)
        return self

    async def _serve(self) -> None:
        with contextlib.suppress(Exception):
            async for raw in self.connection:
                message = json.loads(raw)
                if message.get("type") == "ping":
                    self.pings += 1
                    await self.connection.send('{"type":"pong"}')
                    continue
                self.received.append(message)
                answer = self.answers.get(message["cmd"])
                if answer is None:
                    continue
                if isinstance(answer, Exception):
                    reply = {"id": message["id"], "ok": False, "error": str(answer)}
                else:
                    result = answer(message["args"]) if callable(answer) else answer
                    reply = {"id": message["id"], "ok": True, "result": result}
                await self.connection.send(json.dumps(reply))

    async def __aexit__(self, *_: Any) -> None:
        await self.connection.close()
        if self.task is not None:
            await self.task
        for _ in range(200):
            if not await self.hub.ready():
                break
            await asyncio.sleep(0.01)


async def _status_of(url: str, **kwargs: Any) -> int:
    try:
        async with connect(url, open_timeout=3, **kwargs):
            return 101
    except InvalidStatus as e:
        return e.response.status_code


# -- who may connect -----------------------------------------------------------------


async def test_only_kwamis_extension_may_be_the_browser(hub: BridgeHub) -> None:
    base = f"ws://127.0.0.1:{hub.port}"
    other = "chrome-extension://aaaabbbbccccddddeeeeffffgggghhhh"
    assert await _status_of(f"{base}/extension", origin=other) == 403
    assert await _status_of(f"{base}/extension") == 403
    assert await _status_of(f"{base}/extension", origin=EXTENSION_ORIGIN) == 101


async def test_a_web_page_cannot_drive_it_even_with_the_token(hub: BridgeHub) -> None:
    base = f"ws://127.0.0.1:{hub.port}"
    token = {TOKEN_HEADER: hub.token}
    assert (
        await _status_of(f"{base}/control", additional_headers=token, origin="https://x.io") == 403
    )
    assert await _status_of(f"{base}/control") == 403
    assert await _status_of(f"{base}/control", additional_headers={TOKEN_HEADER: "nope"}) == 403
    assert await _status_of(f"{base}/control", additional_headers=token) == 101
    assert await _status_of(f"{base}/anything") == 404


async def test_it_tells_jarvis_where_it_listens_and_forgets_it_on_stop(tmp_path: Path) -> None:
    state = tmp_path / "chrome-bridge.json"
    hub = BridgeHub(port=0, state_file=state)
    assert await hub.start()
    written = json.loads(state.read_text(encoding="utf-8"))
    assert written["port"] == hub.port != 0
    assert written["token"] == hub.token
    assert chrome_bridge.local_hub() is hub
    await hub.stop()
    assert not state.exists()
    assert chrome_bridge.local_hub() is None
    await hub.stop()  # Twice is harmless.


async def test_a_newer_hubs_state_is_left_alone(tmp_path: Path) -> None:
    state = tmp_path / "chrome-bridge.json"
    first = BridgeHub(port=0, state_file=state)
    second = BridgeHub(port=0, state_file=state)
    assert await first.start()
    assert await second.start()
    await first.stop()
    assert json.loads(state.read_text(encoding="utf-8"))["token"] == second.token
    assert chrome_bridge.local_hub() is second
    state.unlink()
    await second.stop()  # Its file already gone: nothing to remove.


async def test_a_taken_port_is_said_not_raised(hub: BridgeHub, tmp_path: Path) -> None:
    rival = BridgeHub(port=hub.port, state_file=tmp_path / "other.json")
    assert await rival.start() is False


# -- the browser ------------------------------------------------------------------------


async def test_a_command_reaches_chrome_and_its_answer_comes_back(hub: BridgeHub) -> None:
    tabs = {"tabs": [{"id": 7, "url": "https://example.org/"}]}
    async with FakeExtension(hub, {"tabs": tabs, "echo": lambda args: args}) as chrome:
        assert hub.status() == {"connected": True, "version": "1.0.0", "browsers": 1}
        assert await hub.call("tabs") == tabs
        assert await hub.call("echo", {"a": 1}) == {"a": 1}
        assert chrome.received[0]["cmd"] == "tabs"
    assert hub.status()["connected"] is False


async def test_chromes_refusal_is_an_error_with_its_words(hub: BridgeHub) -> None:
    async with FakeExtension(hub, {"type": RuntimeError("secret-field")}):
        with pytest.raises(BridgeError, match="secret-field"):
            await hub.call("type", {"text": "x"})


async def test_without_chrome_nothing_is_sent_but_the_status_answers(hub: BridgeHub) -> None:
    assert await hub.call("status") == {"connected": False, "version": "", "browsers": 0}
    with pytest.raises(BridgeUnavailableError):
        await hub.call("tabs")


async def test_a_chrome_that_does_not_answer_times_out(hub: BridgeHub) -> None:
    async with FakeExtension(hub):
        with pytest.raises(BridgeError, match="in time"):
            await hub.call("tabs", timeout=0.05)
        assert hub._pending == {}


async def test_chrome_leaving_fails_what_waits_on_it(hub: BridgeHub) -> None:
    async with FakeExtension(hub) as chrome:
        waiting = asyncio.create_task(hub.call("tabs", timeout=5))
        for _ in range(200):
            if chrome.received:
                break
            await asyncio.sleep(0.01)
        await chrome.connection.close()
        with pytest.raises(BridgeUnavailableError):
            await waiting


async def test_a_socket_that_fails_to_send_is_a_lost_chrome(hub: BridgeHub) -> None:
    class Broken:
        async def send(self, _data: str) -> None:
            raise OSError("gone")

    hub._links.append(_Link(Broken(), "1.0.0", "", 0.0))
    with pytest.raises(BridgeUnavailableError):
        await hub.call("tabs")
    hub._links.clear()


async def test_stopping_fails_what_waits(tmp_path: Path) -> None:
    hub = BridgeHub(port=0, state_file=tmp_path / "s.json")
    assert await hub.start()
    async with FakeExtension(hub) as chrome:
        waiting = asyncio.create_task(hub.call("tabs", timeout=5))
        for _ in range(200):
            if chrome.received:
                break
            await asyncio.sleep(0.01)
        await hub.stop()
        with pytest.raises(BridgeUnavailableError):
            await waiting


async def test_it_keeps_chrome_awake_and_ignores_what_it_cannot_read(
    hub: BridgeHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chrome_bridge, "PING_SECONDS", 0.01)
    async with FakeExtension(hub, {"tabs": {"tabs": []}}) as chrome:
        for raw in ("not json", "[1, 2]", '{"type": "hello"}', '{"id": 999, "ok": true}'):
            await chrome.connection.send(raw)
        for _ in range(200):
            if chrome.pings >= 2:
                break
            await asyncio.sleep(0.01)
        assert chrome.pings >= 2
        assert await hub.call("tabs") == {"tabs": []}


async def test_a_link_that_never_says_hello_is_not_a_browser(hub: BridgeHub) -> None:
    async with connect(f"ws://127.0.0.1:{hub.port}/extension", origin=EXTENSION_ORIGIN) as ws:
        await ws.send('{"type":"pong"}')
        assert not await hub.ready()


async def test_an_answer_for_a_call_already_over_is_dropped(hub: BridgeHub) -> None:
    done: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    done.set_result("first")
    hub._pending[41] = (done, _Link(None, "", "", 0.0))
    hub._settle({"id": 41, "ok": True, "result": "second"})
    assert done.result() == "first"


# -- Jarvis's processes -------------------------------------------------------------------


async def test_another_process_reaches_chrome_through_the_hub(hub: BridgeHub, tmp_path) -> None:
    client = BridgeClient(state_file=tmp_path / "chrome-bridge.json")
    async with FakeExtension(hub, {"tabs": {"tabs": [1]}, "click": RuntimeError("not-found")}):
        assert await client.ready() is True
        assert await client.call("tabs") == {"tabs": [1]}
        with pytest.raises(BridgeError, match="not-found") as refused:
            await client.call("click", {"elementId": "el-1"})
        assert not isinstance(refused.value, BridgeUnavailableError)
    with pytest.raises(BridgeUnavailableError):
        await client.call("tabs")
    await client.close()
    await client.close()


async def test_the_client_asks_at_most_every_few_seconds(hub: BridgeHub, tmp_path) -> None:
    client = BridgeClient(state_file=tmp_path / "chrome-bridge.json")
    assert await client.ready() is False
    async with FakeExtension(hub):
        assert await client.ready() is False, "the last answer still holds"
        client._ready_at -= chrome_bridge.STATUS_TTL_SECONDS
        assert await client.ready() is True
    await client.close()


async def test_without_a_hub_the_client_says_so(tmp_path: Path) -> None:
    missing = BridgeClient(state_file=tmp_path / "nothing.json")
    with pytest.raises(BridgeUnavailableError, match="not running"):
        await missing.call("tabs")
    assert await missing.ready() is False

    stale = tmp_path / "stale.json"
    hub = BridgeHub(port=0, state_file=stale)
    assert await hub.start()
    port = hub.port
    await hub.stop()
    stale.write_text(json.dumps({"port": port, "token": "t"}), encoding="utf-8")
    with pytest.raises(BridgeUnavailableError, match="did not answer"):
        await BridgeClient(state_file=stale).call("tabs")


async def test_a_wrong_token_is_turned_away(hub: BridgeHub, tmp_path: Path) -> None:
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps({"port": hub.port, "token": "guess"}), encoding="utf-8")
    with pytest.raises(BridgeUnavailableError):
        await BridgeClient(state_file=forged).call("tabs")


async def test_the_hub_answers_odd_control_messages_without_falling(hub: BridgeHub) -> None:
    async with (
        FakeExtension(hub, {"tabs": lambda args: args}),
        connect(
            f"ws://127.0.0.1:{hub.port}/control", additional_headers={TOKEN_HEADER: hub.token}
        ) as control,
    ):
        await control.send("not json")
        await control.send('{"no": "id"}')
        await control.send(json.dumps({"id": 1, "cmd": "tabs", "args": "not a dict"}))
        assert json.loads(await control.recv()) == {"id": 1, "result": {}, "ok": True}
        await control.send(json.dumps({"id": 2, "cmd": "tabs", "timeout": "soon"}))
        second = json.loads(await control.recv())
        assert second["ok"] is False and second["id"] == 2


async def test_a_control_link_that_leaves_mid_call_cancels_it(hub: BridgeHub) -> None:
    async with FakeExtension(hub) as chrome:
        async with connect(
            f"ws://127.0.0.1:{hub.port}/control", additional_headers={TOKEN_HEADER: hub.token}
        ) as control:
            await control.send(json.dumps({"id": 1, "cmd": "tabs", "timeout": 5}))
            for _ in range(200):
                if chrome.received:
                    break
                await asyncio.sleep(0.01)
        for _ in range(200):
            if not hub._pending:
                break
            await asyncio.sleep(0.01)
        assert hub._pending == {}


async def _silent_hub(tmp_path: Path, close_on_message: bool) -> tuple[Any, Path]:
    async def handler(connection: Any) -> None:
        with contextlib.suppress(Exception):
            async for _raw in connection:
                if close_on_message:
                    await connection.close()
                    return
                await connection.send("not json")
                await connection.send('{"id": 12345, "ok": true}')

    server = await serve(handler, "127.0.0.1", 0)
    port = next(iter(server.sockets)).getsockname()[1]
    state = tmp_path / f"silent-{close_on_message}.json"
    state.write_text(json.dumps({"port": port, "token": "t"}), encoding="utf-8")
    return server, state


async def test_the_client_gives_up_on_a_hub_that_never_answers(tmp_path: Path) -> None:
    server, state = await _silent_hub(tmp_path, close_on_message=False)
    try:
        with pytest.raises(BridgeError, match="in time"):
            await BridgeClient(state_file=state).call("tabs", timeout=0.01)
    finally:
        server.close()


async def test_the_client_fails_what_waits_when_the_hub_closes(tmp_path: Path) -> None:
    server, state = await _silent_hub(tmp_path, close_on_message=True)
    try:
        with pytest.raises(BridgeUnavailableError):
            await BridgeClient(state_file=state).call("tabs", timeout=5)
    finally:
        server.close()


async def test_the_client_starts_over_on_a_new_event_loop_or_a_dead_socket() -> None:
    client = BridgeClient()

    class Dead:
        async def send(self, _data: str) -> None:
            raise ConnectionClosedOK(None, None)

    client._loop = asyncio.get_running_loop()
    client._connection = Dead()
    with pytest.raises(BridgeUnavailableError):
        await client.call("tabs")

    client._loop = object()  # type: ignore[assignment]
    client._state_file = Path("does-not-exist.json")
    with pytest.raises(BridgeUnavailableError, match="not running"):
        await client.call("tabs")
    assert client._connection is None


# -- the switches -------------------------------------------------------------------------


def test_the_port_comes_from_the_environment_when_it_makes_sense(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(chrome_bridge.PORT_ENV, "50000")
    assert chrome_bridge.bridge_port() == 50000
    monkeypatch.setenv(chrome_bridge.PORT_ENV, "seventy")
    assert chrome_bridge.bridge_port() == chrome_bridge.DEFAULT_PORT
    monkeypatch.setenv(chrome_bridge.PORT_ENV, "70000")
    assert chrome_bridge.bridge_port() == chrome_bridge.DEFAULT_PORT
    monkeypatch.delenv(chrome_bridge.PORT_ENV)
    assert chrome_bridge.bridge_port() == chrome_bridge.DEFAULT_PORT


def test_only_on_the_users_own_computer_and_unless_turned_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = SimpleNamespace(browser_provider="local")
    monkeypatch.setattr(chrome_bridge, "get_settings", lambda: provider)
    monkeypatch.delenv(chrome_bridge.SWITCH, raising=False)
    assert chrome_bridge.bridge_enabled() is True
    monkeypatch.setenv(chrome_bridge.SWITCH, "off")
    assert chrome_bridge.bridge_enabled() is False
    monkeypatch.setenv(chrome_bridge.SWITCH, "on")
    provider.browser_provider = "browser_use"
    assert chrome_bridge.bridge_enabled() is False


async def test_a_process_uses_its_own_hub_or_one_client(
    hub: BridgeHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert chrome_bridge.chrome_bridge() is hub
    monkeypatch.setattr(chrome_bridge, "_local_hub", None)
    monkeypatch.setattr(chrome_bridge, "_client", None)
    first = chrome_bridge.chrome_bridge()
    assert isinstance(first, BridgeClient)
    assert chrome_bridge.chrome_bridge() is first


async def test_ready_is_false_when_off_or_when_asking_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chrome_bridge, "bridge_enabled", lambda: False)
    assert await chrome_bridge.user_chrome_ready() is False

    class Bridge:
        def __init__(self, answer: Any) -> None:
            self.answer = answer

        async def ready(self) -> bool:
            if isinstance(self.answer, Exception):
                raise self.answer
            return bool(self.answer)

    monkeypatch.setattr(chrome_bridge, "bridge_enabled", lambda: True)
    monkeypatch.setattr(chrome_bridge, "chrome_bridge", lambda: Bridge(True))
    assert await chrome_bridge.user_chrome_ready() is True
    monkeypatch.setattr(chrome_bridge, "chrome_bridge", lambda: Bridge(RuntimeError("x")))
    assert await chrome_bridge.user_chrome_ready() is False


async def test_kwamis_extension_first_then_claude_in_chrome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.browser import claude_chrome

    class Way:
        def __init__(self, there: bool) -> None:
            self.there = there

        async def ready(self) -> bool:
            return self.there

    kwami, claude = Way(True), Way(True)
    monkeypatch.setattr(chrome_bridge, "bridge_enabled", lambda: True)
    monkeypatch.setattr(chrome_bridge, "chrome_bridge", lambda: kwami)
    monkeypatch.setattr(claude_chrome, "claude_chrome", lambda: claude)
    assert await chrome_bridge.user_browser() is kwami
    kwami.there = False
    assert await chrome_bridge.user_browser() is claude
    claude.there = False
    assert await chrome_bridge.user_browser() is None
    monkeypatch.setattr(claude_chrome, "claude_chrome", lambda: None)  # turned off
    assert await chrome_bridge.user_browser() is None
    assert await chrome_bridge.user_chrome_ready() is False
    monkeypatch.setattr(chrome_bridge, "bridge_enabled", lambda: False)
    kwami.there = True
    assert await chrome_bridge.user_browser() is None


async def test_the_work_thread_hosts_the_hub_only_where_it_should(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(chrome_bridge, "bridge_enabled", lambda: False)
    assert await chrome_bridge.start_hub() is None

    monkeypatch.setattr(chrome_bridge, "bridge_enabled", lambda: True)
    monkeypatch.setattr(chrome_bridge, "STATE_FILE", tmp_path / "hub.json")
    monkeypatch.setattr(chrome_bridge, "bridge_port", lambda: 0)
    original = BridgeHub.__init__

    def at_tmp(self: BridgeHub, port: int | None = None, **_: Any) -> None:
        original(self, port=port, state_file=tmp_path / "hub.json")

    monkeypatch.setattr(BridgeHub, "__init__", at_tmp)
    hub = await chrome_bridge.start_hub()
    assert hub is not None
    await hub.stop()

    async def refuse(self: BridgeHub) -> bool:
        return False

    monkeypatch.setattr(BridgeHub, "start", refuse)
    assert await chrome_bridge.start_hub() is None


# -- the rough edges ----------------------------------------------------------------------


async def test_a_hub_on_a_port_of_its_choosing(tmp_path: Path) -> None:
    probe = BridgeHub(port=0, state_file=tmp_path / "probe.json")
    assert await probe.start()
    port = probe.port
    await probe.stop()
    hub = BridgeHub(port=port, state_file=tmp_path / "fixed.json")
    assert await hub.start()
    assert hub.port == port
    await hub.stop()


async def test_stopping_fails_a_call_whatever_held_it(tmp_path: Path) -> None:
    hub = BridgeHub(port=0, state_file=tmp_path / "s.json")
    assert await hub.start()
    waiting: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    answered: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    answered.set_result("in time")
    hub._pending[3] = (waiting, _Link(None, "", "", 0.0))
    hub._pending[4] = (answered, _Link(None, "", "", 0.0))
    await hub.stop()
    assert answered.result() == "in time"
    with pytest.raises(BridgeUnavailableError, match="stopped"):
        waiting.result()


async def test_a_chrome_that_drops_off_fails_only_its_own_calls(hub: BridgeHub) -> None:
    async with FakeExtension(hub) as first, FakeExtension(hub) as second:
        for _ in range(200):
            if len(hub._links) == 2:
                break
            await asyncio.sleep(0.01)
        theirs = asyncio.create_task(hub.call("tabs", timeout=5))
        for _ in range(200):
            if second.received:
                break
            await asyncio.sleep(0.01)
        over: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        over.set_result("answered")
        hub._pending[77] = (over, hub._links[0])
        first.connection.transport.abort()
        for _ in range(200):
            if len(hub._links) == 1:
                break
            await asyncio.sleep(0.01)
        assert not theirs.done(), "the other Chrome's call stays"
        await second.connection.send(
            json.dumps({"id": second.received[0]["id"], "ok": True, "result": "fine"})
        )
        assert await theirs == "fine"
        assert over.result() == "answered"


async def test_a_control_link_cut_off_is_let_go(hub: BridgeHub) -> None:
    control = await connect(
        f"ws://127.0.0.1:{hub.port}/control", additional_headers={TOKEN_HEADER: hub.token}
    )
    control.transport.abort()
    await asyncio.sleep(0.05)
    assert await hub.call("status") == {"connected": False, "version": "", "browsers": 0}


async def test_the_client_lets_go_of_a_hub_cut_off(tmp_path: Path) -> None:
    async def handler(connection: Any) -> None:
        with contextlib.suppress(Exception):
            async for _raw in connection:
                connection.transport.abort()
                return

    server = await serve(handler, "127.0.0.1", 0)
    state = tmp_path / "cut.json"
    port = next(iter(server.sockets)).getsockname()[1]
    state.write_text(json.dumps({"port": port, "token": "t"}), encoding="utf-8")
    client = BridgeClient(state_file=state)
    try:
        await client._connect()
        over: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        over.set_result("answered")
        client._pending[999] = over
        with pytest.raises(BridgeUnavailableError):
            await client.call("tabs", timeout=5)
        assert over.result() == "answered"
    finally:
        server.close()
