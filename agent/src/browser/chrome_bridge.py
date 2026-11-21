"""The user's own Chrome, through Kwami's extension, on their PC (2026-11-17).

Jarvis's own Chrome (`local_chrome`) is driven over a debugging port, and sites that
guard their sign-in (Google, and everything that signs in through it) refuse it. The
user's everyday Chrome is already signed in to all of it. Kwami's extension
(`chrome-extension/` at the root of this repository) lives in that Chrome and answers
Jarvis from there, the way Claude in Chrome answers Claude, without depending on which
claude.ai account anything is signed in to.

The pieces:

* **The hub** (`BridgeHub`) listens on 127.0.0.1 only, on a fixed port the extension
  knows (`KWAMI_CHROME_BRIDGE_PORT`, 47219 by default). The PC's work thread hosts it,
  since it runs for as long as KWAMI does; the voice reaches it as a client.
* **The browser** connects on `/extension`. Only Kwami's extension may: the hub checks
  the `Origin` Chrome sends, `chrome-extension://<id>`, and the id is fixed by the key
  in the extension's manifest. A web page cannot send that origin.
* **Jarvis's processes** connect on `/control` with the token the hub writes in
  `~/.jarvis/chrome-bridge.json` (readable by this Windows user only, like the rest of
  `~/.jarvis`). A connection that sends any `Origin`, as every web page's does, is
  refused, so a site open in a browser cannot drive the bridge.

What a command may do is the extension's business (`background.js`): it runs only its
own functions, never code sent over the socket, and never types into a password, code or
card field. The user pauses it from its popup. `KWAMI_CHROME_BRIDGE=off` turns the hub off.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import itertools
import json
import os
import secrets
import time
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from ..settings import get_settings
from ..utils.logging import get_logger
from .launched import LOCAL

logger = get_logger("browser.chrome_bridge")

#: Fixed by the `key` in the extension's manifest: only this extension can be the browser.
EXTENSION_ID = "bkgbdgdkjbfagjkghfdfeilncobkiiie"
EXTENSION_ORIGIN = f"chrome-extension://{EXTENSION_ID}"
DEFAULT_PORT = 47219
PORT_ENV = "KWAMI_CHROME_BRIDGE_PORT"
SWITCH = "KWAMI_CHROME_BRIDGE"
STATE_FILE = Path.home() / ".jarvis" / "chrome-bridge.json"
#: The header name, not a secret: the token itself is made at random when the hub starts.
TOKEN_HEADER = "X-Kwami-Bridge-Token"  # noqa: S105
#: The extension's service worker sleeps after 30 s without traffic; this keeps it awake.
PING_SECONDS = 20.0
CALL_TIMEOUT_SECONDS = 20.0
MAX_CALL_TIMEOUT_SECONDS = 120.0
#: A picture of a tab is the largest thing that crosses.
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
#: How long "is the user's Chrome there?" is trusted before asking again.
STATUS_TTL_SECONDS = 3.0
STATUS_TIMEOUT_SECONDS = 1.5


class BridgeError(RuntimeError):
    """The user's Chrome answered with an error, or did not answer."""


class BridgeUnavailableError(BridgeError):
    """No bridge on this computer, or no Chrome connected to it."""


class ChromeBridgePort(Protocol):
    """A way into the user's own Chrome: Kwami's extension, or Claude in Chrome's."""

    async def call(
        self, cmd: str, args: dict[str, Any] | None = None, timeout: float = CALL_TIMEOUT_SECONDS
    ) -> Any: ...

    async def ready(self) -> bool: ...


def bridge_port() -> int:
    try:
        port = int(os.environ.get(PORT_ENV, "") or DEFAULT_PORT)
    except ValueError:
        return DEFAULT_PORT
    return port if 0 <= port <= 65535 else DEFAULT_PORT


def bridge_enabled() -> bool:
    """Only where Jarvis runs on the user's own computer, and unless they turned it off."""
    if os.environ.get(SWITCH, "on").strip().lower() in ("off", "0", "false", "no"):
        return False
    return get_settings().browser_provider == LOCAL


def _parse(raw: Any) -> dict[str, Any] | None:
    try:
        message = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return message if isinstance(message, dict) else None


def _closed_errors() -> tuple[type[BaseException], ...]:
    from websockets.exceptions import ConnectionClosed

    return (ConnectionClosed, OSError)


@dataclass
class _Link:
    """One Chrome connected through the extension."""

    connection: Any
    version: str
    browser: str
    since: float


class BridgeHub:
    """Where the user's Chrome and Jarvis's processes meet. One per computer."""

    def __init__(
        self,
        port: int | None = None,
        state_file: Path = STATE_FILE,
        token: str | None = None,
    ) -> None:
        self._port = bridge_port() if port is None else port
        self._state_file = state_file
        self._token = token or secrets.token_urlsafe(32)
        self._links: list[_Link] = []
        self._pending: dict[int, tuple[asyncio.Future[Any], _Link]] = {}
        self._ids = itertools.count(1)
        self._server: Any = None

    @property
    def port(self) -> int:
        return self._port

    @property
    def token(self) -> str:
        return self._token

    async def start(self) -> bool:
        """Listen, and tell Jarvis's processes where. False if the port is taken."""
        from websockets.asyncio.server import serve

        try:
            self._server = await serve(
                self._handle,
                "127.0.0.1",
                self._port,
                process_request=self._check,
                max_size=MAX_MESSAGE_BYTES,
                ping_interval=None,
            )
        except OSError as e:
            logger.warning("The Chrome bridge cannot listen on port %s: %s", self._port, e)
            return False
        if self._port == 0:
            self._port = int(next(iter(self._server.sockets)).getsockname()[1])
        await asyncio.to_thread(self._write_state)
        set_local_hub(self)
        logger.info("Chrome bridge listening on 127.0.0.1:%s", self._port)
        return True

    async def stop(self) -> None:
        server, self._server = self._server, None
        if server is None:
            return
        server.close()
        with contextlib.suppress(Exception):
            await server.wait_closed()
        for future, _link in list(self._pending.values()):
            if not future.done():
                future.set_exception(BridgeUnavailableError("the bridge stopped"))
        self._pending.clear()
        self._links.clear()
        if local_hub() is self:
            set_local_hub(None)
        with contextlib.suppress(OSError):
            current = json.loads(self._state_file.read_text(encoding="utf-8"))
            if current.get("token") == self._token:
                self._state_file.unlink()

    def _write_state(self) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        draft = self._state_file.with_suffix(".tmp")
        draft.write_text(
            json.dumps({"port": self._port, "token": self._token, "pid": os.getpid()}),
            encoding="utf-8",
        )
        os.replace(draft, self._state_file)

    # -- who may connect ------------------------------------------------------------

    def _check(self, connection: Any, request: Any) -> Any:
        path = urlsplit(request.path).path
        origin = request.headers.get("Origin")
        if path == "/extension":
            if origin != EXTENSION_ORIGIN:
                logger.warning("Refused a browser link from origin %r", (origin or "")[:80])
                return connection.respond(HTTPStatus.FORBIDDEN, "Forbidden\n")
            return None
        if path == "/control":
            token = str(request.headers.get(TOKEN_HEADER, ""))
            if origin is not None or not hmac.compare_digest(token, self._token):
                logger.warning("Refused a control link (origin %r)", (origin or "")[:80])
                return connection.respond(HTTPStatus.FORBIDDEN, "Forbidden\n")
            return None
        return connection.respond(HTTPStatus.NOT_FOUND, "Not found\n")

    async def _handle(self, connection: Any) -> None:
        if urlsplit(connection.request.path).path == "/extension":
            await self._serve_extension(connection)
        else:
            await self._serve_control(connection)

    # -- the browser ------------------------------------------------------------------

    async def _serve_extension(self, connection: Any) -> None:
        link: _Link | None = None
        pinger = asyncio.create_task(self._ping(connection), name="chrome_bridge_ping")
        try:
            async for raw in connection:
                message = _parse(raw)
                if message is None:
                    continue
                kind = message.get("type")
                if kind == "hello" and link is None:
                    link = _Link(
                        connection,
                        str(message.get("version", ""))[:20],
                        str(message.get("browser", ""))[:200],
                        time.monotonic(),
                    )
                    self._links.append(link)
                    logger.info("The user's Chrome connected (extension %s)", link.version)
                elif kind != "pong":
                    self._settle(message)
        except _closed_errors():
            pass
        finally:
            pinger.cancel()
            if link is not None:
                self._links.remove(link)
                for key, (future, owner) in list(self._pending.items()):
                    if owner is link:
                        self._pending.pop(key, None)
                        if not future.done():
                            future.set_exception(
                                BridgeUnavailableError("the user's Chrome disconnected")
                            )
                logger.info("The user's Chrome disconnected")

    @staticmethod
    async def _ping(connection: Any) -> None:
        with contextlib.suppress(*_closed_errors()):
            while True:
                await asyncio.sleep(PING_SECONDS)
                await connection.send('{"type":"ping"}')

    def _settle(self, message: dict[str, Any]) -> None:
        entry = self._pending.pop(message.get("id"), None)  # type: ignore[arg-type]
        if entry is None:
            return
        future, _link = entry
        if future.done():
            return
        if message.get("ok"):
            future.set_result(message.get("result"))
        else:
            future.set_exception(BridgeError(str(message.get("error") or "failed")[:300]))

    def status(self) -> dict[str, Any]:
        link = self._links[-1] if self._links else None
        return {
            "connected": link is not None,
            "version": link.version if link else "",
            "browsers": len(self._links),
        }

    async def ready(self) -> bool:
        return bool(self._links)

    async def call(
        self, cmd: str, args: dict[str, Any] | None = None, timeout: float = CALL_TIMEOUT_SECONDS
    ) -> Any:
        """Ask the user's Chrome (the one connected last) to do `cmd`."""
        if cmd == "status":
            return self.status()
        link = self._links[-1] if self._links else None
        if link is None:
            raise BridgeUnavailableError("the user's Chrome is not connected to Kwami")
        request_id = next(self._ids)
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = (future, link)
        try:
            await link.connection.send(
                json.dumps({"id": request_id, "cmd": cmd, "args": args or {}})
            )
            async with asyncio.timeout(min(timeout, MAX_CALL_TIMEOUT_SECONDS)):
                return await future
        except TimeoutError as e:
            raise BridgeError(f"the user's Chrome did not answer '{cmd}' in time") from e
        except _closed_errors() as e:
            raise BridgeUnavailableError("the user's Chrome disconnected") from e
        finally:
            self._pending.pop(request_id, None)

    # -- Jarvis's processes -------------------------------------------------------------

    async def _serve_control(self, connection: Any) -> None:
        running: set[asyncio.Task[None]] = set()
        try:
            async for raw in connection:
                message = _parse(raw)
                if message is None or "id" not in message:
                    continue
                task = asyncio.create_task(self._relay(connection, message))
                running.add(task)
                task.add_done_callback(running.discard)
        except _closed_errors():
            pass
        finally:
            for task in running:
                task.cancel()

    async def _relay(self, connection: Any, message: dict[str, Any]) -> None:
        reply: dict[str, Any] = {"id": message["id"]}
        try:
            timeout = float(message.get("timeout") or CALL_TIMEOUT_SECONDS)
            args = message.get("args")
            reply["result"] = await self.call(
                str(message.get("cmd", "")),
                args if isinstance(args, dict) else {},
                max(0.5, min(timeout, MAX_CALL_TIMEOUT_SECONDS)),
            )
            reply["ok"] = True
        except BridgeUnavailableError as e:
            reply.update(ok=False, error=str(e), unavailable=True)
        except Exception as e:
            reply.update(ok=False, error=str(e)[:300])
        with contextlib.suppress(*_closed_errors()):
            await connection.send(json.dumps(reply))


class BridgeClient:
    """A process of Jarvis's reaching the hub another process hosts (the voice, mostly)."""

    def __init__(self, state_file: Path = STATE_FILE) -> None:
        self._state_file = state_file
        self._connection: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._reader: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._ids = itertools.count(1)
        self._connecting: asyncio.Lock | None = None
        self._ready = False
        self._ready_at = -STATUS_TTL_SECONDS

    def _read_state(self) -> tuple[int, str] | None:
        try:
            state = json.loads(self._state_file.read_text(encoding="utf-8"))
            return int(state["port"]), str(state["token"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    async def _connect(self) -> Any:
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            # Another event loop (a new job, a test): what was open belongs to the old one.
            self._connection, self._reader, self._pending = None, None, {}
            self._connecting = asyncio.Lock()
            self._loop = loop
        lock = self._connecting or asyncio.Lock()
        self._connecting = lock
        async with lock:
            if self._connection is not None:
                return self._connection
            state = await asyncio.to_thread(self._read_state)
            if state is None:
                raise BridgeUnavailableError("Kwami's bridge to Chrome is not running here")
            port, token = state
            from websockets.asyncio.client import connect
            from websockets.exceptions import InvalidHandshake

            try:
                connection = await connect(
                    f"ws://127.0.0.1:{port}/control",
                    additional_headers={TOKEN_HEADER: token},
                    max_size=MAX_MESSAGE_BYTES,
                    open_timeout=2,
                    ping_interval=None,
                )
            except (OSError, TimeoutError, InvalidHandshake) as e:
                raise BridgeUnavailableError("Kwami's bridge to Chrome did not answer") from e
            self._connection = connection
            self._reader = loop.create_task(self._read(connection), name="chrome_bridge_client")
            return connection

    async def _read(self, connection: Any) -> None:
        try:
            async for raw in connection:
                message = _parse(raw)
                if message is None:
                    continue
                future = self._pending.pop(message.get("id"), None)  # type: ignore[arg-type]
                if future is None or future.done():
                    continue
                if message.get("ok"):
                    future.set_result(message.get("result"))
                else:
                    error_type = (
                        BridgeUnavailableError if message.get("unavailable") else BridgeError
                    )
                    future.set_exception(error_type(str(message.get("error") or "failed")))
        except _closed_errors():
            pass
        finally:
            if self._connection is connection:
                self._connection = None
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(BridgeUnavailableError("the bridge closed"))
            self._pending.clear()

    async def call(
        self, cmd: str, args: dict[str, Any] | None = None, timeout: float = CALL_TIMEOUT_SECONDS
    ) -> Any:
        connection = await self._connect()
        request_id = next(self._ids)
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await connection.send(
                json.dumps({"id": request_id, "cmd": cmd, "args": args or {}, "timeout": timeout})
            )
            async with asyncio.timeout(timeout + 2):
                return await future
        except TimeoutError as e:
            raise BridgeError(f"the user's Chrome did not answer '{cmd}' in time") from e
        except _closed_errors() as e:
            raise BridgeUnavailableError("Kwami's bridge to Chrome closed") from e
        finally:
            self._pending.pop(request_id, None)

    async def ready(self) -> bool:
        """Whether the user's Chrome is connected now. Asked at most every few seconds."""
        now = time.monotonic()
        if now - self._ready_at < STATUS_TTL_SECONDS:
            return self._ready
        try:
            status = await self.call("status", timeout=STATUS_TIMEOUT_SECONDS)
            ready = isinstance(status, dict) and bool(status.get("connected"))
        except BridgeError:
            ready = False
        self._ready, self._ready_at = ready, time.monotonic()
        return ready

    async def close(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            with contextlib.suppress(Exception):
                await connection.close()


_local_hub: BridgeHub | None = None
_client: BridgeClient | None = None


def local_hub() -> BridgeHub | None:
    return _local_hub


def set_local_hub(hub: BridgeHub | None) -> None:
    global _local_hub
    _local_hub = hub


def chrome_bridge() -> ChromeBridgePort:
    """The hub when this process hosts it, otherwise a client of the one that does."""
    global _client
    if _local_hub is not None:
        return _local_hub
    if _client is None:
        _client = BridgeClient()
    return _client


async def _answers(port: ChromeBridgePort) -> bool:
    try:
        return await port.ready()
    except Exception:
        logger.debug("Could not ask %s", type(port).__name__, exc_info=True)
        return False


async def user_browser() -> ChromeBridgePort | None:
    """The way into the user's own browser that works now, or None.

    Kwami's extension first: it sees every tab and depends on no account. Then Claude in
    Chrome (`claude_chrome`), the way Claude Code reaches the same Chrome, when the Claude
    extension there is signed in to this PC's Claude account.
    """
    if not bridge_enabled():
        return None
    kwami = chrome_bridge()
    if await _answers(kwami):
        return kwami
    from .claude_chrome import claude_chrome

    claude = claude_chrome()
    if claude is not None and await _answers(claude):
        return claude
    return None


async def user_chrome_ready() -> bool:
    """Whether Jarvis can work in the user's own Chrome right now."""
    return await user_browser() is not None


async def start_hub() -> BridgeHub | None:
    """Host the hub in this process, where the user's PC runs Jarvis. None if it cannot."""
    if not bridge_enabled():
        return None
    hub = BridgeHub()
    return hub if await hub.start() else None
