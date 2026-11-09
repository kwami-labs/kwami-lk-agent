"""The built-in browser's only way out: a proxy that refuses what is not public.

Chromium runs in our own container, so whatever it can reach, our network can. Checking
the URL it landed on comes too late: the request has happened by then, and a screenshot
of the answer is on its way to the user.

So every connection it makes goes through here, one per browser, and this side does the
DNS lookup and connects to the address it checked. A name cannot answer one thing to the
check and another to the connection, which is what rebinding needs. CDP's Fetch domain
would have to be enabled target by target and still not see WebSockets or service
workers; a proxy sees all of them, because they all need a socket.
"""

from __future__ import annotations

import asyncio
from asyncio.staggered import staggered_race
from functools import partial
from typing import cast
from urllib.parse import urlsplit

from ..utils.logging import get_logger
from .safety import UnsafeURLError, public_addresses

logger = get_logger("browser.egress")

_HEAD_SECONDS = 15.0
_CONNECT_SECONDS = 10.0
_NEXT_ADDRESS_SECONDS = 0.25
_MAX_HEAD_BYTES = 64 * 1024
_CHUNK_BYTES = 64 * 1024

_REFUSED = b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
_UNREACHABLE = b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
_BAD_REQUEST = b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
_TUNNEL_OPEN = b"HTTP/1.1 200 Connection established\r\n\r\n"

#: About this hop, not the next. `Connection` is replaced: see `_forwarded`.
_HOP_HEADERS = frozenset({"connection", "keep-alive", "proxy-authorization", "proxy-connection"})


def _forwarded(method: str, target: str, version: str, headers: bytes) -> bytes:
    """A proxy's request as the site expects it.

    One request per connection: the browser may reuse a proxy connection for another
    host, and this one is already open to the host that was checked.
    """
    url = urlsplit(target)
    path = url.path or "/"
    if url.query:
        path = f"{path}?{url.query}"
    kept = [
        line
        for line in headers.split(b"\r\n")
        if line and line.split(b":", 1)[0].strip().lower().decode("latin-1") not in _HOP_HEADERS
    ]
    head = f"{method} {path} {version}".encode("latin-1")
    return b"\r\n".join([head, *kept, b"Connection: close", b"", b""])


async def _open(host: str, port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """A connection to this host, on an address that was checked. Raises when there is none."""
    addresses = await asyncio.to_thread(public_addresses, host)
    # A moment apart and the first to answer wins, as a browser would on its own: where
    # a container has no IPv6 route, an IPv6 address only times out. IPv4 goes first.
    attempts = [
        partial(asyncio.open_connection, address, port)
        for address in sorted(addresses, key=lambda address: ":" in address)
    ]
    connection, _, failures = await asyncio.wait_for(
        staggered_race(attempts, _NEXT_ADDRESS_SECONDS), _CONNECT_SECONDS
    )
    if connection is None:
        raise next((e for e in failures if isinstance(e, OSError)), ConnectionError(host))
    return connection


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(_CHUNK_BYTES):
            writer.write(data)
            await writer.drain()
    except OSError:
        pass
    finally:
        # The other direction reads an end of file from this and stops too.
        writer.close()


class EgressProxy:
    """One browser's forward proxy, on a port only this container can reach."""

    def __init__(self) -> None:
        self._server: asyncio.Server | None = None
        self._connections: set[asyncio.Task] = set()

    async def start(self) -> int:
        """Start listening, and return the port."""
        self._server = await asyncio.start_server(
            self._serve, "127.0.0.1", 0, limit=_MAX_HEAD_BYTES
        )
        return int(self._server.sockets[0].getsockname()[1])

    def close(self) -> None:
        """Stop listening and drop what is open. Safe to call more than once."""
        server, self._server = self._server, None
        if server is not None:
            server.close()
        for connection in list(self._connections):
            connection.cancel()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # A handler always runs in a task of its own.
        task = cast(asyncio.Task, asyncio.current_task())
        self._connections.add(task)
        try:
            await self._handle(reader, writer)
        except (OSError, EOFError, ValueError) as e:
            # A browser hanging up, or a request that never made sense.
            logger.debug("Built-in browser connection ended: %s", type(e).__name__)
        except asyncio.CancelledError:
            pass
        finally:
            self._connections.discard(task)
            writer.close()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), _HEAD_SECONDS)
        line, _, headers = head.partition(b"\r\n")
        method, target, version = line.decode("latin-1").split(" ")

        tunnel = method.upper() == "CONNECT"
        url = urlsplit(f"//{target}" if tunnel else target)
        if not url.hostname or not (tunnel or url.scheme == "http"):
            writer.write(_BAD_REQUEST)
            await writer.drain()
            return

        try:
            upstream_reader, upstream_writer = await _open(
                url.hostname, url.port or (443 if tunnel else 80)
            )
        except UnsafeURLError as e:
            logger.info("Built-in browser refused: %s", e)
            writer.write(_REFUSED)
            await writer.drain()
            return
        except OSError as e:
            logger.debug("Built-in browser could not reach a host: %s", type(e).__name__)
            writer.write(_UNREACHABLE)
            await writer.drain()
            return

        if tunnel:
            writer.write(_TUNNEL_OPEN)
            await writer.drain()
        else:
            upstream_writer.write(_forwarded(method, target, version, headers))
            await upstream_writer.drain()

        try:
            await asyncio.gather(_pipe(reader, upstream_writer), _pipe(upstream_reader, writer))
        finally:
            upstream_writer.close()
