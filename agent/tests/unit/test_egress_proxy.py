"""The built-in browser's proxy: what it lets out, and what never gets a connection."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest

from src.browser import egress_proxy
from src.browser.egress_proxy import EgressProxy, _forwarded
from src.browser.safety import UnsafeURLError

Handler = Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]


@pytest.fixture
async def proxy() -> AsyncIterator[tuple[EgressProxy, int]]:
    instance = EgressProxy()
    port = await instance.start()
    yield instance, port
    instance.close()


async def _origin(handler: Handler) -> tuple[asyncio.Server, int]:
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    return server, int(server.sockets[0].getsockname()[1])


def _public(monkeypatch: pytest.MonkeyPatch, *addresses: str) -> None:
    """Let loopback stand in for a public site: the check itself is tested in safety."""
    monkeypatch.setattr(egress_proxy, "public_addresses", lambda _host: list(addresses))


async def _ask(port: int, request: bytes) -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(request)
    await writer.drain()
    answer = await asyncio.wait_for(reader.read(), 5)
    writer.close()
    return answer


async def test_loopback_is_refused_for_real(proxy: Any) -> None:
    """No stand-in here: this is what a page asking for the debugging port gets."""
    _, port = proxy
    reached: list[int] = []

    async def origin(_reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        reached.append(1)  # pragma: no cover - must not be reached
        writer.close()  # pragma: no cover - must not be reached

    server, origin_port = await _origin(origin)
    async with server:
        tunnel = await _ask(port, f"CONNECT 127.0.0.1:{origin_port} HTTP/1.1\r\n\r\n".encode())
        plain = await _ask(
            port, f"GET http://localhost:{origin_port}/json/list HTTP/1.1\r\n\r\n".encode()
        )

    assert tunnel.startswith(b"HTTP/1.1 403")
    assert plain.startswith(b"HTTP/1.1 403")
    assert reached == []


async def test_a_tunnel_carries_both_directions(proxy: Any, monkeypatch) -> None:
    _, port = proxy
    _public(monkeypatch, "127.0.0.1")

    async def origin(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(await reader.readexactly(5) + b" back")
        await writer.drain()
        writer.close()

    server, origin_port = await _origin(origin)
    async with server:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(f"CONNECT site.example:{origin_port} HTTP/1.1\r\n\r\n".encode())
        await writer.drain()
        opened = await reader.readuntil(b"\r\n\r\n")
        writer.write(b"hello")
        await writer.drain()
        answer = await asyncio.wait_for(reader.read(), 5)
        writer.close()

    assert opened.startswith(b"HTTP/1.1 200")
    assert answer == b"hello back"


async def test_plain_http_reaches_the_site_as_a_normal_request(proxy: Any, monkeypatch) -> None:
    _, port = proxy
    _public(monkeypatch, "127.0.0.1")
    seen: list[bytes] = []

    async def origin(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        seen.append(await reader.readuntil(b"\r\n\r\n"))
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
        await writer.drain()
        writer.close()

    server, origin_port = await _origin(origin)
    async with server:
        answer = await _ask(
            port,
            (
                f"GET http://site.example:{origin_port}/a?b=c HTTP/1.1\r\n"
                f"Host: site.example:{origin_port}\r\n"
                "Proxy-Connection: keep-alive\r\n\r\n"
            ).encode(),
        )

    assert answer.endswith(b"ok")
    assert seen[0].startswith(b"GET /a?b=c HTTP/1.1\r\n")
    assert b"Proxy-Connection" not in seen[0]
    assert b"Connection: close" in seen[0]


def test_a_request_for_no_path_asks_for_the_root() -> None:
    assert _forwarded("GET", "http://a.example", "HTTP/1.1", b"") == (
        b"GET / HTTP/1.1\r\nConnection: close\r\n\r\n"
    )


@pytest.mark.parametrize(
    "request_line",
    [b"GET https://a.example/ HTTP/1.1\r\n\r\n", b"GET /relative HTTP/1.1\r\n\r\n"],
)
async def test_what_a_proxy_is_not_asked_is_a_bad_request(proxy: Any, request_line: bytes) -> None:
    _, port = proxy
    assert (await _ask(port, request_line)).startswith(b"HTTP/1.1 400")


async def test_nonsense_just_ends_the_connection(proxy: Any) -> None:
    _, port = proxy
    assert await _ask(port, b"nonsense\r\n\r\n") == b""


async def test_a_name_the_check_refuses_gets_no_connection(proxy: Any, monkeypatch) -> None:
    _, port = proxy

    def refuse(_host: str) -> list[str]:
        raise UnsafeURLError("'rebound.example' resolves to the non-public address 10.0.0.1")

    monkeypatch.setattr(egress_proxy, "public_addresses", refuse)
    answer = await _ask(port, b"CONNECT rebound.example:443 HTTP/1.1\r\n\r\n")
    assert answer.startswith(b"HTTP/1.1 403")


async def test_a_site_that_does_not_answer_is_a_bad_gateway(proxy: Any, monkeypatch) -> None:
    _, port = proxy
    _public(monkeypatch)
    answer = await _ask(port, b"CONNECT nothing.example:443 HTTP/1.1\r\n\r\n")
    assert answer.startswith(b"HTTP/1.1 502")


async def test_the_next_address_is_tried(monkeypatch) -> None:
    tried: list[str] = []

    async def connect(address: str, _port: int) -> tuple[str, str]:
        tried.append(address)
        if address == "bad":
            raise ConnectionRefusedError
        return "reader", "writer"

    monkeypatch.setattr(egress_proxy.asyncio, "open_connection", connect)
    _public(monkeypatch, "bad", "good")

    assert await egress_proxy._open("site.example", 443) == ("reader", "writer")  # type: ignore[comparison-overlap]
    assert tried == ["bad", "good"]


async def test_ipv4_is_tried_first(monkeypatch) -> None:
    """Where there is no IPv6 route, an IPv6 address only times out."""
    tried: list[str] = []

    async def connect(address: str, _port: int) -> tuple[str, str]:
        tried.append(address)
        return "reader", "writer"

    monkeypatch.setattr(egress_proxy.asyncio, "open_connection", connect)
    _public(monkeypatch, "2606:2800:220:1::1", "93.184.216.34")

    await egress_proxy._open("site.example", 443)

    assert tried == ["93.184.216.34"]


async def test_a_connection_reset_while_piping_ends_quietly() -> None:
    class Reset:
        closed = False

        def write(self, _data: bytes) -> None:
            raise ConnectionResetError

        def close(self) -> None:
            self.closed = True

    reader = asyncio.StreamReader()
    reader.feed_data(b"data")
    writer = Reset()

    await egress_proxy._pipe(reader, writer)  # type: ignore[arg-type]

    assert writer.closed


async def test_closing_drops_what_is_open_and_can_be_repeated(monkeypatch) -> None:
    _public(monkeypatch, "127.0.0.1")
    proxy = EgressProxy()
    port = await proxy.start()

    async def origin(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.read()
        writer.close()

    server, origin_port = await _origin(origin)
    async with server:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(f"CONNECT site.example:{origin_port} HTTP/1.1\r\n\r\n".encode())
        await writer.drain()
        await reader.readuntil(b"\r\n\r\n")

        proxy.close()
        proxy.close()

        assert await asyncio.wait_for(reader.read(), 5) == b""
        writer.close()
