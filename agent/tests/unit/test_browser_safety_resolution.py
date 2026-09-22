"""DNS resolution, truncation, and the store's own plumbing.

These are the last uncovered branches in the two files that carry the security
surface, and they are uncovered precisely because they are the unhappy paths:
a hostname that will not resolve, a name that resolves to somewhere private, a
page too big for the model's context, an HTTP pool that has to be released.

`validate_url`'s literal-IP and scheme rules are covered in
`test_browser_safety.py`; this file is the DNS half, which is the one that
matters for a name an attacker controls. `evil.example.com` resolving to
10.0.0.1 is the shape that gets past a blocklist of literals.
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

from src.browser import safety
from src.browser.context_store import KwamiApiContextStore
from src.browser.safety import (
    MAX_TOOL_OUTPUT_CHARS,
    UnsafeURLError,
    truncate_for_llm,
    validate_url,
)
from src.settings import Settings


def _addrinfo(*addresses: str) -> list[tuple]:
    """Shaped like `socket.getaddrinfo`, which returns 5-tuples."""
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443)) for address in addresses]


# -- DNS resolution ----------------------------------------------------------


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "10.0.0.1", "192.168.1.5", "169.254.169.254", "::1", "0.0.0.0"],
)
def test_a_public_name_resolving_somewhere_private_is_refused(monkeypatch, address: str) -> None:
    """The shape a literal blocklist cannot catch: attacker-controlled DNS."""
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: _addrinfo(address))

    with pytest.raises(UnsafeURLError, match="non-public"):
        validate_url("https://evil.example.com")


def test_a_name_resolving_publicly_is_allowed(monkeypatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: _addrinfo("93.184.216.34"))

    assert validate_url("https://example.com") == "https://example.com"


def test_any_private_answer_refuses_even_among_public_ones(monkeypatch) -> None:
    """DNS rebinding returns several answers; one bad address is enough."""
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: _addrinfo("93.184.216.34", "10.0.0.1")
    )

    with pytest.raises(UnsafeURLError):
        validate_url("https://rebind.example.com")


@pytest.mark.parametrize("error", [socket.gaierror("no such host"), UnicodeError("idna")])
def test_a_name_that_will_not_resolve_is_not_treated_as_private(
    monkeypatch, error: Exception
) -> None:
    """Nothing was reached, so there is nothing to refuse; the request fails later."""

    def raise_it(*args: Any, **kwargs: Any):
        raise error

    monkeypatch.setattr(socket, "getaddrinfo", raise_it)

    assert validate_url("https://nowhere.example.com") == "https://nowhere.example.com"


def test_an_unparseable_resolved_address_is_skipped(monkeypatch) -> None:
    """getaddrinfo can hand back an AF family whose sockaddr is not an IP."""
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: _addrinfo("not-an-ip", "93.184.216.34")
    )

    assert validate_url("https://example.com") == "https://example.com"


def test_dns_can_be_switched_off_for_offline_use(monkeypatch) -> None:
    """Literal IPs are still refused; only the lookup is skipped."""

    def explode(*args: Any, **kwargs: Any):
        raise AssertionError("resolve_dns=False must not hit the resolver")

    monkeypatch.setattr(socket, "getaddrinfo", explode)

    assert validate_url("https://example.com", resolve_dns=False)
    with pytest.raises(UnsafeURLError):
        validate_url("https://127.0.0.1", resolve_dns=False)


def test_resolution_is_skipped_for_a_literal_ip(monkeypatch) -> None:
    def explode(*args: Any, **kwargs: Any):
        raise AssertionError("a literal address needs no lookup")

    monkeypatch.setattr(socket, "getaddrinfo", explode)

    assert validate_url("https://93.184.216.34") == "https://93.184.216.34"


def test_the_async_wrapper_keeps_the_lookup_off_the_event_loop(monkeypatch) -> None:
    """A slow resolver in a voice turn is audible dead air, hence to_thread."""
    import asyncio
    import inspect

    source = inspect.getsource(safety.validate_url_async)
    assert "to_thread" in source
    assert asyncio.iscoroutinefunction(safety.validate_url_async)


# -- Bounding what reaches the model -----------------------------------------


def test_short_output_passes_through_unchanged() -> None:
    assert truncate_for_llm("hello") == "hello"


def test_output_at_the_limit_is_not_truncated() -> None:
    exact = "x" * MAX_TOOL_OUTPUT_CHARS
    assert truncate_for_llm(exact) == exact


def test_oversized_output_is_cut_and_says_how_much_was_lost() -> None:
    result = truncate_for_llm("x" * (MAX_TOOL_OUTPUT_CHARS + 500))

    assert "truncated, 500 more characters" in result
    assert len(result) < MAX_TOOL_OUTPUT_CHARS + 100


def test_none_becomes_an_empty_string() -> None:
    """Callers pass browser helper output straight in; its type is not guaranteed."""
    assert truncate_for_llm(None) == ""


def test_the_limit_can_be_lowered_by_the_caller() -> None:
    assert truncate_for_llm("abcdef", limit=3).startswith("abc")


# -- Context store plumbing --------------------------------------------------


class FakeHttp:
    def __init__(self) -> None:
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


def _settings() -> Settings:
    return Settings(kwami_api_url="https://api.example.test", kwami_api_key="secret")


def test_the_store_builds_its_own_pooled_client_when_not_given_one() -> None:
    """Per-call client construction costs a TLS handshake on every browser open."""
    from src.adapters.http import HttpxClient

    store = KwamiApiContextStore(settings=_settings())

    client = store._client()

    assert isinstance(client, HttpxClient)
    assert store._client() is client, "a second client was built"


async def test_a_store_that_built_its_client_releases_it() -> None:
    store = KwamiApiContextStore(settings=_settings())
    http = FakeHttp()
    store._http = http

    await store.aclose()

    assert http.closed == 1
    assert store._http is None


async def test_closing_twice_is_safe() -> None:
    store = KwamiApiContextStore(settings=_settings())
    store._http = FakeHttp()

    await store.aclose()
    await store.aclose()


async def test_a_borrowed_client_is_not_closed_by_the_store() -> None:
    """The caller owns a client it passed in; closing it would break them."""
    http = FakeHttp()
    store = KwamiApiContextStore(settings=_settings(), http=http)

    await store.aclose()

    assert http.closed == 0


# -- Hostname blocklist ------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/",
        "http://localhost.localdomain/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://metadata/",
        "http://anything.localhost/",
    ],
)
def test_blocked_hostnames_are_refused_before_dns(monkeypatch, url: str) -> None:
    """These must not even be looked up: the answer is irrelevant."""

    def explode(*args: Any, **kwargs: Any):
        raise AssertionError("a blocked hostname must not reach the resolver")

    monkeypatch.setattr(socket, "getaddrinfo", explode)

    with pytest.raises(UnsafeURLError, match="not a public address"):
        validate_url(url)


def test_a_url_with_no_hostname_is_refused() -> None:
    with pytest.raises(UnsafeURLError, match="no hostname"):
        validate_url("https:///path-only")


def test_a_trailing_dot_does_not_evade_the_blocklist() -> None:
    """`localhost.` is the same name; the FQDN root dot is stripped first."""
    with pytest.raises(UnsafeURLError):
        validate_url("http://localhost./")
