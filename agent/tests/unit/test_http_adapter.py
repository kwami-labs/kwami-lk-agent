"""`HttpxClient` is the pooled client the runtime container owns.

It exists so adapters stop paying a fresh TCP+TLS handshake per tool call, so
the properties worth pinning are ownership (who closes the pool) and that the
per-call timeout override actually reaches httpx.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from src.adapters.http import DEFAULT_TIMEOUT_SECONDS, HttpxClient, HttpxResponse


# =============================================================================
# HttpxResponse
# =============================================================================


def test_the_response_wrapper_exposes_status_text_and_json() -> None:
    wrapped = HttpxResponse(httpx.Response(201, json={"ok": True}))

    assert wrapped.status_code == 201
    assert wrapped.json() == {"ok": True}
    assert '"ok"' in wrapped.text


def test_the_wrapper_does_not_swallow_a_json_decode_error() -> None:
    """Callers branch on status before calling json(); hiding the error here
    would turn a malformed body into a silent empty result."""
    wrapped = HttpxResponse(httpx.Response(200, text="not json"))

    with pytest.raises(Exception):
        wrapped.json()


# =============================================================================
# Ownership
# =============================================================================


async def test_a_self_constructed_client_is_closed_by_aclose() -> None:
    client = HttpxClient()

    assert client._owns_client is True
    await client.aclose()

    assert client._client.is_closed


async def test_aclose_is_idempotent() -> None:
    """Cleanup runs on more than one path; a second close must not raise."""
    client = HttpxClient()

    await client.aclose()
    await client.aclose()

    assert client._client.is_closed


async def test_an_injected_client_is_not_closed() -> None:
    """The container may hand in a client it owns; closing someone else's pool
    would break every other adapter sharing it."""
    injected = httpx.AsyncClient()
    client = HttpxClient(client=injected)

    assert client._owns_client is False
    await client.aclose()

    assert not injected.is_closed
    await injected.aclose()


def test_the_default_timeout_is_applied_when_none_is_given() -> None:
    client = HttpxClient()

    assert client._timeout == DEFAULT_TIMEOUT_SECONDS


# =============================================================================
# Requests
# =============================================================================


@respx.mock
async def test_get_returns_a_wrapped_response() -> None:
    respx.get("https://example.test/thing").mock(
        return_value=httpx.Response(200, json={"a": 1})
    )
    client = HttpxClient()

    response = await client.get("https://example.test/thing")

    assert isinstance(response, HttpxResponse)
    assert response.status_code == 200
    assert response.json() == {"a": 1}
    await client.aclose()


@respx.mock
async def test_get_forwards_params_and_headers() -> None:
    route = respx.get("https://example.test/thing").mock(
        return_value=httpx.Response(200, json={})
    )
    client = HttpxClient()

    await client.get(
        "https://example.test/thing",
        params={"q": "search term"},
        headers={"X-Token": "abc"},
    )

    request = route.calls[0].request
    assert request.url.params["q"] == "search term"
    assert request.headers["X-Token"] == "abc"
    await client.aclose()


@respx.mock
async def test_post_sends_json_and_headers() -> None:
    route = respx.post("https://example.test/thing").mock(
        return_value=httpx.Response(202, json={"queued": True})
    )
    client = HttpxClient()

    response = await client.post(
        "https://example.test/thing",
        json={"payload": 1},
        headers={"X-Token": "abc"},
    )

    assert response.status_code == 202
    request = route.calls[0].request
    assert request.headers["X-Token"] == "abc"
    assert b'"payload"' in request.content
    await client.aclose()


@respx.mock
async def test_a_non_2xx_is_returned_rather_than_raised() -> None:
    """No raise_for_status here: callers decide what a 404 means."""
    respx.get("https://example.test/missing").mock(
        return_value=httpx.Response(404, text="nope")
    )
    client = HttpxClient()

    response = await client.get("https://example.test/missing")

    assert response.status_code == 404
    await client.aclose()


class TimeoutRecordingTransport(httpx.AsyncBaseTransport):
    """Captures the timeout httpx actually resolved for each request."""

    def __init__(self) -> None:
        self.timeouts: list[float | None] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        timeout = request.extensions.get("timeout", {})
        self.timeouts.append(timeout.get("connect"))
        return httpx.Response(200, json={})


async def test_the_instance_timeout_is_used_by_default() -> None:
    transport = TimeoutRecordingTransport()
    client = HttpxClient(timeout=9.0, client=httpx.AsyncClient(transport=transport))

    await client.get("https://example.test/thing")

    assert transport.timeouts == [9.0]


async def test_a_per_call_timeout_overrides_the_instance_one() -> None:
    transport = TimeoutRecordingTransport()
    client = HttpxClient(timeout=9.0, client=httpx.AsyncClient(transport=transport))

    await client.get("https://example.test/thing", timeout=2.0)
    await client.post("https://example.test/thing", timeout=3.0)

    assert transport.timeouts == [2.0, 3.0]


async def test_a_zero_per_call_timeout_falls_back_to_the_instance_one() -> None:
    """`timeout or self._timeout` is falsy-checked, so 0 means "unset" here --
    pinned so the behaviour is deliberate rather than discovered."""
    transport = TimeoutRecordingTransport()
    client = HttpxClient(timeout=9.0, client=httpx.AsyncClient(transport=transport))

    await client.get("https://example.test/thing", timeout=0)

    assert transport.timeouts == [9.0]
