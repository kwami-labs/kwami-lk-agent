"""A pooled HTTP client shared by every adapter and tool.

Constructing `httpx.AsyncClient()` per call pays a fresh TCP and TLS handshake
every time -- roughly 100-300ms, on a path where the user is listening to
silence. This module existed to fix that and its docstring claimed it had, but
eighteen call sites across `browser/`, `tools/` and `runtime_bootstrap` still
built their own; the claim was aspirational.

`shared_client()` is the process-wide pool those call sites use now. It is a
module-level singleton rather than something threaded through every signature
because the alternative -- passing a client down into every tool body -- is the
refactor that stalled the first time.

Pool limits are set explicitly. httpx defaults to 100 connections and 20
keepalives, which is sized for a script rather than a worker serving many
concurrent sessions against a handful of hosts: the agent talks to maybe eight
origins (OpenAI, Deepgram, Cartesia, Tavily, SerpAPI, Zep, Browserbase, the
Kwami API), so a large *per-host* keepalive pool is what actually helps, and an
unbounded total is what eventually exhausts file descriptors.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..utils.logging import get_logger

logger = get_logger("http")

DEFAULT_TIMEOUT_SECONDS = 15.0


class HttpxResponse:
    """Adapts httpx.Response to HttpResponsePort."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    @property
    def status_code(self) -> int:
        return self._response.status_code

    @property
    def text(self) -> str:
        return self._response.text

    def json(self) -> Any:
        return self._response.json()


class HttpxClient:
    """Shared, pooled HTTP client.

    Owned by the runtime container and closed with the session. Adapters are
    handed one; they never construct their own.
    """

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._timeout = timeout
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> HttpxResponse:
        response = await self._client.get(
            url, params=params, headers=headers, timeout=timeout or self._timeout
        )
        return HttpxResponse(response)

    async def post(
        self,
        url: str,
        *,
        json: Any = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> HttpxResponse:
        response = await self._client.post(
            url, json=json, headers=headers, timeout=timeout or self._timeout
        )
        return HttpxResponse(response)

    async def aclose(self) -> None:
        """Release the pool. Safe to call more than once."""
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()


#: Sized for a worker talking to a few origins under concurrency, not for a
#: script making one request. `max_keepalive_connections` is what turns the
#: second call to a provider into a resumed connection instead of a new TLS
#: handshake.
POOL_LIMITS = httpx.Limits(
    max_connections=200,
    max_keepalive_connections=50,
    keepalive_expiry=30.0,
)

_shared: httpx.AsyncClient | None = None


def shared_client() -> httpx.AsyncClient:
    """The process-wide pooled client.

    Created on first use. Callers pass a per-request `timeout=` rather than
    getting their own client, because the timeouts here are genuinely different
    per call site -- 10s to check a browser's status, 20s for a research fan-out
    -- and that difference is not a reason to pay another handshake.

    Never used with `async with`: closing it would close the pool out from under
    every other caller. `aclose_shared()` is the only thing that closes it.
    """
    global _shared
    if _shared is None or _shared.is_closed:
        _shared = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS, limits=POOL_LIMITS)
    return _shared


async def aclose_shared() -> None:
    """Release the shared pool. Safe to call more than once."""
    global _shared
    if _shared is not None and not _shared.is_closed:
        await _shared.aclose()
    _shared = None
