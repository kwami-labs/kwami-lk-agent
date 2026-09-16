"""A pooled HTTP client shared by every adapter.

Replaces per-call `httpx.AsyncClient()` construction, which happened at ten
separate call sites and paid a fresh TCP and TLS handshake on every tool
invocation -- roughly 100-300ms of avoidable latency each, on a voice path.
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
