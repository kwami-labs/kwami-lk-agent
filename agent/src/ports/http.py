"""The HTTP boundary.

Narrow on purpose: adapters need `get`/`post` returning status, text and JSON,
and nothing else. Keeping it this small is what lets a test supply a fake
without reaching for a transport-level mocking library.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class HttpResponsePort(Protocol):
    @property
    def status_code(self) -> int: ...

    @property
    def text(self) -> str: ...

    def json(self) -> Any: ...


@runtime_checkable
class HttpClientPort(Protocol):
    """A shared, pooled client. Adapters receive one; they never construct one.

    Per-call client construction was costing a fresh TCP and TLS handshake on
    every tool invocation, at ten separate call sites.
    """

    async def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> HttpResponsePort: ...

    async def post(
        self,
        url: str,
        *,
        json: Any = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> HttpResponsePort: ...
