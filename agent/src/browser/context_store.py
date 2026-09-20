"""Remembering which persisted browser profile belongs to which user.

Browser Use Cloud does not need this: its profiles carry a name, so the user id
is enough to find last week's profile again. Browserbase Contexts have no
lookup-by-name endpoint at all -- `POST /v1/contexts` hands back an opaque id
and that is the only way to ever reach that context again. Names are unique
per project, so creating "kwami-<user>" a second time is rejected rather than
returning the original id.

So the id has to be kept somewhere that outlives the agent process, or every
session silently starts a fresh context: the user is signed out of everything,
and the old context is orphaned and keeps being billed for storage. That store
is the Kwami API, reached with the same internal key as the telephony runtime
config.

When there is no API key -- local development, tests -- the store degrades to
process memory rather than failing. Within one worker that still reuses a
context across several browser opens, which is the case a developer actually
hits; it just cannot survive a restart.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..settings import Settings, get_settings
from ..utils.logging import get_logger

logger = get_logger("browser.context_store")

#: Short, so a slow store never becomes audible dead air before browsing starts.
STORE_TIMEOUT_SECONDS = 5.0


@runtime_checkable
class ContextStorePort(Protocol):
    """Maps a user to the vendor-side handle for their persisted browser state."""

    async def get(self, user_id: str, vendor: str) -> str | None: ...

    async def put(self, user_id: str, vendor: str, context_id: str) -> None: ...


class InMemoryContextStore:
    """Process-local fallback. Survives browser opens, not restarts."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], str] = {}

    async def get(self, user_id: str, vendor: str) -> str | None:
        return self._entries.get((user_id, vendor))

    async def put(self, user_id: str, vendor: str, context_id: str) -> None:
        self._entries[(user_id, vendor)] = context_id


class KwamiApiContextStore:
    """Durable store backed by the Kwami API's internal routes.

    Every failure path returns None (or does nothing) rather than raising: not
    finding a saved context costs the user their logins, but failing the whole
    browser open costs them the feature.
    """

    def __init__(self, *, settings: Settings | None = None, http: Any = None) -> None:
        self._settings = settings or get_settings()
        self._http = http
        self._owns_http = http is None

    @property
    def _enabled(self) -> bool:
        return bool(self._settings.kwami_api_key and self._settings.kwami_api_url)

    def _client(self) -> Any:
        if self._http is None:
            from ..adapters.http import HttpxClient

            self._http = HttpxClient(timeout=STORE_TIMEOUT_SECONDS)
        return self._http

    def _url(self, user_id: str) -> str:
        from urllib.parse import quote

        base = self._settings.kwami_api_url.rstrip("/")
        return f"{base}/internal/browser-contexts/{quote(user_id, safe='')}"

    def _headers(self) -> dict[str, str]:
        return {"X-Kwami-API-Key": self._settings.kwami_api_key}

    async def get(self, user_id: str, vendor: str) -> str | None:
        if not (self._enabled and user_id):
            return None
        try:
            response = await self._client().get(
                self._url(user_id),
                params={"vendor": vendor},
                headers=self._headers(),
                timeout=STORE_TIMEOUT_SECONDS,
            )
        except Exception as e:
            logger.warning("Could not read the saved browser context (%s)", e)
            return None

        if response.status_code == 404:
            # Ambiguous by design on this route: "this user has no saved context
            # yet" and "this endpoint does not exist" look identical. The write
            # path below can tell the difference, and says so loudly.
            return None
        if response.status_code >= 400:
            logger.warning(
                "Kwami API returned %s reading the saved browser context",
                response.status_code,
            )
            return None

        try:
            payload = response.json()
        except Exception:
            return None
        context_id = (payload or {}).get("context_id") if isinstance(payload, dict) else None
        return context_id or None

    async def put(self, user_id: str, vendor: str, context_id: str) -> None:
        if not (self._enabled and user_id and context_id):
            return
        try:
            response = await self._client().post(
                self._url(user_id),
                json={"vendor": vendor, "context_id": context_id},
                headers=self._headers(),
                timeout=STORE_TIMEOUT_SECONDS,
            )
        except Exception as e:
            logger.warning("Could not save the browser context (%s)", e)
            return
        if response.status_code == 404:
            # A write cannot legitimately 404: the route either exists or it
            # does not. This is the one signal that distinguishes "no context
            # saved yet" from "context persistence is not implemented on the
            # API at all", and without it the whole feature fails invisibly --
            # `get` returns None forever, every session mints a fresh
            # Browserbase Context, the user is signed out of every site each
            # time, and the abandoned contexts keep being billed for storage.
            logger.error(
                "Kwami API has no %s endpoint (404 on write). Browser context "
                "persistence is NOT working: every session will start a fresh "
                "context, users will be signed out of every site, and the "
                "orphaned contexts will continue to be billed.",
                self._url("<user_id>"),
            )
            return
        if response.status_code >= 400:
            logger.warning(
                "Kwami API returned %s saving the browser context; the user's "
                "logins will not survive this session",
                response.status_code,
            )

    async def aclose(self) -> None:
        """Release the HTTP pool, if this store built one."""
        if self._owns_http and self._http is not None and hasattr(self._http, "aclose"):
            await self._http.aclose()
            self._http = None


def create_context_store(settings: Settings | None = None) -> ContextStorePort:
    """The durable store when the API is reachable, process memory otherwise."""
    settings = settings or get_settings()
    if settings.kwami_api_key and settings.kwami_api_url:
        return KwamiApiContextStore(settings=settings)
    logger.info(
        "KWAMI_API_KEY is not set; browser contexts are kept in memory only and "
        "will not survive a restart."
    )
    return InMemoryContextStore()
