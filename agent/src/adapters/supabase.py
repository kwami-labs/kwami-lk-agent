"""The user's Supabase, reached directly: PostgREST and Storage over httpx.

On LiveKit Cloud the agent has no Kwami API to go through -- the user runs the
app and the API on their own PC and wants neither a paid server nor a tunnel --
so the few things the agent must keep (brain spend and results, session usage,
browser contexts, the executor's files) are written straight to the user's
free Supabase project with the secret key.

This is deliberately a dozen methods rather than `supabase-py`: the agent needs
select, insert, update, one RPC and one upload, and a second HTTP stack with its
own pool and retry policy is not worth that.

The rule every method follows: **never raise, never wait long**. Each call has
a short timeout and returns None/False on any failure after logging it. Nothing
written here is worth a second of silence on the call, and losing one usage row
is better than losing the session that produced it.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote
from uuid import UUID

import httpx

from ..settings import Settings, get_settings
from ..utils.logging import get_logger
from .http import shared_client

logger = get_logger("supabase")

#: Short on purpose: these calls sit next to a live conversation.
DEFAULT_TIMEOUT_SECONDS = 4.0
#: Uploads carry a file, not a row.
UPLOAD_TIMEOUT_SECONDS = 20.0


def as_user_id(value: Any) -> str | None:
    """`value` as a canonical Supabase user id, or None when it is not one.

    The LiveKit participant identity is the user's id for app sessions; for
    telephony it is a SIP identity, and a uuid column must not receive that.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        return str(UUID(value))
    except ValueError:
        return None


class SupabaseClient:
    """Minimal PostgREST + Storage client authenticated with the secret key."""

    def __init__(
        self,
        url: str,
        key: str,
        *,
        http: httpx.AsyncClient | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._url = url.rstrip("/")
        self._key = key
        self._http = http
        self._timeout = timeout

    @property
    def url(self) -> str:
        return self._url

    def _client(self) -> httpx.AsyncClient:
        # The shared pool, fetched per call: session cleanup closes it, and
        # `shared_client()` hands back a fresh one after that.
        return self._http or shared_client()

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        # Both headers carry the same key. That is what the gateway accepts for
        # the new `sb_secret_...` keys as well as for a legacy service_role JWT.
        headers = {"apikey": self._key, "Authorization": f"Bearer {self._key}"}
        if extra:
            headers.update(extra)
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        what: str,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> httpx.Response | None:
        try:
            response = await self._client().request(
                method, f"{self._url}{path}", timeout=timeout or self._timeout, **kwargs
            )
        except Exception as e:
            logger.warning("Supabase unreachable for %s (%s: %s)", what, type(e).__name__, e)
            return None
        if response.status_code >= 400:
            logger.warning(
                "Supabase returned %s for %s: %s",
                response.status_code,
                what,
                response.text[:300],
            )
            return None
        return response

    # -- PostgREST -------------------------------------------------------------

    async def select(
        self, table: str, params: dict[str, str], *, what: str = ""
    ) -> list[dict[str, Any]] | None:
        """Rows matching PostgREST `params` (``select``, ``col=eq.x``...), or None."""
        response = await self._request(
            "GET",
            f"/rest/v1/{table}",
            what=what or f"reading {table}",
            params=params,
            headers=self._headers({"Accept": "application/json"}),
        )
        if response is None:
            return None
        try:
            data = response.json()
        except ValueError:
            return None
        return data if isinstance(data, list) else None

    async def insert(
        self,
        table: str,
        rows: dict[str, Any] | list[dict[str, Any]],
        *,
        on_conflict: str = "",
        what: str = "",
    ) -> bool:
        """Insert, or upsert on `on_conflict` columns. True when it was stored."""
        prefer = ["return=minimal"]
        params: dict[str, str] = {}
        if on_conflict:
            prefer.append("resolution=merge-duplicates")
            params["on_conflict"] = on_conflict
        response = await self._request(
            "POST",
            f"/rest/v1/{table}",
            what=what or f"writing {table}",
            params=params,
            json=rows,
            headers=self._headers({"Prefer": ",".join(prefer)}),
        )
        return response is not None

    async def update(
        self, table: str, values: dict[str, Any], params: dict[str, str], *, what: str = ""
    ) -> bool:
        """PATCH the rows matching `params`. Refuses an unfiltered update."""
        if not params:
            raise ValueError("an update needs a filter")
        response = await self._request(
            "PATCH",
            f"/rest/v1/{table}",
            what=what or f"updating {table}",
            params=params,
            json=values,
            headers=self._headers({"Prefer": "return=minimal"}),
        )
        return response is not None

    async def rpc(self, function: str, args: dict[str, Any], *, what: str = "") -> Any:
        """Call a Postgres function. Its JSON result, or None on failure."""
        response = await self._request(
            "POST",
            f"/rest/v1/rpc/{function}",
            what=what or f"calling {function}",
            json=args,
            headers=self._headers(),
        )
        if response is None:
            return None
        try:
            return response.json()
        except ValueError:
            return True

    # -- Storage ---------------------------------------------------------------

    async def upload(
        self,
        bucket: str,
        path: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> bool:
        """Store `data` at `bucket/path`, replacing what is there."""
        object_path = "/".join(quote(part, safe="") for part in path.split("/") if part)
        response = await self._request(
            "POST",
            f"/storage/v1/object/{quote(bucket, safe='')}/{object_path}",
            what=f"uploading {bucket}/{path}",
            timeout=UPLOAD_TIMEOUT_SECONDS,
            content=data,
            headers=self._headers({"Content-Type": content_type, "x-upsert": "true"}),
        )
        return response is not None


_clients: dict[tuple[str, str], SupabaseClient] = {}


def get_supabase(settings: Settings | None = None) -> SupabaseClient | None:
    """The client for the configured project, or None when there is none."""
    settings = settings or get_settings()
    if not settings.supabase_enabled:
        return None
    key = (settings.supabase_url, settings.supabase_secret_key)
    client = _clients.get(key)
    if client is None:
        client = _clients[key] = SupabaseClient(*key)
    return client


def direct_supabase(settings: Settings | None = None) -> SupabaseClient | None:
    """The client when the agent should write Supabase itself (no Kwami API)."""
    settings = settings or get_settings()
    return get_supabase(settings) if settings.supabase_direct else None


async def kwami_owner(client: SupabaseClient, kwami_id: str) -> str | None:
    """The user a kwami belongs to, from `user_kwamis`. None when unknown."""
    if as_user_id(kwami_id) is None:
        return None
    rows = await client.select(
        "user_kwamis",
        {"select": "user_id", "id": f"eq.{kwami_id}", "limit": "1"},
        what="the kwami's owner",
    )
    if not rows:
        return None
    return as_user_id(rows[0].get("user_id"))
