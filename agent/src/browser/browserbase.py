"""Browserbase: sessions, and the Contexts that carry a user's logins.

Browserbase is the default cloud-browser vendor because of Contexts. A Context
is an encrypted user-data directory the platform keeps between runs, so a
session started with `browserSettings.context = {id, persist: true}` comes up
already signed in to whatever the user signed in to last time. That is the
whole point of the navigation panel: "open my mail" has to mean *their* mail.

Three details of the REST API shape this module, and each one is a bug if you
assume otherwise:

* **`connectUrl` is a browser-level endpoint**, the one Puppeteer's
  `connectOverCDP` takes -- not a page. `Page.enable` on it fails. `CDPConnection`
  is told to attach to a page target (see `connect_ws`).
* **The live view is a separate call.** `POST /v1/sessions` does not return a
  watchable URL; `GET /v1/sessions/{id}/debug` does, as
  `debuggerFullscreenUrl`.
* **Contexts cannot be looked up by name.** Creating one returns an id, and
  that id is the only handle. Names are unique per project, so re-creating
  "kwami-<user>" is rejected rather than idempotent -- which is why the id is
  persisted through `context_store` instead of being re-derived.

Sessions are billed for as long as they run, so `release` uses
`REQUEST_RELEASE` rather than waiting for the inactivity timeout.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..adapters.http import shared_client
from ..settings import get_settings
from ..utils.logging import get_logger
from .context_store import ContextStorePort, create_context_store
from .providers import BROWSERBASE, LaunchedBrowser

logger = get_logger("browser.browserbase")

BB_API_BASE = "https://api.browserbase.com/v1"

#: Session wall-clock cap, in seconds. The session's own idle timer closes it
#: much sooner; this is the backstop that bounds the bill if the agent dies
#: without running its cleanup.
SESSION_TIMEOUT_SECONDS = 15 * 60

#: Matches the CDP viewport override in `CloudBrowserSession.start`, so the
#: coordinates `page_info` reports are the ones clicks land on.
VIEWPORT = {"width": 1280, "height": 1400}

CONTEXT_NAME_PREFIX = "kwami"
#: The API caps context names at 128 characters.
MAX_CONTEXT_NAME = 128


def context_name_for(user_id: str) -> str:
    """The context name for a user, within the vendor's length limit."""
    return f"{CONTEXT_NAME_PREFIX}-{user_id}"[:MAX_CONTEXT_NAME]


class BrowserbaseClient:
    """Async HTTP client for the Browserbase REST API v1."""

    def __init__(self, api_key: str | None = None, project_id: str | None = None) -> None:
        settings = get_settings()
        self._api_key = api_key if api_key is not None else settings.browserbase_api_key
        self._project_id = project_id if project_id is not None else settings.browserbase_project_id
        if not self._api_key:
            raise ValueError(
                "BROWSERBASE_API_KEY is not set. Get one at https://www.browserbase.com/settings"
            )

    def _headers(self) -> dict[str, str]:
        return {"X-BB-API-Key": self._api_key, "Content-Type": "application/json"}

    def _with_project(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Add projectId when we have one.

        It is optional -- the platform infers the project from the API key --
        but sending it makes a key scoped to several projects unambiguous.
        """
        if self._project_id:
            payload["projectId"] = self._project_id
        return payload

    # -- Contexts ------------------------------------------------------------

    async def create_context(self, name: str) -> str:
        """Create a persistent context and return its id."""
        client = shared_client()
        r = await client.post(
            f"{BB_API_BASE}/contexts",
            json=self._with_project({"name": name}),
            headers=self._headers(),
            timeout=15.0,
        )
        r.raise_for_status()
        data = r.json()
        context_id = (data or {}).get("id") if isinstance(data, dict) else None
        if not context_id:
            raise RuntimeError("Browserbase did not return a context id")
        logger.info("Created Browserbase context %s (%s)", context_id[:8], name)
        return context_id

    async def context_exists(self, context_id: str) -> bool:
        """Whether a stored context id is still live.

        A context can be deleted from the dashboard, or belong to a different
        project after a key rotation. Starting a session against a dead id
        fails the whole browser open, so a stale entry is checked and discarded
        rather than trusted.
        """
        if not context_id:
            return False
        try:
            client = shared_client()
            r = await client.get(
                f"{BB_API_BASE}/contexts/{context_id}",
                headers=self._headers(),
                timeout=10.0,
            )
        except httpx.RequestError as e:
            # Unreachable is not the same as gone: assume it is still there and
            # let the session-create call be the thing that fails loudly.
            logger.warning("Could not verify Browserbase context (%s); assuming it exists", e)
            return True
        return r.status_code < 400

    # -- Sessions ------------------------------------------------------------

    async def create_session(
        self,
        context_id: str | None = None,
        *,
        persist: bool = True,
        timeout_seconds: int = SESSION_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Start a browser session, optionally on a persistent context.

        Returns the raw session dict, which carries `id` and `connectUrl`.
        """
        browser_settings: dict[str, Any] = {
            "viewport": dict(VIEWPORT),
            # The panel is a live view of a real page; ads are noise the user
            # did not ask to watch, and blocking them cuts page weight too.
            "blockAds": True,
            "solveCaptchas": True,
        }
        if context_id:
            browser_settings["context"] = {"id": context_id, "persist": persist}

        payload = self._with_project(
            {"browserSettings": browser_settings, "timeout": timeout_seconds}
        )

        client = shared_client()
        r = await client.post(
            f"{BB_API_BASE}/sessions",
            json=payload,
            headers=self._headers(),
            timeout=30.0,
        )
        r.raise_for_status()
        data = r.json()

        if not isinstance(data, dict) or not data.get("id"):
            raise RuntimeError("Browserbase did not return a session id")
        logger.info(
            "Created Browserbase session %s (context=%s)",
            str(data["id"])[:8],
            (context_id or "none")[:8],
        )
        return data

    async def live_view_url(self, session_id: str) -> str:
        """The embeddable live view for a session, or "" if unavailable.

        Never raises: the browser is already running and usable by the agent at
        this point, so failing to get a URL for the user to watch must not take
        the session down with it.
        """
        try:
            client = shared_client()
            r = await client.get(
                f"{BB_API_BASE}/sessions/{session_id}/debug",
                headers=self._headers(),
                timeout=15.0,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning("Could not fetch the Browserbase live view URL: %s", e)
            return ""
        if not isinstance(data, dict):
            return ""
        return data.get("debuggerFullscreenUrl") or data.get("debuggerUrl") or ""

    async def release_session(self, session_id: str) -> None:
        """Ask the platform to end the session now, rather than at its timeout."""
        client = shared_client()
        r = await client.post(
            f"{BB_API_BASE}/sessions/{session_id}",
            json={"status": "REQUEST_RELEASE"},
            headers=self._headers(),
            timeout=15.0,
        )
        r.raise_for_status()
        logger.info("Released Browserbase session %s", session_id[:8])


class BrowserbaseProvider:
    """Rents Browserbase sessions bound to a user's persistent context."""

    def __init__(
        self,
        api_key: str | None = None,
        project_id: str | None = None,
        *,
        client: BrowserbaseClient | None = None,
        context_store: ContextStorePort | None = None,
    ) -> None:
        self._client = client or BrowserbaseClient(api_key=api_key, project_id=project_id)
        self._store = context_store or create_context_store()

    @property
    def vendor(self) -> str:
        return BROWSERBASE

    async def _resolve_context(self, user_id: str) -> str | None:
        """The user's context id, creating and recording one on first use.

        Returns None when a context could not be obtained. That degrades to an
        ephemeral session -- the user browses, but is not signed in and nothing
        is saved -- which is a better outcome than refusing to browse at all.
        """
        stored = await self._store.get(user_id, self.vendor)
        if stored and await self._client.context_exists(stored):
            logger.info("Reusing Browserbase context %s for this user", stored[:8])
            return stored
        if stored:
            logger.warning(
                "Saved Browserbase context %s is gone; creating a replacement. "
                "The user will need to sign in to sites again.",
                stored[:8],
            )

        try:
            context_id = await self._client.create_context(context_name_for(user_id))
        except Exception as e:
            logger.warning(
                "Could not create a Browserbase context (%s); browsing without "
                "persistence, so logins will not be saved",
                e,
            )
            return None

        await self._store.put(user_id, self.vendor, context_id)
        return context_id

    async def launch(self, user_id: str) -> LaunchedBrowser:
        """Start a session on this user's context and return how to reach it."""
        if not (user_id or "").strip():
            # Contexts carry cookies and logins; a shared default would hand one
            # user's authenticated sessions to the next.
            raise ValueError("a browser session needs a real user id")

        context_id = await self._resolve_context(user_id)
        session = await self._client.create_session(context_id)

        session_id = str(session["id"])
        connect_url = session.get("connectUrl") or ""
        if not connect_url:
            # Nothing can drive this browser, and it is already billing.
            await self.release(session_id)
            raise RuntimeError("Browserbase session has no connectUrl")

        return LaunchedBrowser(
            browser_id=session_id,
            vendor=self.vendor,
            live_url=await self._client.live_view_url(session_id),
            cdp_ws_url=connect_url,
            persistence_id=context_id or "",
        )

    async def release(self, browser_id: str) -> None:
        if not browser_id:
            return
        await self._client.release_session(browser_id)
