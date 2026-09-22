"""Choosing, and talking to, a cloud-browser vendor.

`CloudBrowserSession` used to construct `BrowserUseClient` itself and read
Browser Use Cloud's response shape inline -- `liveUrl`, `cdpUrl`, `profileId`.
That made the vendor a hard dependency of the session lifecycle, so adding a
second one meant either forking the session class or threading `if
vendor == ...` through start, close and the failure paths.

This module is the seam instead. A provider answers one question -- "give me a
running browser bound to this user, and take it away again" -- and returns a
`LaunchedBrowser` describing how to reach it. Everything the session does after
that (CDP, idle timeout, metering, publishing to the frontend) is vendor-neutral.

The two vendors differ in exactly two ways that matter here, and both are
captured in `LaunchedBrowser`:

* **How persistence is named.** Browser Use has *profiles*, addressed by name,
  so the user id is enough to find one again. Browserbase has *contexts*,
  addressed only by an opaque id with no lookup-by-name endpoint, so the id has
  to be stored on our side -- see `context_store`.
* **What kind of CDP endpoint you get.** Browser Use returns an HTTP base to
  discover a page target through; Browserbase returns a browser-level
  WebSocket. `CDPConnection` handles both, but it has to be told which.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..settings import Settings, get_settings
from ..utils.logging import get_logger

logger = get_logger("browser.providers")

BROWSER_USE = "browser_use"
BROWSERBASE = "browserbase"


@dataclass(frozen=True)
class LaunchedBrowser:
    """A running cloud browser, and the two ways to reach it.

    `live_url` is what the user watches; `cdp_http_url` or `cdp_ws_url` is what
    the agent drives. Exactly one of the two CDP fields is set -- which one is
    the vendor's choice, not ours.
    """

    browser_id: str
    vendor: str
    live_url: str = ""
    cdp_http_url: str = ""
    cdp_ws_url: str = ""
    #: The vendor-side handle for this user's persisted cookies and logins
    #: (a Browser Use profile id, or a Browserbase context id). Recorded so a
    #: session can say whether persistence was actually in effect.
    persistence_id: str = ""

    def __post_init__(self) -> None:
        if not (self.cdp_http_url or self.cdp_ws_url):
            raise ValueError(f"{self.vendor} returned a browser with no CDP endpoint")


@runtime_checkable
class BrowserProviderPort(Protocol):
    """A vendor that can rent us a browser carrying one user's logins."""

    @property
    def vendor(self) -> str:
        """Short id used in logs and usage records."""
        ...

    async def launch(self, user_id: str) -> LaunchedBrowser:
        """Start a browser bound to `user_id`'s persisted state.

        Implementations must not fall back to a shared or anonymous profile:
        these carry cookies and logins, so one would hand one user's
        authenticated sessions to the next.
        """
        ...

    async def release(self, browser_id: str) -> None:
        """Stop the browser, persisting its state. Safe to call more than once."""
        pass


class ProviderUnavailableError(RuntimeError):
    """No cloud-browser vendor is configured well enough to use."""


def _browserbase_provider(settings: Settings) -> Any:
    from .browserbase import BrowserbaseProvider

    return BrowserbaseProvider(
        api_key=settings.browserbase_api_key,
        project_id=settings.browserbase_project_id,
    )


def _browser_use_provider(settings: Settings) -> Any:
    from .cloud_browser import BrowserUseProvider

    return BrowserUseProvider(api_key=settings.browser_use_api_key)


#: Vendor id -> (is it configured?, how to build it). Order decides the
#: fallback sequence when the requested vendor has no credentials.
_PROVIDERS: dict[str, tuple[str, Any]] = {
    BROWSERBASE: ("browserbase_api_key", _browserbase_provider),
    BROWSER_USE: ("browser_use_api_key", _browser_use_provider),
}


def configured_vendors(settings: Settings | None = None) -> list[str]:
    """Vendors that have a credential, in preference order."""
    settings = settings or get_settings()
    return [
        vendor for vendor, (key_field, _) in _PROVIDERS.items() if getattr(settings, key_field, "")
    ]


def create_browser_provider(settings: Settings | None = None) -> BrowserProviderPort:
    """Build the cloud-browser provider this deployment is configured for.

    Falls back to any other configured vendor rather than failing, because the
    alternative the user experiences is "the browser panel does not work at
    all". The fallback is logged loudly: running on a vendor nobody selected is
    a billing surprise, and the two do not persist logins to the same place, so
    the user's sites will be logged out.
    """
    settings = settings or get_settings()
    requested = (settings.browser_provider or BROWSERBASE).strip().lower()

    if requested not in _PROVIDERS:
        logger.warning(
            "Unknown KWAMI_BROWSER_PROVIDER %r; known vendors are %s",
            requested,
            ", ".join(sorted(_PROVIDERS)),
        )
        requested = BROWSERBASE

    available = configured_vendors(settings)
    if not available:
        raise ProviderUnavailableError(
            "No cloud browser is configured. Set BROWSERBASE_API_KEY "
            "(and BROWSERBASE_PROJECT_ID), or BROWSER_USE_API_KEY."
        )

    chosen = requested if requested in available else available[0]
    if chosen != requested:
        logger.warning(
            "KWAMI_BROWSER_PROVIDER=%s has no credential; falling back to %s. "
            "Saved logins do not carry across vendors, so the user will be "
            "signed out of sites they were signed in to.",
            requested,
            chosen,
        )

    return _PROVIDERS[chosen][1](settings)
