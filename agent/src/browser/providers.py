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

import time
from typing import Any, Protocol, runtime_checkable

import httpx

from ..settings import Settings, get_settings
from ..utils.logging import get_logger
from .launched import (
    BILLING,
    BROWSER_USE,
    BROWSERBASE,
    BUSY,
    CREDENTIALS,
    DOWN,
    HEADLESS,
    LOCAL,
    MISSING,
    BrowserVendorError,
    LaunchedBrowser,
)

logger = get_logger("browser.providers")

__all__ = [
    "BILLING",
    "BROWSERBASE",
    "BROWSER_USE",
    "BUSY",
    "CREDENTIALS",
    "DOWN",
    "HEADLESS",
    "LOCAL",
    "MISSING",
    "BrowserProviderPort",
    "BrowserVendorError",
    "FailoverBrowserProvider",
    "LaunchedBrowser",
    "ProviderUnavailableError",
    "configured_vendors",
    "create_browser_provider",
    "vendor_failure",
]

#: Vendors that refuse for a reason a retry does not fix are left alone for a while.
_REST_SECONDS = {BILLING: 15 * 60, CREDENTIALS: 15 * 60, BUSY: 60}


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
        ...


class ProviderUnavailableError(RuntimeError):
    """No cloud-browser vendor is configured well enough to use."""


def vendor_failure(vendor: str, error: Exception) -> BrowserVendorError:
    """Name what went wrong, leaving the vendor's URLs and bodies out of it."""
    if isinstance(error, BrowserVendorError):
        return error
    kind = DOWN
    if isinstance(error, httpx.HTTPStatusError):
        kind = {402: BILLING, 401: CREDENTIALS, 403: CREDENTIALS, 429: BUSY}.get(
            error.response.status_code, DOWN
        )
    return BrowserVendorError(vendor, kind)


#: Vendors that recently refused, across this worker's sessions: asking one with
#: no balance again on every "open this" only makes the user wait for the same 402.
_resting: dict[str, tuple[float, BrowserVendorError]] = {}


class FailoverBrowserProvider:
    """Tries each vendor in order until one hands over a browser.

    A vendor can have a credential and still be useless: the balance runs out
    with nothing in our configuration changing, and picking a vendor only by
    its key is what left the panel dead on Browser Use's 402 (2026-11-02).
    """

    def __init__(self, candidates: list[tuple[str, Any]]) -> None:
        self._candidates = candidates
        self._active_vendor = candidates[0][0]
        self._built: dict[str, Any] = {}
        self._owners: dict[str, Any] = {}

    @property
    def vendor(self) -> str:
        return self._active_vendor

    def _provider(self, vendor: str, build: Any) -> Any:
        if vendor not in self._built:
            self._built[vendor] = build()
        return self._built[vendor]

    async def launch(self, user_id: str) -> LaunchedBrowser:
        failures: list[BrowserVendorError] = []
        for vendor, build in self._candidates:
            rest = _resting.get(vendor)
            if rest and time.monotonic() < rest[0]:
                failures.append(rest[1])
                continue
            try:
                provider = self._provider(vendor, build)
                launched = await provider.launch(user_id)
            except ValueError:
                # Not the vendor's fault, and no other vendor would accept it.
                raise
            except Exception as e:
                failure = vendor_failure(vendor, e)
                logger.warning("Browser vendor %s failed: %s", vendor, failure)
                failures.append(failure)
                seconds = _REST_SECONDS.get(failure.kind)
                if seconds:
                    _resting[vendor] = (time.monotonic() + seconds, failure)
                continue
            if vendor != self._candidates[0][0]:
                logger.warning(
                    "Browsing on %s because %s failed. Saved logins do not carry across "
                    "vendors, so the user may have to sign in again.",
                    vendor,
                    self._candidates[0][0],
                )
            self._active_vendor = vendor
            self._owners[launched.browser_id] = provider
            return launched
        raise ProviderUnavailableError(
            "No browser could be opened. "
            + " ".join(str(failure) for failure in failures)
            + " This is the browser service, not the website or the user's account on it."
        )

    async def release(self, browser_id: str) -> None:
        owner = self._owners.get(browser_id)
        if owner is not None:
            await owner.release(browser_id)

    async def quit(self, browser_id: str) -> None:
        """Close a browser the user can see, for a vendor that has such a thing."""
        closer = getattr(self._owners.get(browser_id), "quit", None)
        if closer is not None:
            await closer(browser_id)


def _browserbase_provider(settings: Settings) -> Any:
    from .browserbase import BrowserbaseProvider

    return BrowserbaseProvider(
        api_key=settings.browserbase_api_key,
        project_id=settings.browserbase_project_id,
    )


def _browser_use_provider(settings: Settings) -> Any:
    from .cloud_browser import BrowserUseProvider

    return BrowserUseProvider(api_key=settings.browser_use_api_key)


def _headless_provider(settings: Settings) -> Any:
    from .headless_chrome import HeadlessChromeProvider

    return HeadlessChromeProvider(settings.headless_chromium)


def _local_provider(settings: Settings) -> Any:
    from .local_chrome import LocalChromeProvider

    return LocalChromeProvider()


#: Vendor id -> (is it configured?, how to build it). Order decides the
#: fallback sequence when the requested vendor has no credentials.
_PROVIDERS: dict[str, tuple[str, Any]] = {
    BROWSERBASE: ("browserbase_api_key", _browserbase_provider),
    BROWSER_USE: ("browser_use_api_key", _browser_use_provider),
}


def _cloud_candidates(vendors: list[str], settings: Settings) -> list[tuple[str, Any]]:
    return [(vendor, lambda v=vendor: _PROVIDERS[v][1](settings)) for vendor in vendors]


def _with_builtin(candidates: list[tuple[str, Any]], settings: Settings) -> list[tuple[str, Any]]:
    """The container's own Chromium, last: free, but it remembers no logins."""
    if not settings.headless_chromium:
        return candidates
    return [*candidates, (HEADLESS, lambda: _headless_provider(settings))]


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

    if requested == LOCAL:
        # Cloud vendors stay behind it, for a machine with no Chrome to open.
        return FailoverBrowserProvider(
            _with_builtin(
                [
                    (LOCAL, lambda: _local_provider(settings)),
                    *_cloud_candidates(configured_vendors(settings), settings),
                ],
                settings,
            )
        )

    if requested not in _PROVIDERS:
        logger.warning(
            "Unknown KWAMI_BROWSER_PROVIDER %r; known vendors are %s",
            requested,
            ", ".join(sorted(_PROVIDERS)),
        )
        requested = BROWSERBASE

    available = configured_vendors(settings)
    if not available and settings.headless_chromium:
        return FailoverBrowserProvider(_with_builtin([], settings))
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

    # The chosen vendor first, then the others: a credential says nothing about
    # the balance behind it.
    return FailoverBrowserProvider(
        _with_builtin(
            _cloud_candidates([chosen, *(v for v in available if v != chosen)], settings),
            settings,
        )
    )
