"""What a browser vendor hands back, and how it says it could not.

Apart from `providers` because that module builds the vendors, and the vendors
need these: importing them from there would be a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

BROWSER_USE = "browser_use"
BROWSERBASE = "browserbase"
#: Chrome on the agent's own machine: see `local_chrome`. Never a fallback for a
#: cloud vendor, because on a server it would open a window nobody can see.
LOCAL = "local"
#: Chromium in the agent's own container: see `headless_chrome`.
HEADLESS = "headless"

#: Why a vendor could not give us a browser. The first two do not fix themselves
#: on a retry, so the vendor is left alone for a while.
BILLING = "billing"
CREDENTIALS = "credentials"
BUSY = "busy"
DOWN = "down"
MISSING = "missing"

_LABELS = {
    BROWSERBASE: "Browserbase",
    BROWSER_USE: "Browser Use",
    LOCAL: "The local browser",
    HEADLESS: "The built-in browser",
}


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


class BrowserVendorError(RuntimeError):
    """One vendor could not give us a browser, in words fit for the user."""

    def __init__(self, vendor: str, kind: str, text: str = "") -> None:
        self.vendor = vendor
        self.kind = kind
        super().__init__(text or _describe(vendor, kind))


def _describe(vendor: str, kind: str) -> str:
    label = _LABELS.get(vendor, vendor)
    return {
        BILLING: f"{label} has no credit left (402) and needs topping up.",
        CREDENTIALS: f"{label} rejected our API key.",
        BUSY: f"{label} is at its usage limit for the moment.",
    }.get(kind, f"{label} could not be reached.")
