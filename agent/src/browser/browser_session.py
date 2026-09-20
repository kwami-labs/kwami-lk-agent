"""Cloud browser session lifecycle manager.

Manages a single cloud browser per agent session: creation against the user's
persistent profile, CDP-based navigation and interaction, idle timeout, and
cleanup.

Which vendor supplies the browser is decided in `browser.providers`, not here.
This class used to construct `BrowserUseClient` itself and read that vendor's
response shape inline, which made a second vendor impossible to add without
forking it. Everything below the launch call -- CDP, idle timeout, metering,
publishing to the frontend -- is the same whoever is running the browser.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from ..utils.logging import get_logger
from .cloud_browser import CDPConnection
from .providers import BrowserProviderPort, LaunchedBrowser, create_browser_provider
from .safety import validate_url

logger = get_logger("browser.session")

# Auto-close cloud browser after this many seconds of inactivity
IDLE_TIMEOUT_SECONDS = 5 * 60  # 5 minutes


class CloudBrowserSession:
    """Manages a cloud browser session with CDP control.

    Lifecycle:
        1. start() — rent a browser on the user's profile, connect CDP, return liveUrl
        2. navigate() / read_page() / click() / type_text() — interact via CDP
        3. close() — release the browser (persisting logins), disconnect CDP
    """

    def __init__(
        self,
        room: Any = None,
        usage_tracker: Any = None,
        provider: BrowserProviderPort | None = None,
    ) -> None:
        """Initialize the session manager.

        Args:
            room: LiveKit room for publishing data to the frontend.
            usage_tracker: Records the billed minutes this session consumes.
            provider: Cloud-browser vendor. Resolved from settings on first use
                when not supplied, so constructing a session never needs
                credentials -- only starting one does.
        """
        self._provider: BrowserProviderPort | None = provider
        self._cdp: CDPConnection | None = None
        self._browser_id: str | None = None
        self._live_url: str | None = None
        self._vendor: str = ""
        self._persistence_id: str | None = None
        self._room = room
        self._current_url: str = ""
        self._idle_timer: asyncio.Task | None = None
        # Cloud browsers bill per minute, with a US proxy on top. This is the
        # single most expensive resource the agent can hold, and it was the one
        # resource never metered -- pure, invisible margin loss.
        self._usage_tracker = usage_tracker
        self._started_at: float | None = None

    @property
    def is_active(self) -> bool:
        """Whether a cloud browser is currently running."""
        return self._browser_id is not None and self._cdp is not None and self._cdp.is_connected

    @property
    def live_url(self) -> str | None:
        return self._live_url

    def set_room(self, room: Any) -> None:
        """Update the LiveKit room reference."""
        self._room = room

    def set_usage_tracker(self, usage_tracker: Any) -> None:
        """Attach the tracker that bills this session's minutes."""
        self._usage_tracker = usage_tracker

    def _record_browser_minutes(self) -> None:
        """Bill the wall-clock minutes this browser was held for.

        Called from every path that releases a browser, so a session torn down
        by the idle timer or by a failed CDP connect is metered the same as one
        the user closed.
        """
        started, self._started_at = self._started_at, None
        if started is None or self._usage_tracker is None:
            return
        minutes = max(0.0, (time.monotonic() - started)) / 60.0
        if minutes <= 0:
            return
        try:
            self._usage_tracker.record_external_usage(
                "browser",
                f"{self._vendor or 'browser'}/cloud",
                units_used=round(minutes, 4),
            )
        except Exception as e:  # never let metering break teardown
            logger.warning("Failed to record browser usage: %s", e)

    # -- Lifecycle -----------------------------------------------------------

    async def start(self, user_id: str, url: str | None = None) -> str:
        """Create a cloud browser with the user's profile and connect CDP.

        Args:
            user_id: Kwami user ID (used as profile name for cookie persistence).
            url: Optional initial URL to navigate to.

        Returns:
            The liveUrl for embedding in the frontend iframe.
        """
        # Validate before renting anything. The reuse path below goes through
        # `navigate`, which validates, but the fresh path drove `self._cdp`
        # directly and so had no gate at all -- `navigate_to` happened to
        # validate first, which left the session object safe only because of
        # its caller. A browser profile carries the user's cookies and logins,
        # so this has to hold wherever `start` is called from.
        if url:
            url = validate_url(url)

        if self.is_active:
            # Reuse existing session — just navigate if URL given
            if url:
                await self.navigate(url)
            return self._live_url or ""

        if self._provider is None:
            try:
                self._provider = create_browser_provider()
            except Exception as e:
                logger.warning("Cannot start cloud browser: %s", e)
                raise

        launched: LaunchedBrowser = await self._provider.launch(user_id)
        self._browser_id = launched.browser_id
        self._live_url = launched.live_url
        self._vendor = launched.vendor
        self._persistence_id = launched.persistence_id or None

        # From here on the cloud browser exists and is being billed by the
        # minute. Anything that fails before the session is usable must release
        # it: `is_active` stays False without a CDP connection, and every
        # cleanup path is gated on `is_active`, so a failure here used to strand
        # a paid browser until its own 15-minute server timeout.
        try:
            self._cdp = CDPConnection()
            if launched.cdp_ws_url:
                # A browser-level endpoint: attach to a page, or the Page,
                # Input and Runtime domains are simply absent.
                await self._cdp.connect_ws(launched.cdp_ws_url)
            else:
                await self._cdp.connect(launched.cdp_http_url)
        except Exception:
            await self._release_unusable_browser()
            raise

        # Enable Page domain for navigation events
        try:
            await self._cdp.send("Page.enable")
            await self._cdp.send(
                "Emulation.setDeviceMetricsOverride",
                width=1280,
                height=1400,
                deviceScaleFactor=1,
                mobile=False,
            )
        except Exception as e:
            logger.debug("Failed to set CDP initial overrides: %s", e)

        # Navigate to initial URL if provided
        if url:
            await self._cdp.navigate(url)
            self._current_url = url

        # Publish liveUrl to frontend
        await self._publish_session_event("open", url=url)

        # Start idle timer
        self._started_at = time.monotonic()
        self._reset_idle_timer()

        logger.info(
            "Cloud browser started: vendor=%s, id=%s, profile=%s, url=%s",
            self._vendor or "?",
            self._browser_id[:8] if self._browser_id else "?",
            self._persistence_id[:8] if self._persistence_id else "none",
            (url or "")[:60],
        )
        if not self._persistence_id:
            logger.warning(
                "Browsing without a persistent profile: nothing the user signs "
                "in to during this session will be remembered."
            )
        return self._live_url or ""

    async def close(self) -> None:
        """Stop the cloud browser and disconnect CDP. Profile state is persisted."""
        self._cancel_idle_timer()
        self._record_browser_minutes()

        if self._cdp:
            await self._cdp.close()
            self._cdp = None

        if self._provider and self._browser_id:
            try:
                await self._provider.release(self._browser_id)
            except Exception as e:
                logger.warning("Failed to stop cloud browser %s: %s", self._browser_id[:8], e)

        # Notify frontend
        await self._publish_session_event("close")

        self._browser_id = None
        self._live_url = None
        self._current_url = ""
        logger.info("Cloud browser session closed")

    # -- Navigation ----------------------------------------------------------

    async def navigate(self, url: str) -> str:
        """Navigate to a URL in the cloud browser."""
        self._ensure_active()
        # Revalidated here as well as in `start` and in the `navigate_to` tool:
        # this is the boundary every browsing path eventually crosses, and it
        # must not depend on which of them got here.
        url = validate_url(url)
        await self._connection.navigate(url)
        self._current_url = url
        self._reset_idle_timer()

        # Notify frontend of URL update
        await self._publish_session_event("update", url=url)

        # Wait a moment for page to start loading
        await asyncio.sleep(1.5)
        return f"Navigating to {url}. The user can see the page in their browser panel."

    async def go_back(self) -> str:
        """Navigate back in history."""
        self._ensure_active()
        await self._connection.go_back()
        self._reset_idle_timer()
        return "Going back to the previous page."

    async def go_forward(self) -> str:
        """Navigate forward in history."""
        self._ensure_active()
        await self._connection.go_forward()
        self._reset_idle_timer()
        return "Going forward to the next page."

    # -- Page interaction ----------------------------------------------------

    async def read_page(self) -> str:
        """Read the current page content via CDP."""
        self._ensure_active()
        self._reset_idle_timer()

        info = await self._connection.page_info()
        if not info or not isinstance(info, dict):
            return "Could not read page content."

        title = info.get("title", "")
        text = info.get("text", "")
        elements = info.get("elements", [])

        parts = [f"Page title: {title}"]
        if text:
            parts.append(f"\nPage content:\n{text[:1500]}")
        if elements:
            parts.append("\nInteractive elements:")
            for el in elements[:30]:
                vis = "✓" if el.get("visible") else "✗"
                parts.append(
                    f'  {el["id"]} [{el["type"]}] {vis} "{el["label"]}" '
                    f"(x={el.get('x', 0)}, y={el.get('y', 0)})"
                )
        return "\n".join(parts)

    async def click(
        self,
        element_id: str = "",
        description: str = "",
    ) -> str:
        """Click an element on the page.

        Uses element_id (from read_page) for precise coordinate clicks,
        or falls back to description-based matching.
        """
        self._ensure_active()
        self._reset_idle_timer()

        # Get page elements with coordinates
        info = await self._connection.page_info()
        elements = info.get("elements", []) if isinstance(info, dict) else []

        target = None

        # Priority 1: exact element_id match
        if element_id and element_id.startswith("el-"):
            for el in elements:
                if el.get("id") == element_id and el.get("visible"):
                    target = el
                    break

        # Priority 2: fuzzy description match
        if not target and description:
            desc_lower = description.lower().strip()
            best_score = 0
            for el in elements:
                if not el.get("visible"):
                    continue
                label = (el.get("label") or "").lower()
                if desc_lower in label:
                    score = 4
                elif all(w in label for w in desc_lower.split() if len(w) >= 2):
                    score = 2
                else:
                    score = 0
                if score > best_score:
                    best_score = score
                    target = el

        if not target:
            return f"Could not find element to click: {element_id or description}"

        x = target.get("x", 0)
        y = target.get("y", 0)
        await self._connection.click(float(x), float(y))

        label = (target.get("label") or "")[:60]
        return f'Clicked on "{label}" ({target["id"]}) at ({x}, {y}).'

    async def type_text(
        self,
        text: str,
        element_id: str = "",
        description: str = "",
        clear_first: bool = True,
    ) -> str:
        """Type text into a field."""
        self._ensure_active()
        self._reset_idle_timer()

        # If an element is specified, click it first to focus
        if element_id or description:
            click_result = await self.click(element_id=element_id, description=description)
            if "Could not find" in click_result:
                return click_result
            await asyncio.sleep(0.3)

        # Clear existing content if requested
        if clear_first:
            await self._connection.send(
                "Input.dispatchKeyEvent",
                type="keyDown",
                key="a",
                code="KeyA",
                windowsVirtualKeyCode=65,
                modifiers=2 if not _is_mac() else 4,  # Ctrl/Cmd+A
            )
            await self._connection.send(
                "Input.dispatchKeyEvent",
                type="keyUp",
                key="a",
                code="KeyA",
                windowsVirtualKeyCode=65,
            )
            await self._connection.press_key("Backspace")
            await asyncio.sleep(0.1)

        # Type the text
        await self._connection.type_text(text)
        return f"Typed '{text[:50]}' into the field."

    async def press_key(self, key: str) -> str:
        """Press a keyboard key."""
        self._ensure_active()
        self._reset_idle_timer()
        await self._connection.press_key(key)
        return f"Pressed '{key}'."

    async def scroll(self, direction: str = "down") -> str:
        """Scroll the page up or down."""
        self._ensure_active()
        self._reset_idle_timer()
        delta = 400 if direction.lower() == "down" else -400
        await self._connection.scroll(x=400, y=300, delta_y=delta)
        return f"Scrolled {direction}."

    async def evaluate_js(self, expression: str) -> str:
        """Evaluate arbitrary JavaScript in the page context."""
        self._ensure_active()
        self._reset_idle_timer()
        try:
            result = await self._connection.evaluate(expression)
            return f"JavaScript executed successfully. Result: {result}"
        except Exception as e:
            return f"Failed to execute JavaScript: {e}"

    # -- Internal helpers ----------------------------------------------------

    def _ensure_active(self) -> None:
        """Raise if no active cloud browser session."""
        if not self.is_active:
            raise RuntimeError(
                "No active cloud browser session. Use navigate_to to open a website first."
            )

    @property
    def _connection(self) -> CDPConnection:
        """The live CDP connection, narrowed.

        Every caller has already been through `_ensure_active`, which raises
        when `is_active` is false -- and `is_active` is exactly the check that
        `_cdp is not None`. The type checker cannot follow that across a method
        returning None, so thirteen call sites each read as "None has no
        attribute navigate". This states the invariant in one place instead.

        The raise is not dead: it is what keeps this honest if a future caller
        reaches a driving method without the guard.
        """
        cdp = self._cdp
        if cdp is None:
            raise RuntimeError(
                "No active cloud browser session. Use navigate_to to open a website first."
            )
        return cdp

    async def _publish_session_event(
        self, action: str, url: str | None = None, title: str | None = None
    ) -> None:
        """Send a browser_session event to the frontend via LiveKit data channel."""
        if not self._room:
            return
        msg: dict[str, Any] = {"type": "browser_session", "action": action}
        if self._live_url and action == "open":
            msg["liveUrl"] = self._embeddable_live_url()
            msg["vendor"] = self._vendor
            # The panel shows a "not signed in" hint when persistence is off,
            # rather than letting the user discover it by being logged out.
            msg["persistent"] = bool(self._persistence_id)
        if url:
            msg["url"] = url
        if title:
            msg["title"] = title
        try:
            payload = json.dumps(msg).encode("utf-8")
            await self._room.local_participant.publish_data(payload, reliable=True)
            logger.debug("Published browser_session event: action=%s", action)
        except Exception as e:
            logger.warning("Failed to publish browser_session event: %s", e)

    def _embeddable_live_url(self) -> str:
        """The live view URL, tuned for an iframe.

        Each vendor hides its own chrome differently, and the parameters are
        not interchangeable: Browser Use takes `ui=false`, while Browserbase's
        fullscreen debugger URL is already bare and takes `navbar` to put
        controls *back*. Sending one vendor's parameters to the other is
        harmless but does nothing, which is exactly the kind of silent no-op
        that reads as "the panel is broken".
        """
        live = self._live_url or ""
        if not live:
            return ""
        from .providers import BROWSER_USE

        if self._vendor != BROWSER_USE:
            return live
        sep = "&" if "?" in live else "?"
        return f"{live}{sep}theme=dark&ui=false"

    def _reset_idle_timer(self) -> None:
        """Reset the idle auto-close timer."""
        self._cancel_idle_timer()
        self._idle_timer = asyncio.create_task(self._idle_timeout())

    async def _release_unusable_browser(self) -> None:
        """Stop a cloud browser that was created but never became usable."""
        self._record_browser_minutes()
        browser_id, self._browser_id = self._browser_id, None
        self._live_url = None
        cdp, self._cdp = self._cdp, None
        if cdp is not None:
            try:
                await cdp.close()
            except Exception as e:
                logger.debug("Failed to close half-open CDP connection: %s", e)
        if browser_id and self._provider:
            try:
                await self._provider.release(browser_id)
                logger.info("Released unusable cloud browser %s", browser_id[:8])
            except Exception as e:
                logger.warning("Failed to release cloud browser %s: %s", browser_id[:8], e)

    def _cancel_idle_timer(self) -> None:
        """Cancel the pending idle timer, unless we are running inside it.

        `_idle_timeout` calls `close()`, and `close()` starts by cancelling the
        idle timer -- which is the very task executing it. The CancelledError
        then lands somewhere inside close() and can skip `stop_browser`, so the
        browser is never released and keeps billing.
        """
        timer = self._idle_timer
        if timer is None or timer.done():
            self._idle_timer = None
            return

        try:
            running = asyncio.current_task()
        except RuntimeError:
            running = None
        if running is timer:
            # Let the timer finish its own close(); just drop the reference.
            self._idle_timer = None
            return

        timer.cancel()
        self._idle_timer = None

    async def _idle_timeout(self) -> None:
        """Auto-close the browser after idle timeout."""
        try:
            await asyncio.sleep(IDLE_TIMEOUT_SECONDS)
            if self.is_active:
                logger.info("Cloud browser idle timeout reached, closing")
                await self.close()
        except asyncio.CancelledError:
            pass


def _is_mac() -> bool:
    """Check if running on macOS (for Cmd vs Ctrl key modifier)."""
    import platform

    return platform.system() == "Darwin"
