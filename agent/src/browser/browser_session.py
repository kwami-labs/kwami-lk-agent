"""Cloud browser session lifecycle manager.

Manages a single cloud browser per agent session: creation with per-user
profiles, CDP-based navigation and interaction, idle timeout, and cleanup.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger
from .cloud_browser import BrowserUseClient, CDPConnection

logger = get_logger("browser.session")

# Auto-close cloud browser after this many seconds of inactivity
IDLE_TIMEOUT_SECONDS = 5 * 60  # 5 minutes


class CloudBrowserSession:
    """Manages a cloud browser session with CDP control.

    Lifecycle:
        1. start() — create cloud browser (with profile), connect CDP, return liveUrl
        2. navigate() / read_page() / click() / type_text() — interact via CDP
        3. close() — stop browser (persists profile), disconnect CDP
    """

    def __init__(self, room: Any = None) -> None:
        """Initialize the session manager.

        Args:
            room: LiveKit room for publishing data to the frontend.
        """
        self._client: Optional[BrowserUseClient] = None
        self._cdp: Optional[CDPConnection] = None
        self._browser_id: Optional[str] = None
        self._live_url: Optional[str] = None
        self._profile_id: Optional[str] = None
        self._room = room
        self._current_url: str = ""
        self._idle_timer: Optional[asyncio.Task] = None

    @property
    def is_active(self) -> bool:
        """Whether a cloud browser is currently running."""
        return self._browser_id is not None and self._cdp is not None and self._cdp.is_connected

    @property
    def live_url(self) -> Optional[str]:
        return self._live_url

    def set_room(self, room: Any) -> None:
        """Update the LiveKit room reference."""
        self._room = room

    # -- Lifecycle -----------------------------------------------------------

    async def start(self, user_id: str, url: Optional[str] = None) -> str:
        """Create a cloud browser with the user's profile and connect CDP.

        Args:
            user_id: Kwami user ID (used as profile name for cookie persistence).
            url: Optional initial URL to navigate to.

        Returns:
            The liveUrl for embedding in the frontend iframe.
        """
        if self.is_active:
            # Reuse existing session — just navigate if URL given
            if url:
                await self.navigate(url)
            return self._live_url or ""

        try:
            self._client = BrowserUseClient()
        except ValueError as e:
            logger.warning("Cannot start cloud browser: %s", e)
            raise

        # Get or create a persistent profile for this user
        try:
            self._profile_id = await self._client.get_or_create_profile(user_id)
        except Exception as e:
            logger.warning("Profile lookup failed, continuing without profile: %s", e)
            self._profile_id = None

        # Create the cloud browser
        browser = await self._client.create_browser(
            profile_id=self._profile_id,
            timeout_minutes=15,
        )
        self._browser_id = browser["id"]
        self._live_url = browser.get("liveUrl")
        cdp_url = browser.get("cdpUrl")

        if not cdp_url:
            raise RuntimeError("Browser Use Cloud did not return a cdpUrl")

        # Connect CDP
        self._cdp = CDPConnection()
        await self._cdp.connect(cdp_url)

        # Enable Page domain for navigation events
        try:
            await self._cdp.send("Page.enable")
        except Exception:
            pass

        # Navigate to initial URL if provided
        if url:
            await self._cdp.navigate(url)
            self._current_url = url

        # Publish liveUrl to frontend
        await self._publish_session_event("open", url=url)

        # Start idle timer
        self._reset_idle_timer()

        logger.info(
            "Cloud browser started: id=%s, profile=%s, url=%s",
            self._browser_id[:8] if self._browser_id else "?",
            self._profile_id[:8] if self._profile_id else "none",
            (url or "")[:60],
        )
        return self._live_url or ""

    async def close(self) -> None:
        """Stop the cloud browser and disconnect CDP. Profile state is persisted."""
        self._cancel_idle_timer()

        if self._cdp:
            await self._cdp.close()
            self._cdp = None

        if self._client and self._browser_id:
            try:
                await self._client.stop_browser(self._browser_id)
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
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        await self._cdp.navigate(url)
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
        await self._cdp.go_back()
        self._reset_idle_timer()
        return "Going back to the previous page."

    async def go_forward(self) -> str:
        """Navigate forward in history."""
        self._ensure_active()
        await self._cdp.go_forward()
        self._reset_idle_timer()
        return "Going forward to the next page."

    # -- Page interaction ----------------------------------------------------

    async def read_page(self) -> str:
        """Read the current page content via CDP."""
        self._ensure_active()
        self._reset_idle_timer()

        info = await self._cdp.page_info()
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
                    f"  {el['id']} [{el['type']}] {vis} \"{el['label']}\" "
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
        info = await self._cdp.page_info()
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
        await self._cdp.click(float(x), float(y))

        label = (target.get("label") or "")[:60]
        return f"Clicked on \"{label}\" ({target['id']}) at ({x}, {y})."

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
            await self._cdp.send(
                "Input.dispatchKeyEvent",
                type="keyDown",
                key="a",
                code="KeyA",
                windowsVirtualKeyCode=65,
                modifiers=2 if not _is_mac() else 4,  # Ctrl/Cmd+A
            )
            await self._cdp.send(
                "Input.dispatchKeyEvent",
                type="keyUp",
                key="a",
                code="KeyA",
                windowsVirtualKeyCode=65,
            )
            await self._cdp.press_key("Backspace")
            await asyncio.sleep(0.1)

        # Type the text
        await self._cdp.type_text(text)
        return f"Typed '{text[:50]}' into the field."

    async def press_key(self, key: str) -> str:
        """Press a keyboard key."""
        self._ensure_active()
        self._reset_idle_timer()
        await self._cdp.press_key(key)
        return f"Pressed '{key}'."

    async def scroll(self, direction: str = "down") -> str:
        """Scroll the page up or down."""
        self._ensure_active()
        self._reset_idle_timer()
        delta = 400 if direction.lower() == "down" else -400
        await self._cdp.scroll(x=400, y=300, delta_y=delta)
        return f"Scrolled {direction}."

    # -- Internal helpers ----------------------------------------------------

    def _ensure_active(self) -> None:
        """Raise if no active cloud browser session."""
        if not self.is_active:
            raise RuntimeError(
                "No active cloud browser session. Use navigate_to to open a website first."
            )

    async def _publish_session_event(
        self, action: str, url: Optional[str] = None, title: Optional[str] = None
    ) -> None:
        """Send a browser_session event to the frontend via LiveKit data channel."""
        if not self._room:
            return
        msg: Dict[str, Any] = {"type": "browser_session", "action": action}
        if self._live_url and action == "open":
            # Append dark theme and hide BU UI chrome
            live = self._live_url
            sep = "&" if "?" in live else "?"
            msg["liveUrl"] = f"{live}{sep}theme=dark&ui=false"
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

    def _reset_idle_timer(self) -> None:
        """Reset the idle auto-close timer."""
        self._cancel_idle_timer()
        self._idle_timer = asyncio.create_task(self._idle_timeout())

    def _cancel_idle_timer(self) -> None:
        if self._idle_timer and not self._idle_timer.done():
            self._idle_timer.cancel()
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
