"""Browser Use Cloud API client and CDP WebSocket connection.

Wraps the Browser Use Cloud REST API v3 (https://api.browser-use.com/api/v3)
for creating cloud browsers with per-user profiles, and provides an async
CDP (Chrome DevTools Protocol) client for direct browser control.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import httpx

from ..utils.logging import get_logger

logger = get_logger("browser.cloud")

BU_API_BASE = "https://api.browser-use.com/api/v3"


# ---------------------------------------------------------------------------
# Browser Use Cloud REST API client
# ---------------------------------------------------------------------------


class BrowserUseClient:
    """Async HTTP client for the Browser Use Cloud REST API v3."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or os.environ.get("BROWSER_USE_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "BROWSER_USE_API_KEY is not set. "
                "Get one at https://cloud.browser-use.com/settings?tab=api-keys"
            )

    def _headers(self) -> Dict[str, str]:
        return {
            "X-Browser-Use-API-Key": self._api_key,
            "Content-Type": "application/json",
        }

    # -- Browsers ------------------------------------------------------------

    async def create_browser(
        self,
        profile_id: Optional[str] = None,
        timeout_minutes: int = 15,
        proxy_country: Optional[str] = "us",
    ) -> Dict[str, Any]:
        """Create a cloud browser session.

        Returns dict with keys: id, status, liveUrl, cdpUrl, timeoutAt, startedAt.
        """
        payload: Dict[str, Any] = {"timeout": timeout_minutes}
        if profile_id:
            payload["profileId"] = profile_id
        if proxy_country:
            payload["proxyCountryCode"] = proxy_country
        else:
            payload["proxyCountryCode"] = None

        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{BU_API_BASE}/browsers",
                json=payload,
                headers=self._headers(),
            )
            r.raise_for_status()
            data = r.json()
            logger.info(
                "Created cloud browser: id=%s, liveUrl=%s",
                data.get("id", "?")[:8],
                (data.get("liveUrl") or "")[:60],
            )
            return data

    async def stop_browser(self, browser_id: str) -> Dict[str, Any]:
        """Stop a cloud browser session (persists profile state)."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.patch(
                f"{BU_API_BASE}/browsers/{browser_id}",
                json={"status": "stopped"},
                headers=self._headers(),
            )
            r.raise_for_status()
            logger.info("Stopped cloud browser: %s", browser_id[:8])
            return r.json()

    async def get_browser(self, browser_id: str) -> Dict[str, Any]:
        """Get browser session details."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{BU_API_BASE}/browsers/{browser_id}",
                headers=self._headers(),
            )
            r.raise_for_status()
            return r.json()

    # -- Profiles ------------------------------------------------------------

    async def create_profile(self, name: str) -> Dict[str, Any]:
        """Create a browser profile for persistent auth state."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{BU_API_BASE}/profiles",
                json={"name": name},
                headers=self._headers(),
            )
            r.raise_for_status()
            data = r.json()
            logger.info("Created profile: id=%s name=%s", data.get("id", "?")[:8], name)
            return data

    async def list_profiles(self, query: Optional[str] = None) -> List[Dict[str, Any]]:
        """List profiles, optionally filtered by name query."""
        params: Dict[str, Any] = {"pageSize": 20}
        if query:
            params["query"] = query
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{BU_API_BASE}/profiles",
                params=params,
                headers=self._headers(),
            )
            r.raise_for_status()
            return r.json().get("items", [])

    async def get_or_create_profile(self, user_id: str) -> str:
        """Get existing profile for a user, or create one. Returns profile_id."""
        # Search by name (we use kwami user_id as the profile name)
        existing = await self.list_profiles(query=user_id)
        for p in existing:
            if p.get("name") == user_id:
                logger.debug("Found existing profile for %s: %s", user_id, p["id"][:8])
                return p["id"]
        # Create new
        profile = await self.create_profile(name=user_id)
        return profile["id"]


# ---------------------------------------------------------------------------
# CDP WebSocket connection
# ---------------------------------------------------------------------------


class CDPConnection:
    """Async Chrome DevTools Protocol client over WebSocket.

    Connects to the cloud browser's CDP endpoint and provides high-level
    methods for navigation, interaction, and page inspection.
    """

    def __init__(self) -> None:
        self._ws: Any = None
        self._msg_id: int = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None

    async def connect(self, cdp_url: str) -> None:
        """Connect to the CDP endpoint.

        Args:
            cdp_url: The HTTP CDP URL (e.g. https://cdp-xxx.browser-use.com).
                     We resolve the WS debugger URL from /json/version.
        """
        import websockets

        # Resolve WebSocket URL from /json/version
        ws_url = await self._resolve_ws_url(cdp_url)
        logger.info("Connecting to CDP WebSocket: %s", ws_url[:80])

        self._ws = await websockets.connect(
            ws_url,
            max_size=10 * 1024 * 1024,  # 10 MB for screenshots
            close_timeout=5,
        )
        self._reader_task = asyncio.create_task(self._reader_loop())
        logger.info("CDP WebSocket connected")

    async def _resolve_ws_url(self, cdp_url: str) -> str:
        """Resolve the WebSocket debugger URL for a page target.

        We must connect to a *page* target (not the browser target) so that
        Page.*, Runtime.*, and Input.* domains are available.
        """
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Try /json/list first — gives us page-level targets
            try:
                r = await client.get(f"{cdp_url}/json/list")
                r.raise_for_status()
                targets = r.json()
                # Find the first "page" type target
                for target in targets:
                    if target.get("type") == "page" and target.get("webSocketDebuggerUrl"):
                        ws_url = target["webSocketDebuggerUrl"]
                        logger.debug("Resolved page target: %s", ws_url[:80])
                        return ws_url
            except Exception as e:
                logger.debug("/json/list failed, trying /json/version: %s", e)

            # Fallback: get browser-level WS and use Target.createTarget
            r = await client.get(f"{cdp_url}/json/version")
            r.raise_for_status()
            data = r.json()
            browser_ws = data.get("webSocketDebuggerUrl", "")
            if not browser_ws:
                raise ValueError(f"No webSocketDebuggerUrl in /json/version response")

            # Connect to browser target temporarily to create a page
            import websockets
            async with websockets.connect(browser_ws, close_timeout=5) as browser_conn:
                create_msg = json.dumps({
                    "id": 1,
                    "method": "Target.createTarget",
                    "params": {"url": "about:blank"},
                })
                await browser_conn.send(create_msg)
                resp = json.loads(await asyncio.wait_for(browser_conn.recv(), timeout=10))
                target_id = resp.get("result", {}).get("targetId", "")

            if not target_id:
                raise ValueError("Failed to create a page target via Target.createTarget")

            # Now resolve the page target WS URL
            r2 = await client.get(f"{cdp_url}/json/list")
            r2.raise_for_status()
            for target in r2.json():
                if target.get("id") == target_id and target.get("webSocketDebuggerUrl"):
                    ws_url = target["webSocketDebuggerUrl"]
                    logger.debug("Created and resolved page target: %s", ws_url[:80])
                    return ws_url

            raise ValueError(f"Could not find page target {target_id} in /json/list")

    async def _reader_loop(self) -> None:
        """Read incoming WebSocket messages and resolve pending futures."""
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                msg_id = msg.get("id")
                if msg_id is not None and msg_id in self._pending:
                    self._pending[msg_id].set_result(msg)
        except Exception as e:
            logger.debug("CDP reader loop ended: %s", e)
            # Resolve all pending futures with error
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError(f"CDP connection lost: {e}"))

    async def send(self, method: str, **params: Any) -> Dict[str, Any]:
        """Send a CDP command and wait for the response."""
        if not self._ws:
            raise ConnectionError("CDP WebSocket not connected")

        self._msg_id += 1
        msg_id = self._msg_id
        msg = {"id": msg_id, "method": method, "params": params}

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[msg_id] = future

        await self._ws.send(json.dumps(msg))
        try:
            result = await asyncio.wait_for(future, timeout=30.0)
        finally:
            self._pending.pop(msg_id, None)

        if "error" in result:
            err = result["error"]
            raise RuntimeError(f"CDP error: {err.get('message', err)}")

        return result.get("result", {})

    async def close(self) -> None:
        """Close the CDP WebSocket connection."""
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._pending.clear()
        logger.debug("CDP connection closed")

    @property
    def is_connected(self) -> bool:
        if self._ws is None:
            return False
        try:
            # websockets >= 13: ClientConnection uses .state (State enum)
            from websockets.protocol import State
            return self._ws.state is State.OPEN
        except (ImportError, AttributeError):
            # websockets < 13 fallback
            return getattr(self._ws, "open", False)

    # -- High-level CDP operations -------------------------------------------

    async def navigate(self, url: str) -> Dict[str, Any]:
        """Navigate to a URL."""
        return await self.send("Page.navigate", url=url)

    async def go_back(self) -> None:
        """Navigate back in history."""
        history = await self.send("Page.getNavigationHistory")
        idx = history.get("currentIndex", 0)
        entries = history.get("entries", [])
        if idx > 0:
            await self.send("Page.navigateToHistoryEntry", entryId=entries[idx - 1]["id"])

    async def go_forward(self) -> None:
        """Navigate forward in history."""
        history = await self.send("Page.getNavigationHistory")
        idx = history.get("currentIndex", 0)
        entries = history.get("entries", [])
        if idx < len(entries) - 1:
            await self.send("Page.navigateToHistoryEntry", entryId=entries[idx + 1]["id"])

    async def evaluate(self, expression: str) -> Any:
        """Evaluate JavaScript in the page context."""
        result = await self.send(
            "Runtime.evaluate",
            expression=expression,
            returnByValue=True,
            awaitPromise=True,
        )
        val = result.get("result", {})
        if val.get("type") == "undefined":
            return None
        return val.get("value", val)

    async def screenshot(self, quality: int = 70) -> str:
        """Take a screenshot, return base64 encoded JPEG."""
        result = await self.send(
            "Page.captureScreenshot",
            format="jpeg",
            quality=quality,
        )
        return result.get("data", "")

    async def click(self, x: float, y: float) -> None:
        """Click at page coordinates using compositor-level mouse events."""
        for event_type in ("mousePressed", "mouseReleased"):
            await self.send(
                "Input.dispatchMouseEvent",
                type=event_type,
                x=x,
                y=y,
                button="left",
                clickCount=1,
            )

    async def type_text(self, text: str) -> None:
        """Type text using CDP insertText (works with any focused input)."""
        await self.send("Input.insertText", text=text)

    async def press_key(self, key: str) -> None:
        """Press a keyboard key."""
        # Map common key names to CDP key definitions
        key_map = {
            "Enter": {"key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13},
            "Tab": {"key": "Tab", "code": "Tab", "windowsVirtualKeyCode": 9},
            "Escape": {"key": "Escape", "code": "Escape", "windowsVirtualKeyCode": 27},
            "Backspace": {"key": "Backspace", "code": "Backspace", "windowsVirtualKeyCode": 8},
            "ArrowDown": {"key": "ArrowDown", "code": "ArrowDown", "windowsVirtualKeyCode": 40},
            "ArrowUp": {"key": "ArrowUp", "code": "ArrowUp", "windowsVirtualKeyCode": 38},
            "Space": {"key": " ", "code": "Space", "windowsVirtualKeyCode": 32},
        }
        kdef = key_map.get(key, {"key": key, "code": key, "windowsVirtualKeyCode": ord(key[0]) if key else 0})

        for event_type in ("keyDown", "keyUp"):
            await self.send("Input.dispatchKeyEvent", type=event_type, **kdef)

    async def scroll(self, x: float = 0, y: float = 0, delta_y: float = -400) -> None:
        """Scroll the page. Negative delta_y = scroll down."""
        await self.send(
            "Input.dispatchMouseEvent",
            type="mouseWheel",
            x=x,
            y=y,
            deltaX=0,
            deltaY=delta_y,
        )

    async def page_info(self) -> Dict[str, Any]:
        """Extract page title, text content, and interactive elements via JS evaluation."""
        js = """
        (() => {
            const doc = document;
            const body = doc.body;
            if (!body) return { title: doc.title, text: '', elements: [], html: '' };

            const clickableSelector = [
                'a[href]', 'button', 'input', 'select', 'textarea',
                '[type="submit"]', '[type="button"]',
                '[role="button"]', '[role="link"]', '[role="tab"]', '[role="menuitem"]',
                '[onclick]', '[tabindex="0"]',
            ].join(', ');

            // Clear previous stamps
            doc.querySelectorAll('[data-kwami-id]').forEach(el => el.removeAttribute('data-kwami-id'));

            const main = doc.querySelector('main, article, [role="main"], .content, #content') || body;
            const textContent = (main.innerText || '').slice(0, 5000);

            const items = [];
            doc.querySelectorAll(clickableSelector).forEach((el, i) => {
                if (i >= 80) return;
                const label = (el.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 120)
                    || el.getAttribute('aria-label') || el.getAttribute('title')
                    || el.getAttribute('placeholder') || '';
                if (!label) return;
                const eid = 'el-' + i;
                el.setAttribute('data-kwami-id', eid);
                const rect = el.getBoundingClientRect();
                items.push({
                    id: eid,
                    type: el.tagName.toLowerCase(),
                    label: label.slice(0, 80),
                    x: Math.round(rect.left + rect.width / 2),
                    y: Math.round(rect.top + rect.height / 2),
                    visible: rect.width > 0 && rect.height > 0,
                });
            });

            let html = '';
            try {
                const clone = main.cloneNode(true);
                clone.querySelectorAll('script, style, noscript, iframe').forEach(el => el.remove());
                html = clone.innerHTML.replace(/\\s+/g, ' ').trim().slice(0, 12000);
            } catch (e) {}

            return { title: doc.title, text: textContent, elements: items, html };
        })()
        """
        return await self.evaluate(js)
