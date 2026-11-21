"""Jarvis working in the user's own Chrome: through Kwami's extension (`chrome_bridge`), or
through the Claude extension the way Claude Code does (`claude_chrome`), whichever is there.

The same face as `CloudBrowserSession`, so the voice's tools and the work thread's
`browser_tool` use either without knowing which. Where the two differ:

* **Jarvis's pages open in a window of its own** inside the user's Chrome, grouped as
  "Jarvis", without taking the focus from the user. Their sessions are all there, and
  nothing drives the browser over a debugging port, so sites that refuse Jarvis's own
  Chrome let the user sign in here.
* **A tab of theirs can be worked in** (`use_tab`), after they asked. Jarvis closes only
  the tabs it opened, never one of theirs.
* **No scripts.** `evaluate_js` refuses: a page in their Chrome holds their sessions. The
  media tools' fixed actions go through `run_named`, which picks a function the extension
  already has.
* **Signing in is the user's.** The extension never types into a password, code or card
  field; on a sign-in page Jarvis brings the tab to the front and the user signs in.
* **The panel mirrors it**: while Jarvis works in a tab and someone is looking, a picture
  of it goes to the app every `FRAME_INTERVAL_SECONDS` if it changed, on the same topic as
  the other browsers. Only the tab in front of its window can be pictured.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import re
from collections.abc import Callable
from typing import Any

from ..utils.logging import get_logger
from .browser_session import (
    _SIGN_IN_ADDRESS,
    _SIGN_IN_REFUSED,
    FRAME_INTERVAL_SECONDS,
    FRAME_QUALITY,
    FRAME_TOPIC,
    pick_element,
)
from .chrome_bridge import (
    BridgeError,
    BridgeUnavailableError,
    ChromeBridgePort,
    chrome_bridge,
    user_browser,
)
from .safety import validate_url

logger = get_logger("browser.user_chrome")

VENDOR = "user_chrome"
#: Opening a page waits for it to load, in the extension, for up to 15 s.
OPEN_TIMEOUT_SECONDS = 30.0
MAX_PAGE_TEXT = 12000
#: How much of the page's text one read hands the model, and how many elements.
READ_TEXT_CHARS = 4000
READ_ELEMENTS = 40
#: A click or a key that sends a form starts a navigation; give it a moment.
SETTLE_SECONDS = 0.8
#: The user's Chrome behind KWAMI is not painted and gives no picture (measured): it is
#: asked again this much later, and the panel says the page is in Chrome meanwhile.
FRAME_HIDDEN_SECONDS = 4.0
CAPTURE_TIMEOUT_SECONDS = 5.0

#: What the extension's refusals mean, said for the model.
_REFUSALS = {
    "secret-field": (
        "That field is for a password, a code or a card: the user types it themselves. "
        "Nothing was typed."
    ),
    "no-field": "No field has the focus: say which field to type in.",
    "not-a-field": "That element is not a field you can type in.",
    "not-found": "That element is gone from the page: read it again.",
    "protected-page": "Chrome does not let extensions into this page (its own settings, "
    "the Web Store): the user has to do this one.",
    "paused": "The user has paused Jarvis in their Chrome (the extension's button). "
    "Tell them, and carry on without it.",
}


def explain(error: BaseException) -> str:
    """The extension's error, in words the model can act on."""
    text = str(error)
    for code, words in _REFUSALS.items():
        if code in text:
            return words
    if "No tab with id" in text:
        return "The tab Jarvis was working in was closed."
    return text


def tab_line(number: int, tab: dict[str, Any]) -> str:
    marks = [
        mark
        for mark, on in (
            ("in front", tab.get("focused")),
            ("playing sound", tab.get("audible")),
            ("Jarvis's", tab.get("jarvis")),
        )
        if on
    ]
    title = str(tab.get("title") or "").strip().replace("\n", " ")[:90] or "(no title)"
    address = str(tab.get("url") or "")[:120]
    flags = f" [{', '.join(marks)}]" if marks else ""
    return f"{number}. {title} -- {address}{flags}"


def find_tab(tabs: list[dict[str, Any]], which: str) -> dict[str, Any] | None:
    """A tab by its number in the list (from 1), or by words of its title or address."""
    wanted = which.strip().lower()
    if wanted.isdigit():
        index = int(wanted) - 1
        return tabs[index] if 0 <= index < len(tabs) else None
    words = [w for w in re.split(r"\s+", wanted) if w]
    for tab in sorted(tabs, key=lambda t: not t.get("focused")):
        haystack = f"{tab.get('title') or ''} {tab.get('url') or ''}".lower()
        if words and all(w in haystack for w in words):
            return tab
    return None


class UserChromeSession:
    """One tab of the user's Chrome that Jarvis works in."""

    def __init__(self, room: Any = None, bridge: ChromeBridgePort | None = None) -> None:
        self._bridge = bridge
        #: Whether the way in was chosen here (and is chosen again if it goes away).
        self._picks = bridge is None
        self._room = room
        self._tab_id: int | None = None
        #: Whether Jarvis opened the tab (it may close it) or the user did (it may not).
        self._owned = False
        self._current_url = ""
        self._title = ""
        self._frames_wanted = True
        self._frame_task: asyncio.Task[None] | None = None

    # -- what the tools ask of every session -------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._tab_id is not None

    @property
    def runs_locally(self) -> bool:
        return True

    @property
    def is_builtin(self) -> bool:
        return False

    @property
    def uses_user_chrome(self) -> bool:
        return True

    @property
    def vendor(self) -> str:
        return VENDOR

    @property
    def live_url(self) -> str | None:
        return None

    @property
    def tab_id(self) -> int | None:
        return self._tab_id

    @property
    def owns_tab(self) -> bool:
        return self._owned

    def set_room(self, room: Any) -> None:
        self._room = room

    def set_usage_tracker(self, usage_tracker: Any) -> None:
        """The user's own Chrome costs nothing to meter."""

    @property
    def bridge(self) -> ChromeBridgePort:
        return self._bridge if self._bridge is not None else chrome_bridge()

    @property
    def sees_all_tabs(self) -> bool:
        """Kwami's extension lists every tab; Claude in Chrome only Jarvis's own group."""
        return bool(getattr(self._bridge, "sees_all_tabs", True))

    @property
    def place(self) -> str:
        """Where Jarvis's pages open, said for the model."""
        return str(getattr(self._bridge, "place", "Jarvis's window of the user's own Chrome"))

    async def _port(self) -> ChromeBridgePort:
        """The way in, chosen at the first call when none was given: the one that answers."""
        if self._bridge is None:
            self._bridge = await user_browser()
        return self.bridge

    async def _call(self, cmd: str, timeout: float = 20.0, **args: Any) -> Any:
        port = await self._port()
        try:
            return await port.call(cmd, args, timeout=timeout)
        except BridgeUnavailableError:
            self._forget()
            if self._picks:
                self._bridge = None
            raise
        except BridgeError as e:
            if "No tab with id" in str(e):
                self._forget()
            raise

    def _forget(self) -> None:
        self._tab_id = None
        self._owned = False
        self._stop_frames()

    def _ensure_active(self) -> int:
        if self._tab_id is None:
            raise RuntimeError("No page is open in the user's Chrome. Use navigate_to first.")
        return self._tab_id

    # -- lifecycle ---------------------------------------------------------------------

    async def start(self, user_id: str, url: str | None = None) -> str:
        """Open `url` in Jarvis's window of the user's Chrome, or go there if one is open."""
        if not (url or "").strip():
            raise ValueError("an address to open is needed")
        url = validate_url(url or "")
        if self.is_active:
            await self.navigate(url)
            return ""
        info = await self._call("open", OPEN_TIMEOUT_SECONDS, url=url, focus=False)
        self._adopt(info, owned=True)
        await self._publish_session_event("open", url=self._current_url)
        await self._start_frames()
        logger.info("Working in the user's Chrome: %s", self._current_url[:80])
        return ""

    def _adopt(self, info: Any, *, owned: bool) -> None:
        if not isinstance(info, dict) or not isinstance(info.get("id"), int):
            raise BridgeError("the user's Chrome did not say which tab it opened")
        self._tab_id = int(info["id"])
        self._owned = owned
        self._current_url = str(info.get("url") or "")
        self._title = str(info.get("title") or "")

    async def list_tabs(self) -> list[dict[str, Any]]:
        """Every tab open in the user's Chrome, as the extension sees them."""
        result = await self._call("tabs")
        tabs = result.get("tabs") if isinstance(result, dict) else None
        return [t for t in tabs or [] if isinstance(t, dict)]

    async def use_tab(self, tab_id: int) -> None:
        """Work in one of the user's tabs from now on. It stays theirs: Jarvis never closes it."""
        info = await self._call("info", tabId=int(tab_id))
        jarvis = isinstance(info, dict) and bool(info.get("jarvis"))
        self._stop_frames()
        self._adopt(info, owned=jarvis)
        await self._publish_session_event("open", url=self._current_url, title=self._title)
        await self._start_frames()

    async def close(self, *, quit_browser: bool = False) -> None:
        """Stop working in the tab. `quit_browser` (the user asked) closes it, if it is Jarvis's."""
        tab_id, owned = self._tab_id, self._owned
        self._forget()
        if quit_browser and owned and tab_id is not None:
            with contextlib.suppress(BridgeError):
                await (await self._port()).call("close", {"tabId": tab_id})
        await self._publish_session_event("close")
        self._current_url = ""

    # -- navigation --------------------------------------------------------------------

    async def navigate(self, url: str) -> str:
        tab_id = self._ensure_active()
        url = validate_url(url)
        info = await self._call("navigate", OPEN_TIMEOUT_SECONDS, tabId=tab_id, url=url)
        self._current_url = url
        if isinstance(info, dict):
            self._current_url = str(info.get("url") or url)
            self._title = str(info.get("title") or "")
        await self._publish_session_event("update", url=self._current_url)
        return f"Opened {url} in {self.place}."

    async def go_back(self) -> str:
        await self._call("back", OPEN_TIMEOUT_SECONDS, tabId=self._ensure_active())
        return "Going back to the previous page."

    async def go_forward(self) -> str:
        await self._call("forward", OPEN_TIMEOUT_SECONDS, tabId=self._ensure_active())
        return "Going forward to the next page."

    async def page_address(self) -> str:
        if self._tab_id is None:
            return ""
        try:
            info = await self._call("info", tabId=self._tab_id)
        except BridgeError:
            return self._current_url
        href = str(info.get("url") or "") if isinstance(info, dict) else ""
        if href.startswith(("http://", "https://")):
            self._current_url = href
            return href
        return self._current_url

    async def leave_page(self) -> None:
        """Off a page Jarvis may not be on: its own tab closes, one of theirs is left alone."""
        await self.close(quit_browser=True)

    async def bring_to_front(self) -> None:
        if self._tab_id is None:
            return
        with contextlib.suppress(BridgeError):
            await self._call("activate", tabId=self._tab_id, focus=True)

    # -- the page ------------------------------------------------------------------------

    async def _page(self, max_text: int = MAX_PAGE_TEXT) -> dict[str, Any]:
        page = await self._call("read", tabId=self._ensure_active(), maxText=max_text)
        return page if isinstance(page, dict) else {}

    async def read_page(self) -> str:
        page = await self._page()
        self._title = str(page.get("title") or "")
        parts = [f"Page title: {self._title}", f"Address: {page.get('url') or ''}"]
        text = str(page.get("text") or "")
        if text:
            parts.append(f"\nPage content:\n{text[:READ_TEXT_CHARS]}")
        elements = [e for e in page.get("elements") or [] if isinstance(e, dict)]
        # What is on screen first: that is what the user sees Jarvis look at.
        elements.sort(key=lambda e: (not e.get("inView"), not e.get("visible")))
        if elements:
            parts.append("\nInteractive elements:")
            for el in elements[:READ_ELEMENTS]:
                mark = "✓" if el.get("visible") else "✗"
                parts.append(f'  {el.get("id")} [{el.get("type")}] {mark} "{el.get("label")}"')
        return "\n".join(parts)

    async def read_tab_text(self, tab_id: int, max_text: int = READ_TEXT_CHARS) -> dict[str, Any]:
        """Another tab's title, address and text, without working in it."""
        page = await self._call("read", tabId=int(tab_id), maxText=max_text)
        return page if isinstance(page, dict) else {}

    async def click(
        self,
        element_id: str = "",
        description: str = "",
        refuse: Callable[[str], str | None] | None = None,
    ) -> str:
        tab_id = self._ensure_active()
        elements = [e for e in (await self._page(1)).get("elements") or [] if isinstance(e, dict)]
        target = pick_element(elements, element_id, description)
        if not target:
            return f"Could not find element to click: {element_id or description}"
        label = str(target.get("label") or "")
        refusal = refuse(label) if refuse else None
        if refusal:
            return refusal
        try:
            await self._call("click", tabId=tab_id, elementId=str(target["id"]))
        except BridgeError as e:
            return explain(e)
        await asyncio.sleep(SETTLE_SECONDS)
        return f'Clicked on "{label[:60]}" ({target["id"]}).'

    async def type_text(
        self,
        text: str,
        element_id: str = "",
        description: str = "",
        clear_first: bool = True,
    ) -> str:
        tab_id = self._ensure_active()
        target_id = ""
        if element_id or description:
            elements = [
                e for e in (await self._page(1)).get("elements") or [] if isinstance(e, dict)
            ]
            target = pick_element(elements, element_id, description)
            if not target:
                return f"Could not find element to click: {element_id or description}"
            target_id = str(target["id"])
        try:
            await self._call(
                "type", tabId=tab_id, elementId=target_id, text=text, clear=clear_first
            )
        except BridgeError as e:
            return explain(e)
        return f"Typed '{text[:50]}' into the field."

    async def press_key(self, key: str) -> str:
        try:
            result = await self._call("key", tabId=self._ensure_active(), key=key)
        except BridgeError as e:
            return explain(e)
        if isinstance(result, dict) and result.get("submitted"):
            await asyncio.sleep(SETTLE_SECONDS)
            return f"Pressed '{key}': the form was sent."
        return f"Pressed '{key}'."

    async def scroll(self, direction: str = "down") -> str:
        direction = "up" if (direction or "").lower() == "up" else "down"
        await self._call("scroll", tabId=self._ensure_active(), direction=direction)
        return f"Scrolled {direction}."

    async def evaluate_js(self, expression: str) -> str:
        return (
            "Jarvis does not run scripts in the user's own Chrome: the page holds their "
            "sessions. Use read_navigation_page and click_in_navigation instead."
        )

    async def run_named(self, action: str, level: float = 0.0) -> str:
        """One of the extension's own page actions (the media tools'), by name."""
        result = await self._call(
            "media", tabId=self._ensure_active(), action=action, level=float(level)
        )
        return str(result.get("result") if isinstance(result, dict) else result)

    async def _probe(self) -> dict[str, Any]:
        probe = await self._call("probe", tabId=self._ensure_active())
        return probe if isinstance(probe, dict) else {}

    async def on_sign_in_page(self) -> bool:
        try:
            probe = await self._probe()
        except (BridgeError, RuntimeError):
            return False
        if _SIGN_IN_ADDRESS.search(str(probe.get("address") or "")):
            return True
        return bool(probe.get("password"))

    async def sign_in_refused(self) -> bool:
        try:
            probe = await self._probe()
        except (BridgeError, RuntimeError):
            return False
        return bool(_SIGN_IN_REFUSED.search(str(probe.get("head") or "")))

    # -- the app's panel -------------------------------------------------------------

    async def _publish_session_event(
        self, action: str, url: str | None = None, title: str | None = None
    ) -> None:
        if not self._room:
            return
        message: dict[str, Any] = {"type": "browser_session", "action": action}
        if action == "open":
            message.update(liveUrl="", vendor=VENDOR, persistent=True)
        if url:
            message["url"] = url
        if title:
            message["title"] = title
        try:
            await self._room.local_participant.publish_data(
                json.dumps(message).encode("utf-8"), reliable=True
            )
        except Exception as e:
            logger.debug("Could not publish browser_session %s: %s", action, e)

    def _stop_frames(self) -> None:
        task, self._frame_task = self._frame_task, None
        if task is not None:
            task.cancel()

    async def _start_frames(self) -> None:
        self._stop_frames()
        if self._frames_wanted and self._room is not None and self._tab_id is not None:
            self._frame_task = asyncio.create_task(self._stream_frames(), name="user_chrome_frames")

    async def set_frames_wanted(self, wanted: bool) -> None:
        if wanted == self._frames_wanted:
            return
        self._frames_wanted = wanted
        if wanted:
            await self._start_frames()
        else:
            self._stop_frames()

    async def _stream_frames(self) -> None:
        """A picture of the tab while it is the one in front of its window, sent when it changed."""
        last = ""
        port = await self._port()
        while self._tab_id is not None:
            pause = FRAME_INTERVAL_SECONDS
            try:
                shot = await port.call(
                    "capture",
                    {"tabId": self._tab_id, "quality": FRAME_QUALITY},
                    timeout=CAPTURE_TIMEOUT_SECONDS,
                )
                frame = str(shot.get("jpeg") or "") if isinstance(shot, dict) else ""
                if frame and frame != last:
                    await self._publish_frame(base64.b64decode(frame))
                    last = frame
            except BridgeUnavailableError:
                return
            except Exception as e:
                # Behind other windows, minimised, behind another tab or mid-navigation.
                if "not-visible" in str(e):
                    pause = FRAME_HIDDEN_SECONDS
                logger.debug("No picture of the user's tab: %s", e)
            await asyncio.sleep(pause)

    async def _publish_frame(self, jpeg: bytes) -> None:
        if not self._room:
            return
        writer = await self._room.local_participant.stream_bytes(
            "frame.jpg", total_size=len(jpeg), mime_type="image/jpeg", topic=FRAME_TOPIC
        )
        await writer.write(jpeg)
        await writer.aclose()
