"""The user's own Chrome through Claude in Chrome, the way Claude Code reaches it (2026-11-18).

Claude Code serves Claude in Chrome's tools as an MCP server: `claude --claude-in-chrome-mcp`
is a stdio server that talks to the Claude extension in the user's Chrome, paired by
claude.ai account. Jarvis runs that same server with the Claude Code this PC already has
(the Agent SDK's own copy, the one the work thread thinks with), and answers the commands
of Kwami's extension (`chrome_bridge`) with its tools. `UserChromeSession` works over
either without knowing which.

What it needs from the user: the Claude extension in their Chrome signed in to the same
claude.ai account as this PC's Claude Code. Nothing else is installed.

Where it differs from Kwami's extension:

* **Jarvis sees only its own tab group**, as Claude does. A tab of the user's reaches
  Jarvis when they drag it into that group.
* **Site permissions are the extension's own**, on top of the user's list in Accesos: a
  site the extension asks about waits for the user to allow it there.
* **No password, code or card is ever typed**: the field is checked on the page's last
  reading before anything is typed, as Kwami's extension does in the page.
* **Page actions are Kwami's**: the media tools' fixed snippets (`media_scripts`) go
  through the page tool by name. No text from a model is ever run as code.
* **Pictures for the panel cross Anthropic's bridge**, so at most one every
  `CAPTURE_EVERY_SECONDS`.

`KWAMI_CLAUDE_IN_CHROME=off` turns it off; `KWAMI_CLAUDE_CLI` picks another `claude`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..utils.logging import get_logger
from .chrome_bridge import CALL_TIMEOUT_SECONDS, BridgeError, BridgeUnavailableError
from .desktop import foreground_window, return_focus
from .media_scripts import media_script

logger = get_logger("browser.claude_chrome")

SWITCH = "KWAMI_CLAUDE_IN_CHROME"
CLI_ENV = "KWAMI_CLAUDE_CLI"
MCP_FLAG = "--claude-in-chrome-mcp"
#: Starting Claude Code's server takes a second or two; its first answer, a few more.
START_TIMEOUT_SECONDS = 30.0
#: After it failed to start, it is not tried again for this long.
RETRY_SECONDS = 60.0
#: How long its server has to stop when told to.
STOP_SECONDS = 5.0
READY_TIMEOUT_SECONDS = 8.0
#: "Is the Claude extension paired?" is trusted this long when it is, and when it is not.
READY_TTL_SECONDS = 5.0
NOT_READY_TTL_SECONDS = 15.0
CAPTURE_EVERY_SECONDS = 2.5
SCREENSHOT_SCALE = 0.5
TREE_CHARS = 20000
HEAD_CHARS = 2000
#: Where a scroll lands when the page's size is not known yet.
DEFAULT_VIEWPORT = (1280, 800)

#: What the server answers, without an error flag, when the extension is not paired.
_NOT_CONNECTED = re.compile(
    r"\s*(?:\[[^\]]*\]\s*)?(?:Browser extension is not connected|Claude in Chrome isn't paired)"
)
_CREATED_TAB = re.compile(r"Tab ID:\s*(\d+)")
_TAB_BULLET = re.compile(r'tabId (\d+): "(.*)" \("(.*)"\)\s*$', re.MULTILINE)
_EXECUTED_ON = re.compile(r"Executed on tabId:\s*(\d+)")
_VIEWPORT = re.compile(r"^Viewport:\s*(\d+)x(\d+)", re.MULTILINE)
_ELEMENT = re.compile(
    r'^\s*(?P<role>[\w-]+)(?: "(?P<label>(?:[^"\\]|\\.)*)")? \[(?P<ref>ref_\d+)\](?P<attrs>.*)$'
)
_ATTRIBUTE = re.compile(r'(\w[\w-]*)="((?:[^"\\]|\\.)*)"')
_SECRET_AUTOCOMPLETE = re.compile(r"password|one-time-code|^cc-", re.IGNORECASE)
_FIELD_ROLES = {"textbox", "searchbox", "combobox", "spinbutton"}


@dataclass
class ToolResult:
    """What one of Claude in Chrome's tools answered: its text, its pictures (base64)."""

    text: str = ""
    images: list[str] = field(default_factory=list)
    error: bool = False


class McpTools(Protocol):
    async def call_tool(
        self, name: str, arguments: dict[str, Any], timeout: float
    ) -> ToolResult: ...

    async def close(self) -> None: ...


Connector = Callable[[], Awaitable[McpTools]]


def claude_in_chrome_enabled() -> bool:
    return os.environ.get(SWITCH, "on").strip().lower() not in ("off", "0", "false", "no")


def claude_cli() -> str | None:
    """The `claude` to run: the configured one, the Agent SDK's own, or the one on PATH."""
    configured = os.environ.get(CLI_ENV, "").strip()
    if configured:
        return configured if Path(configured).is_file() else None
    try:
        import claude_agent_sdk
    except ImportError:
        return shutil.which("claude")
    name = "claude.exe" if os.name == "nt" else "claude"
    bundled = Path(claude_agent_sdk.__file__).parent / "_bundled" / name
    return str(bundled) if bundled.is_file() else shutil.which("claude")


class StdioTools:
    """Claude Code's Claude in Chrome server, run as a child process and spoken to over MCP.

    The MCP client lives in one task of its own (its streams are opened and closed in the
    same task); any task may call a tool while it runs.
    """

    def __init__(self, command: str, args: list[str] | None = None) -> None:
        self._command = command
        self._args = list(args if args is not None else [MCP_FLAG])
        self._session: Any = None
        self._up = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._failure: BaseException | None = None

    async def start(self, timeout: float = START_TIMEOUT_SECONDS) -> StdioTools:
        self._task = asyncio.create_task(self._run(), name="claude_in_chrome_mcp")
        try:
            async with asyncio.timeout(timeout):
                await self._up.wait()
        except TimeoutError as e:
            await self.close()
            raise BridgeUnavailableError("Claude in Chrome's server did not start in time") from e
        if self._session is None:
            raise BridgeUnavailableError(f"Claude in Chrome's server stopped: {self._failure}")
        return self

    async def _run(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command=self._command, args=self._args)
        # Its log is of no use to the user; the server's answers say what went wrong.
        errlog = await asyncio.to_thread(open, os.devnull, "w")
        try:
            async with (
                stdio_client(params, errlog=errlog) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                self._session = session
                self._up.set()
                await self._stop.wait()
        except Exception as e:
            self._failure = e
            logger.info("Claude in Chrome's server stopped: %s", e)
        finally:
            self._session = None
            self._up.set()
            errlog.close()

    async def call_tool(self, name: str, arguments: dict[str, Any], timeout: float) -> ToolResult:
        session = self._session
        if session is None:
            raise BridgeUnavailableError("Claude in Chrome's server is not running")
        result = await session.call_tool(name, arguments, read_timeout_seconds=timeout)
        texts: list[str] = []
        images: list[str] = []
        for item in result.content:
            kind = getattr(item, "type", "")
            if kind == "text":
                texts.append(str(item.text))
            elif kind == "image":
                images.append(str(item.data))
        return ToolResult("\n".join(texts), images, bool(result.is_error))

    async def close(self) -> None:
        self._stop.set()
        task, self._task = self._task, None
        if task is not None:
            # It ends on its own once told to; one stuck starting up is cancelled.
            await asyncio.wait({task}, timeout=STOP_SECONDS)
            if not task.done():
                task.cancel()


async def _connect_claude_code() -> McpTools:
    cli = claude_cli()
    if cli is None:
        raise BridgeUnavailableError("there is no Claude Code on this computer")
    return await StdioTools(cli).start()


def _first_json(text: str, marker: str) -> Any:
    start = text.find(marker)
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return None
    return value


def parse_tabs(text: str) -> list[dict[str, Any]]:
    """The tabs of Jarvis's group, from a tool's answer (its JSON, or its tab context)."""
    context = _first_json(text, '{"availableTabs"')
    if isinstance(context, dict) and isinstance(context.get("availableTabs"), list):
        tabs = [
            {
                "id": int(t["tabId"]),
                "title": str(t.get("title") or ""),
                "url": str(t.get("url") or ""),
            }
            for t in context["availableTabs"]
            if isinstance(t, dict) and isinstance(t.get("tabId"), int)
        ]
    else:
        tabs = [
            {"id": int(tab_id), "title": title, "url": url}
            for tab_id, title, url in _TAB_BULLET.findall(text)
        ]
    for tab in tabs:
        tab.update(jarvis=True, focused=False, audible=False)
    return tabs


def parse_elements(text: str) -> list[dict[str, Any]]:
    """The interactive elements of a page, from its accessibility tree."""
    elements: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = _ELEMENT.match(line)
        if not match:
            continue
        attrs = dict(_ATTRIBUTE.findall(match["attrs"]))
        role = match["role"]
        kind = attrs.get("type", "").lower()
        label = (match["label"] or attrs.get("placeholder") or attrs.get("href") or "").strip()
        if not label and role in _FIELD_ROLES:
            label = "field"
        if not label:
            continue
        secret = kind == "password" or bool(
            _SECRET_AUTOCOMPLETE.search(attrs.get("autocomplete", ""))
        )
        elements.append(
            {
                "id": match["ref"],
                "type": f"input:{kind}" if kind and role in _FIELD_ROLES else role,
                "label": label[:80],
                "visible": True,
                "inView": True,
                "secret": secret,
            }
        )
    return elements


def split_page_text(text: str) -> tuple[str, str, str]:
    """`get_page_text`'s answer: its title, its address, and the text under the `---` line."""
    head, sep, body = text.partition("\n---\n")
    if not sep:
        return "", "", text
    title = re.search(r"^Title:\s*(.*)$", head, re.MULTILINE)
    url = re.search(r"^URL:\s*(.*)$", head, re.MULTILINE)
    return (title[1].strip() if title else "", url[1].strip() if url else "", body)


class ClaudeChromePort:
    """Kwami's bridge commands, answered with Claude in Chrome's tools."""

    #: Jarvis sees its own group only, not every tab of the user's.
    sees_all_tabs = False
    place = "Jarvis's tab group of the user's own Chrome (through the Claude extension)"

    def __init__(
        self, connect: Connector | None = None, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._connect = connect or _connect_claude_code
        self._clock = clock
        self._tools: McpTools | None = None
        self._lock = asyncio.Lock()
        self._failed_at = float("-inf")
        self._ready = False
        self._ready_at = float("-inf")
        #: Per tab: the elements of its last reading, its size, the field clicked last.
        self._elements: dict[int, dict[str, dict[str, Any]]] = {}
        self._viewport: dict[int, tuple[int, int]] = {}
        self._clicked: dict[int, str] = {}
        self._frames: dict[int, tuple[float, str]] = {}
        self._commands: dict[str, Callable[[dict[str, Any], float], Awaitable[Any]]] = {
            "status": self._status,
            "tabs": self._list,
            "info": self._info,
            "open": self._open,
            "navigate": self._navigate,
            "back": self._back,
            "forward": self._forward,
            "activate": self._activate,
            "close": self._close,
            "read": self._read,
            "click": self._click,
            "type": self._type,
            "key": self._key,
            "scroll": self._scroll,
            "probe": self._probe,
            "media": self._media,
            "capture": self._capture,
        }

    # -- the server ----------------------------------------------------------------------

    async def _server(self) -> McpTools:
        async with self._lock:
            if self._tools is None:
                if self._clock() - self._failed_at < RETRY_SECONDS:
                    raise BridgeUnavailableError("Claude in Chrome could not be started")
                try:
                    self._tools = await self._connect()
                except Exception as e:
                    self._failed_at = self._clock()
                    logger.info("Claude in Chrome is not available: %s", e)
                    raise BridgeUnavailableError(f"Claude in Chrome could not start: {e}") from e
            return self._tools

    async def _tool(
        self, name: str, args: dict[str, Any] | None = None, timeout: float = CALL_TIMEOUT_SECONDS
    ) -> ToolResult:
        tools = await self._server()
        try:
            async with asyncio.timeout(timeout + 2):
                result = await tools.call_tool(name, args or {}, timeout)
        except BridgeUnavailableError:
            self._tools = None
            raise
        except TimeoutError as e:
            raise BridgeError(f"Claude in Chrome did not answer '{name}' in time") from e
        except Exception as e:
            raise BridgeError(f"Claude in Chrome failed on '{name}': {e}") from e
        if _NOT_CONNECTED.match(result.text):
            self._ready, self._ready_at = False, self._clock()
            raise BridgeUnavailableError(result.text.split("\n", 1)[0][:300])
        if result.error:
            raise BridgeError(result.text[:400] or f"Claude in Chrome refused '{name}'")
        return result

    async def ready(self) -> bool:
        """Whether the Claude extension is paired with this PC's Claude Code right now."""
        ttl = READY_TTL_SECONDS if self._ready else NOT_READY_TTL_SECONDS
        if self._clock() - self._ready_at < ttl:
            return self._ready
        try:
            result = await self._tool("list_connected_browsers", timeout=READY_TIMEOUT_SECONDS)
            browsers = _first_json(result.text, "[")
            ready = isinstance(browsers, list) and len(browsers) > 0
        except BridgeError:
            ready = False
        self._ready, self._ready_at = ready, self._clock()
        return ready

    async def call(
        self, cmd: str, args: dict[str, Any] | None = None, timeout: float = CALL_TIMEOUT_SECONDS
    ) -> Any:
        handler = self._commands.get(cmd)
        if handler is None:
            raise BridgeError(f"Claude in Chrome has no '{cmd}'")
        return await handler(args or {}, timeout)

    async def close(self) -> None:
        tools, self._tools = self._tools, None
        if tools is not None:
            await tools.close()

    # -- tabs ----------------------------------------------------------------------------

    async def _tabs(self) -> list[dict[str, Any]]:
        return parse_tabs((await self._tool("tabs_context_mcp")).text)

    async def _status(self, args: dict[str, Any], timeout: float) -> dict[str, Any]:
        return {"connected": await self.ready()}

    async def _list(self, args: dict[str, Any], timeout: float) -> dict[str, Any]:
        return {"tabs": await self._tabs(), "scope": "group"}

    async def _tab(self, tab_id: int) -> dict[str, Any]:
        for tab in await self._tabs():
            if tab["id"] == tab_id:
                return tab
        raise BridgeError(f"No tab with id: {tab_id}")

    async def _info(self, args: dict[str, Any], timeout: float) -> dict[str, Any]:
        return await self._tab(int(args["tabId"]))

    @staticmethod
    def _after(text: str, tab_id: int, url: str) -> dict[str, Any]:
        """The tab a page tool acted on, from the context it answered with."""
        for tab in parse_tabs(text):
            if tab["id"] == tab_id:
                return tab
        return {"id": tab_id, "url": url, "title": "", "jarvis": True}

    async def _open(self, args: dict[str, Any], timeout: float) -> dict[str, Any]:
        url = str(args["url"])
        tabs = await self._tabs()
        before = await asyncio.to_thread(foreground_window)
        try:
            if tabs:
                created = await self._tool("tabs_create_mcp")
                found = _CREATED_TAB.search(created.text)
                if not found:
                    raise BridgeError("Claude in Chrome did not say which tab it opened")
                tab_id = int(found[1])
            else:
                # Its first tab comes with the group, in a window of its own, which Chrome
                # brings forward: the focus goes back to where the user was.
                context = await self._tool("tabs_context_mcp", {"createIfEmpty": True})
                fresh = parse_tabs(context.text)
                if not fresh:
                    raise BridgeError("Claude in Chrome did not open a tab")
                tab_id = int(fresh[0]["id"])
        finally:
            await asyncio.to_thread(return_focus, before)
        return await self._go(tab_id, url, timeout)

    async def _go(self, tab_id: int, url: str, timeout: float) -> dict[str, Any]:
        result = await self._tool("navigate", {"url": url, "tabId": tab_id}, timeout)
        self._elements.pop(tab_id, None)
        self._clicked.pop(tab_id, None)
        return self._after(result.text, tab_id, url)

    async def _navigate(self, args: dict[str, Any], timeout: float) -> dict[str, Any]:
        return await self._go(int(args["tabId"]), str(args["url"]), timeout)

    async def _back(self, args: dict[str, Any], timeout: float) -> dict[str, Any]:
        return await self._go(int(args["tabId"]), "back", timeout)

    async def _forward(self, args: dict[str, Any], timeout: float) -> dict[str, Any]:
        return await self._go(int(args["tabId"]), "forward", timeout)

    async def _activate(self, args: dict[str, Any], timeout: float) -> dict[str, Any]:
        # Claude in Chrome has no "bring this tab forward": the user finds it in its group.
        return {"ok": True}

    async def _close(self, args: dict[str, Any], timeout: float) -> dict[str, Any]:
        tab_id = int(args["tabId"])
        await self._tool("tabs_close_mcp", {"tabId": tab_id})
        for cache in (self._elements, self._viewport, self._clicked, self._frames):
            cache.pop(tab_id, None)
        return {"ok": True}

    # -- the page ------------------------------------------------------------------------

    async def _tree(self, tab_id: int) -> list[dict[str, Any]]:
        tree = await self._tool(
            "read_page", {"tabId": tab_id, "filter": "interactive", "max_chars": TREE_CHARS}
        )
        size = _VIEWPORT.search(tree.text)
        if size:
            self._viewport[tab_id] = (int(size[1]), int(size[2]))
        elements = parse_elements(tree.text)
        self._elements[tab_id] = {e["id"]: e for e in elements}
        return elements

    async def _text(self, tab_id: int) -> tuple[str, str, str]:
        return split_page_text((await self._tool("get_page_text", {"tabId": tab_id})).text)

    async def _read(self, args: dict[str, Any], timeout: float) -> dict[str, Any]:
        tab_id = int(args["tabId"])
        max_text = int(args.get("maxText") or 8000)
        page: dict[str, Any] = {"title": "", "url": "", "text": ""}
        if max_text > 1:
            title, url, text = await self._text(tab_id)
            page.update(title=title, url=url, text=text[:max_text])
        page["elements"] = await self._tree(tab_id)
        return page

    async def _click(self, args: dict[str, Any], timeout: float) -> dict[str, Any]:
        tab_id, ref = int(args["tabId"]), str(args["elementId"])
        await self._tool("computer", {"action": "left_click", "ref": ref, "tabId": tab_id})
        self._clicked[tab_id] = ref
        return {"ok": True}

    async def _type(self, args: dict[str, Any], timeout: float) -> dict[str, Any]:
        tab_id = int(args["tabId"])
        ref = str(args.get("elementId") or "")
        target = ref or self._clicked.get(tab_id, "")
        if not target:
            raise BridgeError("no-field")
        known = self._elements.get(tab_id, {})
        if target not in known:
            # Typing goes only where the last reading shows what the field is.
            await self._tree(tab_id)
            known = self._elements.get(tab_id, {})
        element = known.get(target)
        if element is None:
            raise BridgeError("not-found")
        if element["secret"]:
            raise BridgeError("secret-field")
        text = str(args.get("text") or "")
        if args.get("clear", True):
            with contextlib.suppress(BridgeError):
                await self._tool("form_input", {"ref": target, "value": text, "tabId": tab_id})
                return {"ok": True}
        await self._tool("computer", {"action": "left_click", "ref": target, "tabId": tab_id})
        await self._tool("computer", {"action": "type", "text": text, "tabId": tab_id})
        return {"ok": True}

    async def _key(self, args: dict[str, Any], timeout: float) -> dict[str, Any]:
        key = "space" if str(args["key"]) in (" ", "Space") else str(args["key"])
        await self._tool("computer", {"action": "key", "text": key, "tabId": int(args["tabId"])})
        return {"ok": True}

    async def _scroll(self, args: dict[str, Any], timeout: float) -> dict[str, Any]:
        tab_id = int(args["tabId"])
        width, height = self._viewport.get(tab_id, DEFAULT_VIEWPORT)
        await self._tool(
            "computer",
            {
                "action": "scroll",
                "coordinate": [width // 2, height // 2],
                "scroll_direction": "up" if args.get("direction") == "up" else "down",
                "scroll_amount": 5,
                "tabId": tab_id,
            },
        )
        return {"ok": True}

    async def _probe(self, args: dict[str, Any], timeout: float) -> dict[str, Any]:
        tab_id = int(args["tabId"])
        _title, url, text = await self._text(tab_id)
        elements = await self._tree(tab_id)
        href = url or str((await self._tab(tab_id)).get("url") or "")
        address = re.sub(r"^[a-z]+://", "", href).split("?", 1)[0].split("#", 1)[0]
        return {
            "href": href,
            "address": address,
            "password": any(e["type"] == "input:password" for e in elements),
            "head": text[:HEAD_CHARS],
        }

    async def _media(self, args: dict[str, Any], timeout: float) -> dict[str, Any]:
        script = media_script(str(args.get("action") or ""), float(args.get("level") or 0.0))
        if script is None:
            return {"result": "unknown-action"}
        result = await self._tool(
            "javascript_tool",
            {"action": "javascript_exec", "text": script, "tabId": int(args["tabId"])},
        )
        return {"result": result.text.split("\n\nTab Context", 1)[0].strip()}

    async def _capture(self, args: dict[str, Any], timeout: float) -> dict[str, Any]:
        tab_id = int(args["tabId"])
        taken_at, jpeg = self._frames.get(tab_id, (float("-inf"), ""))
        if self._clock() - taken_at < CAPTURE_EVERY_SECONDS:
            return {"jpeg": jpeg}
        shot = await self._tool(
            "computer",
            {"action": "screenshot", "scale": SCREENSHOT_SCALE, "tabId": tab_id},
            timeout,
        )
        if not shot.images:
            raise BridgeError("not-visible")
        self._frames[tab_id] = (self._clock(), shot.images[0])
        return {"jpeg": shot.images[0]}


_port: ClaudeChromePort | None = None


def claude_chrome() -> ClaudeChromePort | None:
    """This process's way to Claude in Chrome, or None where it is off."""
    global _port
    if not claude_in_chrome_enabled():
        return None
    if _port is None:
        _port = ClaudeChromePort()
    return _port


async def close_claude_chrome() -> None:
    global _port
    port, _port = _port, None
    if port is not None:
        await port.close()
