"""A real Chrome window on the machine the agent runs on.

Only for an agent running on the user's own computer, and only when asked for
with `KWAMI_BROWSER_PROVIDER=local`: on a server it would open a browser nobody
can see. It costs nothing and does not depend on a vendor's balance, which is
what a personal install wants when the cloud browsers have no credit (a 402 from
Browser Use, 2026-11-02).

Three things differ from a rented browser:

* **The profile is its own directory**, never the user's everyday one: Chrome
  refuses remote debugging on its default profile, and handing the agent every
  login the user has would be a much bigger grant than opening a page. Logins
  made in this window stay in this directory for next time.
* **The port is Chrome's choice.** Port 0 makes it write `DevToolsActivePort`
  in the profile, which is also how a window left open is found again: starting
  a second Chrome on the same profile only hands over to the first and exits.
* **The window belongs to the user.** Hanging up or going idle leaves it open;
  only an explicit close (`quit`) shuts it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from ..adapters.http import shared_client
from ..utils.logging import get_logger
from .launched import LOCAL, MISSING, BrowserVendorError, LaunchedBrowser

logger = get_logger("browser.local")

#: Where profiles live: the same folder as the rest of the local install.
PROFILES_DIR = Path.home() / ".jarvis" / "browser-profiles"
START_TIMEOUT_SECONDS = 20.0
_POLL_SECONDS = 0.25

_WINDOWS_PATHS = (
    r"Google\Chrome\Application\chrome.exe",
    r"Microsoft\Edge\Application\msedge.exe",
)
_MAC_PATHS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
)
_PATH_NAMES = ("chrome", "google-chrome", "chromium", "chromium-browser", "msedge")


def find_chrome() -> str:
    """The Chrome (or Edge) to run, or "" when this machine has neither."""
    for relative in _WINDOWS_PATHS:
        for root_var in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
            root = os.environ.get(root_var)
            if root and (Path(root) / relative).is_file():
                return str(Path(root) / relative)
    for candidate in _MAC_PATHS:
        if Path(candidate).is_file():
            return candidate
    for name in _PATH_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return ""


def profile_dir_for(user_id: str) -> Path:
    """This user's profile directory. Hashed: ids do not belong in file names."""
    return PROFILES_DIR / hashlib.sha256(user_id.encode()).hexdigest()[:16]


def _active_port(profile: Path) -> int | None:
    try:
        return int((profile / "DevToolsActivePort").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


async def _version_info(port: int) -> dict | None:
    """What Chrome says about itself on `port`, or None if nothing answers."""
    try:
        r = await shared_client().get(f"http://127.0.0.1:{port}/json/version", timeout=2.0)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None
    return data if isinstance(data, dict) else None


async def wait_for_port(process: subprocess.Popen, profile: Path) -> int | None:
    """The debugging port once Chrome answers on it, or None if it never does."""
    deadline = time.monotonic() + START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        port = _active_port(profile)
        if port and await _version_info(port):
            return port
        if process.poll() not in (None, 0):
            return None
        await asyncio.sleep(_POLL_SECONDS)
    return None


class LocalChromeProvider:
    """Opens, or finds again, the user's Chrome window on this machine."""

    def __init__(self, chrome_path: str | None = None) -> None:
        self._chrome = chrome_path
        self._ports: dict[str, int] = {}

    @property
    def vendor(self) -> str:
        return LOCAL

    async def _start_chrome(self, chrome: str, profile: Path) -> int:
        (profile / "DevToolsActivePort").unlink(missing_ok=True)
        process = await asyncio.to_thread(
            subprocess.Popen,
            [
                chrome,
                "--remote-debugging-port=0",
                f"--user-data-dir={profile}",
                "--no-first-run",
                "--no-default-browser-check",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        port = await wait_for_port(process, profile)
        if port:
            return port
        raise BrowserVendorError(LOCAL, MISSING, "Chrome did not start on this computer.")

    async def launch(self, user_id: str) -> LaunchedBrowser:
        if not (user_id or "").strip():
            # The profile holds logins; a shared one would hand them to the next user.
            raise ValueError("a browser session needs a real user id")

        chrome = self._chrome if self._chrome is not None else find_chrome()
        if not chrome:
            raise BrowserVendorError(
                LOCAL, MISSING, "Neither Chrome nor Edge is installed on this computer."
            )

        profile = profile_dir_for(user_id)
        profile.mkdir(parents=True, exist_ok=True)

        port = _active_port(profile)
        if port and await _version_info(port):
            logger.info("Reusing the Chrome window already open on port %s", port)
        else:
            port = await self._start_chrome(chrome, profile)
            logger.info("Started Chrome for this user on port %s", port)

        browser_id = f"local-{port}"
        self._ports[browser_id] = port
        return LaunchedBrowser(
            browser_id=browser_id,
            vendor=LOCAL,
            cdp_http_url=f"http://127.0.0.1:{port}",
            persistence_id=profile.name,
        )

    async def release(self, browser_id: str) -> None:
        """Leave the window where it is: the user may still be working in it."""

    async def quit(self, browser_id: str) -> None:
        """Close the window, because the user asked for that."""
        port = self._ports.pop(browser_id, None)
        info = await _version_info(port) if port else None
        ws_url = (info or {}).get("webSocketDebuggerUrl")
        if not ws_url:
            return
        import websockets

        async with websockets.connect(ws_url, close_timeout=2) as connection:
            await connection.send(json.dumps({"id": 1, "method": "Browser.close"}))
