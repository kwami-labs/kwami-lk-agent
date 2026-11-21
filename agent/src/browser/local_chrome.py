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

Signing in happens in a **clean window** (2026-11-16). Google, and every site that
signs in through it, refuses a browser that a program drives over the debugging
port ("this browser or app may not be secure"), whoever types the password. So for
a sign-in the window is reopened on the same profile *without* that port: a plain
Chrome that no program controls, where the user signs in themselves. Nothing is
disguised and nothing is typed for them. Until they close it nothing drives the
profile (`SignInWindowOpenError`); once they do, Jarvis opens it again as before and the
session they started is in the profile.
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
from .desktop import foreground_window, keep_behind, quiet_startup
from .launched import LOCAL, MISSING, BrowserVendorError, LaunchedBrowser

logger = get_logger("browser.local")

#: Where profiles live: the same folder as the rest of the local install.
PROFILES_DIR = Path.home() / ".jarvis" / "browser-profiles"
START_TIMEOUT_SECONDS = 20.0
_POLL_SECONDS = 0.25
#: How long Chrome gets to let go of a profile after it is asked to close.
RELEASE_TIMEOUT_SECONDS = 10.0
#: How often a clean window is looked at, to know when the user closed it.
CLEAN_WINDOW_POLL_SECONDS = 2.0

#: Clean windows open now, by profile directory name.
_clean_windows: dict[str, subprocess.Popen] = {}
#: Sending a fresh window back runs beside the start; held here so it is not collected.
_behind: set[asyncio.Task[int]] = set()


class SignInWindowOpenError(ValueError):
    """The user is signing in in a window no program controls: driving it would undo that.

    A ValueError so the vendor failover stops here: another vendor would only open a
    browser without the session the user is creating.
    """

    def __init__(self) -> None:
        super().__init__(
            "the user is signing in in a Chrome window that Jarvis does not control. "
            "Wait until they close it: what they sign in to stays saved for Jarvis."
        )


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


async def window_open(user_id: str) -> bool:
    """Whether this user's Chrome is open now: someone may be working in it."""
    if clean_window_open(user_id):
        return True
    port = _active_port(profile_dir_for(user_id))
    return bool(port and await _version_info(port))


def _profile_in_use(profile: Path, windows: bool | None = None) -> bool:
    """Whether a Chrome still holds this profile, whichever process started it.

    Chrome locks the profile for as long as it runs: `lockfile`, held open, on
    Windows; the `SingletonLock` link elsewhere. A second Chrome on a locked profile
    only hands its address to the first one and exits, which is why the process
    started for a clean window is not enough to know whether the window is there.
    """
    if windows is None:
        windows = os.name == "nt"
    if windows:
        lock = profile / "lockfile"
        if not lock.exists():
            return False
        try:
            lock.unlink()  # Only possible when no Chrome has it open.
        except OSError:
            return True
        return False
    singleton = profile / "SingletonLock"
    return singleton.is_symlink() or singleton.exists()


def clean_window_open(user_id: str) -> bool:
    """Whether the user is still in the clean window opened for them to sign in."""
    profile = profile_dir_for(user_id)
    process = _clean_windows.get(profile.name)
    if process is None:
        return False
    if process.poll() is None or _profile_in_use(profile):
        return True
    _clean_windows.pop(profile.name, None)
    return False


async def _close_debug_window(profile: Path) -> None:
    """Close the Chrome that Jarvis drives on this profile, and wait until it lets go."""
    port = _active_port(profile)
    info = await _version_info(port) if port else None
    ws_url = (info or {}).get("webSocketDebuggerUrl")
    if ws_url:
        import websockets

        try:
            async with websockets.connect(ws_url, close_timeout=2) as connection:
                await connection.send(json.dumps({"id": 1, "method": "Browser.close"}))
        except Exception as e:
            logger.debug("Chrome did not take the close: %s", e)
    deadline = time.monotonic() + RELEASE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if not _profile_in_use(profile) and not (port and await _version_info(port)):
            break
        await asyncio.sleep(_POLL_SECONDS)
    (profile / "DevToolsActivePort").unlink(missing_ok=True)


async def open_clean_window(user_id: str, url: str, chrome: str | None = None) -> None:
    """Open `url` in a Chrome on this user's profile that no program controls.

    The caller has validated `url`. A clean window already open gets the page as a
    new tab, which is what Chrome does with a second start on a profile in use.
    """
    if not (user_id or "").strip():
        raise ValueError("a browser session needs a real user id")
    chrome = chrome if chrome is not None else find_chrome()
    if not chrome:
        raise BrowserVendorError(
            LOCAL, MISSING, "Neither Chrome nor Edge is installed on this computer."
        )
    profile = profile_dir_for(user_id)
    profile.mkdir(parents=True, exist_ok=True)
    already = clean_window_open(user_id)
    if not already:
        await _close_debug_window(profile)
    process = await asyncio.to_thread(
        subprocess.Popen,
        [
            chrome,
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--new-window",
            url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not already:
        _clean_windows[profile.name] = process
    logger.info("Opened a clean window for the user to sign in")


async def wait_clean_window_closed(user_id: str, timeout: float) -> bool:
    """Wait until the user closes the clean window. False if it is still open at `timeout`."""
    deadline = time.monotonic() + timeout
    while clean_window_open(user_id):
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(CLEAN_WINDOW_POLL_SECONDS)
    return True


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
        """Chrome on this profile, started without the focus and kept behind (`desktop`).

        Its page is mirrored on Jarvis's screen, so the window has no business covering
        KWAMI: it comes forward only for the user, on a sign-in page or from the taskbar.
        """
        (profile / "DevToolsActivePort").unlink(missing_ok=True)
        focus = await asyncio.to_thread(foreground_window)
        process = await asyncio.to_thread(
            subprocess.Popen,
            [
                chrome,
                "--remote-debugging-port=0",
                f"--user-data-dir={profile}",
                "--no-first-run",
                "--no-default-browser-check",
                # Killed with the agent, it would greet the user with "Restore pages?".
                "--hide-crash-restore-bubble",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            startupinfo=quiet_startup(),
        )
        port = await wait_for_port(process, profile)
        if port:
            task = asyncio.create_task(asyncio.to_thread(keep_behind, process.pid, focus=focus))
            _behind.add(task)
            task.add_done_callback(_behind.discard)
            return port
        raise BrowserVendorError(LOCAL, MISSING, "Chrome did not start on this computer.")

    async def launch(self, user_id: str) -> LaunchedBrowser:
        if not (user_id or "").strip():
            # The profile holds logins; a shared one would hand them to the next user.
            raise ValueError("a browser session needs a real user id")
        if clean_window_open(user_id):
            raise SignInWindowOpenError()

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
