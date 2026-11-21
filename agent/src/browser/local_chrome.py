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

Since 2026-11-21 the user does not have to close it. Measured that day: Google refuses the
driven window as soon as its sign-in page loads, even with no program attached and nobody
typing (the debugging port is enough), and lets the user in through the clean one. That
window's address is read through Windows' accessibility layer (`window_reader`), and once
the user has gone through a sign-in page and stayed off one for a few seconds, Jarvis
closes the window as its close button would, so Chrome saves the session first.

**The profile stays Jarvis's own.** The same day, signing in to Google in the clean window
switched on Chrome's account sync, which installed the user's extensions in this profile (a
crypto wallet among them) and would bring their saved passwords to a window a program
drives. Both windows now start with sync and extensions off (`PROFILE_FLAGS`), and the
profile does not save or fill passwords, addresses or cards (`harden_profile`): the user
types their own passwords, and an autofilled one would be Jarvis signing in for them.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from ..adapters.http import shared_client
from ..utils.logging import get_logger
from . import window_reader
from .browser_session import _SIGN_IN_ADDRESS
from .desktop import close_windows, foreground_window, keep_behind, quiet_startup
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
#: How often the clean window's address is read while the user signs in.
ADDRESS_POLL_SECONDS = 1.0
#: Off every sign-in page this long, the user is in: redirects after a sign-in take less.
SIGNED_IN_SECONDS = 4.0
#: Both windows on a profile. Sync off and extensions off keep the user's account out of a
#: browser a program drives (see the module's docstring); the restore bubble would greet them
#: after Jarvis's window was closed with the agent.
PROFILE_FLAGS = (
    "--no-first-run",
    "--no-default-browser-check",
    "--hide-crash-restore-bubble",
    "--disable-sync",
    "--disable-extensions",
)
#: Chrome's own settings, in the profile's Preferences: nothing saved or filled for the user.
QUIET_PREFERENCES: dict[str, Any] = {
    "credentials_enable_service": False,
    "credentials_enable_autosignin": False,
    "profile": {"password_manager_enabled": False},
    "autofill": {"profile_enabled": False, "credit_card_enabled": False},
}
#: What the address of the clean window says when it is how the user was told no.
_REFUSED_ADDRESS = re.compile(r"/signin/rejected|/signin/v2/deniedsigninrejected", re.IGNORECASE)
#: Hosts that only sign people in, whatever the path: accounts.google.com, login.live.com...
_SIGN_IN_HOST = re.compile(
    r"^(?:accounts|login|signin|auth|id|sso|secure|passport|account)\.", re.IGNORECASE
)
#: Signed in: the user went through a sign-in page and the window is off one (and closed).
SIGNED_IN = "signed_in"
#: The window is gone, closed by the user, whether or not they finished.
CLOSED = "closed"
#: Still open when the wait ran out.
STILL_OPEN = "open"

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


def _merge(into: dict[str, Any], wanted: dict[str, Any]) -> bool:
    """`wanted`'s values into `into`, nested; whether anything changed."""
    changed = False
    for key, value in wanted.items():
        if isinstance(value, dict):
            inner = into.get(key)
            if not isinstance(inner, dict):
                inner = {}
                into[key] = inner
                changed = True
            changed = _merge(inner, value) or changed
        elif into.get(key) != value:
            into[key] = value
            changed = True
    return changed


def harden_profile(profile: Path) -> bool:
    """Chrome's password manager and autofill off in this profile. Whether it wrote.

    Called before a Chrome starts on the profile: a running one would overwrite the file.
    A Preferences file that cannot be read is left alone, so Chrome's own is never lost.
    """
    prefs = profile / "Default" / "Preferences"
    try:
        data = json.loads(prefs.read_text(encoding="utf-8"))
    except FileNotFoundError:
        data = {}
    except (OSError, ValueError):
        logger.warning("Could not read the Chrome profile's preferences; left as they are")
        return False
    if not isinstance(data, dict) or not _merge(data, QUIET_PREFERENCES):
        return False
    try:
        prefs.parent.mkdir(parents=True, exist_ok=True)
        prefs.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        logger.warning("Could not write the Chrome profile's preferences", exc_info=True)
        return False
    return True


def _host_and_path(address: str) -> tuple[str, str]:
    """An address bar's text split into host and the rest. Chrome shows it without scheme."""
    rest = re.sub(r"^[a-z][a-z0-9+.-]*://", "", address.strip(), flags=re.IGNORECASE)
    host, _, path = rest.partition("/")
    return host.lower(), "/" + path


def is_web_address(address: str) -> bool:
    """A page's address, not what the user is typing or one of Chrome's own pages."""
    text = (address or "").strip()
    if not text or any(ch.isspace() for ch in text):
        return False
    if re.match(r"^[a-z][a-z0-9+.-]*://", text, re.IGNORECASE) and not re.match(
        r"^https?://", text, re.IGNORECASE
    ):
        return False
    host, _ = _host_and_path(text)
    return "." in host and not host.startswith(".")


def is_sign_in_address(address: str) -> bool:
    """Whether an address is a sign-in page (or a refusal of one)."""
    host, path = _host_and_path(address)
    return bool(
        _SIGN_IN_HOST.match(host)
        or _SIGN_IN_ADDRESS.search(host + path)
        or _REFUSED_ADDRESS.search(path)
    )


def is_refused_address(address: str) -> bool:
    """Whether the site's answer to the sign-in is that this browser may not sign in."""
    return bool(_REFUSED_ADDRESS.search(_host_and_path(address)[1]))


class SignInProgress:
    """Whether the user is in, from the addresses the clean window shows one after another.

    In once they have been on a sign-in page and then off every one for `SIGNED_IN_SECONDS`.
    What they type in the address bar, and Chrome's own pages, say nothing either way. A
    window that never shows a sign-in page is never taken as done: they close it themselves.
    """

    def __init__(self, settle: float | None = None) -> None:
        self._settle = SIGNED_IN_SECONDS if settle is None else settle
        self.seen_sign_in = False
        self.refused = False
        self._off_since: float | None = None

    def see(self, address: str, *, typing: bool, now: float) -> bool:
        if typing or not is_web_address(address):
            return False
        if is_sign_in_address(address):
            self.seen_sign_in = True
            self.refused = is_refused_address(address)
            self._off_since = None
            return False
        if not self.seen_sign_in:
            return False
        if self._off_since is None:
            self._off_since = now
        return now - self._off_since >= self._settle


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
        await asyncio.to_thread(harden_profile, profile)
    process = await asyncio.to_thread(
        subprocess.Popen,
        [chrome, f"--user-data-dir={profile}", *PROFILE_FLAGS, "--new-window", url],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not already:
        _clean_windows[profile.name] = process
    logger.info("Opened a clean window for the user to sign in")


def close_clean_window(user_id: str) -> bool:
    """Close the clean window as its close button would (Chrome saves the session first)."""
    process = _clean_windows.get(profile_dir_for(user_id).name)
    if process is None or process.poll() is not None:
        return False
    return close_windows(process.pid) > 0


async def _start_watch(user_id: str, timeout: float) -> window_reader.AddressWatch | None:
    """A reader of the clean window's address, or None where windows cannot be read."""
    if not window_reader.available():
        return None
    watch = window_reader.AddressWatch(
        profile_dir_for(user_id).name, every=ADDRESS_POLL_SECONDS, limit=timeout + 5
    )
    try:
        await watch.start()
    except Exception:
        logger.warning("Could not read the sign-in window's address", exc_info=True)
        await watch.close()
        return None
    return watch


async def wait_sign_in(user_id: str, timeout: float) -> str:
    """Wait for the user in the clean window: `SIGNED_IN`, `CLOSED` or `STILL_OPEN`.

    Where its address can be read, the window is closed for them once they are in. Where it
    cannot, or when they never meet a sign-in page there, they close it themselves.
    """
    deadline = time.monotonic() + timeout
    watch = await _start_watch(user_id, timeout)
    progress = SignInProgress()
    closed_it = False
    try:
        while clean_window_open(user_id):
            if time.monotonic() >= deadline:
                return STILL_OPEN
            if watch is None:
                await asyncio.sleep(CLEAN_WINDOW_POLL_SECONDS)
                continue
            seen = await watch.next(CLEAN_WINDOW_POLL_SECONDS)
            if seen is None or seen.get("closed") or closed_it:
                continue
            address = str(seen.get("address") or "")
            if progress.see(address, typing=bool(seen.get("typing")), now=time.monotonic()):
                closed_it = await asyncio.to_thread(close_clean_window, user_id)
                if closed_it:
                    logger.info("The user is signed in; closed the clean window for them")
        return SIGNED_IN if closed_it else CLOSED
    finally:
        if watch is not None:
            await watch.close()


async def wait_clean_window_closed(user_id: str, timeout: float) -> bool:
    """Wait until the clean window is gone, closed by the user or, once they are in, by
    Jarvis. False if it is still open at `timeout`."""
    return await wait_sign_in(user_id, timeout) != STILL_OPEN


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
        await asyncio.to_thread(harden_profile, profile)
        focus = await asyncio.to_thread(foreground_window)
        process = await asyncio.to_thread(
            subprocess.Popen,
            [
                chrome,
                "--remote-debugging-port=0",
                f"--user-data-dir={profile}",
                *PROFILE_FLAGS,
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
