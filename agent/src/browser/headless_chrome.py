"""Chromium inside the agent's own container, for when no vendor can provide one.

Only present where the image installs it and says so with
`KWAMI_HEADLESS_CHROMIUM`. It needs no account and no balance, which is the point:
it is the last vendor the failover tries, so kwami.io can still browse when
Browser Use answers 402.

What it gives up against a rented browser:

* **Nothing is remembered.** Each launch gets a new profile, deleted on release:
  this container serves every user, so logins must not outlive the session.
* **No live view.** The panel gets screenshots instead (`CloudBrowserSession`).

What it risks is in docs/security.md: the browser shares a container with the
agent's credentials. So it starts with an environment holding no secrets, inside
Chromium's sandbox, and its connections go through `EgressProxy`. Where the
container forbids the sandbox it does not start, and the failover says so, unless
an operator accepts that with KWAMI_HEADLESS_ALLOW_NO_SANDBOX.
"""

from __future__ import annotations

import asyncio
import atexit
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from ..utils.logging import get_logger
from .egress_proxy import EgressProxy
from .launched import BUSY, HEADLESS, MISSING, BrowserVendorError, LaunchedBrowser
from .local_chrome import wait_for_port

logger = get_logger("browser.headless")

_STOP_TIMEOUT_SECONDS = 5.0

_PROC = Path("/proc")
_PROFILE_PREFIX = "kwami-browser-"
#: Next to the profile, naming the Chromium using it. Every worker in the container
#: reads these: each job is its own process, and they share the machine.
_PID_FILE = "kwami-browser.pid"
#: A profile with no pid yet is being started by another worker, if it is this new.
_STARTING_SECONDS = 60.0
#: Each one holds hundreds of MB on the machine the voice pipelines run on, so a day
#: when every vendor refuses must not be able to starve them.
MAX_BROWSERS = 4

_FLAGS = (
    "--headless=new",
    "--remote-debugging-port=0",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-gpu",
    # /dev/shm is small in a container and a full one crashes the renderer.
    "--disable-dev-shm-usage",
    "--disable-extensions",
    "--disable-sync",
    "--window-size=1280,1400",
    "--disable-background-networking",
    # WebRTC's UDP goes round a proxy, and it can be pointed at internal addresses.
    "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
)


def _clean_environment(profile: Path) -> dict[str, str]:
    """All Chromium gets: the agent's environment is full of API keys."""
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(profile),
        "TMPDIR": str(profile),
        "LANG": "C.UTF-8",
        # Windows cannot start a process without it, and a developer may try this there.
        **({"SYSTEMROOT": os.environ["SYSTEMROOT"]} if "SYSTEMROOT" in os.environ else {}),
    }


def _stop(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _discard(process: subprocess.Popen, profile: Path) -> None:
    _stop(process)
    shutil.rmtree(profile, ignore_errors=True)


def _browsers_running() -> int:
    """How many Chromiums this container holds, deleting the profiles of dead ones.

    A worker killed mid-session leaves its profile behind, with the logins of that
    session in it, and nothing else would ever remove it.
    """
    if not (_PROC / "self").exists():
        return 0
    running = 0
    for profile in Path(tempfile.gettempdir()).glob(f"{_PROFILE_PREFIX}*"):
        try:
            pid = (profile / _PID_FILE).read_text().strip()
            alive = pid.isdigit() and (_PROC / pid).exists()
        except FileNotFoundError:
            try:
                alive = time.time() - profile.stat().st_mtime < _STARTING_SECONDS
            except OSError:
                continue
        except OSError:
            continue
        if alive:
            running += 1
        else:
            shutil.rmtree(profile, ignore_errors=True)
    return running


class HeadlessChromeProvider:
    """Starts one throwaway Chromium per session, in this container."""

    def __init__(self, chromium_path: str, *, allow_no_sandbox: bool = False) -> None:
        self._chromium = chromium_path
        self._allow_no_sandbox = allow_no_sandbox
        self._running: dict[str, tuple[subprocess.Popen, Path, EgressProxy]] = {}
        atexit.register(self._stop_all)

    @property
    def vendor(self) -> str:
        return HEADLESS

    def _spawn(self, profile: Path, proxy_port: int, *, sandboxed: bool) -> subprocess.Popen:
        args = [
            self._chromium,
            *_FLAGS,
            f"--user-data-dir={profile}",
            f"--proxy-server=http://127.0.0.1:{proxy_port}",
            # Chromium skips a proxy for loopback unless told not to, and loopback is
            # where its own debugging port and every other session's are.
            "--proxy-bypass-list=<-loopback>",
        ]
        if not sandboxed:
            args.append("--no-sandbox")
        process = subprocess.Popen(  # noqa: S603
            [*args, "about:blank"],
            env=_clean_environment(profile),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        (profile / _PID_FILE).write_text(str(process.pid))
        return process

    async def _start(self, profile: Path, proxy_port: int) -> tuple[subprocess.Popen, int]:
        # A renderer with no sandbox can read the agent's credentials out of /proc, so
        # going without is an operator's decision and never what a failure falls back to.
        attempts = (True, False) if self._allow_no_sandbox else (True,)
        for sandboxed in attempts:
            process = await asyncio.to_thread(self._spawn, profile, proxy_port, sandboxed=sandboxed)
            port = await wait_for_port(process, profile)
            if port:
                if not sandboxed:
                    logger.warning(
                        "Chromium is running without its sandbox: "
                        "KWAMI_HEADLESS_ALLOW_NO_SANDBOX is set."
                    )
                return process, port
            await asyncio.to_thread(_stop, process)
        logger.warning(
            "Chromium did not start with its sandbox, and running without one is not allowed."
        )
        raise BrowserVendorError(HEADLESS, MISSING, "The built-in browser did not start.")

    async def launch(self, user_id: str) -> LaunchedBrowser:
        if not (user_id or "").strip():
            raise ValueError("a browser session needs a real user id")

        if await asyncio.to_thread(_browsers_running) >= MAX_BROWSERS:
            raise BrowserVendorError(HEADLESS, BUSY)

        profile = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix=_PROFILE_PREFIX))
        proxy = EgressProxy()
        try:
            process, port = await self._start(profile, await proxy.start())
        except BaseException:
            proxy.close()
            await asyncio.to_thread(shutil.rmtree, profile, True)
            raise

        browser_id = f"headless-{port}"
        self._running[browser_id] = (process, profile, proxy)
        logger.info("Started built-in Chromium on port %s", port)
        return LaunchedBrowser(
            browser_id=browser_id,
            vendor=HEADLESS,
            cdp_http_url=f"http://127.0.0.1:{port}",
        )

    async def release(self, browser_id: str) -> None:
        """Stop it and delete everything it saw. Safe to call more than once."""
        entry = self._running.pop(browser_id, None)
        if entry is not None:
            process, profile, proxy = entry
            await asyncio.to_thread(_discard, process, profile)
            proxy.close()

    def _stop_all(self) -> None:
        """A worker exiting mid-session must not leave a browser behind."""
        for process, profile, _proxy in self._running.values():
            _discard(process, profile)
        self._running.clear()
