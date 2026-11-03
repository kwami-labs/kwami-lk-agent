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
agent's credentials. So it starts with an environment holding no secrets, and
asks for Chromium's sandbox first. Without one (a container may forbid it) it
still runs, and says so in the logs.
"""

from __future__ import annotations

import asyncio
import atexit
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..utils.logging import get_logger
from .launched import HEADLESS, MISSING, BrowserVendorError, LaunchedBrowser
from .local_chrome import wait_for_port

logger = get_logger("browser.headless")

_STOP_TIMEOUT_SECONDS = 5.0

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


class HeadlessChromeProvider:
    """Starts one throwaway Chromium per session, in this container."""

    def __init__(self, chromium_path: str) -> None:
        self._chromium = chromium_path
        self._running: dict[str, tuple[subprocess.Popen, Path]] = {}
        atexit.register(self._stop_all)

    @property
    def vendor(self) -> str:
        return HEADLESS

    def _spawn(self, profile: Path, *, sandboxed: bool) -> subprocess.Popen:
        args = [self._chromium, *_FLAGS, f"--user-data-dir={profile}"]
        if not sandboxed:
            args.append("--no-sandbox")
        return subprocess.Popen(  # noqa: S603
            [*args, "about:blank"],
            env=_clean_environment(profile),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    async def _start(self, profile: Path) -> tuple[subprocess.Popen, int]:
        for sandboxed in (True, False):
            process = await asyncio.to_thread(self._spawn, profile, sandboxed=sandboxed)
            port = await wait_for_port(process, profile)
            if port:
                if not sandboxed:
                    logger.warning("Chromium is running without its sandbox in this container.")
                return process, port
            await asyncio.to_thread(_stop, process)
        raise BrowserVendorError(HEADLESS, MISSING, "The built-in browser did not start.")

    async def launch(self, user_id: str) -> LaunchedBrowser:
        if not (user_id or "").strip():
            raise ValueError("a browser session needs a real user id")

        profile = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="kwami-browser-"))
        try:
            process, port = await self._start(profile)
        except BaseException:
            await asyncio.to_thread(shutil.rmtree, profile, True)
            raise

        browser_id = f"headless-{port}"
        self._running[browser_id] = (process, profile)
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
            await asyncio.to_thread(_discard, *entry)

    def _stop_all(self) -> None:
        """A worker exiting mid-session must not leave a browser behind."""
        for process, profile in self._running.values():
            _discard(process, profile)
        self._running.clear()
