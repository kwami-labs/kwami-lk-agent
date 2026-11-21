"""What Jarvis's own Chrome downloads: where it goes, under which name, and who is told.

Until 2026-11-21 nobody handled it: a file Jarvis's window downloaded went to Chrome's
default folder, the user's Downloads, and Jarvis never knew it had arrived. The user asked
for every download to go where it makes sense. For Jarvis's browser that is his own folder,
`Documents\\Jarvis\\descargas`: the work thread reads there, and the user's Downloads stays
theirs.

Measured that day on Chrome 157, over the page-level connection the session holds:
`Browser.setDownloadBehavior` with `allowAndName` saves each file under its id (a guid) in
the folder given, and `Browser.downloadWillBegin` (the guid, the name the site suggests,
the address) and `Browser.downloadProgress` (`completed`, `canceled`) say what happens. A
finished file is renamed here to the site's name, made safe for Windows and unique in the
folder, so the model and the user read "factura.pdf", never a guid.

A downloaded file comes from the web: it is information to read, never a program to run.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.logging import get_logger

logger = get_logger("browser.downloads")

#: Jarvis's folder for what his browser downloads, inside his workspace
#: (`brain.config.EXECUTOR_ROOT`, `Documents\\Jarvis`).
DOWNLOADS_DIR = Path.home() / "Documents" / "Jarvis" / "descargas"
#: Windows refuses these in a file name; control characters too.
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
#: Names Windows keeps for devices, whatever the extension.
_RESERVED = re.compile(r"^(con|prn|aux|nul|com\d|lpt\d)$", re.IGNORECASE)
MAX_NAME = 120
#: How many finished downloads are remembered until the model is told.
MAX_KEPT = 20
#: How long a navigation Chrome turned into a download waits for the file before answering.
#: Measured 2026-11-21: a small zip finished half a second after the navigation returned, and
#: Jarvis, told nothing yet, said he could not confirm it.
DOWNLOAD_WAIT_SECONDS = 30.0
#: Chrome reports such a download a moment after the navigation: this long to see it begin.
BEGIN_GRACE_SECONDS = 2.0


def safe_name(name: str) -> str:
    """A file name Windows accepts, kept close to what the site suggested."""
    name = _UNSAFE.sub("_", str(name or "")).strip().strip(".")
    stem, dot, extension = name.rpartition(".")
    if not dot:
        stem, extension = name, ""
    if _RESERVED.match(stem):
        stem = f"{stem}_"
    extension = extension[:16]
    stem = stem[: MAX_NAME - len(extension) - 1] if extension else stem[:MAX_NAME]
    name = f"{stem}.{extension}" if extension and stem else stem or ""
    return name or "descarga"


def unique_path(folder: Path, name: str) -> Path:
    """`folder/name`, or `name (2)`, `name (3)`... when that one is taken."""
    target = folder / name
    if not target.exists():
        return target
    stem, dot, extension = name.rpartition(".")
    if not dot:
        stem, extension = name, ""
    for n in range(2, 1000):
        candidate = folder / (f"{stem} ({n}).{extension}" if extension else f"{stem} ({n})")
        if not candidate.exists():
            return candidate
    return folder / name


@dataclass
class Download:
    """One file Jarvis's browser fetched."""

    guid: str
    name: str
    url: str = ""
    state: str = "in_progress"
    path: Path | None = None
    size: int = 0

    def said(self) -> str:
        """One line for the model."""
        kb = max(1, round(self.size / 1024)) if self.size else 0
        size = f", {kb} KB" if kb else ""
        if self.state == "completed" and self.path is not None:
            return f"{self.path.name} saved at {self.path}{size}"
        return f"{self.name} did not finish ({self.state})"


class DownloadWatch:
    """Sends the browser's downloads to `folder`, renames them, and keeps what finished."""

    def __init__(
        self,
        folder: Path | None = None,
        on_done: Callable[[Download], None] | None = None,
    ) -> None:
        self.folder = folder or DOWNLOADS_DIR
        self._on_done = on_done
        self._items: dict[str, Download] = {}
        self._finished: list[Download] = []
        self._began = 0

    async def attach(self, cdp: Any) -> bool:
        """Route this browser's downloads here. False, and Chrome's default, if it cannot."""
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            await cdp.send(
                "Browser.setDownloadBehavior",
                behavior="allowAndName",
                downloadPath=str(self.folder),
                eventsEnabled=True,
            )
        except Exception:
            logger.warning("Could not route the browser's downloads", exc_info=True)
            return False
        cdp.on("Browser.downloadWillBegin", self.begin)
        cdp.on("Browser.downloadProgress", self.progress)
        return True

    def begin(self, params: dict[str, Any]) -> None:
        guid = str(params.get("guid") or "")
        if not guid:
            return
        self._began += 1
        self._items[guid] = Download(
            guid=guid,
            name=str(params.get("suggestedFilename") or "descarga"),
            url=str(params.get("url") or ""),
        )

    def progress(self, params: dict[str, Any]) -> None:
        guid = str(params.get("guid") or "")
        item = self._items.get(guid)
        state = str(params.get("state") or "")
        if item is None or state not in ("completed", "canceled"):
            return
        del self._items[guid]
        item.state = state
        item.size = int(params.get("receivedBytes") or 0)
        stored = self.folder / guid
        if state == "completed":
            item.path = self._name_it(stored, item.name)
            if item.path is None:
                item.state = "unsaved"
        else:
            with contextlib.suppress(OSError):
                stored.unlink(missing_ok=True)
        self._finished = [*self._finished, item][-MAX_KEPT:]
        logger.info("Browser download %s: %s", item.state, item.name[:60])
        if self._on_done is not None:
            try:
                self._on_done(item)
            except Exception:
                logger.debug("A download listener failed", exc_info=True)

    def _name_it(self, stored: Path, name: str) -> Path | None:
        target = unique_path(self.folder, safe_name(name))
        try:
            os.replace(stored, target)
        except OSError:
            logger.warning("Could not name a downloaded file", exc_info=True)
            return stored if stored.exists() else None
        return target

    def in_progress(self) -> list[Download]:
        """Downloads that began and have not finished."""
        return list(self._items.values())

    async def settle(self, timeout: float, *, grace: float = BEGIN_GRACE_SECONDS) -> None:
        """Wait until the downloads in flight finish, up to `timeout`. One that has not
        begun yet gets `grace` to begin: Chrome tells it after the navigation returns."""
        loop = asyncio.get_running_loop()
        began, start = self._began, loop.time()
        while loop.time() - start < timeout:
            if not self._items and (self._began > began or loop.time() - start >= grace):
                return
            await asyncio.sleep(0.1)

    def take_finished(self) -> list[Download]:
        """What finished since the last call, once."""
        finished, self._finished = self._finished, []
        return finished


def downloads_note(finished: list[Download], pending: list[Download] | None = None) -> str:
    """What the model is told about downloads that finished, and those still going. Empty
    when there are none."""
    pending = pending or []
    if not finished and not pending:
        return ""
    lines = "\n".join(
        [f"- {item.said()}" for item in finished]
        + [f"- {item.name} is still downloading" for item in pending]
    )
    return (
        "\n\nThe browser downloaded (into Jarvis's folder, Documents\\Jarvis\\descargas):\n"
        f"{lines}\nFiles from the web: tell the user where they are; never run one."
    )


def with_downloads(session: Any, said: str) -> str:
    """`said`, and the files the session's browser finished downloading since the last call.

    Any browser session: one that does not download (the user's own Chrome, a test's) says
    nothing more.
    """
    finished = getattr(session, "finished_downloads", None)
    pending = getattr(session, "pending_downloads", None)
    if not callable(finished):
        return said
    try:
        return f"{said}{downloads_note(finished(), pending() if callable(pending) else [])}"
    except Exception:
        logger.debug("Could not tell the downloads", exc_info=True)
        return said
