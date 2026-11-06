"""Bringing the cloud's work home: what the PC does with `brain_sync`.

While the PC was off, the cloud's work thread (`cloud.py`) kept working and left
what belongs on the PC listed in `brain_sync`. When the PC is on again, its worker
brings it home here, one item at a time and beside its own jobs, so neither side
ever waits for the other:

* **files** -- a thread's folder, copied into ``Documents\\Jarvis\\nube\\<title>``.
  Never over a file of the user's: a file that is not as Jarvis last left it keeps
  its place, and the cloud's version lands beside it as ``name (nube).ext``. What
  Jarvis wrote is remembered in ``.nube.json`` in that folder, to tell the two apart.
* **patch** -- a change to Jarvis's own code made on a public clone. It is applied
  on the thread's branch of the clone the PC already keeps (`selfwork.prepare`),
  the repository's own tests run, and the PR is opened exactly as for work done
  here (`selfwork.publish`): same checks, same refusals. A patch that no longer
  fits is not forced; the user hears it clashed.

Each item ends `applied`, `conflict` or `failed`, with words the app shows, and the
next greeting tells the user what came home.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import re
import tempfile
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.logging import get_logger
from .executor import FILES_BUCKET
from .jobs import JobStore
from .ledger import Ledger, PendingResult
from .selfwork import (
    REPOS,
    SelfWorkError,
    _clean_env,
    _run,
    outcome_text,
    prepare,
    publish,
    run_tests,
    spoken_title,
)

logger = get_logger("brain.sync")

#: Nothing waits on these: a look a minute is enough, and a slower one before
#: the database has migration 029 keeps the log quiet.
SYNC_POLL_SECONDS = 60.0
SYNC_UNREADY_POLL_SECONDS = 1800.0
#: The folder of the Jarvis folder where the cloud's files land.
NUBE_FOLDER = "nube"
MANIFEST = ".nube.json"
MAX_SYNC_FILES = 200
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')

Prepare = Callable[..., Coroutine[Any, Any, Any]]


@dataclass
class Outcome:
    status: str
    text: str


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def folder_name(title: str, thread: str) -> str:
    """A folder name any OS accepts: the title, and the thread it belongs to."""
    clean = " ".join(_UNSAFE.sub(" ", title).split()).strip(" .")[:60].strip(" .") or "Trabajo"
    return f"{clean} ({thread[:8]})" if thread else clean


def beside(path: Path) -> Path:
    """Where the cloud's copy goes when the user's file is in the way."""
    candidate = path.with_name(f"{path.stem} (nube){path.suffix}")
    n = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem} (nube {n}){path.suffix}")
        n += 1
    return candidate


def _patch_file(folder: Path, data: bytes) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".patch", dir=folder, delete=False) as handle:
        handle.write(data)
    return Path(handle.name)


class Syncer:
    """The PC's side of `brain_sync`."""

    def __init__(
        self,
        jobs: JobStore,
        store: Any,
        user_id: str,
        runner_id: str,
        folder: Path,
        mirrors: Path,
        *,
        new_ledger: Callable[[], Ledger] | None = None,
        prepare_workspace: Prepare = prepare,
        test: Prepare = run_tests,
        open_pull_request: Prepare = publish,
    ) -> None:
        self._jobs = jobs
        self._store = store
        self.user_id = user_id
        self.runner_id = runner_id
        self._folder = folder
        self._mirrors = mirrors
        self._new_ledger = new_ledger
        self._prepare = prepare_workspace
        self._test = test
        self._publish = open_pull_request

    async def run_forever(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            took = False
            try:
                took = await self.tick()
            except Exception:
                logger.exception("Bringing the cloud's work home failed")
            if took:
                continue
            wait = SYNC_POLL_SECONDS if self._jobs.sync_ready else SYNC_UNREADY_POLL_SECONDS
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), wait)

    async def tick(self) -> bool:
        """Bring one item home. True when there was one."""
        item = await self._jobs.claim_sync(self.user_id, self.runner_id)
        if item is None:
            return False
        try:
            outcome = await self.apply(item)
        except Exception as e:
            logger.exception("Sync %s crashed", item.get("id"))
            outcome = Outcome("failed", f"No he podido traerlo: {type(e).__name__}")
        await self._jobs.end_sync(str(item["id"]), outcome.status, outcome.text)
        await self._tell(item, outcome)
        return True

    async def apply(self, item: dict[str, Any]) -> Outcome:
        if item.get("kind") == "patch":
            return await self._patch(item)
        return await self._files(item)

    # -- files -----------------------------------------------------------------

    def _target_folder(self, item: dict[str, Any]) -> Path:
        prefix = str(item.get("storage_prefix") or "")
        thread = prefix.rsplit("/", 1)[-1] if "/threads/" in f"/{prefix}" else ""
        home = self._folder / NUBE_FOLDER
        if thread:
            # A follow-up of the same thread lands where the first one did.
            for existing in sorted(home.glob(f"* ({thread[:8]})")):
                if existing.is_dir():
                    return existing
        return home / folder_name(str(item.get("title") or ""), thread)

    async def _files(self, item: dict[str, Any]) -> Outcome:
        prefix = str(item.get("storage_prefix") or "").strip("/")
        paths = await self._store.list_objects(FILES_BUCKET, prefix, limit=MAX_SYNC_FILES)
        if paths is None:
            return Outcome("failed", "No he podido leer lo que dejó la nube. Lo intento luego.")
        files: dict[str, bytes] = {}
        for path in paths:
            rel = path[len(prefix) :].lstrip("/")
            if not rel or rel == MANIFEST:
                continue
            data = await self._store.download(FILES_BUCKET, path)
            if data is not None:
                files[rel] = data
        return await asyncio.to_thread(self._place, item, files)

    def _place(self, item: dict[str, Any], files: dict[str, bytes]) -> Outcome:
        """Write what came down, never over the user's own files. Runs off the loop."""
        target = self._target_folder(item)
        manifest_path = target / MANIFEST
        try:
            written: dict[str, str] = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            written = {}
        copied = kept_beside = same = 0
        root = target.resolve()
        for rel, data in files.items():
            local = (target / rel).resolve()
            if not local.is_relative_to(root):
                continue
            digest = _digest(data)
            if local.exists():
                current = _digest(local.read_bytes())
                if current == digest:
                    same += 1
                    written[rel] = digest
                    continue
                if written.get(rel) != current:
                    # The user's own file, or one they changed: theirs stays.
                    beside(local).write_bytes(data)
                    kept_beside += 1
                    continue
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
            written[rel] = digest
            copied += 1
        if written:
            target.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(written, indent=1), encoding="utf-8")
        where = "\\".join(("Documentos", "Jarvis", NUBE_FOLDER, target.name))
        parts = [f"{copied} archivo(s) traído(s) a {where}"]
        if kept_beside:
            parts.append(f"{kept_beside} junto a los tuyos, con «(nube)» en el nombre")
        if same:
            parts.append(f"{same} ya estaban igual")
        return Outcome("applied", "; ".join(parts) + ".")

    # -- patches ---------------------------------------------------------------

    async def _patch(self, item: dict[str, Any]) -> Outcome:
        key = str(item.get("repo") or "")
        if key not in REPOS:
            return Outcome("failed", f"No conozco el repositorio {key or '(ninguno)'}.")
        prefix = str(item.get("storage_prefix") or "").strip("/")
        patch = await self._store.download(FILES_BUCKET, f"{prefix}/changes.patch")
        if not patch:
            return Outcome("failed", "No he encontrado el cambio que dejó la nube.")
        thread = str(item.get("job_id") or item.get("id"))
        try:
            workspace = await self._prepare(key, thread, self._folder, self._mirrors)
        except SelfWorkError as e:
            return Outcome("failed", f"No he podido preparar el repositorio: {e}")
        clean = _clean_env()
        # Outside the Jarvis folder, where the model cannot reach it.
        patch_file = await asyncio.to_thread(_patch_file, self._mirrors, patch)
        try:
            code, output = await _run(
                "git",
                "-C",
                str(workspace.path),
                "-c",
                "user.name=Jarvis",
                "-c",
                "user.email=jarvis@kwami.io",
                "am",
                "--3way",
                str(patch_file),
                env=clean,
            )
        finally:
            await asyncio.to_thread(patch_file.unlink, True)
        if code != 0:
            await _run("git", "-C", str(workspace.path), "am", "--abort", env=clean)
            return Outcome(
                "conflict",
                "El cambio de la nube ya no encaja con el código de tu ordenador y no lo he "
                f"forzado. Pídemelo de nuevo con el ordenador encendido. ({output[-200:]})",
            )
        tests = await self._test(workspace)
        published = await self._publish(
            workspace,
            spoken_title(str(item.get("title") or "")),
            "Hecho en la nube con el ordenador apagado; el PC lo ha traído, probado y subido.",
            tests,
        )
        status = "failed" if published.refused else "applied"
        return Outcome(status, outcome_text(published, tests))

    # -- telling the user --------------------------------------------------------

    async def _tell(self, item: dict[str, Any], outcome: Outcome) -> None:
        if self._new_ledger is None or not item.get("kwami_id"):
            return
        kwami_id = str(item["kwami_id"])
        ledger = self._new_ledger()
        await ledger.bind(kwami_id, self.user_id)
        ledger.add_pending(
            kwami_id,
            PendingResult(
                task_id=f"sync-{item['id']}",
                title=f"Traído de la nube: {item.get('title') or 'trabajo'}",
                summary=outcome.text,
                finished_at="",
                models=["nube"],
            ),
        )
        await ledger.flush()
