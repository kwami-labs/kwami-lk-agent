"""Bringing the cloud's work home: files never over the user's, patches as pull requests.

The queue and Storage are in-memory stand-ins; a patch is applied by real git to a
real repository, standing in for the clone the PC keeps of Jarvis's own code.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.brain import sync as sync_module
from src.brain.selfwork import Published, SelfWorkError, TestRun, Workspace
from src.brain.sync import MANIFEST, Syncer, beside, folder_name

USER = "11111111-1111-4111-8111-111111111111"
KWAMI = "22222222-2222-4222-8222-222222222222"
PREFIX = f"{USER}/{KWAMI}/threads/abcdef12-3456"


class FakeJobs:
    def __init__(self, items: list[dict[str, Any]] | None = None) -> None:
        self.items = list(items or [])
        self.ended: list[tuple[str, str, str]] = []
        self.sync_ready = True

    async def claim_sync(self, user_id: str, runner_id: str) -> dict[str, Any] | None:
        return self.items.pop(0) if self.items else None

    async def end_sync(self, sync_id: str, status: str, result: str) -> bool:
        self.ended.append((sync_id, status, result))
        return True


class FakeStore:
    def __init__(self, files: dict[str, bytes | None], listing_ok: bool = True) -> None:
        self.files = files
        self.listing_ok = listing_ok

    async def list_objects(
        self, bucket: str, prefix: str, *, limit: int = 1000
    ) -> list[str] | None:
        return [p for p in self.files if p.startswith(prefix)] if self.listing_ok else None

    async def download(self, bucket: str, path: str) -> bytes | None:
        return self.files.get(path)


class FakeLedger:
    told: list[Any] = []

    async def bind(self, kwami_id: str, user_id: str | None = None) -> None:
        return None

    def add_pending(self, kwami_id: str, item: Any) -> None:
        FakeLedger.told.append(item)

    async def flush(self, timeout: float = 5.0) -> None:
        return None


def _item(**extra: Any) -> dict[str, Any]:
    return {
        "id": "s1",
        "kwami_id": KWAMI,
        "job_id": "j1",
        "kind": "files",
        "repo": "",
        "storage_prefix": PREFIX,
        "title": "Informe: semana 45",
        **extra,
    }


def _syncer(jobs: FakeJobs, store: Any, tmp_path: Path, **extra: Any) -> Syncer:
    FakeLedger.told = []
    return Syncer(
        jobs,  # type: ignore[arg-type]
        store,
        USER,
        "pc-1",
        tmp_path / "Jarvis",
        tmp_path / "mirrors",
        new_ledger=FakeLedger,  # type: ignore[arg-type]
        **extra,
    )


# -- names ---------------------------------------------------------------------------


def test_a_folder_name_any_system_accepts() -> None:
    assert folder_name('Informe: "semana" 45?', "abcdef1234") == "Informe semana 45 (abcdef12)"
    assert folder_name("...", "") == "Trabajo"
    assert len(folder_name("x" * 200, "t")) <= 64


def test_the_clouds_copy_never_takes_another_name(tmp_path: Path) -> None:
    mine = tmp_path / "notas.md"
    assert beside(mine).name == "notas (nube).md"
    beside(mine).write_text("1", encoding="utf-8")
    assert beside(mine).name == "notas (nube 2).md"


# -- files -----------------------------------------------------------------------------


async def test_files_come_home_into_the_nube_folder(tmp_path: Path) -> None:
    store = FakeStore(
        {
            f"{PREFIX}/informe.md": b"hola",
            f"{PREFIX}/datos/tabla.csv": b"a,b",
            f"{PREFIX}/{MANIFEST}": b"{}",
            f"{PREFIX}/perdido.md": None,
            PREFIX: b"the folder itself",
        }
    )
    jobs = FakeJobs([_item()])
    syncer = _syncer(jobs, store, tmp_path)

    assert await syncer.tick() is True

    home = tmp_path / "Jarvis" / "nube" / "Informe semana 45 (abcdef12)"
    assert (home / "informe.md").read_bytes() == b"hola"
    assert (home / "datos" / "tabla.csv").read_bytes() == b"a,b"
    assert not (home / "perdido.md").exists()
    assert set(json.loads((home / MANIFEST).read_text(encoding="utf-8"))) == {
        "informe.md",
        "datos/tabla.csv",
    }
    sync_id, status, text = jobs.ended[0]
    assert (sync_id, status) == ("s1", "applied")
    assert "2 archivo(s)" in text and "nube" in text
    assert FakeLedger.told[0].summary == text and FakeLedger.told[0].models == ["nube"]


async def test_the_users_own_files_are_never_overwritten(tmp_path: Path) -> None:
    home = tmp_path / "Jarvis" / "nube" / "Informe semana 45 (abcdef12)"
    home.mkdir(parents=True)
    (home / "suyo.md").write_text("del usuario", encoding="utf-8")
    (home / "igual.md").write_bytes(b"igual")
    (home / "mio.md").write_bytes(b"version 1")
    (home / "tocado.md").write_bytes(b"cambiado por el usuario")
    digest = sync_module._digest
    (home / MANIFEST).write_text(
        json.dumps({"mio.md": digest(b"version 1"), "tocado.md": digest(b"original")}),
        encoding="utf-8",
    )
    store = FakeStore(
        {
            f"{PREFIX}/suyo.md": b"de la nube",
            f"{PREFIX}/igual.md": b"igual",
            f"{PREFIX}/mio.md": b"version 2",
            f"{PREFIX}/tocado.md": b"version nube",
        }
    )
    jobs = FakeJobs([_item(title="Otro título")])

    await _syncer(jobs, store, tmp_path).tick()

    # A follow-up of the same thread lands in the folder of the first.
    assert (home / "suyo.md").read_text(encoding="utf-8") == "del usuario"
    assert (home / "suyo (nube).md").read_bytes() == b"de la nube"
    assert (home / "mio.md").read_bytes() == b"version 2"
    assert (home / "tocado.md").read_bytes() == b"cambiado por el usuario"
    assert (home / "tocado (nube).md").read_bytes() == b"version nube"
    text = jobs.ended[0][2]
    assert "1 archivo(s)" in text and "2 junto a los tuyos" in text and "1 ya estaban" in text


async def test_a_name_that_would_leave_the_folder_is_left_out(tmp_path: Path) -> None:
    store = FakeStore({f"{PREFIX}/../../fuera.md": b"x", f"{PREFIX}/dentro.md": b"y"})
    jobs = FakeJobs([_item()])
    await _syncer(jobs, store, tmp_path).tick()
    assert not list(tmp_path.rglob("fuera.md"))  # noqa: ASYNC240
    assert "1 archivo(s)" in jobs.ended[0][2]


async def test_files_outside_a_thread_get_a_folder_of_their_own(tmp_path: Path) -> None:
    prefix = f"{USER}/{KWAMI}/sync/j9"
    store = FakeStore({f"{prefix}/a.md": b"a"})
    jobs = FakeJobs([_item(storage_prefix=prefix, title="")])
    await _syncer(jobs, store, tmp_path).tick()
    assert (tmp_path / "Jarvis" / "nube" / "Trabajo" / "a.md").exists()


async def test_an_unreadable_listing_is_tried_again_later(tmp_path: Path) -> None:
    jobs = FakeJobs([_item()])
    await _syncer(jobs, FakeStore({}, listing_ok=False), tmp_path).tick()
    assert jobs.ended[0][1] == "failed"


async def test_nothing_to_bring_home_is_nothing(tmp_path: Path) -> None:
    assert await _syncer(FakeJobs(), FakeStore({}), tmp_path).tick() is False


async def test_a_crash_ends_the_item_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = FakeJobs([_item()])
    syncer = _syncer(jobs, FakeStore({}), tmp_path)

    async def boom(item: dict[str, Any]) -> Any:
        raise RuntimeError("x")

    monkeypatch.setattr(syncer, "apply", boom)
    assert await syncer.tick() is True
    assert jobs.ended[0][1:] == ("failed", "No he podido traerlo: RuntimeError")


async def test_without_a_ledger_or_a_kwami_nobody_is_told(tmp_path: Path) -> None:
    jobs = FakeJobs([_item(kwami_id="")])
    syncer = Syncer(jobs, FakeStore({}), USER, "pc-1", tmp_path, tmp_path / "m")  # type: ignore[arg-type]
    FakeLedger.told = []
    await syncer.tick()
    assert FakeLedger.told == []


# -- patches ---------------------------------------------------------------------------


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(  # noqa: S603
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],  # noqa: S607
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, bytes, bytes]:
    """The PC's clone, and two patches made elsewhere: one that fits and one that clashes."""
    pc = tmp_path / "pc"
    pc.mkdir()
    _git("init", "-q", "-b", "main", cwd=pc)
    (pc / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git("add", "-A", cwd=pc)
    _git("commit", "-q", "-m", "init", cwd=pc)
    cloud = tmp_path / "cloud"
    _git("clone", "-q", str(pc), str(cloud), cwd=tmp_path)
    (cloud / "b.py").write_text("y = 2\n", encoding="utf-8")
    _git("add", "-A", cwd=cloud)
    _git("commit", "-q", "-m", "feat: add b", cwd=cloud)
    fits = _git("format-patch", "--stdout", "HEAD~1..HEAD", cwd=cloud).encode()
    (cloud / "a.py").write_text("x = 3\n", encoding="utf-8")
    _git("commit", "-qam", "fix: change a", cwd=cloud)
    clashes = _git("format-patch", "--stdout", "HEAD~1..HEAD", cwd=cloud).encode()
    (pc / "a.py").write_text("x = 99\n", encoding="utf-8")
    _git("commit", "-qam", "chore: the PC moved on", cwd=pc)
    return pc, fits, clashes


def _patch_syncer(
    jobs: FakeJobs, patch: bytes | None, pc: Path, tmp_path: Path, refused: str = ""
) -> tuple[Syncer, list[Any]]:
    calls: list[Any] = []

    async def prepare(key: str, thread: str, root: Path, mirrors: Path) -> Workspace:
        calls.append(("prepare", key, thread))
        return Workspace(key, None, pc, mirrors, "jarvis/j1")  # type: ignore[arg-type]

    async def test(workspace: Workspace) -> TestRun:
        calls.append(("test",))
        return TestRun(ran=True, passed=True)

    async def publish(workspace: Workspace, title: str, summary: str, tests: TestRun) -> Published:
        calls.append(("publish", title, summary))
        if refused:
            return Published(refused=refused)
        return Published(url="https://github.com/o/r/pull/9", commits=1)

    store = FakeStore({f"{PREFIX}/changes.patch": patch})
    syncer = _syncer(
        jobs, store, tmp_path, prepare_workspace=prepare, test=test, open_pull_request=publish
    )
    return syncer, calls


async def test_a_patch_is_applied_tested_and_opened_as_a_pr(repo, tmp_path: Path) -> None:
    pc, fits, _ = repo
    jobs = FakeJobs([_item(kind="patch", repo="agent", title="Mejora de voz")])
    syncer, calls = _patch_syncer(jobs, fits, pc, tmp_path)

    await syncer.tick()

    assert (pc / "b.py").read_text(encoding="utf-8") == "y = 2\n"
    assert "feat: add b" in _git("log", "-1", "--format=%s", cwd=pc)
    assert [c[0] for c in calls] == ["prepare", "test", "publish"]
    assert calls[0] == ("prepare", "agent", "j1") and calls[2][1] == "Mejora de voz"
    assert jobs.ended[0][1] == "applied" and "pull/9" in jobs.ended[0][2]
    assert not list((tmp_path / "mirrors").glob("*.patch"))


async def test_a_patch_that_clashes_is_not_forced(repo, tmp_path: Path) -> None:
    pc, _, clashes = repo
    jobs = FakeJobs([_item(kind="patch", repo="agent")])
    syncer, calls = _patch_syncer(jobs, clashes, pc, tmp_path)

    await syncer.tick()

    assert (pc / "a.py").read_text(encoding="utf-8") == "x = 99\n"
    assert not (pc / ".git" / "rebase-apply").exists()
    assert jobs.ended[0][1] == "conflict" and [c[0] for c in calls] == ["prepare"]


async def test_a_refused_pr_ends_failed(repo, tmp_path: Path) -> None:
    pc, fits, _ = repo
    jobs = FakeJobs([_item(kind="patch", repo="agent")])
    syncer, _ = _patch_syncer(jobs, fits, pc, tmp_path, refused="toca archivos prohibidos: x")
    await syncer.tick()
    assert jobs.ended[0][1] == "failed" and "prohibidos" in jobs.ended[0][2]


@pytest.mark.parametrize(
    ("item", "patch", "words"),
    [
        ({"repo": "nope"}, b"x", "No conozco el repositorio nope"),
        ({"repo": ""}, b"x", "(ninguno)"),
        ({"repo": "agent"}, None, "No he encontrado el cambio"),
    ],
)
async def test_what_cannot_be_applied_says_why(
    tmp_path: Path, item: dict[str, Any], patch: bytes | None, words: str
) -> None:
    jobs = FakeJobs([_item(kind="patch", **item)])
    syncer, calls = _patch_syncer(jobs, patch, tmp_path, tmp_path)
    await syncer.tick()
    assert jobs.ended[0][1] == "failed" and words in jobs.ended[0][2] and calls == []


async def test_a_repository_that_cannot_be_prepared_says_so(tmp_path: Path) -> None:
    async def prepare(*args: Any) -> Workspace:
        raise SelfWorkError("traer main: sin red")

    jobs = FakeJobs([_item(kind="patch", repo="agent")])
    store = FakeStore({f"{PREFIX}/changes.patch": b"x"})
    await _syncer(jobs, store, tmp_path, prepare_workspace=prepare).tick()
    assert jobs.ended[0][1:] == (
        "failed",
        "No he podido preparar el repositorio: traer main: sin red",
    )


# -- the loop --------------------------------------------------------------------------


async def test_the_loop_brings_all_home_then_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sync_module, "SYNC_POLL_SECONDS", 0.01)
    monkeypatch.setattr(sync_module, "SYNC_UNREADY_POLL_SECONDS", 0.01)
    jobs = FakeJobs([_item(id="s1"), _item(id="s2")])
    syncer = _syncer(jobs, FakeStore({}), tmp_path)
    stop = asyncio.Event()
    looks = 0
    real_tick = syncer.tick

    async def tick() -> bool:
        nonlocal looks
        looks += 1
        if looks == 4:
            jobs.sync_ready = False
        if looks == 6:
            stop.set()
        if looks == 5:
            raise RuntimeError("blip")
        return await real_tick()

    monkeypatch.setattr(syncer, "tick", tick)
    await asyncio.wait_for(syncer.run_forever(stop), 2)
    assert [e[0] for e in jobs.ended] == ["s1", "s2"] and looks == 6


async def test_a_file_with_the_threads_name_is_not_taken_for_its_folder(tmp_path: Path) -> None:
    nube = tmp_path / "Jarvis" / "nube"
    nube.mkdir(parents=True)
    (nube / "Suelto (abcdef12)").write_text("no es carpeta", encoding="utf-8")
    jobs = FakeJobs([_item()])
    await _syncer(jobs, FakeStore({f"{PREFIX}/a.md": b"a"}), tmp_path).tick()
    assert (nube / "Informe semana 45 (abcdef12)" / "a.md").exists()
