"""The whole trip: the PC is off, the cloud works, the PC comes back and brings it home.

`test_brain_cloud.py` and `test_brain_sync.py` pin each half on its own. This runs
the real `CloudWorker` and the real `Syncer` against one shared in-memory database
and Storage, so what is pinned is the contract between them: the folder a thread
keeps, the rows the cloud leaves, and what lands on the PC -- across a follow-up,
with the user editing a file in between, and for a change to Jarvis's own code that
crosses from a public clone to the PC's own clone as a patch.

Only the model is a stand-in: it writes files where the executor would.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

import pytest

from src.brain import cloud as cloud_module
from src.brain import selfwork
from src.brain import worker as worker_module
from src.brain.cloud import CloudWorker
from src.brain.executor import (
    FILES_BUCKET,
    CloudWorkspace,
    ExecutorResult,
    restore_files,
    upload_results,
)
from src.brain.jobs import BrainJob
from src.brain.selfwork import Published, SelfRepo, TestRun
from src.brain.sync import MANIFEST, Syncer

USER = "11111111-1111-4111-8111-111111111111"
KWAMI = "22222222-2222-4222-8222-222222222222"


class Cloud:
    """Supabase as both halves see it: brain_jobs, brain_sync and the jarvis-files bucket."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.jobs: dict[str, BrainJob] = {}
        self.queue: list[str] = []
        self.sync: list[dict[str, Any]] = []
        self.sync_ready = True

    # Storage
    async def upload(self, bucket: str, path: str, data: bytes, **_kwargs: Any) -> bool:
        self.objects[f"{bucket}/{path}"] = data
        return True

    async def list_objects(self, bucket: str, prefix: str, *, limit: int = 1000) -> list[str]:
        start = f"{bucket}/{prefix.strip('/')}/"
        return [key[len(bucket) + 1 :] for key in self.objects if key.startswith(start)][:limit]

    async def download(self, bucket: str, path: str) -> bytes | None:
        return self.objects.get(f"{bucket}/{path}")

    # The queue, as the cloud sees it
    def ask(self, task: str, thread_id: str = "", repo: str = "") -> BrainJob:
        job = BrainJob(
            USER,
            KWAMI,
            f"Encargo {len(self.jobs) + 1}",
            task,
            id=str(uuid.uuid4()),
            thread_id=thread_id or str(uuid.uuid4()),
            repo=repo,
            target="cloud",
        )
        self.jobs[job.id] = job
        self.queue.append(job.id)
        return job

    async def claim_cloud(self, runner_id: str) -> BrainJob | None:
        if not self.queue:
            return None
        job = self.jobs[self.queue.pop(0)]
        job.status, job.runner_id = "running", runner_id
        return job

    async def stale_cloud_jobs(self) -> list[BrainJob]:
        return []

    async def month_spend(self, user_id: str) -> float:
        return 0.0

    async def agents(self, user_id: str) -> list[dict[str, Any]]:
        return []

    async def session_of_thread(self, thread_id: str) -> str:
        return ""

    async def thread_results(self, thread_id: str, limit: int = 3) -> list[str]:
        done = [j for j in self.jobs.values() if j.thread_id == thread_id and j.status == "done"]
        return [j.result for j in done][-limit:]

    async def update_running(self, job_id: str, progress: str) -> bool:
        return True

    async def get(self, job_id: str) -> BrainJob | None:
        return self.jobs.get(job_id)

    async def finish(
        self, job_id: str, status: str, result: str, cost_usd: float, session_id: str
    ) -> bool:
        job = self.jobs[job_id]
        job.status, job.result, job.sdk_session_id = status, result, session_id
        return True

    async def mark_delivered(self, job_id: str) -> bool:
        return True

    async def add_sync(self, job: BrainJob, kind: str, prefix: str, **extra: Any) -> bool:
        self.sync.append(
            {
                "id": str(uuid.uuid4()),
                "kwami_id": job.kwami_id,
                "job_id": job.id,
                "kind": kind,
                "storage_prefix": prefix,
                "title": job.title,
                "status": "pending",
                **extra,
            }
        )
        return True

    # The PC bringing it home
    async def claim_sync(self, user_id: str, runner_id: str) -> dict[str, Any] | None:
        waiting = next((s for s in self.sync if s["status"] == "pending"), None)
        if waiting is not None:
            waiting["status"] = "applying"
        return waiting

    async def end_sync(self, sync_id: str, status: str, result: str) -> bool:
        item = next(s for s in self.sync if s["id"] == sync_id)
        item["status"], item["result"] = status, result
        return True


class Ledger:
    async def bind(self, *args: Any) -> None:
        return None

    def add_spend(self, *args: Any) -> None:
        return None

    def add_pending(self, *args: Any) -> None:
        return None

    async def flush(self, timeout: float = 5.0) -> None:
        return None


@pytest.fixture(autouse=True)
def quick(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_module, "SupabaseLedger", lambda store: Ledger())
    monkeypatch.setattr(worker_module, "HANDOVER_SECONDS", 0.0)
    monkeypatch.setattr(worker_module, "RUNNING_POLL_SECONDS", 0.01)


def model_writes(files: dict[str, str]) -> Any:
    """The executor's cloud path, with a model that writes `files` and nothing else."""

    async def run(api_key: str, model: str, task: str, config: Any, **kwargs: Any) -> Any:
        cloud: CloudWorkspace | None = kwargs.get("cloud")
        root: Path | None = kwargs.get("root")
        if root is not None:  # a clone of Jarvis's own code
            for name, text in files.items():
                (root / name).write_text(text, encoding="utf-8")
            return ExecutorResult("Cambiado.", 0.1, True)
        assert cloud is not None and cloud.folder
        folder = Path(tempfile.mkdtemp(prefix="jarvis-test-"))
        try:
            known = await restore_files(folder, cloud, cloud.folder)
            for name, text in files.items():
                (folder / name).parent.mkdir(parents=True, exist_ok=True)
                (folder / name).write_text(text, encoding="utf-8")
            prefix, stored = await upload_results(folder, cloud, cloud.folder, known)
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        return ExecutorResult(
            f"Hecho: {', '.join(files)}.",
            0.2,
            True,
            uploaded_to=f"{FILES_BUCKET}/{prefix}" if stored else "",
            uploaded=stored,
            session_id="cloud-session",
        )

    return run


def _cloud_worker(cloud: Cloud, run: Any, **extra: Any) -> CloudWorker:
    async def own_key(store: Any, user_id: str, secret: str) -> dict[str, str]:
        return {"ANTHROPIC_API_KEY": "user-key"}

    return CloudWorker(
        cloud,  # type: ignore[arg-type]
        cloud,  # type: ignore[arg-type]
        "",
        runner_id="cloud-test",
        run=run,
        load_keys=own_key,
        **extra,
    )


def _pc(cloud: Cloud, tmp_path: Path, **extra: Any) -> Syncer:
    return Syncer(
        cloud,  # type: ignore[arg-type]
        cloud,
        USER,
        "pc-1",
        tmp_path / "Jarvis",
        tmp_path / "mirrors",
        **extra,
    )


async def test_files_made_in_the_cloud_land_on_the_pc_and_follow_ups_keep_the_users_edits(
    tmp_path: Path,
) -> None:
    cloud = Cloud()

    # The PC is off. The user asks from the phone; the cloud does it.
    first = cloud.ask("Escribe el informe y el anexo")
    await _cloud_worker(cloud, model_writes({"informe.md": "v1", "anexo.md": "a"})).tick()
    assert first.status == "done" and first.sdk_session_id == ""
    assert [s["kind"] for s in cloud.sync] == ["files"]

    # The PC comes back and brings it home.
    pc = _pc(cloud, tmp_path)
    assert await pc.tick() is True
    home = next((tmp_path / "Jarvis" / "nube").iterdir())
    assert (home / "informe.md").read_text(encoding="utf-8") == "v1"
    assert cloud.sync[0]["status"] == "applied"

    # The user edits the annex on the PC; the PC goes off; a follow-up in the cloud
    # finds the thread's files, rewrites both, and only what changed goes up.
    (home / "anexo.md").write_text("editado por el usuario", encoding="utf-8")
    uploads_before = len(cloud.objects)
    cloud.ask("Mejora el informe y el anexo", thread_id=first.thread_id)
    await _cloud_worker(cloud, model_writes({"informe.md": "v2", "anexo.md": "a2"})).tick()
    assert len(cloud.objects) == uploads_before  # same names, replaced in place

    await pc.tick()

    assert (home / "informe.md").read_text(encoding="utf-8") == "v2"
    assert (home / "anexo.md").read_text(encoding="utf-8") == "editado por el usuario"
    assert (home / "anexo (nube).md").read_text(encoding="utf-8") == "a2"
    assert MANIFEST in {p.name for p in home.iterdir()}
    assert [s["status"] for s in cloud.sync] == ["applied", "applied"]
    assert await pc.tick() is False


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(  # noqa: S603
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],  # noqa: S607
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


async def test_a_change_to_jarvis_crosses_from_the_public_clone_to_a_pr_from_the_pc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One repository standing in for both the public one and the user's own.
    seed = tmp_path / "seed"
    seed.mkdir()
    _git("init", "-q", "-b", "main", cwd=seed)
    (seed / "README.md").write_text("hola\n", encoding="utf-8")
    _git("add", "-A", cwd=seed)
    _git("commit", "-q", "-m", "init", cwd=seed)
    _git("branch", "dev", cwd=seed)
    public = tmp_path / "public" / "kwami-labs" / "kwami-lk-agent.git"
    public.parent.mkdir(parents=True)
    _git("clone", "-q", "--bare", str(seed), str(public), cwd=tmp_path)
    monkeypatch.setitem(
        selfwork.REPOS, "agent", SelfRepo("owner/kwami-lk-agent", remote=str(public))
    )
    monkeypatch.setattr(cloud_module, "REPOS", selfwork.REPOS)

    cloud = Cloud()
    cloud.ask("Añade un saludo", repo="agent")
    worker = _cloud_worker(
        cloud,
        model_writes({"saludo.py": "print('hola')\n"}),
        repo_base=(tmp_path / "public").as_uri(),
    )
    await worker.tick()
    assert cloud.sync, [j.result for j in cloud.jobs.values()]
    patch_row = cloud.sync[0]
    assert (patch_row["kind"], patch_row["repo"], patch_row["files"]) == ("patch", "agent", 1)

    opened: list[tuple[str, str]] = []

    async def no_tests(workspace: Any) -> TestRun:
        return TestRun(ran=False)

    async def publish(workspace: Any, title: str, summary: str, tests: TestRun) -> Published:
        opened.append((workspace.branch, (workspace.path / "saludo.py").read_text("utf-8")))
        return Published(url="https://github.com/owner/kwami-lk-agent/pull/1", commits=1)

    pc = _pc(cloud, tmp_path, test=no_tests, open_pull_request=publish)
    await pc.tick()

    assert opened and opened[0][1] == "print('hola')\n"
    assert opened[0][0].startswith("jarvis/")
    assert patch_row["status"] == "applied" and "pull/1" in patch_row["result"]
