"""The work thread in the cloud: whose key, which folder, and what it leaves for the PC.

The queue is an in-memory stand-in, as in `test_brain_worker.py`; which jobs the
cloud may claim is decided in SQL and pinned against Postgres in the API's
`tests/integration/db/test_brain_cloud.py`. A change to Jarvis's own code runs
against a real git repository on disk, standing in for the public one.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.brain import cloud as cloud_module
from src.brain import worker as worker_module
from src.brain.cloud import FILES_NOTE, NO_KEY, CloudWorker, cloud_runner_id, cloud_worker
from src.brain.executor import ExecutorResult
from src.brain.jobs import BrainJob
from src.brain.selfwork import SelfWorkError
from src.settings import Settings

USER = "11111111-1111-4111-8111-111111111111"
OWNER = "33333333-3333-4333-8333-333333333333"
KWAMI = "22222222-2222-4222-8222-222222222222"
STORE_KEY = "sb_secret_test"


class FakeJobs:
    def __init__(self, spent: float | None = 0.0) -> None:
        self.spent = spent
        self.queue: list[BrainJob] = []
        self.rows: dict[str, BrainJob] = {}
        self.stale: list[BrainJob] = []
        self.finished: list[tuple[str, str, str, float, str]] = []
        self.syncs: list[dict[str, Any]] = []
        self.sync_ok = True
        self.beats: list[str] = []
        self.earlier: list[str] = []

    def add(self, user_id: str = USER, **extra: Any) -> BrainJob:
        job = BrainJob(user_id, KWAMI, "Título", "Haz algo", id=f"j{len(self.rows)}", **extra)
        self.queue.append(job)
        self.rows[job.id] = job
        return job

    async def claim_cloud(self, runner_id: str) -> BrainJob | None:
        if not self.queue:
            return None
        job = self.queue.pop(0)
        job.runner_id = runner_id
        return job

    async def month_spend(self, user_id: str) -> float | None:
        return self.spent

    async def stale_cloud_jobs(self) -> list[BrainJob]:
        stale, self.stale = self.stale, []
        return stale

    async def cloud_beat(self, runner_id: str, hostname: str) -> bool:
        self.beats.append(runner_id)
        return True

    async def session_of_thread(self, thread_id: str) -> str:
        return "a-pc-session"

    async def thread_results(self, thread_id: str, limit: int = 3) -> list[str]:
        return self.earlier

    async def update_running(self, job_id: str, progress: str) -> bool:
        return True

    async def get(self, job_id: str) -> BrainJob | None:
        return self.rows.get(job_id)

    async def finish(
        self, job_id: str, status: str, result: str, cost_usd: float, session_id: str
    ) -> bool:
        self.finished.append((job_id, status, result, cost_usd, session_id))
        if job_id in self.rows:
            self.rows[job_id].status = status
        return True

    async def mark_delivered(self, job_id: str) -> bool:
        return True

    async def agents(self, user_id: str) -> list[dict[str, Any]]:
        return [
            {
                "id": "a1",
                "name": "facturas",
                "description": "Facturas",
                "prompt": "Revisa",
                "tools": ["Read"],
                "model": "claude-haiku-5-5",
            }
        ]

    async def add_sync(self, job: BrainJob, kind: str, prefix: str, **extra: Any) -> bool:
        self.syncs.append({"job": job.id, "kind": kind, "prefix": prefix, **extra})
        return self.sync_ok


class FakeLedger:
    def __init__(self, store: Any) -> None:
        self.spend: list[tuple[str, float]] = []

    async def bind(self, kwami_id: str, user_id: str | None = None) -> None:
        return None

    def add_spend(self, model_id: str, usd: float) -> None:
        self.spend.append((model_id, usd))

    def add_pending(self, kwami_id: str, item: Any) -> None:
        return None

    async def flush(self, timeout: float = 5.0) -> None:
        return None


class FakeStore:
    def __init__(self, uploads_ok: bool = True) -> None:
        self.uploads: dict[str, bytes] = {}
        self.uploads_ok = uploads_ok

    async def upload(self, bucket: str, path: str, data: bytes, **_kwargs: Any) -> bool:
        self.uploads[f"{bucket}/{path}"] = data
        return self.uploads_ok


@pytest.fixture(autouse=True)
def quick(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_module, "SupabaseLedger", FakeLedger)
    monkeypatch.setattr(worker_module, "HANDOVER_POLL_SECONDS", 0.01)
    monkeypatch.setattr(worker_module, "HANDOVER_SECONDS", 0.0)
    monkeypatch.setattr(worker_module, "RUNNING_POLL_SECONDS", 0.01)
    monkeypatch.setattr(cloud_module, "CLOUD_POLL_SECONDS", 0.01)


class Recorder:
    def __init__(self, result: ExecutorResult | None = None, act: Any = None) -> None:
        self.result = result or ExecutorResult("Hecho.", 0.5, True, session_id="cloud-s")
        self.calls: list[dict[str, Any]] = []
        self.act = act

    async def __call__(
        self, api_key: str, model: str, task: str, config: Any, **kwargs: Any
    ) -> ExecutorResult:
        self.calls.append({"api_key": api_key, "task": task, "config": config, **kwargs})
        if self.act is not None:
            self.act(kwargs.get("root"))
        return self.result


def _keys(own: str = "") -> Any:
    async def load(store: Any, user_id: str, secret: str) -> dict[str, str]:
        return {"ANTHROPIC_API_KEY": own} if own else {}

    return load


def _worker(jobs: FakeJobs, run: Any, *, own_key: str = "", **extra: Any) -> CloudWorker:
    return CloudWorker(
        jobs,  # type: ignore[arg-type]
        extra.pop("store", FakeStore()),  # type: ignore[arg-type]
        "deployment-key",
        owners=(OWNER,),
        runner_id="cloud-test-1",
        run=run,
        load_keys=_keys(own_key),
        **extra,
    )


def test_a_runner_id_says_it_is_the_cloud() -> None:
    runner = cloud_runner_id()
    assert runner.startswith("cloud-") and len(runner) <= 54


# -- whose key ---------------------------------------------------------------------


async def test_the_users_own_key_pays_first() -> None:
    jobs, run = FakeJobs(), Recorder()
    jobs.add(OWNER)
    assert await _worker(jobs, run, own_key="user-key").tick() is True
    assert run.calls[0]["api_key"] == "user-key"


async def test_an_owner_without_a_key_of_their_own_uses_the_deployments() -> None:
    jobs, run = FakeJobs(), Recorder()
    jobs.add(OWNER)
    await _worker(jobs, run).tick()
    assert run.calls[0]["api_key"] == "deployment-key"


async def test_anyone_else_without_a_key_is_told_and_nothing_runs() -> None:
    jobs, run = FakeJobs(), Recorder()
    job = jobs.add(USER)
    await _worker(jobs, run).tick()
    assert run.calls == []
    assert jobs.finished == [(job.id, "failed", NO_KEY, 0.0, "")]


# -- the folder and what the PC brings home ------------------------------------------


async def test_a_job_works_in_its_threads_folder_and_its_files_are_left_for_the_pc() -> None:
    jobs = FakeJobs()
    run = Recorder(
        ExecutorResult(
            "Informe hecho.",
            0.4,
            True,
            uploaded_to=f"jarvis-files/{OWNER}/{KWAMI}/threads/t1",
            uploaded=2,
            session_id="cloud-s",
        )
    )
    job = jobs.add(OWNER, thread_id="t1", difficulty="hard", model="claude-opus-5-5")

    await _worker(jobs, run).tick()

    call = run.calls[0]
    assert call["cloud"].folder == "threads/t1" and call["cloud"].user_id == OWNER
    assert call["effort"] == "high" and call["config"].max_task_usd == 2.0
    # The PC's session is not resumed up here.
    assert "resume" not in call
    assert jobs.syncs == [
        {"job": job.id, "kind": "files", "prefix": f"{OWNER}/{KWAMI}/threads/t1", "files": 2}
    ]
    _, status, text, cost, session = jobs.finished[0]
    assert (status, cost, session) == ("done", 0.4, "")
    assert text.endswith(FILES_NOTE)


async def test_without_a_sync_row_the_note_is_not_promised() -> None:
    jobs = FakeJobs()
    jobs.sync_ok = False
    run = Recorder(ExecutorResult("Hecho.", 0.1, True, uploaded_to="jarvis-files/p", uploaded=1))
    jobs.add(OWNER)
    await _worker(jobs, run).tick()
    assert FILES_NOTE not in jobs.finished[0][2]


async def test_a_follow_up_is_told_what_the_thread_found() -> None:
    jobs, run = FakeJobs(), Recorder()
    jobs.earlier = ["Encontré tres vuelos.", "x" * 5000]
    jobs.add(OWNER, thread_id="t1")
    await _worker(jobs, run).tick()
    task = run.calls[0]["task"]
    assert task.startswith("Haz algo") and "Encontré tres vuelos." in task
    assert "x" * 1500 in task and "x" * 1501 not in task


async def test_a_first_job_has_nothing_earlier_to_be_told() -> None:
    jobs, run = FakeJobs(), Recorder()
    jobs.add(OWNER, thread_id="t1")
    await _worker(jobs, run).tick()
    assert run.calls[0]["task"] == "Haz algo"


# -- the queue -----------------------------------------------------------------------


async def test_unknown_spend_ends_the_job_it_took() -> None:
    jobs, run = FakeJobs(spent=None), Recorder()
    job = jobs.add(OWNER)
    assert await _worker(jobs, run).tick() is False
    assert run.calls == [] and jobs.finished[0][:2] == (job.id, "failed")


async def test_an_empty_queue_does_nothing() -> None:
    assert await _worker(FakeJobs(), Recorder()).tick() is False


async def test_jobs_a_vanished_cloud_worker_left_are_failed_not_rerun() -> None:
    jobs, run = FakeJobs(), Recorder()
    stale = BrainJob(USER, KWAMI, "T", "t", id="old", runner_id="cloud-gone")
    jobs.rows["old"] = stale
    jobs.stale = [stale]
    await _worker(jobs, run).tick()
    assert jobs.finished[0][:2] == ("old", "failed")
    assert "nube" in jobs.finished[0][2]


async def test_it_beats_takes_work_in_its_slots_and_stops() -> None:
    jobs = FakeJobs()
    release = asyncio.Event()

    async def slow(*_args: Any, **_kwargs: Any) -> ExecutorResult:
        await release.wait()
        return ExecutorResult("Hecho.", 0.0, True)

    for _ in range(3):
        jobs.add(OWNER)
    worker = _worker(jobs, slow, slots=2)
    stop = asyncio.Event()
    loop = asyncio.create_task(worker.run_forever(stop))
    for _ in range(200):
        if len(worker._working) == 2:
            break
        await asyncio.sleep(0.005)
    assert len(worker._working) == 2 and len(jobs.queue) == 1
    assert jobs.beats == ["cloud-test-1"]
    release.set()
    for _ in range(200):
        if len(jobs.finished) == 3:
            break
        await asyncio.sleep(0.005)
    stop.set()
    await loop
    assert len(jobs.finished) == 3


async def test_a_failing_tick_does_not_stop_the_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _worker(FakeJobs(), Recorder())
    stop = asyncio.Event()
    calls = 0

    async def broken() -> bool:
        nonlocal calls
        calls += 1
        if calls >= 2:
            stop.set()
        raise RuntimeError("boom")

    monkeypatch.setattr(worker, "start_next", broken)
    await worker.run_forever(stop)
    assert calls >= 2


# -- Jarvis's own code ------------------------------------------------------------------


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def public_repo(tmp_path: Path) -> str:
    """`<base>/kwami-labs/kwami-lk-agent.git` with a `dev` branch, as a file:// base."""
    work = tmp_path / "seed"
    work.mkdir()
    _git("init", "-q", "-b", "dev", cwd=work)
    (work / "README.md").write_text("hola\n", encoding="utf-8")
    _git("add", "-A", cwd=work)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init", cwd=work)
    bare = tmp_path / "public" / "kwami-labs" / "kwami-lk-agent.git"
    bare.parent.mkdir(parents=True)
    _git("clone", "-q", "--bare", str(work), str(bare), cwd=tmp_path)
    return (tmp_path / "public").as_uri()


def _writes(name: str, text: str = "nuevo\n", commit: bool = False) -> Any:
    def act(root: Path) -> None:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
        if commit:
            _git("add", "-A", cwd=root)
            _git(
                "-c",
                "user.name=m",
                "-c",
                "user.email=m@m",
                "commit",
                "-q",
                "-m",
                "feat: x",
                cwd=root,
            )

    return act


async def test_a_code_change_becomes_a_patch_for_the_pc(public_repo: str) -> None:
    jobs, store = FakeJobs(), FakeStore()
    run = Recorder(act=_writes("src/nuevo.py", commit=True))
    job = jobs.add(OWNER, repo="agent")

    await _worker(jobs, run, store=store, repo_base=public_repo).tick()

    call = run.calls[0]
    assert call["root"].name == "kwami-lk-agent" and "kwami-labs/kwami-lk-agent" in call["task"]
    assert "jarvis/nube-" in call["task"]
    sync = jobs.syncs[0]
    assert (sync["kind"], sync["repo"], sync["files"]) == ("patch", "agent", 1)
    assert len(sync["base_ref"]) == 40
    patch = store.uploads[f"jarvis-files/{sync['prefix']}/changes.patch"].decode()
    assert "src/nuevo.py" in patch and "feat: x" in patch
    _, status, text, _, session = jobs.finished[0]
    assert status == "done" and session == "" and "abrirá el PR" in text
    # The clone is gone once the patch is kept.
    assert not call["root"].exists()
    assert sync["prefix"] == f"{OWNER}/{KWAMI}/sync/{job.id}"


async def test_work_left_uncommitted_is_kept_too(public_repo: str) -> None:
    jobs, store = FakeJobs(), FakeStore()
    jobs.add(OWNER, repo="agent")
    await _worker(
        jobs, Recorder(act=_writes("notas.md")), store=store, repo_base=public_repo
    ).tick()
    patch = next(iter(store.uploads.values())).decode()
    assert "notas.md" in patch and "chore(jarvis)" in patch


async def test_no_change_means_no_patch(public_repo: str) -> None:
    jobs = FakeJobs()
    jobs.add(OWNER, repo="agent")
    await _worker(jobs, Recorder(), repo_base=public_repo).tick()
    assert jobs.syncs == [] and "No ha habido cambios" in jobs.finished[0][2]


async def test_a_forbidden_file_is_never_kept(public_repo: str) -> None:
    jobs, store = FakeJobs(), FakeStore()
    jobs.add(OWNER, repo="agent")
    run = Recorder(act=_writes(".github/workflows/steal.yml", commit=True))
    await _worker(jobs, run, store=store, repo_base=public_repo).tick()
    assert store.uploads == {} and jobs.syncs == []
    assert jobs.finished[0][1] == "failed" and "prohibidos" in jobs.finished[0][2]


@pytest.mark.parametrize(("uploads_ok", "sync_ok"), [(False, True), (True, False)])
async def test_a_patch_that_cannot_be_kept_is_said(
    public_repo: str, uploads_ok: bool, sync_ok: bool
) -> None:
    jobs = FakeJobs()
    jobs.sync_ok = sync_ok
    jobs.add(OWNER, repo="agent")
    run = Recorder(act=_writes("a.py", commit=True))
    await _worker(jobs, run, store=FakeStore(uploads_ok), repo_base=public_repo).tick()
    assert jobs.finished[0][1] == "failed" and "no he podido guardar" in jobs.finished[0][2]


async def test_a_failed_run_is_reported_as_it_is(public_repo: str) -> None:
    jobs = FakeJobs()
    jobs.add(OWNER, repo="agent")
    await _worker(
        jobs, Recorder(ExecutorResult("No pude", 0.2, False)), repo_base=public_repo
    ).tick()
    assert jobs.finished[0][1:3] == ("failed", "No pude") and jobs.syncs == []


async def test_a_repository_it_cannot_clone_waits_for_the_pc(tmp_path: Path) -> None:
    jobs, run = FakeJobs(), Recorder()
    jobs.add(OWNER, repo="agent")
    await _worker(jobs, run, repo_base=tmp_path.as_uri()).tick()
    assert run.calls == [] and "esperar a que el ordenador" in jobs.finished[0][2]


async def test_an_unknown_repository_or_no_key_stops_it_first() -> None:
    jobs, run = FakeJobs(), Recorder()
    jobs.add(OWNER, repo="nope")
    jobs.add(USER, repo="agent")
    worker = _worker(jobs, run)
    await worker.tick()
    await worker.tick()
    assert run.calls == []
    assert "No conozco" in jobs.finished[0][2] and jobs.finished[1][2] == NO_KEY


async def test_a_git_step_that_fails_is_told(
    public_repo: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(step: str, *args: Any, **kwargs: Any) -> str:
        raise SelfWorkError(f"{step}: fatal")

    monkeypatch.setattr(cloud_module, "_must", broken)
    jobs = FakeJobs()
    jobs.add(OWNER, repo="agent")
    await _worker(jobs, Recorder(), repo_base=public_repo).tick()
    assert "No he podido terminar en la nube: leer la base: fatal" in jobs.finished[0][2]


# -- set up from the settings -----------------------------------------------------------


def test_off_unless_the_deployment_turns_it_on() -> None:
    assert cloud_worker(Settings()) is None


def test_on_without_a_database_it_cannot_run() -> None:
    assert cloud_worker(Settings(cloud_worker=True)) is None


def test_on_with_a_database_it_takes_the_deployments_choices() -> None:
    worker = cloud_worker(
        Settings(
            cloud_worker=True,
            supabase_url="https://p.supabase.co",
            supabase_secret_key=STORE_KEY,
            cloud_owners=(OWNER,),
            cloud_repo_owner="someone",
            provider_keys={"ANTHROPIC_API_KEY": "sk-ant-x"},
        )
    )
    assert worker is not None
    assert worker._owners == frozenset({OWNER}) and worker._repo_owner == "someone"
    assert worker._api_key == "sk-ant-x" and worker.runner_id.startswith("cloud-")


async def test_the_entry_point_needs_a_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cloud_module, "get_settings", lambda: Settings())
    assert await cloud_module._main([]) == 2


async def test_the_entry_point_runs_the_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[bool] = []

    async def run_forever(self: CloudWorker, stop: asyncio.Event) -> None:
        ran.append(True)

    monkeypatch.setattr(CloudWorker, "run_forever", run_forever)
    monkeypatch.setattr(
        cloud_module,
        "get_settings",
        lambda: Settings(supabase_url="https://p.supabase.co", supabase_secret_key=STORE_KEY),
    )
    assert await cloud_module._main([]) == 0 and ran == [True]


def test_the_process_exits_with_the_code(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fine(argv: Any) -> int:
        return 0

    loaded: list[Path] = []
    configured: list[Settings] = []
    monkeypatch.setattr(cloud_module, "load_env_file", loaded.append)
    monkeypatch.setattr(cloud_module, "set_settings", configured.append)
    monkeypatch.setattr(cloud_module, "_main", fine)
    with pytest.raises(SystemExit) as stopped:
        cloud_module.main(["--env-file", "otro.env"])
    assert loaded == [Path("otro.env")] and len(configured) == 1
    assert stopped.value.code == 0


def test_control_c_stops_it_quietly(monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupted(coro: Any) -> int:
        coro.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(cloud_module, "load_env_file", lambda path: False)
    monkeypatch.setattr(cloud_module, "set_settings", lambda settings: None)
    monkeypatch.setattr(cloud_module.asyncio, "run", interrupted)
    cloud_module.main([])


async def test_the_users_agents_join_the_cloud_team_too() -> None:
    jobs, run = FakeJobs(), Recorder()
    jobs.add(OWNER, agent_id="a1")
    await _worker(jobs, run).tick()
    config = run.calls[0]["config"]
    assert set(config.members) == {"facturas"} and config.permissions == "read_only"
    assert run.calls[0]["task"].startswith("You are facturas")


async def test_the_sandboxs_placeholders_never_reach_the_patch(public_repo: str) -> None:
    jobs, store = FakeJobs(), FakeStore()
    jobs.add(OWNER, repo="agent")

    def act(root: Path) -> None:
        (root / "real.py").write_text("x = 1\n", encoding="utf-8")
        (root / "package.json").write_bytes(b"")  # left untracked
        (root / "sub").mkdir()
        (root / "sub" / ".npmrc").write_bytes(b"")
        _git("add", "sub/.npmrc", "real.py", cwd=root)  # one the model committed
        _git("-c", "user.name=m", "-c", "user.email=m@m", "commit", "-q", "-m", "feat: x", cwd=root)
        (root / "yarn.lock").write_bytes(b"")  # untracked, but stays out too

    await _worker(jobs, Recorder(act=act), store=store, repo_base=public_repo).tick()

    patch = next(iter(store.uploads.values())).decode()
    assert "real.py" in patch
    assert "package.json" not in patch and "yarn.lock" not in patch
    assert jobs.syncs[0]["files"] == 1


async def test_no_placeholders_means_nothing_extra_to_commit(public_repo: str) -> None:
    jobs, store = FakeJobs(), FakeStore()
    jobs.add(OWNER, repo="agent")
    await _worker(
        jobs, Recorder(act=_writes("a.py", commit=True)), store=store, repo_base=public_repo
    ).tick()
    assert "drop the sandbox" not in next(iter(store.uploads.values())).decode()


def test_the_cloud_brief_does_not_ask_for_what_it_cannot_do() -> None:
    brief = cloud_module.CLOUD_SELF_TASK.format(slug="o/r", branch="jarvis/nube-1", task="x")
    assert "There is no shell here" in brief
    assert "Commit your work" not in brief and "run them" not in brief


def test_the_commit_is_named_after_the_job() -> None:
    assert cloud_module.commit_subject("Añade un  saludo") == "chore(jarvis): añade un saludo"
    assert cloud_module.commit_subject("   ") == "chore(jarvis): change asked from the cloud"
    assert len(cloud_module.commit_subject("x" * 200)) == cloud_module.COMMIT_SUBJECT_CHARS
