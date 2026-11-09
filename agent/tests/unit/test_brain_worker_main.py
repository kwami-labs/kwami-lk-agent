"""The work thread as a process: what it needs to start, and how its loops end."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from claude_agent_sdk import TextBlock, ToolUseBlock

from src.brain import local as local_module
from src.brain import selfwork
from src.brain import worker as worker_module
from src.brain.bridge import Bridge
from src.brain.config import ExecutorConfig, default_brain
from src.brain.executor import ExecutorResult, run_executor
from src.brain.jobs import BrainJob
from src.brain.ledger import Ledger, SupabaseLedger
from src.brain.local import LocalJobs
from src.brain.selfwork import SelfRepo, Workspace
from src.brain.worker import Worker, _arguments, _main, main
from src.settings import Settings
from tests.unit.test_brain_executor import _assistant, _result, jarvis, sdk  # noqa: F401

USER = "11111111-1111-4111-8111-111111111111"
KWAMI = "22222222-2222-4222-8222-222222222222"


@pytest.fixture(autouse=True)
def no_real_login(monkeypatch: pytest.MonkeyPatch) -> None:
    """The heartbeat would otherwise start the real Claude Code, on every test."""

    async def signed_out() -> bool:
        return False

    monkeypatch.setattr(worker_module, "account_login", signed_out)


class Jobs:
    thinking_ready = False
    sync_ready = True

    def __init__(self, stored: bool = True) -> None:
        self.stored = stored
        self.beats = 0
        self.sync_looks = 0

    async def claim_sync(self, user_id: str, runner_id: str) -> None:
        self.sync_looks += 1

    async def beat(self, *args: Any) -> bool:
        self.beats += 1
        return True

    async def finish(self, *args: Any) -> bool:
        return self.stored

    async def get(self, job_id: str) -> BrainJob | None:
        return None


def _worker(jobs: Any, tmp_path: Path) -> Worker:
    return Worker(jobs, object(), "ak", USER, runner_id="r", folder=tmp_path)  # type: ignore[arg-type]


async def test_after_a_job_it_looks_at_the_queue_again_at_once(tmp_path: Path) -> None:
    worker = _worker(Jobs(), tmp_path)
    stop = asyncio.Event()
    looks = 0

    async def start_next() -> bool:
        nonlocal looks
        looks += 1
        if looks == 2:
            stop.set()
        return True

    worker.start_next = start_next  # type: ignore[method-assign]
    await asyncio.wait_for(worker.run_forever(stop), 2)
    assert looks == 2


async def test_the_heartbeat_ends_with_the_stop(tmp_path: Path) -> None:
    jobs = Jobs()
    stop = asyncio.Event()
    stop.set()
    await _worker(jobs, tmp_path)._beat_forever(stop)
    assert jobs.beats == 0


async def test_a_job_already_over_needs_no_watching(tmp_path: Path) -> None:
    async def done() -> None:
        return None

    running = asyncio.create_task(done())
    await running
    job = BrainJob(USER, KWAMI, "T", "t", id="j")
    assert await _worker(Jobs(), tmp_path)._watch(job, running, []) is False


async def test_a_result_that_cannot_be_recorded_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    job = BrainJob(USER, KWAMI, "T", "t", id="j")
    await _worker(Jobs(stored=False), tmp_path)._close(job, "cancelled", "x", 0.0, "")
    assert "could not be recorded" in caplog.text


# -- starting ---------------------------------------------------------------------


def test_the_user_can_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JARVIS_USER_ID", USER)
    assert _arguments([]).user == USER
    assert _arguments(["--user", "other"]).user == "other"


@pytest.fixture
def startable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    started: list[Any] = []

    class FakeWorker:
        def __init__(self, jobs: Any, store: Any, api_key: str, user_id: str) -> None:
            started.append((api_key, user_id))

        async def run_forever(self, stop: asyncio.Event) -> None:
            return None

    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")
    monkeypatch.setattr(worker_module, "get_settings", lambda: Settings(executor_mode="local"))
    monkeypatch.setattr(worker_module, "get_supabase", lambda settings: object())
    monkeypatch.setattr(worker_module, "executor_in_cloud", lambda mode: False)
    monkeypatch.setattr(worker_module, "EXECUTOR_ROOT", tmp_path / "Jarvis")
    monkeypatch.setattr(worker_module, "Worker", FakeWorker)
    return started


async def test_it_starts_when_it_has_everything(startable: list[Any]) -> None:
    assert await _main(["--user", USER]) == 0
    assert startable == [("ak", USER)]


async def test_it_needs_a_user_id(startable: list[Any]) -> None:
    assert await _main(["--user", "not-a-uuid"]) == 2
    assert startable == []


async def test_it_refuses_the_cloud(startable: list[Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_module, "executor_in_cloud", lambda mode: True)
    assert await _main(["--user", USER]) == 2


async def test_it_needs_supabase(startable: list[Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_module, "get_supabase", lambda settings: None)
    assert await _main(["--user", USER]) == 2


async def test_login_signs_in_and_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    async def sign_in() -> int:
        return 7

    monkeypatch.setattr(worker_module, "sign_in", sign_in)
    assert await _main(["--login"]) == 7


async def test_it_runs_on_the_login_alone(
    startable: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert await _main(["--user", USER]) == 0


def test_the_process_exits_with_the_code(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fine(argv: Any) -> int:
        return 0

    loaded: list[Path] = []
    monkeypatch.setattr(worker_module, "load_env_file", loaded.append)
    monkeypatch.setattr(worker_module, "_main", fine)
    with pytest.raises(SystemExit) as stopped:
        main(["--env-file", "otro.env"])
    assert loaded == [Path("otro.env")]
    assert stopped.value.code == 0


def test_control_c_stops_it_quietly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def interrupted(coro: Any) -> int:
        coro.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(worker_module, "load_env_file", lambda path: False)
    monkeypatch.setattr(worker_module.asyncio, "run", interrupted)
    main([])


# -- the corners elsewhere ----------------------------------------------------------


async def test_text_beside_a_tool_call_is_not_progress(sdk: Any, jarvis: Path) -> None:  # noqa: F811
    sdk.messages = [
        _assistant(TextBlock(text="Voy"), ToolUseBlock(id="1", name="Bash", input={})),
        _result("Hecho"),
    ]
    seen: list[str] = []
    await run_executor(
        "ak", "claude-sonnet-5-5", "t", ExecutorConfig(str(jarvis)), on_progress=seen.append
    )
    assert seen == ["Bash"]


def test_a_ledger_only_notes_what_the_worker_recorded(tmp_path: Path) -> None:
    ledger = SupabaseLedger(object(), tmp_path)  # type: ignore[arg-type]
    ledger.note_spend("claude-code", 0.5)
    ledger.note_spend("claude-code", 0.0)
    assert ledger.month_spend()["usd"] == 0.5
    disk = Ledger(tmp_path)
    disk.note_spend("claude-code", 0.5)
    assert disk.month_spend()["usd"] == 0.0


async def test_without_a_code_model_long_work_stays_in_the_call(tmp_path: Path) -> None:
    keys = {"ANTHROPIC_API_KEY": "a"}
    bridge = Bridge(keys=keys, ledger=Ledger(tmp_path), settings=Settings(executor_mode="local"))
    bridge.config = default_brain()
    bridge.config.models = [m for m in bridge.config.models if m.role != "code"]
    bridge.config.split = False
    bridge.kwami_id, bridge.user_id = KWAMI, USER

    async def online(user_id: str) -> bool:
        return True

    bridge.jobs = SimpleNamespace(runner_online=online)  # type: ignore[assignment]

    async def answer(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            text="Aquí",
            sources=[],
            ok=True,
            cost_usd=0.0,
            model_id="m",
            model="m",
            rate_limited=False,
            via_account=False,
        )

    bridge._ask = answer  # type: ignore[method-assign]
    task = bridge.start(
        task="t", title="T", kind="analysis", difficulty="normal", long_running=True
    )
    assert not isinstance(task, str)
    await asyncio.gather(task.handle, return_exceptions=True)  # type: ignore[arg-type]
    await asyncio.gather(*bridge._background, return_exceptions=True)
    assert task.result == "Aquí" and task.job_id == ""


async def test_cancelling_stops_the_tests(tmp_path: Path) -> None:
    import sys

    workspace = Workspace(
        "agent",
        SelfRepo("x", tests=f'"{sys.executable}" -c "import time; time.sleep(30)"'),
        tmp_path,
        tmp_path,
        "jarvis/ab",
    )
    running = asyncio.create_task(selfwork.run_tests(workspace))
    await asyncio.sleep(0.2)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running


async def test_a_ready_pull_request_is_not_a_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []

    async def fake_run(*args: str, **kwargs: Any) -> tuple[int, str]:
        calls.append(args)
        return (0, "") if "list" in args else (0, "https://github.com/x/y/pull/8")

    monkeypatch.setattr(selfwork, "_run", fake_run)
    workspace = Workspace("agent", SelfRepo("x/y"), Path("."), Path("."), "jarvis/ab")
    assert (await selfwork._open_pull_request(workspace, "T", "B", False)).endswith("/8")
    assert "--draft" not in calls[1]


# -- --local: one job, no Supabase ----------------------------------------------------


@pytest.fixture
def local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {"runs": []}
    data = tmp_path / "data"

    async def run(self: Worker, job: BrainJob, spent: float) -> None:
        calls["runs"].append((job, spent, self._api_key))
        await self._jobs.update_running(job.id, "Bash")
        await self._jobs.update_running(job.id, "Bash")
        await self._close(job, calls.get("status", "done"), "Hecho.", 0.25, "s1")

    monkeypatch.setattr(Worker, "process", run)
    monkeypatch.setattr(worker_module, "Ledger", lambda: Ledger(data))
    monkeypatch.setattr(worker_module, "get_settings", lambda: Settings(executor_mode="local"))
    monkeypatch.setattr(worker_module, "executor_in_cloud", lambda mode: False)
    monkeypatch.setattr(worker_module, "EXECUTOR_ROOT", tmp_path / "Jarvis")
    monkeypatch.setattr(local_module, "DATA_DIR", data)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")
    return calls


async def test_local_runs_one_job_and_says_how_it_ended(
    local: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert await _main(["--local", "--repo", "agent", "Arregla X\ncon detalle"]) == 0
    job, spent, api_key = local["runs"][0]
    assert (job.repo, job.title, spent, api_key) == ("agent", "Arregla X", 0.0, "ak")
    assert job.task == "Arregla X\ncon detalle"
    out = capsys.readouterr().out
    assert out.count("Bash") == 1
    assert "Hecho." in out and "0.2500 $" in out and job.thread_id in out


async def test_local_says_when_it_failed(local: dict[str, Any]) -> None:
    local["status"] = "failed"
    assert await _main(["--local", "--repo", "agent", "X"]) == 1


async def test_a_local_thread_can_be_carried_on(local: dict[str, Any], tmp_path: Path) -> None:
    await _main(["--local", "--repo", "agent", "--thread", "t1", "X"])
    jobs = LocalJobs(Ledger(tmp_path), tmp_path / "data" / "local_threads.json")
    assert await jobs.session_of_thread("t1") == "s1"
    assert await jobs.session_of_thread("t2") == ""


@pytest.mark.parametrize(
    "argv", [["--local", "X"], ["--local", "--repo", "agent"], ["--local", "--repo", "labs", "X"]]
)
async def test_local_needs_a_known_repo_and_a_task(local: dict[str, Any], argv: list[str]) -> None:
    assert await _main(argv) == 2
    assert local["runs"] == []


async def test_local_refuses_the_cloud(
    local: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker_module, "executor_in_cloud", lambda mode: True)
    assert await _main(["--local", "--repo", "agent", "X"]) == 2


async def test_local_spend_counts_toward_the_ceiling(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path)
    ledger.add_spend("claude-code", 3.0)
    assert await LocalJobs(ledger, tmp_path / "t.json").month_spend("local") == 3.0


async def test_local_jobs_tolerate_corners(tmp_path: Path) -> None:
    threads = tmp_path / "t.json"
    jobs = LocalJobs(Ledger(tmp_path), threads)
    assert await jobs.get("x") is None
    assert await jobs.finish("x", "done", "r", 0.0, "s") is False
    assert await jobs.thread_cost("t") == 0.0
    threads.write_text("[1]", encoding="utf-8")
    assert await jobs.session_of_thread("t") == ""
    threads.write_text("{roto", encoding="utf-8")
    assert await jobs.session_of_thread("t") == ""
    job = jobs.new_job("agent", "X")
    assert job.delivered_at == "terminal"
    assert await jobs.finish(job.id, "failed", "r", 0.0, "") is True
    assert await jobs.update_running(job.id, "") is True


async def test_with_no_key_a_failed_login_stays_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    async def run(api_key: str, *args: Any, **kwargs: Any) -> ExecutorResult:
        calls.append(api_key)
        return ExecutorResult("Not logged in", 0.0, False)

    async def ready(*args: Any) -> Any:
        return SimpleNamespace(
            repo=SelfRepo("x"), branch="jarvis/ab", path=tmp_path, mirror=tmp_path
        )

    monkeypatch.setattr(worker_module, "prepare", ready)
    jobs = LocalJobs(Ledger(tmp_path), tmp_path / "t.json")
    job = jobs.new_job("agent", "X")
    worker = Worker(jobs, None, "", "local", run=run, new_ledger=lambda: Ledger(tmp_path))  # type: ignore[arg-type]
    await worker.process(job, 0.0)
    assert calls == [""]
    assert job.status == "failed" and "Not logged in" in job.result
