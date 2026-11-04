"""The work thread: what it takes, what it refuses, and who hears the result.

`JobStore` is replaced by an in-memory table with the same methods, so what is
pinned is the worker's decisions. The wire format is in `test_brain_jobs.py`.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from src.brain import worker as worker_module
from src.brain.config import HARD_MONTHLY_CAP_USD, budget_notice
from src.brain.executor import ExecutorResult
from src.brain.jobs import BrainJob
from src.brain.worker import Worker, spoken_summary

USER = "11111111-1111-4111-8111-111111111111"
KWAMI = "22222222-2222-4222-8222-222222222222"


@pytest.fixture(autouse=True)
def no_real_login(monkeypatch: pytest.MonkeyPatch) -> None:
    """The heartbeat would otherwise start the real Claude Code, on every test."""

    async def signed_out() -> bool:
        return False

    monkeypatch.setattr(worker_module, "account_login", signed_out)


class FakeJobs:
    def __init__(self, spent: float | None = 0.0) -> None:
        self.thinking_ready = False
        self.spent = spent
        self.queue: list[BrainJob] = []
        self.rows: dict[str, BrainJob] = {}
        self.stale: list[BrainJob] = []
        self.sessions: dict[str, str] = {}
        self.finished: list[tuple[str, str, str, float, str]] = []
        self.progress: list[str] = []
        self.thread_spent = 0.0

    def add(self, **extra: Any) -> BrainJob:
        job = BrainJob(USER, KWAMI, "Título", "Haz algo", id=f"j{len(self.rows)}", **extra)
        self.queue.append(job)
        self.rows[job.id] = job
        return job

    async def month_spend(self, user_id: str) -> float | None:
        return self.spent

    async def claim(self, user_id: str, runner_id: str, thinking: bool = False) -> BrainJob | None:
        return self.queue.pop(0) if self.queue and not thinking else None

    async def stale_jobs(self, user_id: str) -> list[BrainJob]:
        stale, self.stale = self.stale, []
        return stale

    async def session_of_thread(self, thread_id: str) -> str:
        return self.sessions.get(thread_id, "")

    async def thread_cost(self, thread_id: str) -> float:
        return self.thread_spent

    async def update_running(self, job_id: str, progress: str) -> bool:
        self.progress.append(progress)
        return True

    async def get(self, job_id: str) -> BrainJob | None:
        return self.rows.get(job_id)

    async def finish(
        self, job_id: str, status: str, result: str, cost_usd: float, session_id: str
    ) -> bool:
        self.finished.append((job_id, status, result, cost_usd, session_id))
        self.rows[job_id].status = status
        return True

    async def mark_delivered(self, job_id: str) -> bool:
        self.rows[job_id].delivered_at = "now"
        return True

    async def beat(self, *args: Any) -> bool:
        return True


class FakeLedger:
    instances: list[FakeLedger] = []

    def __init__(self, store: Any) -> None:
        self.spend: list[tuple[str, float]] = []
        self.pending: list[Any] = []
        self.bound: tuple[str, str | None] = ("", None)
        FakeLedger.instances.append(self)

    async def bind(self, kwami_id: str, user_id: str | None = None) -> None:
        self.bound = (kwami_id, user_id)

    def add_spend(self, model_id: str, usd: float) -> None:
        self.spend.append((model_id, usd))

    def add_pending(self, kwami_id: str, item: Any) -> None:
        self.pending.append(item)

    async def flush(self, timeout: float = 5.0) -> None:
        return None

    @classmethod
    def all_pending(cls) -> list[Any]:
        return [p for ledger in cls.instances for p in ledger.pending]

    @classmethod
    def all_spend(cls) -> list[tuple[str, float]]:
        return [s for ledger in cls.instances for s in ledger.spend]


@pytest.fixture(autouse=True)
def ledger(monkeypatch: pytest.MonkeyPatch) -> type[FakeLedger]:
    FakeLedger.instances = []
    monkeypatch.setattr(worker_module, "SupabaseLedger", FakeLedger)
    monkeypatch.setattr(worker_module, "HANDOVER_POLL_SECONDS", 0.01)
    monkeypatch.setattr(worker_module, "RUNNING_POLL_SECONDS", 0.01)
    return FakeLedger


class Recorder:
    def __init__(self, result: ExecutorResult | Exception | None = None) -> None:
        self.result = result or ExecutorResult("Hecho.\n\nDetalle: x", 0.5, True, session_id="s1")
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self, api_key: str, model: str, task: str, config: Any, **kwargs: Any
    ) -> ExecutorResult:
        self.calls.append(
            {"api_key": api_key, "model": model, "task": task, "config": config, **kwargs}
        )
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _worker(jobs: FakeJobs, run: Any, tmp_path: Path) -> Worker:
    return Worker(jobs, object(), "ak", USER, runner_id="r1", run=run, folder=tmp_path)  # type: ignore[arg-type]


async def test_an_empty_queue_does_nothing(tmp_path: Path) -> None:
    run = Recorder()
    assert await _worker(FakeJobs(), run, tmp_path).tick() is False
    assert run.calls == []


async def test_it_will_not_start_without_knowing_the_spend(tmp_path: Path) -> None:
    jobs, run = FakeJobs(spent=None), Recorder()
    jobs.add()
    assert await _worker(jobs, run, tmp_path).tick() is False
    assert run.calls == [] and len(jobs.queue) == 1


async def test_a_job_runs_in_the_folder_and_its_spend_is_recorded(
    tmp_path: Path, ledger: type[FakeLedger]
) -> None:
    jobs, run = FakeJobs(), Recorder()
    job = jobs.add(model="claude-opus-5-5")

    assert await _worker(jobs, run, tmp_path).tick() is True

    call = run.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["task"] == "Haz algo"
    assert call["config"].folder == str(tmp_path)
    assert call["config"].permissions == "full_no_push"
    # On the PC's login: no key, and no money moved whatever the SDK reports.
    assert call["api_key"] == ""
    assert jobs.finished == [(job.id, "done", "Hecho.\n\nDetalle: x", 0.0, "s1")]
    assert ledger.all_spend() == []


async def test_when_the_login_cannot_the_key_takes_over_and_that_is_spend(
    tmp_path: Path, ledger: type[FakeLedger]
) -> None:
    jobs = FakeJobs(spent=HARD_MONTHLY_CAP_USD - 0.5)
    results = iter(
        [ExecutorResult("Usage limit reached", 3.0, False), ExecutorResult("Hecho.", 0.25, True)]
    )
    calls: list[tuple[str, float, str]] = []

    async def run(api_key: str, model: str, task: str, config: Any, **kwargs: Any) -> Any:
        calls.append((api_key, config.max_task_usd, kwargs.get("resume", "")))
        return next(results)

    jobs.sessions["t1"] = "s1"
    job = jobs.add(thread_id="t1")
    await _worker(jobs, run, tmp_path).tick()

    assert calls[0] == ("", 25.0, "s1")
    # The key's run never resumes, and never spends past the month's ceiling.
    assert calls[1][0] != "" and calls[1][1] == pytest.approx(0.5) and calls[1][2] == ""
    assert jobs.finished[0][:2] == (job.id, "done")
    assert ledger.all_spend() == [("claude-code", 0.25)]


async def test_no_model_means_the_default_and_hard_work_thinks_harder(tmp_path: Path) -> None:
    jobs, run = FakeJobs(), Recorder()
    jobs.add(difficulty="hard")
    await _worker(jobs, run, tmp_path).tick()
    assert run.calls[0]["model"] == "claude-sonnet-5-5"
    assert run.calls[0]["effort"] == "high"


async def test_a_follow_up_resumes_its_thread(tmp_path: Path) -> None:
    jobs, run = FakeJobs(), Recorder()
    jobs.sessions["t1"] = "earlier"
    jobs.add(thread_id="t1")
    await _worker(jobs, run, tmp_path).tick()
    assert run.calls[0]["resume"] == "earlier"


async def test_past_the_ceiling_it_stops_and_says_so(
    tmp_path: Path, ledger: type[FakeLedger]
) -> None:
    jobs, run = FakeJobs(spent=HARD_MONTHLY_CAP_USD), Recorder()
    job = jobs.add()

    assert await _worker(jobs, run, tmp_path).tick() is True

    assert run.calls == []
    assert jobs.finished[0][:2] == (job.id, "failed")
    assert "tope" in jobs.finished[0][2]
    assert "tope" in ledger.all_pending()[0].summary


@pytest.mark.parametrize(
    ("text", "takes_over"),
    [
        ("Not logged in · Please run /login", True),
        ("Invalid API key", True),
        ('{"type":"authentication_error"}', True),
        ("OAuth token has expired", True),
        ("Credit balance is too low", True),
        ("Claude usage limit reached", True),
        ("El ejecutor local falló: ProcessError", True),
        ("La página pide log in y no tengo la contraseña.", False),
        ("El author del commit no coincide.", False),
        ("Falta el flujo OAuth en la API.", False),
    ],
)
def test_only_what_a_login_says_hands_the_job_to_the_api_key(text: str, takes_over: bool) -> None:
    """Handing over runs the whole job again, with everything it already did."""
    assert bool(worker_module._LOGIN_TROUBLE.search(text)) is takes_over


async def test_crossing_a_line_warns_in_the_result(tmp_path: Path) -> None:
    jobs = FakeJobs(spent=119.9)
    results = iter([ExecutorResult("Sin credit", 0.0, False), ExecutorResult("Hecho.", 0.5, True)])

    async def run(*args: Any, **kwargs: Any) -> Any:
        return next(results)

    jobs.add()
    await _worker(jobs, run, tmp_path).tick()
    assert jobs.finished[0][2].startswith("Aviso: este mes ya van 120 de los 150")


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        (10.0, 20.0, ""),
        (119.0, 121.0, "ya van 121"),
        (121.0, 122.0, ""),
        (149.0, 150.5, "se ha llegado al tope"),
        (151.0, 152.0, ""),
    ],
)
def test_the_notice_comes_once_per_line(before: float, after: float, expected: str) -> None:
    notice = budget_notice(before, after)
    assert expected in notice if expected else notice == ""


async def test_a_failed_run_is_failed_with_its_session_kept(tmp_path: Path) -> None:
    jobs = FakeJobs()
    run = Recorder(ExecutorResult("No pude.", 0.1, False, session_id="s2"))
    job = jobs.add()
    await _worker(jobs, run, tmp_path).tick()
    assert jobs.finished == [(job.id, "failed", "No pude.", 0.0, "s2")]


async def test_a_crash_does_not_take_the_worker_down(tmp_path: Path) -> None:
    jobs, run = FakeJobs(), Recorder(RuntimeError("boom"))
    jobs.add()
    assert await _worker(jobs, run, tmp_path).tick() is True
    assert jobs.finished[0][1] == "failed" and "boom" in jobs.finished[0][2]


async def test_cancelling_stops_the_run(tmp_path: Path, ledger: type[FakeLedger]) -> None:
    jobs = FakeJobs()
    job = jobs.add()
    started = asyncio.Event()

    async def slow(*args: Any, **kwargs: Any) -> ExecutorResult:
        kwargs["on_progress"]("Bash")
        started.set()
        await asyncio.sleep(60)
        raise AssertionError("not cancelled")

    worker = _worker(jobs, slow, tmp_path)
    ticking = asyncio.create_task(worker.tick())
    await started.wait()
    await asyncio.sleep(0.03)
    assert "Bash" in jobs.progress
    jobs.rows[job.id].cancel_requested = True
    await asyncio.wait_for(ticking, 2)

    assert jobs.finished[0][:3] == (job.id, "cancelled", "Cancelada.")
    assert ledger.all_pending() == []


# -- who tells the user ---------------------------------------------------------


async def test_with_nobody_watching_it_waits_for_the_greeting(
    tmp_path: Path, ledger: type[FakeLedger]
) -> None:
    jobs, run = FakeJobs(), Recorder()
    job = jobs.add()
    await _worker(jobs, run, tmp_path).tick()
    pending = ledger.all_pending()
    assert [p.summary for p in pending] == ["Hecho."]
    assert pending[0].task_id == job.id
    assert jobs.rows[job.id].delivered_at == "now"


async def test_a_watching_call_takes_charge(tmp_path: Path, ledger: type[FakeLedger]) -> None:
    jobs, run = FakeJobs(), Recorder()
    job = jobs.add()
    job.watcher_seen_at = datetime.now(UTC).isoformat()
    job.delivered_at = ""

    original = jobs.finish

    async def finish_and_take(*args: Any) -> bool:
        done = await original(*args)
        job.delivered_at = "by the call"
        return done

    jobs.finish = finish_and_take  # type: ignore[method-assign]
    await _worker(jobs, run, tmp_path).tick()
    assert ledger.all_pending() == []


async def test_a_call_that_never_takes_it_loses_it_to_the_greeting(
    tmp_path: Path, ledger: type[FakeLedger], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker_module, "HANDOVER_SECONDS", 0.02)
    jobs, run = FakeJobs(), Recorder()
    job = jobs.add()
    job.watcher_seen_at = datetime.now(UTC).isoformat()
    await _worker(jobs, run, tmp_path).tick()
    assert len(ledger.all_pending()) == 1


async def test_a_job_whose_pc_vanished_fails_and_is_not_rerun(
    tmp_path: Path, ledger: type[FakeLedger]
) -> None:
    jobs, run = FakeJobs(), Recorder()
    job = BrainJob(USER, KWAMI, "Vieja", "x", id="old", status="running", sdk_session_id="s9")
    jobs.rows[job.id] = job
    jobs.stale = [job]

    assert await _worker(jobs, run, tmp_path).tick() is False

    assert run.calls == []
    assert jobs.finished[0][1] == "failed" and jobs.finished[0][4] == "s9"
    assert "a medias" in ledger.all_pending()[0].summary


def test_only_the_spoken_part_goes_to_the_greeting() -> None:
    assert spoken_summary("Hablado.\n\nDetalle: largo") == "Hablado."


async def test_the_loop_survives_a_failing_tick_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(worker_module, "IDLE_POLL_SECONDS", 0.01)
    worker = _worker(FakeJobs(), Recorder(), tmp_path)
    stop = asyncio.Event()
    calls = 0

    async def failing() -> bool:
        nonlocal calls
        calls += 1
        if calls >= 2:
            stop.set()
        raise RuntimeError("down")

    worker.start_next = failing  # type: ignore[method-assign]
    await asyncio.wait_for(worker.run_forever(stop), 2)
    assert calls == 2
    assert "tick failed" in caplog.text


async def test_with_the_slots_full_the_loop_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker_module, "IDLE_POLL_SECONDS", 0.001)
    monkeypatch.setattr(worker_module, "WORK_SLOTS", 0)
    worker = _worker(FakeJobs(), Recorder(), tmp_path)
    assert await worker.start_next() is False
