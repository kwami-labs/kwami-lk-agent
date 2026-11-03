"""The work thread: the process on the user's PC that does what the brain is asked.

    uv run python -m src.brain.worker --user <supabase user id>

With `--local --repo <repo> "<task>"` it does one job and exits, with no Supabase
(`brain/local.py`).

One job at a time, in Documents\\Jarvis, under the executor's guard. It takes only
this user's jobs, and only the service role can write them (migration 021), so a
row reaching this loop came from the user's own agent or clock.

Everything runs on the PC's Claude Code login first, which moves no money, and the
API key takes over only when the login cannot do it. Without a key there is simply
no such fallback. Beside the work runs a second, smaller line of jobs that only
answer (`mode` think or research): no files, no commands, a few at once, so the
brain's thinking does not wait behind a long job.

A job outlives the call that asked for it. If a call is still watching when it
ends, the call says the result; otherwise it goes to the next greeting, and the
API's clock tells the user about scheduled ones on the phone.

The monthly cap is checked here against the database, not a cached figure: this is
the one place work can start with nobody on the line.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import platform
import re
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..adapters.supabase import SupabaseClient, as_user_id, get_supabase
from ..envfile import load_env_file
from ..settings import get_settings
from ..utils.logging import get_logger
from .account import account_login, sign_in
from .config import (
    EXECUTOR_ROOT,
    HARD_MONTHLY_CAP_USD,
    ExecutorConfig,
    budget_notice,
    executor_in_cloud,
)
from .executor import ExecutorResult, run_executor
from .jobs import BEAT_SECONDS, BrainJob, JobStore
from .ledger import DATA_DIR, Ledger, PendingResult, SupabaseLedger
from .local import LOCAL_ID, LocalJobs
from .selfwork import (
    REPOS,
    SELF_TASK,
    SelfWorkError,
    outcome_text,
    prepare,
    publish,
    run_tests,
    spoken_title,
)
from .team import SONNET

if TYPE_CHECKING:
    from claude_agent_sdk.types import EffortLevel

logger = get_logger("brain.worker")

IDLE_POLL_SECONDS = 10.0
#: Someone is on the line waiting for these.
THINK_POLL_SECONDS = 2.0
#: Right after a thought more usually follow: a call is under way.
THINK_POLL_FAST_SECONDS = 0.4
THINK_POLL_FAST_FOR_SECONDS = 120.0
THINK_SLOTS = 3
THINK_MAX_TURNS = 12
#: How often the worker checks the PC is still signed in to Claude.
LOGIN_CHECK_SECONDS = 600.0
#: How often a running job reports, and looks for a cancellation.
RUNNING_POLL_SECONDS = 5.0
#: How long a watching call gets to take charge of a result.
HANDOVER_SECONDS = 15.0
HANDOVER_POLL_SECONDS = 2.0
#: Long work is bounded by the money, which is the limit the user set.
THREAD_MAX_TURNS = 100
#: The user's ceiling per task (2026-11-01).
THREAD_MAX_TASK_USD = 2.0
#: On the PC's Claude Code login no money moves and the plan's limits apply, but the SDK
#: still needs a figure, in what the run would have cost on the API.
LOGIN_NOTIONAL_USD = 25.0
#: A login that cannot do the work: the API key takes over. What Claude Code and the API
#: say, not single words: taking over runs the whole job again, and a job's own text can
#: mention OAuth, an author, or logging in to some site.
_LOGIN_TROUBLE = re.compile(
    r"not logged in|/login\b|invalid api key|authentication[_ ](?:error|failed)|oauth token"
    r"|\bcredits?\b|rate.?limit|usage limit|quota|ejecutor local falló",
    re.IGNORECASE,
)

Runner = Callable[..., Coroutine[Any, Any, ExecutorResult]]


def spoken_summary(result: str) -> str:
    return result.split("Detalle:", 1)[0].strip()[:1200]


class Worker:
    def __init__(
        self,
        jobs: JobStore,
        store: SupabaseClient,
        api_key: str,
        user_id: str,
        *,
        runner_id: str = "",
        run: Runner = run_executor,
        folder: Path | None = None,
        mirrors: Path | None = None,
        new_ledger: Callable[[], Ledger] | None = None,
        check_login: Callable[[], Coroutine[Any, Any, bool]] | None = None,
    ) -> None:
        self._jobs = jobs
        self._store = store
        self._api_key = api_key
        self.user_id = user_id
        self.runner_id = runner_id or f"{platform.node()}-{uuid.uuid4().hex[:6]}"
        self._run = run
        self._folder = folder or EXECUTOR_ROOT
        #: Outside the Jarvis folder, so the guard keeps the model out of them.
        self._mirrors = mirrors or DATA_DIR / "mirrors"
        self._new_ledger = new_ledger or (lambda: SupabaseLedger(self._store))
        self._check_login = check_login
        self.claude_login = False
        self._login_checked_at: float | None = None
        self._thinking: set[asyncio.Task[None]] = set()

    # -- the loop --------------------------------------------------------------

    async def run_forever(self, stop: asyncio.Event) -> None:
        beat = asyncio.create_task(self._beat_forever(stop), name="brain_worker_beat")
        think = asyncio.create_task(self._think_forever(stop), name="brain_worker_think")
        logger.info("Work thread ready on %s, in %s", self.runner_id, self._folder)
        try:
            while not stop.is_set():
                worked = False
                try:
                    worked = await self.tick()
                except Exception:
                    logger.exception("The work thread's tick failed")
                if not worked:
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(stop.wait(), IDLE_POLL_SECONDS)
        finally:
            beat.cancel()
            think.cancel()
            await asyncio.gather(beat, think, *self._thinking, return_exceptions=True)

    async def _beat_forever(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await self._note_login()
            await self._jobs.beat(self.user_id, self.runner_id, platform.node(), self.claude_login)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), BEAT_SECONDS)

    async def _note_login(self) -> None:
        now = asyncio.get_running_loop().time()
        if self._login_checked_at is None or now - self._login_checked_at >= LOGIN_CHECK_SECONDS:
            self._login_checked_at = now
            self.claude_login = await (self._check_login or account_login)()
            if not self.claude_login:
                logger.warning("Claude Code is not signed in here: run this with --login")

    async def tick(self) -> bool:
        """One look at the queue. True when a job was taken."""
        await self._fail_stale()
        spent = await self._jobs.month_spend(self.user_id)
        if spent is None:
            # Not knowing the spend is not the same as having money left.
            logger.warning("Monthly spend unreadable; leaving the queue alone")
            return False
        job = await self._jobs.claim(self.user_id, self.runner_id)
        if job is None:
            return False
        await self.process(job, spent)
        return True

    # -- thinking --------------------------------------------------------------

    async def _think_forever(self, stop: asyncio.Event) -> None:
        last_thought: float | None = None
        while not stop.is_set():
            took = False
            try:
                took = await self.think_tick()
            except Exception:
                logger.exception("The thinking line's tick failed")
            now = asyncio.get_running_loop().time()
            if took:
                last_thought = now
            if not took:
                recent = (
                    last_thought is not None and now - last_thought < THINK_POLL_FAST_FOR_SECONDS
                )
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        stop.wait(), THINK_POLL_FAST_SECONDS if recent else THINK_POLL_SECONDS
                    )

    async def think_tick(self) -> bool:
        """Start the next job that only answers, if a slot is free. True when one started."""
        if not self._jobs.thinking_ready or len(self._thinking) >= THINK_SLOTS:
            return False
        job = await self._jobs.claim(self.user_id, self.runner_id, thinking=True)
        if job is None:
            return False
        running = asyncio.create_task(self.think(job), name=f"brain_think_{job.id}")
        self._thinking.add(running)
        running.add_done_callback(self._thinking.discard)
        return True

    async def think(self, job: BrainJob) -> None:
        """Answer on the login, with the web at most. The call that asked hears the result.

        No fallback to the API key here: the bridge decides that, by what the user
        chose for this model. Nor a pending result: a thought is a part of a task the
        bridge is still running, and it stores what the user should hear.
        """
        if not job.watched():
            # Queued by a call that is gone, or that gave up: it would spend the plan
            # on an answer nobody will hear.
            await self._jobs.finish(job.id, "cancelled", "Caducada.", 0.0, "")
            return
        progress: list[str] = []
        running: asyncio.Task[ExecutorResult] = asyncio.create_task(
            self._run(
                "",
                job.model or SONNET,
                job.task,
                ExecutorConfig(
                    folder=str(self._folder),
                    permissions="think_web" if job.mode == "research" else "think",
                    max_turns=THINK_MAX_TURNS,
                    max_task_usd=LOGIN_NOTIONAL_USD,
                ),
                on_progress=progress.append,
            ),
            name=f"brain_think_run_{job.id}",
        )
        if await self._watch(job, running, progress):
            await self._jobs.finish(job.id, "cancelled", "Cancelada.", 0.0, "")
            return
        try:
            result = running.result()
        except Exception as e:
            logger.exception("Thinking job %s crashed", job.id)
            result = ExecutorResult(f"La tarea falló: {e}", 0.0, False)
        status = "done" if result.ok else "failed"
        if not await self._jobs.finish(job.id, status, result.text, 0.0, ""):
            logger.error("Thinking job %s ended %s but could not be recorded", job.id, status)

    # -- one job ---------------------------------------------------------------

    async def process(self, job: BrainJob, spent: float) -> None:
        left = HARD_MONTHLY_CAP_USD - spent
        if left <= 0:
            await self._close(
                job,
                "failed",
                f"No la he hecho: este mes ya se ha llegado al tope de "
                f"{HARD_MONTHLY_CAP_USD:.0f} dólares. El tope se reinicia el mes que viene.",
                0.0,
                "",
            )
            return

        logger.info("Job %s started: %s", job.id, job.title)
        progress: list[str] = []
        resume = await self._jobs.session_of_thread(job.thread_id) if job.thread_id else ""
        running: asyncio.Task[ExecutorResult] = asyncio.create_task(
            self._self_work(job, resume, left, progress.append)
            if job.repo
            else self._on_login_first(job, job.task, self._folder, resume, left, progress.append),
            name=f"brain_job_{job.id}",
        )
        if await self._watch(job, running, progress):
            await self._close(job, "cancelled", "Cancelada.", 0.0, resume)
            return
        try:
            result = running.result()
        except Exception as e:
            logger.exception("Job %s crashed", job.id)
            result = ExecutorResult(f"La tarea falló: {e}", 0.0, False, session_id=resume)

        cost = result.cost_usd
        await self._record_spend(job, cost)
        notice = budget_notice(spent, spent + cost)
        text = f"{notice}\n\n{result.text}" if notice else result.text
        await self._close(job, "done" if result.ok else "failed", text, cost, result.session_id)

    async def _on_login_first(
        self,
        job: BrainJob,
        task: str,
        folder: Path,
        resume: str,
        left: float,
        on_progress: Callable[[str], None],
        *,
        google: bool = True,
    ) -> ExecutorResult:
        """Run on the PC's login, and on the API key only if the login cannot do it.

        The cost is what the API run cost, and nothing for the login's. The API key
        never resumes a session: a resumed run reports the whole session, login
        included, and that was never paid. It starts afresh on the same files.
        """
        effort: EffortLevel | None = "high" if job.difficulty == "hard" else None
        result = await self._run(
            "",
            job.model or SONNET,
            task,
            ExecutorConfig(
                folder=str(folder),
                max_turns=THREAD_MAX_TURNS,
                max_task_usd=LOGIN_NOTIONAL_USD,
                google_user_id=job.user_id if google else "",
            ),
            resume=resume,
            effort=effort,
            on_progress=on_progress,
        )
        if result.ok or not self._api_key or not _LOGIN_TROUBLE.search(result.text):
            return replace(result, cost_usd=0.0)
        logger.info("Job %s: the PC's login could not do it; using the API key", job.id)
        return await self._run(
            self._api_key,
            job.model or SONNET,
            task,
            ExecutorConfig(
                folder=str(folder),
                max_turns=THREAD_MAX_TURNS,
                max_task_usd=min(THREAD_MAX_TASK_USD, left),
                google_user_id=job.user_id if google else "",
            ),
            effort=effort,
            on_progress=on_progress,
        )

    async def _self_work(
        self, job: BrainJob, resume: str, left: float, on_progress: Callable[[str], None]
    ) -> ExecutorResult:
        """Change Jarvis's own code, on the PC's login if it can, and open the pull request."""
        if job.repo not in REPOS:
            return ExecutorResult(f"No conozco el repositorio {job.repo}.", 0.0, False)
        try:
            workspace = await prepare(
                job.repo, job.thread_id or job.id, self._folder, self._mirrors
            )
            task = SELF_TASK.format(
                slug=workspace.repo.slug, branch=workspace.branch, task=job.task
            )
            # Code that gets pushed is written with nothing read from anyone's mail.
            result = await self._on_login_first(
                job, task, workspace.path, resume, left, on_progress, google=False
            )
            if not result.ok:
                return result
            on_progress("Comprobando y abriendo el PR")
            tests = await run_tests(workspace)
            published = await publish(workspace, spoken_title(job.title), result.text, tests)
        except SelfWorkError as e:
            logger.warning("Job %s: %s", job.id, e)
            return ExecutorResult(f"No he podido terminar: {e}", 0.0, False, session_id=resume)
        outcome = outcome_text(published, tests)
        return replace(result, text=f"{outcome}\n\n{result.text}", ok=not published.refused)

    async def _watch(self, job: BrainJob, running: asyncio.Task, progress: list[str]) -> bool:
        """Report until the job ends. True when it was cancelled instead."""
        while not running.done():
            await asyncio.wait({running}, timeout=RUNNING_POLL_SECONDS)
            if running.done():
                break
            await self._jobs.update_running(job.id, progress[-1] if progress else "")
            current = await self._jobs.get(job.id)
            if current is not None and current.cancel_requested:
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
                logger.info("Job %s cancelled", job.id)
                return True
        return False

    async def _record_spend(self, job: BrainJob, usd: float) -> None:
        if usd <= 0:
            return
        ledger = self._new_ledger()
        await ledger.bind(job.kwami_id, job.user_id)
        ledger.add_spend("claude-code", usd)
        await ledger.flush()

    async def _close(
        self, job: BrainJob, status: str, text: str, cost_usd: float, session_id: str
    ) -> None:
        stored = await self._jobs.finish(job.id, status, text, cost_usd, session_id)
        if not stored:
            logger.error("Job %s ended %s but could not be recorded", job.id, status)
        logger.info("Job %s %s (%.4f $)", job.id, status, cost_usd)
        # A thought is a part of a task its call is running: nothing to hand over.
        if status != "cancelled" and job.mode == "work":
            await self._hand_over(job, text)

    async def _hand_over(self, job: BrainJob, text: str) -> None:
        """Leave the result for the greeting, unless a live call takes charge of it."""
        current = await self._jobs.get(job.id)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + HANDOVER_SECONDS
        while current is not None and current.watched() and not current.delivered_at:
            if loop.time() >= deadline:
                break
            await asyncio.sleep(HANDOVER_POLL_SECONDS)
            current = await self._jobs.get(job.id)
        if current is not None and current.delivered_at:
            return
        ledger = self._new_ledger()
        await ledger.bind(job.kwami_id, job.user_id)
        ledger.add_pending(
            job.kwami_id,
            PendingResult(
                task_id=job.id,
                title=job.title,
                summary=spoken_summary(text),
                finished_at=(current.finished_at if current else "") or "",
                models=["Claude Code"],
            ),
        )
        await ledger.flush()
        await self._jobs.mark_delivered(job.id)

    async def _fail_stale(self) -> None:
        """A job whose worker vanished is failed, never rerun: it may have had effects."""
        for job in await self._jobs.stale_jobs(self.user_id):
            logger.warning("Job %s lost its worker", job.id)
            await self._close(
                job,
                "failed",
                "Se interrumpió porque el ordenador dejó de responder. Puede haber quedado a "
                "medias; revísela antes de repetirla.",
                0.0,
                job.sdk_session_id,
            )


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jarvis's work thread, on this PC.")
    parser.add_argument(
        "--user",
        default=os.environ.get("JARVIS_USER_ID", ""),
        help="The Supabase user id whose jobs this process takes.",
    )
    parser.add_argument("--local", action="store_true", help="One job now, with no Supabase.")
    parser.add_argument(
        "--login",
        action="store_true",
        help="Sign this PC in to its Claude account, in Anthropic's own page, and exit.",
    )
    parser.add_argument("--repo", default="", help=f"With --local: {', '.join(REPOS)}.")
    parser.add_argument("--thread", default="", help="With --local: carry on this thread.")
    parser.add_argument("--env-file", default="", help="The .env, if not at the repo's root.")
    parser.add_argument("task", nargs="?", default="", help="With --local: what to change.")
    return parser.parse_args(argv)


async def _run_local(arguments: argparse.Namespace) -> int:
    """One job, told on the terminal. 0 when it ended well."""
    if arguments.repo not in REPOS or not arguments.task.strip():
        print(f"--local needs --repo ({', '.join(REPOS)}) and the task.")
        return 2
    if executor_in_cloud(get_settings().executor_mode):
        print("This runs on the user's PC, not in the cloud.")
        return 2
    ledger = Ledger()
    jobs = LocalJobs(ledger)
    job = jobs.new_job(arguments.repo, arguments.task, arguments.thread)
    await asyncio.to_thread(EXECUTOR_ROOT.mkdir, parents=True, exist_ok=True)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    worker = Worker(jobs, None, api_key, LOCAL_ID, new_ledger=lambda: ledger)  # type: ignore[arg-type]
    print(f"Hilo {job.thread_id}, en {arguments.repo}. Trabajando...", flush=True)
    await worker.process(job, await jobs.month_spend(LOCAL_ID))
    print(f"\n{job.result}\n\nCoste: {job.cost_usd:.4f} $. Hilo: {job.thread_id}", flush=True)
    return 0 if job.status == "done" else 1


async def _main(argv: list[str] | None = None) -> int:
    arguments = _arguments(argv)
    if arguments.login:
        return await sign_in()
    if arguments.local:
        return await _run_local(arguments)
    settings = get_settings()
    user_id = as_user_id(arguments.user)
    if user_id is None:
        logger.error("--user needs the Supabase user id")
        return 2
    if executor_in_cloud(settings.executor_mode):
        logger.error("The work thread runs on the user's PC, not in the cloud")
        return 2
    store = get_supabase(settings)
    if store is None:
        logger.error("SUPABASE_URL and SUPABASE_SECRET_KEY are needed")
        return 2
    # Optional: without one, the PC's login is all there is.
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    await asyncio.to_thread(EXECUTOR_ROOT.mkdir, parents=True, exist_ok=True)
    stop = asyncio.Event()
    await Worker(JobStore(store), store, api_key, user_id).run_forever(stop)
    return 0


def main(argv: list[str] | None = None) -> None:
    arguments = _arguments(argv)
    root_env = Path(__file__).resolve().parents[3] / ".env"
    load_env_file(Path(arguments.env_file) if arguments.env_file else root_env)
    try:
        raise SystemExit(asyncio.run(_main(argv)))
    except KeyboardInterrupt:
        logger.info("Work thread stopped")


if __name__ == "__main__":  # pragma: no cover - process entry point
    main()
