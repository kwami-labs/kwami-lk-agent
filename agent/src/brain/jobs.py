"""The queue between whoever asks for work and the process on the user's PC.

A call on LiveKit Cloud cannot reach the PC, and the PC cannot be reached from
outside without a tunnel the user does not want. Both can reach Supabase, so the
call writes a row in `brain_jobs` and `brain/worker.py`, on the PC, picks it up.

Three clocks keep it honest, all in the table:

* `brain_runners.last_seen_at` -- the PC is there. Without it, a call does not
  queue the work: it runs it in the cloud, which is what the user chose.
* `heartbeat_at` -- a job is still being worked on. A stale one means the PC went
  away mid-job, and it is failed rather than run twice: half-done work may already
  have had effects.
* `watcher_seen_at` -- a live call is waiting for this result and will say it.
  When it lapses, the result goes to the next greeting instead.

With migration 029 a second kind of worker joins, in the cloud and shared by every
user who allows it (`brain/cloud.py`). It beats in `brain_cloud_runners`, takes
work with `brain_claim_cloud_job` -- which leaves a user's work to their PC while
the PC is there -- and leaves what the PC should bring home in `brain_sync`.

Like every Supabase call in the agent, nothing here raises: None and False mean
it could not be done, and the caller decides what that means.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse

from ..adapters.supabase import SupabaseClient

#: A worker beats every `BEAT_SECONDS`, and is gone after missing a few.
BEAT_SECONDS = 30.0
RUNNER_FRESH_SECONDS = 90.0
#: How long without a heartbeat before a running job is given up on.
JOB_STALE_SECONDS = 180.0
#: How long a call's interest in a job lasts unless it renews it.
WATCH_FRESH_SECONDS = 30.0
#: A cloud worker beats as often as a PC's, and is gone as soon.
CLOUD_FRESH_SECONDS = RUNNER_FRESH_SECONDS
#: How long `auto` work waits for a PC that may be switching on before the cloud takes it.
CLOUD_PATIENCE_SECONDS = 60
#: Jobs of one user the cloud runs at once.
CLOUD_PER_USER = 2
#: Every cloud worker's runner id starts so: the claim function counts them by it.
CLOUD_RUNNER_PREFIX = "cloud-"

TERMINAL = frozenset({"done", "failed", "cancelled"})


def _ago(seconds: float, now: datetime | None = None) -> str:
    return ((now or datetime.now(UTC)) - timedelta(seconds=seconds)).isoformat()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _month() -> str:
    return datetime.now(UTC).strftime("%Y-%m")


def _text(value: Any) -> str:
    return str(value) if value is not None else ""


@dataclass
class BrainJob:
    user_id: str
    kwami_id: str
    title: str
    task: str
    id: str = ""
    thread_id: str = ""
    source: str = "call"
    difficulty: str = "normal"
    model: str = ""
    #: Set when the job changes Jarvis's own code: a key of `selfwork.REPOS`.
    repo: str = ""
    #: work: acts, on the PC. think, research: only answers, with no tools or only
    #: the web's, and does not wait behind the work.
    mode: str = "work"
    #: auto: the PC if it is there, else the cloud if allowed. pc, cloud: only there.
    target: str = "auto"
    #: The user's agent this job runs as (`brain_agents`), if any.
    agent_id: str = ""
    runner_id: str = ""
    #: Set by the worker before the run, never stored: the executor's mode for the
    #: agent the job runs as, and the user's agents that join the team.
    permissions: str = "full_no_push"
    members: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"
    cancel_requested: bool = False
    progress: str = ""
    result: str = ""
    cost_usd: float = 0.0
    sdk_session_id: str = ""
    finished_at: str = ""
    watcher_seen_at: str = ""
    delivered_at: str = ""

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> BrainJob:
        return cls(
            user_id=_text(row.get("user_id")),
            kwami_id=_text(row.get("kwami_id")),
            title=_text(row.get("title")),
            task=_text(row.get("task")),
            id=_text(row.get("id")),
            thread_id=_text(row.get("thread_id")),
            source=_text(row.get("source")) or "call",
            difficulty=_text(row.get("difficulty")) or "normal",
            model=_text(row.get("model")),
            repo=_text(row.get("repo")),
            mode=_text(row.get("mode")) or "work",
            target=_text(row.get("target")) or "auto",
            agent_id=_text(row.get("agent_id")),
            runner_id=_text(row.get("runner_id")),
            status=_text(row.get("status")) or "queued",
            cancel_requested=row.get("cancel_requested") is True,
            progress=_text(row.get("progress")),
            result=_text(row.get("result")),
            cost_usd=float(row.get("cost_usd") or 0.0),
            sdk_session_id=_text(row.get("sdk_session_id")),
            finished_at=_text(row.get("finished_at")),
            watcher_seen_at=_text(row.get("watcher_seen_at")),
            delivered_at=_text(row.get("delivered_at")),
        )

    @property
    def finished(self) -> bool:
        return self.status in TERMINAL

    @property
    def in_cloud(self) -> bool:
        return self.runner_id.startswith(CLOUD_RUNNER_PREFIX)

    def watched(self, now: datetime | None = None) -> bool:
        """Whether a live call renewed its interest recently."""
        if not self.watcher_seen_at:
            return False
        seen = datetime.fromisoformat(self.watcher_seen_at.replace("Z", "+00:00"))
        return ((now or datetime.now(UTC)) - seen).total_seconds() < WATCH_FRESH_SECONDS


class JobStore:
    def __init__(self, store: SupabaseClient) -> None:
        self._store = store
        #: Whether the database has migration 024. Learnt from the heartbeat, so a
        #: thinking line with nothing to claim does not fail every few seconds.
        self.thinking_ready = False
        #: Whether it has migration 029, learnt from the last look for what the cloud
        #: left: the PC then looks far less often until it does.
        self.sync_ready = True

    # -- the side that asks --------------------------------------------------

    async def runner_online(self, user_id: str) -> bool:
        rows = await self._store.select(
            "brain_runners",
            {
                "select": "runner_id",
                "user_id": f"eq.{user_id}",
                "last_seen_at": f"gte.{_ago(RUNNER_FRESH_SECONDS)}",
                "limit": "1",
            },
            what="whether the PC's worker is there",
        )
        return bool(rows)

    async def cloud_online(self) -> bool:
        """Whether a cloud worker is there to take work (migration 029)."""
        rows = await self._store.select(
            "brain_cloud_runners",
            {
                "select": "runner_id",
                "last_seen_at": f"gte.{_ago(CLOUD_FRESH_SECONDS)}",
                "limit": "1",
            },
            what="whether a cloud worker is there",
        )
        return bool(rows)

    async def cloud_allowed(self, user_id: str) -> bool:
        """Whether this user lets the cloud work for them. No, if it cannot be read."""
        rows = await self._store.select(
            "brain_cloud_settings",
            {"select": "enabled", "user_id": f"eq.{user_id}", "limit": "1"},
            what="whether the user allows the cloud",
        )
        return bool(rows and rows[0].get("enabled") is True)

    async def account_ready(self, user_id: str) -> bool:
        """Whether a fresh worker of this user reports a Claude login on its PC."""
        rows = await self._store.select(
            "brain_runners",
            {
                "select": "runner_id",
                "user_id": f"eq.{user_id}",
                "claude_login": "is.true",
                "last_seen_at": f"gte.{_ago(RUNNER_FRESH_SECONDS)}",
                "limit": "1",
            },
            what="whether the PC has a Claude login",
        )
        return bool(rows)

    async def enqueue(self, job: BrainJob) -> bool:
        """Queue `job`, minting its ids here: the insert returns nothing to read them from."""
        job.id = job.id or str(uuid.uuid4())
        job.thread_id = job.thread_id or str(uuid.uuid4())
        # Named only when they are not the default: work keeps queueing on a database
        # that has not had migration 024 or 029 yet.
        extra: dict[str, Any] = {"mode": job.mode} if job.mode != "work" else {}
        if job.target != "auto":
            extra["target"] = job.target
        if job.agent_id:
            extra["agent_id"] = job.agent_id
        return await self._store.insert(
            "brain_jobs",
            {
                **extra,
                "id": job.id,
                "user_id": job.user_id,
                "kwami_id": job.kwami_id,
                "thread_id": job.thread_id,
                "source": job.source,
                "title": job.title[:200],
                "task": job.task,
                "difficulty": job.difficulty,
                "model": job.model,
                "repo": job.repo,
                "watcher_seen_at": _now(),
            },
            what="a job for the PC",
        )

    async def get(self, job_id: str) -> BrainJob | None:
        rows = await self._store.select(
            "brain_jobs", {"select": "*", "id": f"eq.{job_id}", "limit": "1"}, what="a job"
        )
        return BrainJob.from_row(rows[0]) if rows else None

    async def renew_watch(self, job_id: str) -> bool:
        return await self._store.update(
            "brain_jobs",
            {"watcher_seen_at": _now()},
            {"id": f"eq.{job_id}"},
            what="a job's watcher",
        )

    async def end_watch(self, job_id: str) -> bool:
        """The call is over: the result is for the next greeting."""
        return await self._store.update(
            "brain_jobs", {"watcher_seen_at": None}, {"id": f"eq.{job_id}"}, what="a job's watcher"
        )

    async def mark_delivered(self, job_id: str) -> bool:
        return await self._store.update(
            "brain_jobs",
            {"delivered_at": _now()},
            {"id": f"eq.{job_id}", "delivered_at": "is.null"},
            what="who tells the user about a job",
        )

    async def request_cancel(self, job_id: str) -> bool:
        """Cancel it. One still queued ends here; one running ends on the worker's next look."""
        queued = await self._store.update(
            "brain_jobs",
            {"status": "cancelled", "finished_at": _now(), "result": "Cancelada."},
            {"id": f"eq.{job_id}", "status": "eq.queued"},
            what="a queued job",
        )
        running = await self._store.update(
            "brain_jobs",
            {"cancel_requested": True},
            {"id": f"eq.{job_id}", "status": "eq.running"},
            what="a running job",
        )
        return queued and running

    async def last_thread(self, kwami_id: str) -> str:
        """The thread of this kwami's latest job, for "carry on with that"."""
        rows = await self._store.select(
            "brain_jobs",
            {
                "select": "thread_id",
                "kwami_id": f"eq.{kwami_id}",
                # A thought is no work to carry on with.
                "mode": "eq.work",
                "order": "created_at.desc",
                "limit": "1",
            },
            what="the latest job",
        )
        return _text(rows[0].get("thread_id")) if rows else ""

    async def agent(self, user_id: str, agent_id: str) -> dict[str, Any] | None:
        """One of this user's agents, or None (gone, switched off, or not theirs)."""
        rows = await self._store.select(
            "brain_agents",
            {
                "select": "*",
                "id": f"eq.{agent_id}",
                "user_id": f"eq.{user_id}",
                "enabled": "is.true",
                "limit": "1",
            },
            what="the user's agent",
        )
        return rows[0] if rows else None

    async def agents(self, user_id: str) -> list[dict[str, Any]]:
        """This user's agents that are switched on. Empty when there are none or no table."""
        rows = await self._store.select(
            "brain_agents",
            {"select": "*", "user_id": f"eq.{user_id}", "enabled": "is.true", "order": "name.asc"},
            what="the user's agents",
        )
        return list(rows or [])

    # -- the worker's side -----------------------------------------------------

    async def beat(
        self, user_id: str, runner_id: str, hostname: str, claude_login: bool = False
    ) -> bool:
        row = {
            "user_id": user_id,
            "runner_id": runner_id,
            "hostname": hostname[:120],
            "last_seen_at": _now(),
        }
        stored = await self._store.insert(
            "brain_runners",
            {**row, "claude_login": claude_login},
            on_conflict="user_id,runner_id",
            what="the worker's heartbeat",
        )
        self.thinking_ready = stored
        if stored:
            return True
        # Before migration 024: being seen matters more than saying what it can do.
        return await self._store.insert(
            "brain_runners", row, on_conflict="user_id,runner_id", what="the worker's heartbeat"
        )

    async def sites(self, user_id: str) -> list[dict[str, Any]]:
        rows = await self._store.select(
            "brain_sites",
            {"select": "*", "user_id": f"eq.{user_id}", "order": "created_at.asc"},
            what="the user's sites",
        )
        return list(rows or [])

    async def due_sites(self, user_id: str, older_than_seconds: float) -> list[dict[str, Any]]:
        """Sites worth another look: the ones that refuse this browser never change."""
        rows = await self._store.select(
            "brain_sites",
            {
                "select": "*",
                "user_id": f"eq.{user_id}",
                "state": "in.(connected,signed_out)",
                "checked_at": f"lt.{_ago(older_than_seconds)}",
            },
            what="the sites due a check",
        )
        return list(rows or [])

    async def note_site(
        self, user_id: str, kwami_id: str, url: str, state: str, host: str = ""
    ) -> bool:
        return await self._store.insert(
            "brain_sites",
            {
                "user_id": user_id,
                "kwami_id": kwami_id,
                "host": host or (urlparse(url).hostname or "").lower(),
                "url": url,
                "state": state,
                "checked_at": _now(),
            },
            on_conflict="user_id,host",
            what="a site's state",
        )

    async def claim(self, user_id: str, runner_id: str, thinking: bool = False) -> BrainJob | None:
        """The next job. `thinking` takes only the jobs that just answer."""
        args: dict[str, Any] = {"p_user_id": user_id, "p_runner_id": runner_id}
        if thinking:
            args["p_thinking"] = True
        rows = await self._store.rpc("brain_claim_job", args, what="the next job")
        return BrainJob.from_row(rows[0]) if isinstance(rows, list) and rows else None

    async def update_running(self, job_id: str, progress: str) -> bool:
        return await self._store.update(
            "brain_jobs",
            {"heartbeat_at": _now(), "progress": progress[:300]},
            {"id": f"eq.{job_id}", "status": "eq.running"},
            what="a job's progress",
        )

    async def finish(
        self, job_id: str, status: str, result: str, cost_usd: float, session_id: str
    ) -> bool:
        return await self._store.update(
            "brain_jobs",
            {
                "status": status,
                "result": result,
                "cost_usd": round(cost_usd, 6),
                "sdk_session_id": session_id,
                "finished_at": _now(),
            },
            {"id": f"eq.{job_id}"},
            what="a finished job",
        )

    async def session_of_thread(self, thread_id: str) -> str:
        """The Claude session this thread left off in. Empty starts a new one."""
        rows = await self._store.select(
            "brain_jobs",
            {
                "select": "sdk_session_id",
                "thread_id": f"eq.{thread_id}",
                "sdk_session_id": "neq.",
                "order": "created_at.desc",
                "limit": "1",
            },
            what="where a thread left off",
        )
        return _text(rows[0].get("sdk_session_id")) if rows else ""

    # -- the cloud's side ---------------------------------------------------------

    async def cloud_beat(self, runner_id: str, hostname: str) -> bool:
        return await self._store.insert(
            "brain_cloud_runners",
            {"runner_id": runner_id, "hostname": hostname[:120], "last_seen_at": _now()},
            on_conflict="runner_id",
            what="the cloud worker's heartbeat",
        )

    async def claim_cloud(self, runner_id: str) -> BrainJob | None:
        """The next job of any user the cloud may take, or None."""
        rows = await self._store.rpc(
            "brain_claim_cloud_job",
            {
                "p_runner_id": runner_id,
                "p_patience_seconds": CLOUD_PATIENCE_SECONDS,
                "p_per_user": CLOUD_PER_USER,
            },
            what="the next job for the cloud",
        )
        return BrainJob.from_row(rows[0]) if isinstance(rows, list) and rows else None

    async def stale_cloud_jobs(self) -> list[BrainJob]:
        """Jobs some cloud worker took and stopped beating for: it went away."""
        rows = await self._store.select(
            "brain_jobs",
            {
                "select": "*",
                "status": "eq.running",
                "runner_id": f"like.{CLOUD_RUNNER_PREFIX}*",
                "heartbeat_at": f"lt.{_ago(JOB_STALE_SECONDS)}",
            },
            what="jobs a cloud worker left behind",
        )
        return [BrainJob.from_row(row) for row in rows or []]

    async def thread_results(self, thread_id: str, limit: int = 3) -> list[str]:
        """What the latest finished jobs of a thread said, oldest first.

        The cloud cannot resume a Claude session it ran on another machine, so a
        follow-up there starts afresh with this, and the thread's files.
        """
        rows = await self._store.select(
            "brain_jobs",
            {
                "select": "result",
                "thread_id": f"eq.{thread_id}",
                "status": "eq.done",
                "order": "created_at.desc",
                "limit": str(limit),
            },
            what="what a thread found so far",
        )
        return [_text(row.get("result")) for row in reversed(rows or []) if row.get("result")]

    async def add_sync(
        self,
        job: BrainJob,
        kind: str,
        storage_prefix: str,
        *,
        files: int = 0,
        repo: str = "",
        base_ref: str = "",
    ) -> bool:
        """Leave what the cloud made for the user's PC to bring home."""
        return await self._store.insert(
            "brain_sync",
            {
                "id": str(uuid.uuid4()),
                "user_id": job.user_id,
                "kwami_id": job.kwami_id,
                "job_id": job.id,
                "kind": kind,
                "repo": repo,
                "base_ref": base_ref,
                "storage_prefix": storage_prefix,
                "files": files,
                "title": job.title[:200],
            },
            what="what the PC should bring home",
        )

    # -- the PC bringing the cloud's work home ------------------------------------

    async def claim_sync(self, user_id: str, runner_id: str) -> dict[str, Any] | None:
        rows = await self._store.rpc(
            "brain_claim_sync",
            {"p_user_id": user_id, "p_runner_id": runner_id},
            what="what the cloud left for this PC",
        )
        self.sync_ready = rows is not None
        return rows[0] if isinstance(rows, list) and rows else None

    async def end_sync(self, sync_id: str, status: str, result: str) -> bool:
        return await self._store.update(
            "brain_sync",
            {"status": status, "result": result[:2000], "applied_at": _now()},
            {"id": f"eq.{sync_id}"},
            what="how bringing the cloud's work home ended",
        )

    async def stale_jobs(self, user_id: str) -> list[BrainJob]:
        rows = await self._store.select(
            "brain_jobs",
            {
                "select": "*",
                "user_id": f"eq.{user_id}",
                "status": "eq.running",
                "heartbeat_at": f"lt.{_ago(JOB_STALE_SECONDS)}",
            },
            what="jobs a worker left behind",
        )
        return [BrainJob.from_row(row) for row in rows or []]

    async def month_spend(self, user_id: str) -> float | None:
        """What this user spent this month, fresh. None when it could not be read."""
        rows = await self._store.select(
            "brain_spend",
            {"select": "usd", "user_id": f"eq.{user_id}", "month": f"eq.{_month()}"},
            what="the brain's monthly spend",
        )
        if rows is None:
            return None
        return sum(float(row.get("usd") or 0.0) for row in rows)
