"""One job from the command line, with no Supabase and no call.

    scripts\\brain-worker.cmd --local --repo agent "what to change"

The work thread normally takes its jobs from a queue in Supabase. This stands in for
that queue, so Jarvis can change its own code from the user's PC when the database
cannot be reached. Everything else is the real thing: the same worker, guard, tests
and checks on the diff, and it still ends in a pull request the user decides on.

Spend goes to the disk ledger, which knows nothing of what the cloud spent, so here
the monthly ceiling counts this PC's work only.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from .jobs import BrainJob
from .ledger import DATA_DIR, Ledger

#: Not a real kwami: nothing here goes to a database.
LOCAL_ID = "local"


class LocalJobs:
    """The methods of `JobStore` the worker uses, for one job, on this PC."""

    def __init__(self, ledger: Ledger, threads: Path | None = None) -> None:
        self._ledger = ledger
        #: thread id -> Claude session, so `--thread` can carry on earlier work.
        self._threads = threads or DATA_DIR / "local_threads.json"
        self.job: BrainJob | None = None
        self._shown = ""

    def _sessions(self) -> dict[str, str]:
        try:
            data = json.loads(self._threads.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def new_job(self, repo: str, task: str, thread_id: str = "") -> BrainJob:
        self.job = BrainJob(
            user_id=LOCAL_ID,
            kwami_id=LOCAL_ID,
            title=task.strip().splitlines()[0][:100],
            task=task,
            id=str(uuid.uuid4()),
            thread_id=thread_id or uuid.uuid4().hex[:8],
            repo=repo,
            status="running",
            # Told on the terminal: nothing is left for a greeting.
            delivered_at="terminal",
        )
        return self.job

    async def session_of_thread(self, thread_id: str) -> str:
        return self._sessions().get(thread_id, "")

    async def thread_cost(self, thread_id: str) -> float:
        return 0.0

    async def update_running(self, job_id: str, progress: str) -> bool:
        if progress and progress != self._shown:
            self._shown = progress
            print(f"  {progress}", flush=True)
        return True

    async def get(self, job_id: str) -> BrainJob | None:
        return self.job

    async def finish(
        self, job_id: str, status: str, result: str, cost_usd: float, session_id: str
    ) -> bool:
        job = self.job
        if job is None:
            return False
        job.status, job.result, job.cost_usd = status, result, cost_usd
        job.sdk_session_id = session_id
        if session_id:
            sessions = self._sessions()
            sessions[job.thread_id] = session_id
            self._threads.parent.mkdir(parents=True, exist_ok=True)
            self._threads.write_text(json.dumps(sessions), encoding="utf-8")
        return True

    async def month_spend(self, user_id: str) -> float:
        return float(self._ledger.month_spend()["usd"])

    async def agents(self, user_id: str) -> list[dict[str, Any]]:
        """No database here: the agents of the app are not reachable, the folder's still are."""
        return []
