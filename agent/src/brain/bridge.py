"""The bridge: the voice model hands work over, other models do it.

The user's rule (2026-10-09) is that the voice model only reads and speaks.
So a request that takes work becomes a `BrainTask`: the voice says "on it",
the conversation carries on, and the task runs in the background on whichever
model the router picks -- or on several, when the council meets. When it
finishes, the bridge waits for the next pause and has the voice say it.

The bridge belongs to the job, not to an agent: agents are rebuilt on every
voice or soul change, and a task must not die because the user switched voice
mid-research. It is reached through `AgentDeps.bridge`.

If the call ends first, the task finishes anyway (the user chose that) and its
result is written to the ledger's pending list, which the next greeting reads.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from ..adapters.supabase import as_user_id, direct_supabase, get_supabase
from ..domain.pipelines import TEXT_REALTIME_PREFIXES
from ..factories.availability import voice_provider_keys
from ..settings import Settings, get_settings
from ..utils.logging import get_logger
from .browser_tool import SIGN_IN_TOOL
from .config import (
    HARD_MONTHLY_CAP_USD,
    BrainConfig,
    BrainModel,
    budget_notice,
    default_brain,
    executor_in_cloud,
)
from .executor import CloudWorkspace, run_executor
from .jobs import BrainJob, JobStore
from .ledger import Ledger, PendingResult, create_ledger
from .orders import LOGIN, MERGE, merge_approved
from .planner import (
    PLAN_FORMAT,
    PLANNER_SYSTEM,
    Plan,
    PlanStep,
    final_step,
    plan_from_text,
    plan_task,
    waves,
)
from .providers import (
    COMPATIBLE_URLS,
    BrainAnswer,
    BrainRequest,
    run_anthropic,
    run_compatible,
    run_gemini,
    run_openai,
    run_xai,
)
from .router import (
    ACCOUNT_MARK,
    PROVIDER_KEYS,
    can_delegate,
    council_members,
    director_of,
    find_model,
    pick,
    provider_available,
    should_convene,
    tavily_writer,
    usable,
)
from .selfwork import REPOS
from .user_keys import VOICE_PROVIDER_KEYS, load_user_keys

logger = get_logger("brain.bridge")

#: How long a model that answered "out of quota" is left alone. A daily quota
#: does not come back sooner; a per-minute one will have by then.
QUOTA_REST_SECONDS = 15 * 60

#: How long a finished result waits for the user to stop talking.
PAUSE_WAIT_SECONDS = 120.0
#: A moment of quiet, on top of knowing no turn is open (`_user_mid_turn`).
PAUSE_QUIET_SECONDS = 0.4
#: Results longer than this are summarised in speech; the full text stays in
#: `check_tasks` and in the app.
SPOKEN_RESULT_CHARS = 3500
MAX_TASKS_KEPT = 30
#: How long the greeting waits for the ledger to load from Supabase.
LEDGER_READY_SECONDS = 1.5
#: How long the first pipeline waits for the user's keys before it is built without them.
VOICE_KEYS_READY_SECONDS = 2.0
#: How long the end of a call waits for ledger writes to land.
LEDGER_FLUSH_SECONDS = 4.0
#: How often a call looks at a job on the PC, and how many looks between renewals
#: of its interest in the result.
JOB_POLL_SECONDS = 5.0
JOB_RENEW_EVERY = 3
#: How long what the PC last said about itself stands.
ACCOUNT_FRESH_SECONDS = 30.0
#: A thought waits this long for a free slot on the PC before going elsewhere.
THINK_QUEUE_PATIENCE_SECONDS = 25.0
THINK_TIMEOUT_SECONDS = 600.0
THINK_POLL_SECONDS = 1.5
#: The first seconds of a thought, when the answer to something simple arrives.
THINK_POLL_FAST_SECONDS = 0.4
THINK_POLL_FAST_FOR_SECONDS = 30.0
#: Merging takes seconds, and a sign-in only has a window to open.
ORDER_TIMEOUT_SECONDS = 120.0
#: What a task carries of the call. Enough to resolve "eso", not a transcript to pay for.
CONTEXT_TURNS = 8
CONTEXT_TURN_CHARS = 240
CONTEXT_MEMORY_CHARS = 1200
#: A quick answer is spoken as the worker wrote it. Measured 2026-11-02, the voice model
#: took 2 to 3 s more to say it again in its own words, and could only make it worse.
QUICK_ANSWER_SECONDS = 25.0
QUICK_ANSWER_CHARS = 600
#: With no "I'm on it" first, how long a simple answer may take before something is said.
STILL_WORKING_SECONDS = 6.0
#: How often, while the PC thinks, to check that the PC is still there.
THINK_PC_CHECK_SECONDS = 30.0
#: How long a job may sit queued before asking whether the PC is still there.
JOB_QUEUE_PATIENCE_SECONDS = 30.0

WORKER_SYSTEM = """You are part of the brain of {name}, a voice assistant. The voice model
delegated this task to you and will read your answer aloud to the user.

Now: {now} ({zone}). Answer in {language}.

How to answer:
- Start with the answer itself in two to four spoken sentences: no markdown, no
  tables, no lists, no URLs. Numbers as they would be said.
- Then, after a blank line and the heading "Detalle:", anything the user may ask
  for next: figures, sources, caveats. Keep it tight.
- If the task needs a decision from the user, end by saying exactly what to decide.
- Search the web whenever the answer depends on facts that could have changed.
{profile}"""

STEP_SYSTEM = """You do one part of a larger task for a voice assistant's brain. Another model
will combine your result with the other parts, so be complete and plain: facts,
figures, sources and caveats. No greeting, no spoken style, no summary of the
whole task. Search the web when the part needs current information."""

COUNCIL_SYSTEM = """You chair the council of {name}'s brain. Several models answered the same
task independently. Compare them and give the user one answer, in {language}.

- Start with the recommendation in two to four spoken sentences (no markdown).
- Then "Detalle:" with where the models agree, where they differ and why, and
  which one you trust more on each point.
- Never hide a disagreement that matters for money, security or the decision."""


@dataclass
class BrainTask:
    id: str
    title: str
    task: str
    kind: str
    difficulty: str
    status: str = "running"  # running | done | failed | cancelled
    council: bool = False
    models: list[str] = field(default_factory=list)
    result: str = ""
    cost_usd: float = 0.0
    sources: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str = ""
    seconds: float = 0.0
    delivered: bool = False
    #: The answer came from the Claude account on the user's PC, not an API key.
    via_account: bool = False
    #: When the planner split the task: what each part was and who did it.
    steps: list[dict[str, Any]] = field(default_factory=list)
    #: Work that should outlive the call: it goes to the PC whenever the PC is there.
    long_running: bool = False
    #: A task id from this call, or "last": carry on that thread.
    follow_up: str = ""
    #: The user was told to sign in once, for this task.
    asked_to_sign_in: bool = False
    #: Changing Jarvis's own code, in this repository (`selfwork.REPOS`).
    repo: str = ""
    #: Set once the work is on the PC.
    job_id: str = ""
    thread_id: str = ""
    progress: str = ""
    handle: asyncio.Task | None = field(default=None, repr=False)

    def public(self, full: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "difficulty": self.difficulty,
            "status": self.status,
            "council": self.council,
            "models": self.models,
            "costUsd": round(self.cost_usd, 4),
            "viaAccount": self.via_account,
            "steps": self.steps,
            "onPc": bool(self.job_id),
            "progress": self.progress,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
        }
        if full or self.status != "running":
            out["result"] = self.result if full else self.result[:1500]
            out["sources"] = self.sources[:8]
        return out


class Bridge:
    def __init__(
        self,
        keys: Mapping[str, str],
        tavily_key: str = "",
        ledger: Ledger | None = None,
        *,
        settings: Settings | None = None,
    ):
        settings = settings or get_settings()
        self._keys = dict(keys)
        #: What the deployment itself holds, apart from any user's keys.
        self._deployment_keys = dict(keys)
        self._tavily_key = tavily_key
        self.ledger = ledger if ledger is not None else create_ledger(settings)
        #: The user's Supabase when the agent writes it directly. Where the
        #: cloud executor uploads its files.
        self._store = direct_supabase(settings)
        #: The queue to the PC. It needs no API in between, so any configured
        #: Supabase will do, not only when the agent writes it directly.
        jobs_store = get_supabase(settings)
        self.jobs: JobStore | None = JobStore(jobs_store) if jobs_store is not None else None
        self._keys_store = jobs_store
        self._keys_secret = settings.provider_keys_secret
        #: The user's own keys for the voice pipeline, by environment name.
        self.voice_keys: dict[str, str] = {}
        #: Whose keys this call may hold. The first person only: the usage is billed to
        #: them, so nobody who joins later may bring their keys in, or spend theirs.
        self._keys_owner = ""
        self._voice_keys_loaded = asyncio.Event()
        self.pc_online = False
        self._account_checked_at: float | None = None
        self._executor_in_cloud = executor_in_cloud(settings.executor_mode)
        self.config: BrainConfig = default_brain()
        self.session: Any = None
        self.publisher: Any = None
        self.kwami_id = ""
        #: The Supabase user id, once the room or the kwami's row tells us.
        self.user_id = ""
        self.assistant_name = "Jarvis"
        self.language = "Spanish (Spain)"
        self.timezone = "Europe/Madrid"
        self.user_profile = ""
        self.tasks: dict[str, BrainTask] = {}
        self._ids = itertools.count(1)
        #: Models out of quota, and until when (monotonic) they are skipped.
        self._resting: dict[str, float] = {}
        self._closed = False
        self._background: set[asyncio.Task] = set()

    # -- wiring --------------------------------------------------------------

    def configure(self, config: BrainConfig) -> None:
        self.config = config
        logger.info(
            "Brain configured: %d model(s), %d usable, council=%s",
            len(config.models),
            len(usable(config, self._keys)),
            config.council,
        )
        self._schedule(self.publish_status())
        self._schedule(self._regate_voice())

    def set_context(
        self,
        *,
        kwami_id: str = "",
        name: str = "",
        language: str = "",
        timezone: str = "",
        user_profile: str = "",
    ) -> None:
        """What a worker needs to know about who it is working for."""
        if kwami_id:
            self.kwami_id = kwami_id
            # Start loading what the ledger keeps for this kwami now, so it is
            # in memory by the time the greeting asks (see `ready`).
            self._schedule(self.ledger.bind(kwami_id, self.user_id or None))
        if name:
            self.assistant_name = name
        if language:
            self.language = language
        if timezone:
            self.timezone = timezone
        self.user_profile = user_profile[:3000]

    def set_user(self, user_id: str) -> None:
        """The room's participant identity: the user's id for app sessions."""
        resolved = as_user_id(user_id)
        if not resolved or resolved == self.user_id:
            return
        self.user_id = resolved
        self._schedule(self.ledger.bind(self.kwami_id, resolved))
        self._schedule(self._on_user_known())

    async def ready(self, kwami_id: str, timeout: float = LEDGER_READY_SECONDS) -> None:
        """Wait, briefly, for the ledger to load this kwami. Never raises."""
        if not kwami_id:
            return
        try:
            await asyncio.wait_for(self.ledger.bind(kwami_id, self.user_id or None), timeout)
        except TimeoutError:
            logger.warning("Brain ledger not loaded after %.1fs; greeting without it", timeout)
        except Exception:
            logger.exception("Brain ledger failed to load")

    @property
    def voice_key_providers(self) -> frozenset[str]:
        """The providers the voice pipeline runs on with the user's own key."""
        return frozenset(p for p, k in VOICE_PROVIDER_KEYS.items() if self.voice_keys.get(k))

    async def keys_ready(self, timeout: float = VOICE_KEYS_READY_SECONDS) -> None:
        """Wait, briefly, so the first pipeline is built with the user's keys. Never raises."""
        if self._keys_store is None:
            return
        try:
            await asyncio.wait_for(self._voice_keys_loaded.wait(), timeout)
        except TimeoutError:
            logger.warning("The user's keys not loaded after %.1fs; building without", timeout)

    async def _on_user_known(self) -> None:
        """Bring in what depends on who this is: their keys, and their PC."""
        if self._keys_store is not None and self._keys_owner in ("", self.user_id):
            self._keys_owner = owner = self.user_id
            try:
                keys = await load_user_keys(self._keys_store, owner, self._keys_secret)
                voice_keys = await load_user_keys(
                    self._keys_store, owner, self._keys_secret, VOICE_PROVIDER_KEYS
                )
                # They may have been replaced while these were on their way.
                if self.user_id == owner:
                    self._keys.update(keys)
                    self.voice_keys = voice_keys
            finally:
                self._voice_keys_loaded.set()
        elif self._keys_store is not None:
            # Somebody else is speaking now, and the first person's keys are not theirs.
            self._keys = dict(self._deployment_keys)
            self.voice_keys = {}
        await self._refresh_account(force=True)
        await self.publish_status()
        await self._regate_voice()

    async def _refresh_account(self, force: bool = False) -> None:
        """Whether the Claude account of this user's PC can take work right now.

        Only for the room's own user, not the kwami's owner: on a phone line anyone
        can talk, and they must not spend the owner's plan.
        """
        if self.jobs is None or not self.user_id:
            return
        loop = asyncio.get_running_loop()
        now = loop.time()
        fresh = (
            self._account_checked_at is not None
            and now - self._account_checked_at < ACCOUNT_FRESH_SECONDS
        )
        if fresh and not force:
            return
        self._account_checked_at = now
        self.pc_online = await self.jobs.runner_online(self.user_id)
        if self.pc_online and await self.jobs.account_ready(self.user_id):
            self._keys[ACCOUNT_MARK] = "1"
        else:
            self._keys.pop(ACCOUNT_MARK, None)

    def _owner(self) -> str:
        return self.user_id or getattr(self.ledger, "user_id", None) or ""

    def _cloud_workspace(self) -> CloudWorkspace | None:
        if not self._executor_in_cloud:
            return None
        return CloudWorkspace(store=self._store, user_id=self._owner(), kwami_id=self.kwami_id)

    def attach(self, session: Any, publisher: Any) -> None:
        self.session = session
        self.publisher = publisher

    def delegates(self) -> bool:
        """True when searches go to the brain rather than to the voice's own tools."""
        return can_delegate(self.config, self._keys)

    async def _regate_voice(self) -> None:
        """Re-offer or withhold the voice's search tools after a brain change.

        Switching the brain off in the Models panel must hand searching back to
        the voice in the same call, not from the next agent rebuild on.
        """
        try:
            agent = self.session.current_agent if self.session is not None else None
        except Exception:
            agent = None
        refresh = getattr(agent, "refresh_brain_gating", None)
        if refresh is None:
            return
        try:
            await refresh()
        except Exception:
            logger.warning("Could not re-gate the voice's search tools", exc_info=True)

    def providers(self) -> dict[str, bool]:
        return {p: provider_available(p, self._keys) for p in PROVIDER_KEYS}

    def voice_providers(self) -> dict[str, Any]:
        """Every STT, TTS, LLM and realtime provider, and whether it has its key.

        The flat realtime entries (`google`, `openai`, `aws`) are what the app
        read before the per-stage maps existed; they stay until no build reads
        them.
        """
        stages = voice_provider_keys({**self._deployment_keys, **self.voice_keys})
        return {**stages["realtime"], **stages}

    def status(self) -> dict[str, Any]:
        spend = self.ledger.month_spend()
        return {
            "type": "brain_status",
            "enabled": self.config.enabled,
            "providers": self.providers(),
            # Whether each provider has an API key, apart from the Claude account.
            "apiKeys": {p: bool(self._keys.get(k)) for p, k in PROVIDER_KEYS.items()},
            "account": {
                "pcOnline": self.pc_online,
                "loggedIn": bool(self._keys.get(ACCOUNT_MARK)),
            },
            "director": getattr(director_of(self.config, self._keys), "id", ""),
            "voiceProviders": self.voice_providers(),
            # Realtime models (by name prefix) that can answer in text for a
            # TTS to speak: what the panel allows on "Live + TTS".
            "hybridRealtime": {p: list(v) for p, v in TEXT_REALTIME_PREFIXES.items()},
            "tavily": bool(self._tavily_key),
            "usableModels": [m.id for m in usable(self.config, self._keys)],
            "spend": {**spend, "budgetUsd": self.config.monthly_budget_usd},
            "tasks": [t.public() for t in list(self.tasks.values())[-10:]],
        }

    async def publish_status(self) -> None:
        if self.publisher is None:
            return
        try:
            await self._refresh_account()
            await self.publisher.publish(self.status())
        except Exception:
            logger.debug("Could not publish brain status", exc_info=True)

    def _schedule(self, coro: Any) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            coro.close()
            return
        task = loop.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    # -- starting work ---------------------------------------------------------

    def start(
        self,
        *,
        task: str,
        title: str,
        kind: str,
        difficulty: str,
        important: bool = False,
        council: bool = False,
        model: str = "",
        context: str = "",
        long_running: bool = False,
        follow_up: str = "",
        repo: str = "",
    ) -> BrainTask | str:
        """Queue a task. Returns it, or a sentence saying why it cannot run."""
        if repo:
            if repo not in REPOS:
                return f"The repository must be one of: {', '.join(REPOS)}."
            # The room's identity, not the kwami's owner: on a phone line anyone can talk.
            if not self.user_id:
                return "Changing my own code can only be ordered from the app, not by phone."
            kind, important, council = "code", False, False
        if not self.config.enabled:
            return "The brain is switched off in the Models panel."
        if not usable(self.config, self._keys):
            return "No brain model is usable: none is enabled with its API key or account present."

        spend = self.ledger.month_spend()["usd"]
        if spend >= HARD_MONTHLY_CAP_USD:
            return (
                f"This month's ceiling of {HARD_MONTHLY_CAP_USD:.0f} $ is reached. "
                "No more tasks until next month."
            )
        over_budget = self.config.monthly_budget_usd > 0 and spend >= self.config.monthly_budget_usd
        if over_budget and self.config.budget_policy == "stop":
            return f"The monthly brain budget ({self.config.monthly_budget_usd:.0f} $) is spent."
        if over_budget and self.config.budget_policy == "downgrade":
            difficulty, council, important, model = "simple", False, False, ""

        convene = should_convene(self.config, kind, important, council)
        if convene and len(council_members(self.config, self._keys)) < 2:
            convene = False
        if not convene and pick(self.config, self._keys, kind, difficulty, model) is None:
            if kind == "code":
                return "The local executor is not enabled or has no Anthropic key."
            if kind == "x_search":
                return (
                    "Nothing can read X: no enabled model has that job. "
                    "It needs Grok, connected in the Models panel."
                )
            return "No usable model fits this task."

        brain_task = BrainTask(
            id=f"t{next(self._ids)}",
            title=(title or task)[:120],
            task=self._with_conversation(task, context),
            kind=kind,
            difficulty=difficulty,
            council=convene,
            long_running=long_running,
            follow_up=follow_up,
            repo=repo,
        )
        self.tasks[brain_task.id] = brain_task
        while len(self.tasks) > MAX_TASKS_KEPT:
            oldest = next(iter(self.tasks))
            if self.tasks[oldest].status == "running":
                break
            self.tasks.pop(oldest)
        brain_task.handle = asyncio.create_task(
            self._run(brain_task, model), name=f"brain_{brain_task.id}"
        )
        if self.config.relay == "everything" and difficulty == "simple" and self.session:
            self._schedule(self._say_still_working(brain_task))
        self._schedule(self.publish_status())
        return brain_task

    def _with_conversation(self, task: str, context: str) -> str:
        """The task, with what the worker cannot otherwise know.

        A worker starts from nothing, and the voice model mentions only what it thinks
        of: "that company" or "como te dije" reached it with nothing to resolve them.
        """
        parts = [task]
        if context:
            parts.append(f"Context from the conversation:\n{context}")
        recent = self._recent_conversation()
        if recent:
            parts.append(f"The conversation so far, oldest first:\n{recent}")
        remembered = self._remembered()
        if remembered:
            parts.append(f"What is remembered about the user:\n{remembered}")
        return "\n\n".join(parts)

    def _voice(self) -> Any:
        """The agent of the call, or None: a scheduled task has no call, a new one no agent."""
        try:
            return self.session.current_agent
        except (AttributeError, RuntimeError):
            return None

    def _user_mid_turn(self) -> bool:
        return getattr(self._voice(), "turn_open", False) is True

    def _recent_conversation(self) -> str:
        history = getattr(self._voice(), "_chat_ctx", None)
        lines = []
        for item in getattr(history, "items", []):
            role = getattr(item, "role", None)
            said = getattr(item, "text_content", None)
            if role in ("user", "assistant") and said:
                lines.append(f"{role}: {said[:CONTEXT_TURN_CHARS]}")
        return "\n".join(lines[-CONTEXT_TURNS:])

    def _last_user_turn(self) -> str:
        history = getattr(self._voice(), "_chat_ctx", None)
        turns = [
            said
            for item in getattr(history, "items", [])
            if getattr(item, "role", None) == "user"
            and (said := getattr(item, "text_content", None))
        ]
        return turns[-1] if turns else ""

    def _remembered(self) -> str:
        text = getattr(self._voice(), "_memory_text", "")
        return text[:CONTEXT_MEMORY_CHARS] if isinstance(text, str) else ""

    def cancel(self, task_id: str) -> bool:
        brain_task = self.tasks.get(task_id)
        if brain_task is None or brain_task.status != "running" or brain_task.handle is None:
            return False
        if brain_task.job_id and self.jobs is not None:
            self._schedule(self.jobs.request_cancel(brain_task.job_id))
        brain_task.handle.cancel()
        return True

    # -- running -------------------------------------------------------------

    def _system(self) -> str:
        try:
            zone = ZoneInfo(self.timezone)
        except Exception:
            zone = ZoneInfo("UTC")
        now = datetime.now(zone).strftime("%A %d %B %Y, %H:%M")
        profile = (
            f"\nWhat the user has told the assistant about themselves:\n{self.user_profile}\n"
            if self.user_profile
            else ""
        )
        return WORKER_SYSTEM.format(
            name=self.assistant_name,
            now=now,
            zone=self.timezone,
            language=self.language,
            profile=profile,
        )

    def _request(self, text: str, difficulty: str, system: str | None = None) -> BrainRequest:
        tavily = self._tavily_key if self.config.search in ("tavily", "both") else ""
        return BrainRequest(
            task=text,
            system=system or self._system(),
            difficulty=difficulty,
            native_search=self.config.search != "tavily" or not tavily,
            tavily_key=tavily,
        )

    def _brief(self, model: BrainModel) -> str:
        """What the user told the brain, and this model, to always keep in mind."""
        parts = []
        if self.config.instructions:
            parts.append(
                f"The user's instructions for the whole brain:\n{self.config.instructions}"
            )
        if model.instructions:
            parts.append(f"The user's instructions for you:\n{model.instructions}")
        return "\n\n".join(parts)

    def _on_account(self, model: BrainModel) -> bool:
        return (
            model.provider == "anthropic"
            and model.access != "api"
            and bool(self._keys.get(ACCOUNT_MARK))
        )

    async def _think_on_account(
        self, model: BrainModel, request: BrainRequest
    ) -> BrainAnswer | None:
        """Ask the PC, on the user's Claude account. None when it did not get to it."""
        if self.jobs is None or as_user_id(self.kwami_id) is None:
            return None
        job = BrainJob(
            user_id=self.user_id,
            kwami_id=self.kwami_id,
            title=request.task[:120],
            task=f"{request.system}\n\n---\n\n{request.task}",
            difficulty=request.difficulty,
            model=model.model,
            mode="research" if request.native_search else "think",
        )
        if not await self.jobs.enqueue(job):
            return None
        loop = asyncio.get_running_loop()
        started = checked = loop.time()
        try:
            while loop.time() - started < THINK_TIMEOUT_SECONDS:
                await asyncio.sleep(
                    THINK_POLL_FAST_SECONDS
                    if loop.time() - started < THINK_POLL_FAST_FOR_SECONDS
                    else THINK_POLL_SECONDS
                )
                if loop.time() - checked >= THINK_PC_CHECK_SECONDS:
                    checked = loop.time()
                    if not await self.jobs.runner_online(self.user_id):
                        logger.info("The PC went away during a thought on %s", model.id)
                        break
                current = await self.jobs.get(job.id)
                if current is None:
                    continue
                if current.finished:
                    await self.jobs.mark_delivered(job.id)
                    return BrainAnswer(
                        current.result,
                        model.id,
                        model.model,
                        ok=current.status == "done",
                        via_account=True,
                    )
                if (
                    current.status == "queued"
                    and loop.time() - started >= THINK_QUEUE_PATIENCE_SECONDS
                ):
                    break
        except asyncio.CancelledError:
            self._schedule(self.jobs.request_cancel(job.id))
            raise
        logger.info("No answer from the PC on %s in time", model.id)
        await self.jobs.request_cancel(job.id)
        return None

    async def _ask(self, model: BrainModel, request: BrainRequest) -> BrainAnswer:
        brief = self._brief(model)
        if brief:
            request = replace(request, system=f"{request.system}\n\n{brief}")
        api_key = self._keys.get(PROVIDER_KEYS[model.provider], "")
        if self._on_account(model):
            answer = await self._think_on_account(model, request)
            if answer is not None and (answer.ok or model.access == "account" or not api_key):
                return answer
            if answer is not None:
                logger.info("The account could not answer on %s; using the API key", model.id)
        if model.provider in ("anthropic", "local") and (not api_key or model.access == "account"):
            # Chosen as "account only", or the PC went away since it was picked.
            return BrainAnswer(
                "La cuenta de Claude de tu ordenador no está disponible ahora.",
                model.id,
                model.model,
                ok=False,
            )
        if model.provider == "anthropic":
            answer = await run_anthropic(api_key, model.id, model.model, request)
        elif model.provider == "google":
            answer = await run_gemini(api_key, model.id, model.model, request)
        elif model.provider == "openai":
            answer = await run_openai(api_key, model.id, model.model, request)
        elif model.provider == "xai":
            answer = await run_xai(api_key, model.id, model.model, request)
        elif model.provider in COMPATIBLE_URLS:
            answer = await run_compatible(model.provider, api_key, model.id, model.model, request)
        else:
            result = await run_executor(
                api_key,
                model.model,
                f"{request.task}\n\n{brief}" if brief else request.task,
                self.config.executor,
                cloud=self._cloud_workspace(),
            )
            answer = BrainAnswer(result.text, model.id, model.model, result.cost_usd, ok=result.ok)
        self.ledger.add_spend(model.id, answer.cost_usd)
        return answer

    async def _plan(self, director: BrainModel, task: str) -> Plan | None:
        """Ask the director how to split the work. None when it could not say."""
        api_key = self._keys.get("ANTHROPIC_API_KEY", "")
        if director.provider == "anthropic" and api_key and not self._on_account(director):
            plan = await plan_task(api_key, director.model, task)
            if plan is not None:
                self.ledger.add_spend(director.id, plan.cost_usd)
            return plan
        # The account, or a director that is not Claude: a plain answer in JSON.
        answer = await self._ask(
            director,
            BrainRequest(
                task=task,
                system=PLANNER_SYSTEM + PLAN_FORMAT,
                difficulty="simple",
                native_search=False,
            ),
        )
        if not answer.ok:
            return None
        plan = plan_from_text(answer.text)
        plan.cost_usd = answer.cost_usd
        return plan

    def _note_model(self, brain_task: BrainTask, label: str) -> None:
        if label not in brain_task.models:
            brain_task.models.append(label)

    def _rest(self, model: BrainModel) -> None:
        """Skip a model whose quota is spent, for a while.

        A free-tier Gemini key allows Gemini 3.5 Flash 20 requests a day
        (2026-10-20). Each task used to try it first anyway, pay a failed
        round trip and only then fall back -- on every step of every task.
        """
        self._resting[model.id] = time.monotonic() + QUOTA_REST_SECONDS
        logger.info(
            "Brain model %s is out of quota; skipped for %.0f min",
            model.id,
            QUOTA_REST_SECONDS / 60,
        )

    def _resting_ids(self) -> set[str]:
        now = time.monotonic()
        for model_id, until in list(self._resting.items()):
            if until <= now:
                del self._resting[model_id]
        return set(self._resting)

    async def _ask_role(
        self,
        brain_task: BrainTask,
        kind: str,
        difficulty: str,
        request: BrainRequest,
        prefer: str = "",
    ) -> tuple[BrainAnswer, BrainModel | None]:
        """Run on the model the router picks, with one fallback on failure."""
        resting = self._resting_ids()
        # Shared with the Tavily rescue, so a writer that just failed there is
        # not asked again by the fallback below.
        tried: set[str] = set(resting)
        if resting:
            # The search Gemini would have done goes straight to Tavily, the
            # cheap rescue, rather than to another model's paid web search.
            first = pick(self.config, self._keys, kind, difficulty, prefer)
            if (
                first is not None
                and first.id in resting
                and first.provider == "google"
                and request.native_search
                and self._tavily_key
            ):
                rescued = await self._search_on_tavily(brain_task, kind, difficulty, request, tried)
                if rescued is not None:
                    return rescued
        answer: BrainAnswer | None = None
        model: BrainModel | None = None
        for _ in range(2):  # the chosen model, then one fallback
            model = pick(
                self.config,
                self._keys,
                kind,
                difficulty,
                prefer if not tried else "",
                frozenset(tried),
            )
            if model is None:
                break
            tried.add(model.id)
            self._note_model(brain_task, model.label or model.model)
            answer = await self._ask(model, request)
            brain_task.cost_usd += answer.cost_usd
            if answer.ok:
                return answer, model
            logger.info("Brain model %s failed (%s); trying a fallback", model.id, answer.text)
            if answer.rate_limited:
                self._rest(model)
            if model.provider == "google" and request.native_search and self._tavily_key:
                rescued = await self._search_on_tavily(brain_task, kind, difficulty, request, tried)
                if rescued is not None:
                    return rescued
        failed = answer or BrainAnswer("No usable model could take the task.", "", "", ok=False)
        return failed, model

    async def _search_on_tavily(
        self,
        brain_task: BrainTask,
        kind: str,
        difficulty: str,
        request: BrainRequest,
        tried: set[str],
    ) -> tuple[BrainAnswer, BrainModel] | None:
        """Redo a failed Gemini search on Tavily, written up by Claude or GPT.

        Gemini only searches with Google; when that fails -- a 429 on a spent
        quota is the usual cause -- the search itself moves to Tavily rather
        than to another model's own search, so a quota hit costs one Tavily
        call instead of a second paid web search. Returns None when it fails
        too, or when no writer is usable, and the normal fallback goes on.
        """
        writer = tavily_writer(self.config, self._keys, kind, difficulty, frozenset(tried))
        if writer is None:
            return None
        tried.add(writer.id)
        self._note_model(brain_task, f"{writer.label or writer.model} + Tavily")
        answer = await self._ask(
            writer, replace(request, native_search=False, tavily_key=self._tavily_key)
        )
        brain_task.cost_usd += answer.cost_usd
        if not answer.ok:
            logger.info("Tavily fallback on %s failed too (%s)", writer.id, answer.text)
            return None
        logger.info("Gemini search failed; answered through Tavily by %s", writer.id)
        return answer, writer

    async def merge_pull_request(self, repo: str, number: int, user_words: str) -> str:
        """Merge a pull request of Jarvis's, on the user's own words. What to tell them."""
        if not merge_approved(user_words, self._last_user_turn()):
            return (
                "Not merged. Ask the user whether they tried the change by hand and want it "
                "merged. It needs their own last words saying both."
            )
        return await self._order_on_pc(MERGE, f"Fusionar {repo}#{number}", f"{repo}#{number}")

    async def connect_claude_account(self) -> str:
        """Open Claude's own sign-in on the PC. What to tell the user."""
        return await self._order_on_pc(LOGIN, "Conectar la cuenta de Claude", "login")

    async def _order_on_pc(self, mode: str, title: str, order: str) -> str:
        """Only the room's own user, as with any work on the PC: anyone can talk on a phone."""
        if self.jobs is None or not self.user_id or as_user_id(self.kwami_id) is None:
            return "This can only be ordered from the app."
        if not await self.jobs.runner_online(self.user_id):
            return "The user's PC is not there now, so this cannot be done."
        job = BrainJob(self.user_id, self.kwami_id, title, order, mode=mode)
        if not await self.jobs.enqueue(job):
            return "The order could not be queued."
        loop = asyncio.get_running_loop()
        started = loop.time()
        while loop.time() - started < ORDER_TIMEOUT_SECONDS:
            await asyncio.sleep(THINK_POLL_FAST_SECONDS)
            current = await self.jobs.get(job.id)
            if current is not None and current.finished:
                await self.jobs.mark_delivered(job.id)
                return current.result
        await self.jobs.request_cancel(job.id)
        return "The PC did not get to it in time."

    async def sites_to_connect(self) -> str:
        """The user's sites and how each one stands, for the voice to go through."""
        if self.jobs is None or not self.user_id:
            return "This can only be asked from the app."
        sites = await self.jobs.sites(self.user_id)
        if not sites:
            return "The user has signed in to no sites through you yet."
        return "\n".join(f"{site['host']} ({site['url']}): {site['state']}" for site in sites)

    async def runner_online(self) -> bool:
        return bool(self.jobs and self.user_id and await self.jobs.runner_online(self.user_id))

    async def _on_the_pc(self, brain_task: BrainTask, prefer: str) -> BrainAnswer | None:
        """Send the work to the PC. None when it is not there: the caller runs it here.

        Only the room's own user, never the kwami's owner: on a phone line anyone can
        talk, and the PC runs commands on its owner's login.
        """
        if self.jobs is None or not self.user_id or as_user_id(self.kwami_id) is None:
            return None
        if not await self.jobs.runner_online(self.user_id):
            return None
        model = pick(self.config, self._keys, "code", brain_task.difficulty, prefer)
        if model is None:
            return None
        earlier = self.tasks.get(brain_task.follow_up)
        thread_id = earlier.thread_id if earlier is not None else ""
        if not thread_id and brain_task.follow_up == "last":
            thread_id = await self.jobs.last_thread(self.kwami_id)
        brief = self._brief(model)
        job = BrainJob(
            user_id=self.user_id,
            kwami_id=self.kwami_id,
            title=brain_task.title,
            task=f"{brain_task.task}\n\n{brief}" if brief else brain_task.task,
            thread_id=thread_id,
            difficulty=brain_task.difficulty,
            model=model.model,
            repo=brain_task.repo,
        )
        if not await self.jobs.enqueue(job):
            return None
        brain_task.job_id, brain_task.thread_id = job.id, job.thread_id
        self._note_model(brain_task, model.label or model.model)
        self._schedule(self.publish_status())
        return await self._watch_job(self.jobs, brain_task, job.id, model)

    async def _watch_job(
        self, jobs: JobStore, brain_task: BrainTask, job_id: str, model: BrainModel
    ) -> BrainAnswer | None:
        """Wait for the PC. None when it vanished before starting: the job is withdrawn."""
        loop = asyncio.get_running_loop()
        queued_since = loop.time()
        polls = 0
        while True:
            await asyncio.sleep(JOB_POLL_SECONDS)
            polls += 1
            if polls % JOB_RENEW_EVERY == 0:
                await jobs.renew_watch(job_id)
            job = await jobs.get(job_id)
            if job is None:  # a blip: the row is still there
                continue
            brain_task.progress = job.progress
            if job.progress.startswith(SIGN_IN_TOOL) and not brain_task.asked_to_sign_in:
                brain_task.asked_to_sign_in = True
                self._ask_to_sign_in(job.progress)
            if job.finished:
                await jobs.mark_delivered(job_id)
                self.ledger.note_spend(model.id, job.cost_usd)
                brain_task.cost_usd += job.cost_usd
                return BrainAnswer(
                    job.result,
                    model.id,
                    model.model,
                    job.cost_usd,
                    ok=job.status == "done",
                    via_account=job.cost_usd == 0,
                )
            if (
                job.status == "queued"
                and loop.time() - queued_since >= JOB_QUEUE_PATIENCE_SECONDS
                and not await jobs.runner_online(job.user_id)
            ):
                await jobs.request_cancel(job_id)
                job = await jobs.get(job_id)
                if job is not None and job.status == "cancelled":
                    logger.info("The PC went away; running %s here", brain_task.id)
                    brain_task.job_id = ""
                    return None
                queued_since = loop.time()

    async def _single(self, brain_task: BrainTask, prefer: str) -> BrainAnswer:
        request = self._request(brain_task.task, brain_task.difficulty)
        if brain_task.kind == "x_search":
            # Whatever the search setting says: Tavily cannot read X.
            request.native_search, request.tavily_key = True, ""
        answer, _ = await self._ask_role(
            brain_task, brain_task.kind, brain_task.difficulty, request, prefer
        )
        return answer

    async def _staged(self, brain_task: BrainTask, prefer: str) -> BrainAnswer:
        """Plan, run each part on its own model, combine (planner.py)."""
        if (
            prefer
            or not self.config.split
            or brain_task.kind in ("code", "x_search")
            or brain_task.difficulty == "simple"
        ):
            return await self._single(brain_task, prefer)
        director = director_of(self.config, self._keys)
        if director is None:
            return await self._single(brain_task, prefer)

        plan = await self._plan(director, brain_task.task)
        if plan is not None:
            brain_task.cost_usd += plan.cost_usd
        if plan is None or not plan.steps:
            return await self._single(brain_task, prefer)

        if not plan.split:
            # Indivisible: run whole, at the difficulty the planner judged.
            step = plan.steps[0]
            brain_task.difficulty = step.difficulty
            if step.web:
                brain_task.kind = "research"
            elif brain_task.kind == "research":
                brain_task.kind = "analysis"
            return await self._single(brain_task, "")

        # When the last step already builds on every other one, it answers the
        # user itself. Measured 2026-10-09: a separate combine call repeated the
        # Opus work and cost about a third of the whole task.
        final_id = final_step(plan.steps)
        final_answer: BrainAnswer | None = None
        results: dict[str, str] = {}
        for wave in waves(plan.steps):
            answers = await asyncio.gather(
                *(
                    self._run_step(brain_task, step, results, final=step.id == final_id)
                    for step in wave
                )
            )
            for step, (answer, model) in zip(wave, answers, strict=True):
                results[step.id] = (
                    answer.text if answer.ok else f"(this part failed: {answer.text})"
                )
                if step.id == final_id and answer.ok:
                    final_answer = answer
                brain_task.steps.append(
                    {
                        "goal": step.goal[:160],
                        "difficulty": step.difficulty,
                        "model": (model.label or model.model) if model else "",
                        "ok": answer.ok,
                    }
                )

        if final_answer is not None:
            return final_answer

        hard = any(s.difficulty == "hard" for s in plan.steps)
        parts = "\n\n".join(f"### {s.id}: {s.goal}\n{results.get(s.id, '')}" for s in plan.steps)
        combined, _ = await self._ask_role(
            brain_task,
            "analysis",
            "hard" if hard else "normal",
            BrainRequest(
                task=(
                    f"Task:\n{brain_task.task}\n\nThe parts were done separately; combine "
                    f"them into one answer.\n\n{parts}"
                ),
                system=self._system(),
                difficulty="hard" if hard else "normal",
                native_search=False,
            ),
        )
        return combined

    async def _run_step(
        self,
        brain_task: BrainTask,
        step: PlanStep,
        results: dict[str, str],
        final: bool = False,
    ) -> tuple[BrainAnswer, BrainModel | None]:
        if final:
            # The last part answers the whole task, in the spoken format.
            earlier = "\n\n".join(f"### {k}\n{v}" for k, v in results.items())
            text = (
                f"Task:\n{brain_task.task}\n\nWhat the earlier parts found:\n{earlier}\n\n"
                f"Finish it -- {step.goal}"
            )
            request = self._request(text, step.difficulty)
        else:
            earlier = "\n\n".join(f"### {a}\n{results[a]}" for a in step.after if a in results)
            text = f"Overall task, for context only:\n{brain_task.task}\n\nYour part:\n{step.goal}"
            if earlier:
                text += f"\n\nWhat earlier parts found:\n{earlier}"
            request = self._request(text, step.difficulty, system=STEP_SYSTEM)
        request.native_search = request.native_search and step.web
        if not step.web:
            request.tavily_key = ""
        kind = "research" if step.web else "analysis"
        return await self._ask_role(brain_task, kind, step.difficulty, request)

    async def _council(self, brain_task: BrainTask) -> BrainAnswer:
        members = council_members(self.config, self._keys)
        brain_task.models = [m.label or m.model for m in members]
        request = self._request(brain_task.task, "hard")
        answers = await asyncio.gather(*(self._ask(m, request) for m in members))
        brain_task.cost_usd += sum(a.cost_usd for a in answers)
        good = [a for a in answers if a.ok]
        if len(good) < 2:
            return good[0] if good else answers[0]

        chair = pick(self.config, self._keys, "analysis", "hard") or members[0]
        transcript = "\n\n".join(
            f"### {next((m.label for m in members if m.id == a.model_id), a.model)}\n{a.text}"
            for a in good
        )
        synthesis = await self._ask(
            chair,
            BrainRequest(
                task=f"Task:\n{brain_task.task}\n\nAnswers from the council:\n\n{transcript}",
                system=COUNCIL_SYSTEM.format(name=self.assistant_name, language=self.language),
                difficulty="hard",
                native_search=False,
            ),
        )
        brain_task.cost_usd += synthesis.cost_usd
        if not synthesis.ok:
            return good[0]
        synthesis.sources = [s for a in good for s in a.sources][:12]
        return synthesis

    async def _run(self, brain_task: BrainTask, prefer: str) -> None:
        spent_before = self.ledger.month_spend()["usd"]
        started = asyncio.get_running_loop().time()
        try:
            answer: BrainAnswer | None = None
            if (
                not brain_task.council
                and brain_task.kind != "x_search"
                and (brain_task.kind == "code" or brain_task.long_running)
            ):
                answer = await self._on_the_pc(brain_task, prefer)
            if answer is None and brain_task.repo:
                # Only the PC can publish: the cloud must not hold GitHub credentials.
                answer = BrainAnswer(
                    "Cambiar mi propio código necesita el ordenador encendido con el proceso "
                    "del hilo de trabajo, y ahora no está.",
                    "",
                    "",
                    ok=False,
                )
            if answer is None:
                answer = await (
                    self._council(brain_task)
                    if brain_task.council
                    else self._staged(brain_task, prefer)
                )
            brain_task.result = answer.text
            brain_task.sources = answer.sources
            brain_task.via_account = answer.via_account
            brain_task.status = "done" if answer.ok else "failed"
            if not brain_task.job_id:  # the worker warned about its own spend
                notice = budget_notice(spent_before, self.ledger.month_spend()["usd"])
                if notice:
                    brain_task.result = f"{notice}\n\n{brain_task.result}"
        except asyncio.CancelledError:
            brain_task.status = "cancelled"
            brain_task.result = "Cancelada."
            brain_task.finished_at = datetime.now(UTC).isoformat()
            await self.publish_status()
            raise
        except Exception as e:
            logger.exception("Brain task %s crashed", brain_task.id)
            brain_task.status = "failed"
            brain_task.result = f"La tarea falló: {e}"
        brain_task.finished_at = datetime.now(UTC).isoformat()
        brain_task.seconds = asyncio.get_running_loop().time() - started
        logger.info(
            "Brain task %s %s on %s in %.1f s (%.4f $, %s, %s)",
            brain_task.id,
            brain_task.status,
            ", ".join(brain_task.models),
            brain_task.seconds,
            brain_task.cost_usd,
            brain_task.kind,
            brain_task.difficulty,
        )
        await self.publish_status()
        await self._deliver(brain_task)

    # -- delivering --------------------------------------------------------------

    async def _wait_for_pause(self) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + PAUSE_WAIT_SECONDS
        quiet_since: float | None = None
        while loop.time() < deadline:
            session = self.session
            if session is None or self._closed:
                return False
            # The user going quiet is not the turn ending: the end-of-turn wait comes
            # after it, and a result spoken in that wait took the place of the answer.
            quiet = (
                session.user_state != "speaking"
                and session.agent_state in ("listening", "idle")
                and not self._user_mid_turn()
            )
            now = loop.time()
            if quiet:
                quiet_since = quiet_since or now
                if now - quiet_since >= PAUSE_QUIET_SECONDS:
                    return True
            else:
                quiet_since = None
            await asyncio.sleep(0.25)
        return self.session is not None and not self._closed

    def _spoken(self, brain_task: BrainTask) -> str:
        who = " y ".join(brain_task.models) or "el cerebro"
        result = brain_task.result
        if len(result) > SPOKEN_RESULT_CHARS:
            result = result[:SPOKEN_RESULT_CHARS] + " [...]"
        if brain_task.status != "done":
            return (
                f"A delegated task failed. Task: {brain_task.title}. What happened: {result}\n"
                "Tell the user briefly and offer to retry or try another model."
            )
        if self.config.delivery == "on_request":
            return (
                f"A delegated task is ready: {brain_task.title} (task id {brain_task.id}). "
                "Tell the user in one short sentence that it is ready and ask whether they "
                "want it now. Do not give the result yet."
            )
        council = (
            " The council met; mention that the models were compared." if brain_task.council else ""
        )
        return (
            f"A delegated task just finished (task id {brain_task.id}), answered by {who}.{council}\n"
            f"Task: {brain_task.title}\nResult:\n{result}\n\n"
            "Tell the user now, in your own voice and briefly: the spoken part first, details "
            "only if they matter. Do not read headings, URLs or markdown. Offer the detail if "
            "there is more."
        )

    def _spoken_as_written(self, brain_task: BrainTask) -> str:
        """The worker's spoken part, when nothing needs adding to it. Else empty.

        A result that took long arrives after the talk moved on and needs introducing;
        a council's needs comparing; a program's output was not written to be heard;
        and without a TTS of its own a realtime model has to say it itself.
        """
        if (
            brain_task.status != "done"
            or self.config.delivery == "on_request"
            or brain_task.council
            or brain_task.kind == "code"
            or brain_task.seconds > QUICK_ANSWER_SECONDS
            or getattr(self._voice(), "tts", None) is None
        ):
            return ""
        spoken = brain_task.result.split("Detalle:")[0].strip()
        if len(spoken) > QUICK_ANSWER_CHARS or any(m in spoken for m in ("http", "**", "#", "|")):
            return ""
        return spoken

    def _ask_to_sign_in(self, progress: str) -> None:
        """A run is waiting on the user, and only speaking up gets them to the window."""
        if self.session is None:
            return
        site = progress.partition(": ")[2] or "A website"
        self.session.generate_reply(
            instructions=(
                f"Tell the user, in very few words, that {site} asks them to sign in, that "
                "the page is open in the Chrome window on their computer, and that you carry "
                "on by yourself once they are in. You never see what they type."
            )
        )

    async def _say_still_working(self, brain_task: BrainTask) -> None:
        """A simple answer goes unannounced, so if it is slow the silence needs breaking."""
        await asyncio.sleep(STILL_WORKING_SECONDS)
        session = self.session
        if (
            brain_task.status != "running"
            or session is None
            or session.agent_state not in ("listening", "idle")
            or session.user_state == "speaking"
            or self._user_mid_turn()
        ):
            return
        session.generate_reply(
            instructions="Say in very few words that the answer is coming. Nothing else."
        )

    async def _deliver(self, brain_task: BrainTask) -> None:
        if brain_task.status == "cancelled":
            return
        if self.session is None or self._closed:
            self._store_pending(brain_task)
            return
        if self.config.delivery == "interrupt":
            try:
                await self.session.interrupt()
            except Exception:
                logger.debug("Interrupt before delivery failed", exc_info=True)
        elif not await self._wait_for_pause():
            self._store_pending(brain_task)
            return
        self._remember(brain_task)
        try:
            as_written = self._spoken_as_written(brain_task)
            if as_written:
                self.session.say(as_written, allow_interruptions=True)
            else:
                self.session.generate_reply(
                    instructions=self._spoken(brain_task), allow_interruptions=True
                )
            brain_task.delivered = True
        except Exception:
            logger.exception("Could not speak the result of %s", brain_task.id)
            self._store_pending(brain_task)

    def _remember(self, brain_task: BrainTask) -> None:
        """Keep the result in the voice's history, not only in the reply's instructions.

        Instructions last one reply. Without this the next turn saw an answer with no
        source, and on a real call the voice took it for a guess of its own and apologised
        for it, right answers included. It also lets the voice answer a follow-up. Only
        `_chat_ctx`, which is what generation reads: a realtime model keeps its own.
        """
        try:
            history = getattr(self.session.current_agent, "_chat_ctx", None)
            if history is None or brain_task.status != "done":
                return
            result = brain_task.result[:SPOKEN_RESULT_CHARS]
            history.add_message(
                role="system",
                content=(
                    f"Delegated task {brain_task.id} ({brain_task.title}) finished. Its result, "
                    f"which you tell the user next:\n{result}"
                ),
            )
        except Exception:
            logger.debug("Could not keep a result in the voice's history", exc_info=True)

    def _store_pending(self, brain_task: BrainTask) -> None:
        if brain_task.delivered or not self.kwami_id:
            return
        summary = brain_task.result.split("Detalle:", 1)[0].strip()[:1200]
        self.ledger.add_pending(
            self.kwami_id,
            PendingResult(
                task_id=brain_task.id,
                title=brain_task.title,
                summary=summary,
                finished_at=brain_task.finished_at,
                models=list(brain_task.models),
            ),
        )
        brain_task.delivered = True

    def take_pending(self, kwami_id: str) -> list[PendingResult]:
        """Results left over from earlier calls, for the greeting. Clears them."""
        return self.ledger.take_pending(kwami_id) if kwami_id else []

    # -- ending -------------------------------------------------------------------

    async def close(self, timeout: float) -> None:
        """The call ended. Finish or cancel what is running, keep what is owed."""
        self._closed = True
        self.session = None
        self.publisher = None
        running = [t for t in self.tasks.values() if t.status == "running" and t.handle]
        await self._let_go_of_jobs([t for t in running if t.job_id])
        running = [t for t in running if not t.job_id]
        if self.config.on_hangup == "cancel":
            for brain_task in running:
                brain_task.handle.cancel()  # type: ignore[union-attr]
        elif running:
            logger.info("Call ended with %d brain task(s) running; finishing them", len(running))
            _done, pending = await asyncio.wait(
                [t.handle for t in running if t.handle], timeout=timeout
            )
            for handle in pending:
                handle.cancel()
        for brain_task in self.tasks.values():
            if brain_task.status in ("done", "failed") and not brain_task.delivered:
                self._store_pending(brain_task)
        # Spend and owed results go out in the background; on the cloud the
        # worker is gone soon after this returns.
        await self.ledger.flush(timeout=LEDGER_FLUSH_SECONDS)

    async def _let_go_of_jobs(self, on_pc: list[BrainTask]) -> None:
        """Stop watching. The PC carries on, and the next greeting gets the result."""
        for brain_task in on_pc:
            brain_task.handle.cancel()  # type: ignore[union-attr]
        await asyncio.gather(*(t.handle for t in on_pc if t.handle), return_exceptions=True)
        if self.jobs is None:
            return
        stop = self.config.on_hangup == "cancel"
        for brain_task in on_pc:
            await (
                self.jobs.request_cancel(brain_task.job_id)
                if stop
                else self.jobs.end_watch(brain_task.job_id)
            )

    def find(self, wanted: str) -> BrainModel | None:
        return find_model(self.config.models, wanted)
