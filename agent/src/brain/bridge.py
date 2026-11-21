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
import re
import secrets
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, cast
from zoneinfo import ZoneInfo

from ..adapters.supabase import as_user_id, direct_supabase, get_supabase, kwami_row_id
from ..browser.access import SiteAccess
from ..domain.pipelines import TEXT_REALTIME_PREFIXES
from ..factories.availability import voice_provider_keys
from ..settings import Settings, get_settings
from ..utils.logging import get_logger
from .browser_tool import SIGN_IN_TOOL
from .config import (
    BILLING_MODES,
    HARD_MONTHLY_CAP_USD,
    MAX_INSTRUCTIONS_CHARS,
    Billing,
    BrainConfig,
    BrainModel,
    budget_notice,
    default_brain,
    executor_in_cloud,
    reach,
)
from .executor import CloudWorkspace, run_executor
from .jobs import BrainJob, JobStore
from .ledger import Ledger, PendingResult, create_ledger
from .orchestrator import orchestrate
from .orders import LOGIN, MERGE, merge_approved
from .plan import from_failure, out_until, out_words
from .planner import (
    PLAN_FORMAT,
    PLANNER_SYSTEM,
    Plan,
    PlanStep,
    TeamMember,
    final_step,
    plan_from_text,
    plan_task,
    team_sheet,
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
    ACCOUNT_PROVIDERS,
    PROVIDER_KEYS,
    WEB_PROVIDERS,
    can_delegate,
    council_members,
    director_of,
    find_model,
    on_account,
    pick,
    provider_available,
    should_convene,
    tavily_writer,
    usable,
)
from .selfwork import REPOS
from .team import AGENT_NAME, agent_fields_from_voice
from .user_keys import VOICE_PROVIDER_KEYS, load_user_keys

logger = get_logger("brain.bridge")

#: The kinds of work the director holds and hands pieces of (orchestrator.py).
ORCHESTRATED_KINDS = frozenset({"question", "analysis", "research"})
#: Where the user may say work runs. Empty: wherever fits (the PC, else the cloud).
WHERE_TO_RUN = frozenset({"", "pc", "cloud"})
#: An API key that could not answer, when the user put the key before the plan.
_KEY_TROUBLE = re.compile(
    r"invalid (?:x-)?api.?key|authentication|credit balance|\bbilling\b|insufficient|\b40[12]\b",
    re.IGNORECASE,
)
#: The user chose the plan only, and the cloud runs on the API key.
ONLY_PLAN_NO_CLOUD = (
    "La nube solo trabaja con la clave de API, y has elegido pagar solo con la suscripción "
    "de Claude. Se cambia en Ajustes, Modelos, en cómo paga el cerebro."
)
#: Failures a call keeps for the voice to offer a fix of its own code for.
MAX_FAILURES_KEPT = 8
FAILURE_CHARS = 300
#: What each place is, for the voice to say truthfully (and for `brain_status`).
HOW_WORK_RUNS = {
    "pc": (
        "On the user's PC: Claude Code on the Claude account signed in there (their "
        "plan, nothing per token). Only if the account cannot do it does it fall back "
        "to the API key. It works in the folders the user opened to it, and can open "
        "pull requests on Jarvis's own code."
    ),
    "cloud": (
        "In the cloud: Jarvis's work thread on the server, with the PC off or on. "
        "Claude Code on the API key (paid per token), with no access to the PC's "
        "files; what it writes is brought to the PC the next time it is on."
    ),
    "here": (
        "Inside the voice agent's own process, when neither the PC's work thread nor "
        "the cloud takes it: on the API key, paid per token, and only for this call."
    ),
}
#: How long a model that answered "out of quota" is left alone. A daily quota
#: does not come back sooner; a per-minute one will have by then.
QUOTA_REST_SECONDS = 15 * 60
#: How many models one task, or one part of it, is tried on: the chosen one and two
#: fallbacks. One fallback gave up while a third keyed provider sat idle.
MAX_ATTEMPTS = 3

#: How long a finished result waits for the user to stop talking.
PAUSE_WAIT_SECONDS = 120.0
#: A moment of quiet, on top of knowing no turn is open (`_user_mid_turn`).
PAUSE_QUIET_SECONDS = 0.4
#: Results longer than this are summarised in speech; the full text stays in
#: `check_tasks` and in the app.
SPOKEN_RESULT_CHARS = 3500
MAX_TASKS_KEPT = 30
#: Tasks one session runs at the same time. Each can call several paid models, and
#: a voice model that loops on `delegate_task` should hit a wall, not the budget.
MAX_RUNNING_TASKS = 4
#: How long the council waits for one member. A provider that hangs used to hold
#: the whole council -- and the answer -- for its ten-minute client timeout.
COUNCIL_MEMBER_SECONDS = 240.0
#: What a task keeps of what it did, step by step, and how much of it the app is sent:
#: the app draws it as the task's live log.
ACTIVITY_KEPT = 40
ACTIVITY_SHOWN = 14
ACTIVITY_CHARS = 200
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

#: How a spoken request changes instructions (`Bridge.set_instructions`).
INSTRUCTION_MODES = ("replace", "append", "clear")
#: Words that mean the instructions for every model, not one of them.
BRAIN_TARGETS = ("brain", "all", "everyone", "cerebro", "todos", "todo")
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


#: The heading that parts what the voice says from the report behind it (WORKER_SYSTEM).
_REPORT_HEADING = re.compile(r"(?:^|\n)[ \t]*\**Detalle:?\**[ \t]*(?:\n|$)")


def spoken_and_report(result: str) -> tuple[str, str]:
    """What to say, and the report after the "Detalle:" heading (empty if there is none)."""
    match = _REPORT_HEADING.search(result)
    if match is None:
        return result.strip(), ""
    return result[: match.start()].strip(), result[match.end() :].strip()


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
    #: One of the user's agents does it, by name (`brain_agents`).
    agent: str = ""
    #: Where the user said it must run: "pc", "cloud", or "" for wherever fits.
    where: str = ""
    #: Where it did run, once it ends: "pc", "cloud" or "here" (inside this call).
    ran_on: str = ""
    #: Who paid for it: "account" (the Claude plan on the PC), "api" (a key, per
    #: token), or "" when nothing was spent.
    paid_by: str = ""
    #: Set once the work is on the PC.
    job_id: str = ""
    thread_id: str = ""
    progress: str = ""
    #: Models whose prompt carried the user's instructions (the brain's or their own).
    briefed: list[str] = field(default_factory=list)
    #: What it did, line by line ("model: Opus", "Bash: npm test"): the app's live log.
    activity: list[dict[str, str]] = field(default_factory=list)
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
            "where": self.where or "auto",
            "ranOn": self.ran_on,
            "paidBy": self.paid_by,
            "progress": self.progress,
            "briefed": self.briefed,
            "activity": self.activity[-ACTIVITY_SHOWN:],
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
        }
        if full or self.status != "running":
            out["result"] = self.result if full else self.result[:1500]
            out["sources"] = self.sources[:8]
        return out

    def note(self, text: str) -> bool:
        """Add a line to what the task did. False when there is nothing new to add."""
        line = " ".join(text.split())[:ACTIVITY_CHARS]
        if not line or (self.activity and self.activity[-1]["text"] == line):
            return False
        self.activity.append({"at": datetime.now(UTC).isoformat(), "text": line})
        del self.activity[:-ACTIVITY_KEPT]
        return True


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
        #: The user's list of sites Jarvis may visit, and what it may do on each.
        self.site_access = SiteAccess(jobs_store)
        self._keys_secret = settings.provider_keys_secret
        #: The user's own keys for the voice pipeline, by environment name.
        self.voice_keys: dict[str, str] = {}
        #: Whose keys this call may hold. The first person only: the usage is billed to
        #: them, so nobody who joins later may bring their keys in, or spend theirs.
        self._keys_owner = ""
        self._voice_keys_loaded = asyncio.Event()
        self.pc_online = False
        #: Whether the PC's Claude Code is signed in, whatever is left of its plan.
        self.claude_signed_in = False
        #: What is left of that plan (`plan.py`): the PC's word, or this call's own
        #: when a thought there came back saying it ran out.
        self.claude_plan: dict[str, Any] | None = None
        #: The user has heard, in this call, that the plan ran out and the key pays.
        self._told_plan_out = False
        self._account_checked_at: float | None = None
        self._executor_in_cloud = executor_in_cloud(settings.executor_mode)
        self.config: BrainConfig = default_brain()
        self.session: Any = None
        self.publisher: Any = None
        #: The kwami's row in `user_kwamis` when it can be told, else as it came.
        self.kwami_id = ""
        #: As the app sent it: the memory's long name, `kwami_<user>_<kwami>`.
        self._kwami_given = ""
        #: The Supabase user id, once the room or the kwami's row tells us.
        self.user_id = ""
        self.assistant_name = "Jarvis"
        self.language = "Spanish (Spain)"
        self.timezone = "Europe/Madrid"
        self.user_profile = ""
        self.tasks: dict[str, BrainTask] = {}
        self._ids = itertools.count(1)
        #: Ids are numbered per call (`t1`, `t2`...) and a result can outlive its call
        #: (`brain_pending`, the app's windows): this keeps one call's t1 from another's.
        self._call_tag = secrets.token_hex(2)
        #: Models out of quota, and until when (monotonic) they are skipped.
        self._resting: dict[str, float] = {}
        self._closed = False
        self._background: set[asyncio.Task] = set()
        #: What could not be done in this call, newest last: the voice offers to fix
        #: its own code for it (`brain_status`, `own_code_repo`).
        self.failures: deque[dict[str, str]] = deque(maxlen=MAX_FAILURES_KEPT)

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
            self._kwami_given = kwami_id
            # The kwami's row: every brain table keys a kwami by it (`kwami_row_id`).
            self.kwami_id = kwami_row_id(kwami_id, self.user_id or None) or kwami_id
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
        if self._kwami_given:
            # Known now whose the kwami's long name must be.
            self.kwami_id = kwami_row_id(self._kwami_given, resolved) or self._kwami_given
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
        signed_in, plan = (
            await self.jobs.account_state(self.user_id) if self.pc_online else (False, None)
        )
        self.claude_signed_in = signed_in
        self._keep_plan(plan)
        # A plan that has run out takes nothing until it resets: the thoughts go straight
        # to the key, or are refused saying when it comes back, as the user chose.
        if signed_in and out_until(self.claude_plan) is None:
            self._keys[ACCOUNT_MARK] = "1"
        else:
            self._keys.pop(ACCOUNT_MARK, None)

    def _keep_plan(self, plan: dict[str, Any] | None) -> None:
        """Keep the newer of what the PC said and what this call saw."""
        if plan is None:
            return
        mine = self.claude_plan or {}
        if str(plan.get("seenAt") or "") >= str(mine.get("seenAt") or ""):
            self.claude_plan = plan

    def _note_plan_out(self, text: str) -> None:
        """A thought on the PC came back saying the plan ran out: stop sending them."""
        plan = from_failure(text)
        if plan is None:
            return
        logger.info("The PC's Claude plan is out: %s", plan)
        self._keep_plan(plan)
        self._keys.pop(ACCOUNT_MARK, None)

    def _account_unavailable(self, way: str = "") -> str:
        """Why the plan cannot answer, and why the key does not either."""
        until = out_until(self.claude_plan)
        if until is not None:
            why = out_words(until, self._zone())
        elif not self.pc_online:
            why = "El hilo de trabajo de tu ordenador no está en marcha."
        elif not self.claude_signed_in:
            why = "Claude Code no tiene la sesión iniciada en tu ordenador."
        else:
            why = "La cuenta de Claude de tu ordenador no está disponible ahora."
        if self.config.billing == "account_only":
            return (
                f"{why} Has elegido pagar solo con la suscripción, así que no lo hago por la API."
            )
        if way == "account":
            return f"{why} Ese modelo está puesto para usar solo la cuenta, no la clave de API."
        return f"{why} No hay clave de API de Anthropic para hacerlo mientras tanto."

    def billing_state(self) -> dict[str, Any]:
        """How the brain pays now, as the user chose and as things stand (`plan.py`).

        The models panel shows it and the voice answers from it: `now` is what a piece
        of Claude work would use this moment -- the plan, the key, or nothing -- and
        `why` says what moved it off the user's first choice.
        """
        mode = self.config.billing
        key = bool(self._keys.get("ANTHROPIC_API_KEY"))
        plan_has_room = bool(self._keys.get(ACCOUNT_MARK))
        until = out_until(self.claude_plan)
        why = ""
        if mode in ("api_only", "api_first") and key:
            now = "api"
        elif mode == "api_only":
            now, why = "none", "No hay clave de API de Anthropic."
        elif plan_has_room:
            now = "account"
        else:
            if until is not None:
                why = out_words(until, self._zone())
            elif not self.pc_online:
                why = "El hilo de trabajo de tu ordenador no está en marcha."
            else:
                why = "Claude Code no tiene la sesión iniciada en tu ordenador."
            now = "api" if mode != "account_only" and key else "none"
        plan = self.claude_plan or {}
        return {
            "mode": mode,
            "now": now,
            "why": why,
            "apiKey": key,
            "plan": {
                "out": until is not None,
                "until": until.isoformat() if until is not None else None,
                "status": plan.get("status") or "",
                "window": plan.get("type") or "",
                "utilization": plan.get("utilization"),
                "resetsAt": plan.get("resetsAt"),
                "seenAt": plan.get("seenAt"),
            },
            # Gemini, GPT, Grok and Tavily are paid per use: off with the plan only.
            "otherProvidersPaid": mode != "account_only",
        }

    def _owner(self) -> str:
        return self.user_id or getattr(self.ledger, "user_id", None) or ""

    def _cloud_workspace(self) -> CloudWorkspace | None:
        if not self._executor_in_cloud:
            return None
        return CloudWorkspace(store=self._store, user_id=self._owner(), kwami_id=self.kwami_id)

    def attach(self, session: Any, publisher: Any) -> None:
        self.session = session
        self.publisher = publisher

    @property
    def working(self) -> bool:
        """Whether a delegated task is still running: the user may be waiting on it."""
        return any(t.status == "running" for t in self.tasks.values())

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
                "loggedIn": self.claude_signed_in,
                # Signed in with room on the plan: thoughts go there now.
                "usable": bool(self._keys.get(ACCOUNT_MARK)),
            },
            "billing": self.billing_state(),
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

    def instructions(self) -> dict[str, Any]:
        """What the user has told the brain and each model, as it stands now."""
        return {
            "brain": self.config.instructions,
            "models": {
                (m.label or m.model): m.instructions for m in self.config.models if m.instructions
            },
        }

    async def set_instructions(self, target: str, text: str, mode: str = "replace") -> str:
        """Change the brain's or one model's instructions, by the user's spoken request.

        Applied at once to this session and sent to the app as
        `brain_instructions`, which saves it in the Models panel like a typed
        change. Returns what to tell the user.
        """
        if mode not in INSTRUCTION_MODES:
            return f"Unknown mode {mode!r}: use replace, append or clear."
        wanted = target.strip()
        model: BrainModel | None = None
        if wanted.lower() not in BRAIN_TARGETS:
            model = find_model(self.config.models, wanted)
            if model is None:
                names = ", ".join(m.label or m.model for m in self.config.models)
                return f"No model called {target!r}. The models are: {names}."
        current = model.instructions if model is not None else self.config.instructions
        addition = text.strip()
        if mode == "clear":
            updated = ""
        elif mode == "append" and current:
            updated = f"{current}\n{addition}"
        else:
            updated = addition
        if mode != "clear" and not addition:
            return "Say what the instructions should be."
        cut = len(updated) > MAX_INSTRUCTIONS_CHARS
        updated = updated[:MAX_INSTRUCTIONS_CHARS]
        if model is not None:
            model.instructions = updated
        else:
            self.config.instructions = updated
        who = model.label or model.model if model is not None else "the whole brain"
        logger.info("Instructions for %s changed by voice (%s)", who, mode)
        if self.publisher is not None:
            try:
                await self.publisher.publish(
                    {
                        "type": "brain_instructions",
                        "target": model.id if model is not None else "brain",
                        "instructions": updated,
                    }
                )
            except Exception:
                logger.debug("Could not send the new instructions to the app", exc_info=True)
        if not updated:
            return f"Instructions for {who} cleared."
        limit = f" (cut to the {MAX_INSTRUCTIONS_CHARS}-character limit)" if cut else ""
        return f"Instructions for {who} saved{limit}: {updated}"

    async def set_billing(self, billing: str) -> str:
        """Change how the brain pays, by the user's spoken request.

        Applied at once and sent to the app as `brain_billing`, which saves it in the
        Models panel like a click there. Returns what to tell the user.
        """
        wanted = billing.strip().lower()
        if wanted not in BILLING_MODES:
            return f"Unknown billing {billing!r}: use one of {', '.join(BILLING_MODES)}."
        self.config.billing = cast(Billing, wanted)
        logger.info("Billing changed by voice to %s", wanted)
        if self.publisher is not None:
            try:
                await self.publisher.publish({"type": "brain_billing", "billing": wanted})
            except Exception:
                logger.debug("Could not send the new billing to the app", exc_info=True)
        await self._regate_voice()
        state = self.billing_state()
        return f"Billing is now {wanted}; right now Claude work would use: {state['now']}." + (
            f" ({state['why']})" if state["why"] else ""
        )

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
        agent: str = "",
        where: str = "",
    ) -> BrainTask | str:
        """Queue a task. Returns it, or a sentence saying why it cannot run."""
        where = "" if where in ("", "auto") else where
        if where not in WHERE_TO_RUN:
            return "Where it runs must be 'pc', 'cloud', or left empty."
        if where:
            # The PC and the cloud run commands on their owner's plan or key: never by phone.
            if not self.user_id:
                return "Work on the PC or in the cloud can only be asked from the app."
            council = important = False
        if agent:
            if not AGENT_NAME.fullmatch(agent):
                return "An agent's name is lowercase letters, digits and dashes."
            # The room's identity, as for own code: on a phone line anyone can talk.
            if not self.user_id:
                return "The user's agents can only be asked from the app, not by phone."
            important, council = False, False
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

        running = sum(1 for t in self.tasks.values() if t.status == "running")
        if running >= MAX_RUNNING_TASKS:
            return (
                f"{running} tasks are already running. Wait for one to finish, or cancel "
                "one with cancel_task, before starting another."
            )

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
            id=f"t{next(self._ids)}-{self._call_tag}",
            title=(title or task)[:120],
            task=self._with_conversation(task, context),
            kind=kind,
            difficulty=difficulty,
            council=convene,
            long_running=long_running,
            follow_up=follow_up,
            repo=repo,
            agent=agent,
            where=where,
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

    def _zone(self) -> ZoneInfo:
        try:
            return ZoneInfo(self.timezone)
        except Exception:
            return ZoneInfo("UTC")

    def _now_line(self) -> str:
        """The user's date and time, and UTC, for work that runs where no clock is asked."""
        now = datetime.now(self._zone())
        utc = now.astimezone(UTC).strftime("%Y-%m-%d %H:%M")
        return f"Now: {now.strftime('%A %d %B %Y, %H:%M')} ({self.timezone}); {utc} UTC."

    def _system(self) -> str:
        zone = self._zone()
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
        # Tavily is paid per search: not with the plan only.
        paid_ok = self.config.billing != "account_only"
        tavily = self._tavily_key if paid_ok and self.config.search in ("tavily", "both") else ""
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
        return model.provider == "anthropic" and on_account(model, self._keys, self.config.billing)

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

    def _note_brief(self, brain_task: BrainTask | None, model: BrainModel, briefed: bool) -> None:
        """Record that the user's instructions went out with this model's prompt."""
        label = model.label or model.model
        if briefed and brain_task is not None and label not in brain_task.briefed:
            brain_task.briefed.append(label)

    async def _ask(self, model: BrainModel, request: BrainRequest) -> BrainAnswer:
        answer = await self._ask_model(model, request)
        # Only an answer counts: a call that never reached the model carried nothing.
        answer.briefed = answer.ok and bool(self._brief(model))
        return answer

    async def _ask_model(self, model: BrainModel, request: BrainRequest) -> BrainAnswer:
        brief = self._brief(model)
        if brief:
            request = replace(request, system=f"{request.system}\n\n{brief}")
        api_key = self._keys.get(PROVIDER_KEYS[model.provider], "")
        way = reach(model, self.config.billing) if model.provider in ACCOUNT_PROVIDERS else "api"
        if self._on_account(model):
            answer = await self._think_on_account(model, request)
            if answer is not None and not answer.ok:
                self._note_plan_out(answer.text)
            if answer is not None and (answer.ok or way == "account" or not api_key):
                return answer
            if answer is not None:
                logger.info("The account could not answer on %s; using the API key", model.id)
        if model.provider in ACCOUNT_PROVIDERS and (not api_key or way == "account"):
            # The plan only, or the PC went away (or its plan ran out) since it was picked.
            return BrainAnswer(self._account_unavailable(way), model.id, model.model, ok=False)
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
        if (
            not answer.ok
            and way == "api_first"
            and model.provider == "anthropic"
            and self._keys.get(ACCOUNT_MARK)
            and _KEY_TROUBLE.search(answer.text)
        ):
            # The key first, and the key could not: the plan is the user's second choice.
            logger.info("The API key could not answer on %s; the PC's plan", model.id)
            on_plan = await self._think_on_account(model, request)
            if on_plan is not None and on_plan.ok:
                return on_plan
        return answer

    async def _plan(
        self, director: BrainModel, task: str, brain_task: BrainTask | None = None
    ) -> Plan | None:
        """Ask the director how to split the work, and whom to give each part.

        None when it could not say.
        """
        api_key = self._keys.get("ANTHROPIC_API_KEY", "")
        team = self._team()
        ids = [member.id for member in team]
        planner = PLANNER_SYSTEM + team_sheet(team)
        if director.provider == "anthropic" and api_key and not self._on_account(director):
            # The structured-output call does not go through _ask, so the user's
            # instructions for the brain and for the director are added here.
            brief = self._brief(director)
            self._note_brief(brain_task, director, bool(brief))
            system = f"{planner}\n\n{brief}" if brief else planner
            plan = await plan_task(api_key, director.model, task, system=system, team=ids)
            if plan is not None:
                self.ledger.add_spend(director.id, plan.cost_usd)
            return plan
        # The account, or a director that is not Claude: a plain answer in JSON.
        answer = await self._ask(
            director,
            BrainRequest(
                task=task,
                system=planner + PLAN_FORMAT,
                difficulty="simple",
                native_search=False,
            ),
        )
        self._note_brief(brain_task, director, getattr(answer, "briefed", False))
        if not answer.ok:
            return None
        plan = plan_from_text(answer.text, ids)
        plan.cost_usd = answer.cost_usd
        return plan

    def _team(self) -> list[TeamMember]:
        """The models the director may hand a step to: usable, and not the code executor."""
        return [
            TeamMember(
                model.id,
                model.label or model.model,
                tuple(model.roles or [model.role]),
                model.instructions,
            )
            for model in usable(self.config, self._keys)
            if model.provider != "local"
        ]

    def _note_model(self, brain_task: BrainTask, label: str) -> None:
        if label not in brain_task.models:
            brain_task.models.append(label)
            self._log(brain_task, f"model: {label}")

    def _log(self, brain_task: BrainTask, text: str) -> None:
        """A line for the task's live log, sent to the app when it is new."""
        if brain_task.note(text):
            self._schedule(self.publish_status())

    async def publish_result(self, brain_task: BrainTask) -> None:
        """The whole answer of a finished task, for the app to open as a document.

        `brain_status` carries only the start of each result; a report is read in full.
        """
        if self.publisher is None or brain_task.status != "done" or not brain_task.result.strip():
            return
        try:
            spoken, report = spoken_and_report(brain_task.result)
            await self.publisher.publish(
                {
                    "type": "brain_result",
                    **brain_task.public(full=True),
                    "spoken": spoken,
                    "report": report,
                }
            )
        except Exception:
            logger.debug("Could not send a task's result to the app", exc_info=True)

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
        # Providers that failed as a whole this task: a bad key, a quota, an outage.
        down: set[str] = set()
        for _ in range(MAX_ATTEMPTS):  # the chosen model, then the fallbacks
            model = self._pick_fallback(kind, difficulty, prefer if not tried else "", tried, down)
            if model is None:
                break
            tried.add(model.id)
            self._note_model(brain_task, model.label or model.model)
            answer = await self._ask(model, request)
            self._note_brief(brain_task, model, getattr(answer, "briefed", False))
            brain_task.cost_usd += answer.cost_usd
            if answer.ok:
                return answer, model
            logger.info("Brain model %s failed (%s); trying a fallback", model.id, answer.text)
            if answer.provider_down:
                down.add(model.provider)
            if answer.rate_limited:
                self._rest(model)
            if model.provider == "google" and request.native_search and self._tavily_key:
                rescued = await self._search_on_tavily(brain_task, kind, difficulty, request, tried)
                if rescued is not None:
                    return rescued
        failed = answer or BrainAnswer("No usable model could take the task.", "", "", ok=False)
        return failed, model

    def _pick_fallback(
        self, kind: str, difficulty: str, prefer: str, tried: set[str], down: set[str]
    ) -> BrainModel | None:
        """The next model to try, from another provider when one failed as a whole.

        When Anthropic is down, the next model the roles name is usually another
        Claude, which fails the same way: the task went to two Claude models and
        then gave up, with Gemini and GPT keyed and idle. A model of a provider
        that failed is still used when nothing else can take the task.
        """
        if down:
            elsewhere = tried | {m.id for m in self.config.models if m.provider in down}
            model = pick(self.config, self._keys, kind, difficulty, prefer, frozenset(elsewhere))
            if model is not None:
                return model
        return pick(self.config, self._keys, kind, difficulty, prefer, frozenset(tried))

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

    async def list_agents(self) -> dict[str, Any]:
        """The user's own agents, for the voice: who they are and what each is for."""
        if self.jobs is None or not self.user_id:
            return {"error": "The user's agents can only be reached from the app."}
        rows = await self.jobs.all_agents(self.user_id)
        if rows is None:
            return {"error": "Your agents cannot be reached right now."}
        return {
            "agents": [
                {
                    "name": row.get("name"),
                    "description": row.get("description"),
                    "model": row.get("model"),
                    "tools": row.get("tools"),
                    "target": row.get("target"),
                    "enabled": row.get("enabled"),
                }
                for row in rows
            ]
        }

    async def save_agent(self, fields: dict[str, Any]) -> str:
        """Create or change one of the user's agents, as the voice asked. What to say.

        The room's own user only, as with their PC: anyone can talk on a phone line.
        """
        if self.jobs is None or not self.user_id:
            return "The user's agents can only be changed from the app, not by phone."
        cleaned = agent_fields_from_voice(fields)
        if isinstance(cleaned, str):
            return cleaned
        return await self.jobs.save_agent(self.user_id, cleaned)

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

    async def _cloud_takes_work(self) -> bool:
        """Whether the cloud's work thread is there and this user lets it work for them."""
        return bool(
            self.jobs
            and await self.jobs.cloud_online()
            and await self.jobs.cloud_allowed(self.user_id)
        )

    async def placement(self, where: str = "") -> tuple[str, str]:
        """Where a job asked to run at ``where`` would go now, and why not if nowhere.

        ("auto" | "pc" | "cloud", "") when it can go; ("", reason) when it cannot, the
        reason in the user's language for the voice to say as it is. "" asks for
        wherever fits: the PC when its work thread is there, else the cloud if the
        user allowed it. "pc" and "cloud" go there or nowhere: on 2026-11-16 "en la
        nube" ran on the PC, and the voice could not tell.

        Only the room's own user, never the kwami's owner: on a phone line anyone can
        talk, and the PC runs commands on its owner's login.
        """
        if self.jobs is None:
            return "", "Esta sesión no tiene la cola de trabajo (falta Supabase)."
        if not self.user_id:
            return "", "El ordenador y la nube solo se encargan desde la aplicación."
        if as_user_id(self.kwami_id) is None:
            return "", "Esta sesión no sabe qué kwami es, así que no puede encargar trabajo fuera."
        only_plan = self.config.billing == "account_only"
        if where == "cloud":
            if only_plan:
                return "", ONLY_PLAN_NO_CLOUD
            if not await self.jobs.cloud_online():
                return "", (
                    "El hilo de trabajo de la nube no está en marcha ahora: el servidor no "
                    "da señales."
                )
            if not await self.jobs.cloud_allowed(self.user_id):
                return "", "La nube no tiene su permiso: se activa en Ajustes, en el cerebro."
            return "cloud", ""
        if await self.jobs.runner_online(self.user_id):
            return ("pc" if where == "pc" else "auto"), ""
        if where == "pc":
            return "", (
                "El hilo de trabajo del ordenador no está en marcha: el ordenador está "
                "apagado o KWAMI no está abierto en él."
            )
        if only_plan:
            return "", (
                "El hilo de trabajo del ordenador no está en marcha, y la nube no cuenta: "
                "trabaja con la clave de API y has elegido pagar solo con la suscripción."
            )
        if await self._cloud_takes_work():
            return "cloud", ""
        return "", "Ni el hilo de trabajo del ordenador ni el de la nube están en marcha."

    def note_failure(self, what: str, why: str) -> None:
        """Keep what could not be done, for `brain_status` and an offer to improve."""
        self.failures.append(
            {
                "what": what[:120],
                "why": why[:FAILURE_CHARS],
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )

    async def where_work_runs(self) -> dict[str, Any]:
        """Where work can go now, and how each place is paid: the voice answers from this.

        Asked "¿lo hace Claude con la cuenta del PC o por la API?", the voice once sent
        a web search to find out, and the brain could not see the PC (2026-11-16). The
        worker reports its login with every heartbeat; this is that, and the cloud's.
        """
        await self._refresh_account(force=True)
        cloud_on = allowed = False
        if self.jobs is not None:
            cloud_on = await self.jobs.cloud_online()
            allowed = bool(self.user_id) and await self.jobs.cloud_allowed(self.user_id)
        target, why_not = await self.placement()
        return {
            "pc": {
                "workThreadOn": self.pc_online,
                "claudeAccountSignedIn": self.claude_signed_in,
                "planHasRoom": bool(self._keys.get(ACCOUNT_MARK)),
                "apiKeyFallback": bool(self._keys.get("ANTHROPIC_API_KEY")),
                "how": HOW_WORK_RUNS["pc"],
            },
            # The user's choice of plan or key, and what it means this moment.
            "billing": self.billing_state(),
            "cloud": {
                "workThreadOn": cloud_on,
                "allowedByUser": allowed,
                "how": HOW_WORK_RUNS["cloud"],
            },
            "here": HOW_WORK_RUNS["here"],
            # Where a task with nothing said about where would go right now.
            "wouldGoNow": {"auto": "pc"}.get(target, target) or "here",
            "whyNotPcOrCloud": why_not,
        }

    @staticmethod
    def _insisted(brain_task: BrainTask) -> bool:
        """Work that runs on the PC or in the cloud or not at all, never stood in for here.

        Where the user said; one of their agents, their own specialist; and Jarvis's own
        code, which only the PC publishes (the cloud holds no GitHub credentials: it
        leaves a patch the PC publishes later).
        """
        return bool(brain_task.where or brain_task.agent or brain_task.repo)

    async def _on_the_pc(self, brain_task: BrainTask, prefer: str) -> BrainAnswer | None:
        """Send the work to the PC, or to the cloud's work thread (`placement`).

        None when neither will take it and the user did not say where: the caller runs
        it here. Said where, or named an agent, it runs there or fails saying why. The
        cloud only when the user allowed it (migration 029); there the job outlives the
        call, keeps its files in Storage, and the PC brings them home later
        (`cloud.py`, `sync.py`).
        """
        agent_id, where = "", brain_task.where
        row: dict[str, Any] | None = None
        if brain_task.agent and self.jobs is not None and self.user_id:
            agents = await self.jobs.agents(self.user_id)
            row = next((a for a in agents if a.get("name") == brain_task.agent), None)
            if row is None:
                return BrainAnswer(
                    f"No tienes ningún agente activo llamado {brain_task.agent}.", "", "", ok=False
                )
            agent_id = str(row.get("id") or "")
            # The user's words win over where the agent usually works.
            if not where and row.get("target") in ("pc", "cloud"):
                where = str(row["target"])
        target, why_not = await self.placement(where)
        if not target:
            if not self._insisted(brain_task):
                return None
            who = (
                f"Su agente {brain_task.agent}"
                if brain_task.agent
                else ("Cambiar mi propio código" if brain_task.repo else "La tarea")
            )
            place = {"pc": "en el ordenador", "cloud": "en la nube"}.get(where, "fuera")
            return BrainAnswer(
                f"{who} no puede trabajar {place} ahora. {why_not}", "", "", ok=False
            )
        model = pick(self.config, self._keys, "code", brain_task.difficulty, prefer)
        if model is None:
            if not self._insisted(brain_task):
                return None
            return BrainAnswer(
                "No hay ningún modelo de Claude activo para el trabajo con herramientas "
                "(el ejecutor está apagado en el panel de modelos).",
                "",
                "",
                ok=False,
            )
        jobs = cast(JobStore, self.jobs)  # `placement` found the queue
        earlier = self.tasks.get(brain_task.follow_up)
        thread_id = earlier.thread_id if earlier is not None else ""
        if not thread_id and brain_task.follow_up == "last":
            thread_id = await jobs.last_thread(self.kwami_id)
        brief = self._brief(model)
        self._note_brief(brain_task, model, bool(brief))
        # The cloud has no shell to ask the clock, and wrote a file "without the time,
        # because I have no clock here" (2026-11-16). The user's time travels with it.
        task = f"{brain_task.task}\n\n{self._now_line()}"
        job = BrainJob(
            user_id=self.user_id,
            kwami_id=self.kwami_id,
            title=brain_task.title,
            task=f"{task}\n\n{brief}" if brief else task,
            thread_id=thread_id,
            difficulty=brain_task.difficulty,
            model=model.model,
            repo=brain_task.repo,
            target=target,
            agent_id=agent_id,
        )
        said_where = self._insisted(brain_task)
        if not await jobs.enqueue(job):
            if not said_where:
                return None
            return BrainAnswer(
                "No he podido dejar el encargo en la cola de trabajo; inténtelo de nuevo.",
                "",
                "",
                ok=False,
            )
        brain_task.job_id, brain_task.thread_id = job.id, job.thread_id
        self._note_model(brain_task, model.label or model.model)
        if target == "cloud":
            self._note_model(brain_task, "nube")
        self._schedule(self.publish_status())
        answer = await self._watch_job(jobs, brain_task, job.id, model)
        if answer is None and said_where:
            place = "la nube" if target == "cloud" else "el ordenador"
            return BrainAnswer(
                f"Nadie recogió el encargo: {place} dejó de dar señales antes de empezar. "
                "No lo he hecho en otro sitio.",
                "",
                "",
                ok=False,
            )
        return answer

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
            self._log(brain_task, job.progress)
            if job.progress.startswith(SIGN_IN_TOOL) and not brain_task.asked_to_sign_in:
                brain_task.asked_to_sign_in = True
                self._ask_to_sign_in(job.progress)
            if job.finished:
                await jobs.mark_delivered(job_id)
                self.ledger.note_spend(model.id, job.cost_usd)
                brain_task.cost_usd += job.cost_usd
                # Who took it, not where it was asked to go: "auto" may end anywhere.
                in_cloud = job.in_cloud or (not job.runner_id and job.target == "cloud")
                # The PC's login costs nothing per run; the cloud is always on a key. A job
                # cancelled before anyone took it ran nowhere and cost nothing.
                ran = job.status != "cancelled"
                on_account = ran and not in_cloud and job.cost_usd == 0
                if ran:
                    brain_task.ran_on = "cloud" if in_cloud else "pc"
                    brain_task.paid_by = (
                        "account" if on_account else ("api" if job.cost_usd > 0 else "")
                    )
                return BrainAnswer(
                    job.result,
                    model.id,
                    model.model,
                    job.cost_usd,
                    ok=job.status == "done",
                    via_account=on_account,
                )
            if (
                job.status == "queued"
                and loop.time() - queued_since >= JOB_QUEUE_PATIENCE_SECONDS
                and not await (
                    jobs.cloud_online()
                    if job.target == "cloud"
                    else jobs.runner_online(job.user_id)
                )
            ):
                await jobs.request_cancel(job_id)
                job = await jobs.get(job_id)
                if job is not None and job.status == "cancelled":
                    logger.info("Its worker went away; running %s here", brain_task.id)
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
            self.config.orchestrate
            and not prefer
            # Reasoning work. Writing and reading go straight to the model that writes
            # and reads (Sonnet): a director in between only adds its own tokens.
            and brain_task.kind in ORCHESTRATED_KINDS
            and brain_task.difficulty != "simple"
        ):
            orchestrated = await self._orchestrated(brain_task)
            if orchestrated is not None:
                return orchestrated
        if (
            prefer
            or not self.config.split
            # Writing and reading are one piece of text for the model that writes and
            # reads: a plan would only add a director's call.
            or brain_task.kind not in ORCHESTRATED_KINDS
            or brain_task.difficulty == "simple"
        ):
            return await self._single(brain_task, prefer)
        director = director_of(self.config, self._keys)
        if director is None:
            return await self._single(brain_task, prefer)

        plan = await self._plan(director, brain_task.task, brain_task)
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
            # The director may also have named who should do it.
            return await self._single(brain_task, step.model)

        # When the last step already builds on every other one, it answers the
        # user itself. Measured 2026-10-09: a separate combine call repeated the
        # Opus work and cost about a third of the whole task.
        final_id = final_step(plan.steps)
        final_answer: BrainAnswer | None = None
        results: dict[str, str] = {}
        self._log(brain_task, f"plan: {len(plan.steps)}")
        for wave in waves(plan.steps):
            for step in wave:
                self._log(brain_task, f"step: {step.goal}")
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
                self._log(brain_task, f"{'step_done' if answer.ok else 'step_failed'}: {step.goal}")
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

    def _searches(self, model: BrainModel) -> bool:
        """Whether a worker can look at the web: its own search, or Tavily run here."""
        if model.provider in WEB_PROVIDERS and self.config.search != "tavily":
            return True
        return bool(self._tavily_key) and self.config.search in ("tavily", "both")

    async def _orchestrated(self, brain_task: BrainTask) -> BrainAnswer | None:
        """The director holds the task and hands pieces to the others (orchestrator.py).

        None when it cannot run here -- a director that is not Claude on an API key
        (the PC's account has no tool loop), or nobody to hand work to -- or when the
        director could not be reached; the plan made up front runs instead.
        """
        director = director_of(self.config, self._keys)
        api_key = self._keys.get("ANTHROPIC_API_KEY", "")
        if (
            director is None
            or director.provider != "anthropic"
            or not api_key
            or self._on_account(director)
        ):
            return None
        resting = self._resting_ids()
        workers = [
            m
            for m in usable(self.config, self._keys)
            if m.id != director.id and m.provider != "local" and m.id not in resting
        ]
        if not workers:
            return None

        async def delegate(worker: BrainModel, instructions: str, web: bool) -> BrainAnswer:
            self._note_model(brain_task, worker.label or worker.model)
            # A piece is never the hard part: the director keeps the judgement. Run at
            # the hard effort, the workers searched and wrote at length on every piece.
            request = self._request(instructions, "normal", system=STEP_SYSTEM)
            request.native_search = request.native_search and web
            if not web:
                request.tavily_key = ""
            answer = await self._ask(worker, request)
            self._note_brief(brain_task, worker, getattr(answer, "briefed", False))
            if answer.rate_limited:
                self._rest(worker)
            return answer

        self._note_model(brain_task, director.label or director.model)
        brief = self._brief(director)
        self._note_brief(brain_task, director, bool(brief))
        system = f"{self._system()}\n\n{brief}" if brief else self._system()
        result = await orchestrate(
            api_key,
            director,
            workers,
            brain_task.task,
            system,
            brain_task.difficulty,
            delegate,
            self._searches,
            budget_usd=self.config.executor.max_task_usd,
        )
        if result is None:
            return None
        director_cost = result.answer.cost_usd - sum(d.cost_usd for d in result.delegations)
        self.ledger.add_spend(director.id, director_cost)
        brain_task.cost_usd += result.answer.cost_usd
        for piece in result.delegations:
            model = next((m for m in workers if m.id == piece.worker), None)
            brain_task.steps.append(
                {
                    "goal": piece.goal,
                    "difficulty": brain_task.difficulty,
                    "model": (model.label or model.model) if model else piece.worker,
                    "ok": piece.ok,
                }
            )
        return result.answer

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
        # The director's choice for this step goes first; the router covers a
        # model it cannot use (no web for a web step) and the fallback.
        return await self._ask_role(brain_task, kind, step.difficulty, request, step.model)

    async def _council(self, brain_task: BrainTask) -> BrainAnswer:
        members = council_members(self.config, self._keys)
        brain_task.models = [m.label or m.model for m in members]
        request = self._request(brain_task.task, "hard")
        answers = await asyncio.gather(*(self._ask_member(m, request) for m in members))
        for member, member_answer in zip(members, answers, strict=True):
            self._note_brief(brain_task, member, getattr(member_answer, "briefed", False))
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
        self._note_brief(brain_task, chair, getattr(synthesis, "briefed", False))
        brain_task.cost_usd += synthesis.cost_usd
        if not synthesis.ok:
            return good[0]
        synthesis.sources = [s for a in good for s in a.sources][:12]
        return synthesis

    async def _ask_member(self, model: BrainModel, request: BrainRequest) -> BrainAnswer:
        """One council member's answer, or a failed one when it takes too long."""
        try:
            return await asyncio.wait_for(self._ask(model, request), COUNCIL_MEMBER_SECONDS)
        except TimeoutError:
            logger.info("Council member %s gave no answer in time", model.id)
            return BrainAnswer(
                f"{model.label or model.model} no respondió a tiempo.",
                model.id,
                model.model,
                ok=False,
            )

    async def _run(self, brain_task: BrainTask, prefer: str) -> None:
        spent_before = self.ledger.month_spend()["usd"]
        started = asyncio.get_running_loop().time()
        try:
            answer: BrainAnswer | None = None
            if brain_task.where or (
                not brain_task.council
                and brain_task.kind != "x_search"
                and (brain_task.kind == "code" or brain_task.long_running or brain_task.agent)
            ):
                answer = await self._on_the_pc(brain_task, prefer)
            if answer is None:
                answer = await (
                    self._council(brain_task)
                    if brain_task.council
                    else self._staged(brain_task, prefer)
                )
                # A thought on the account was the PC's, as the work on it is.
                brain_task.ran_on = "pc" if answer.via_account else "here"
                brain_task.paid_by = (
                    "account" if answer.via_account else ("api" if brain_task.cost_usd else "")
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
        if brain_task.status == "failed":
            self.note_failure(brain_task.title, brain_task.result)
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
        await self.publish_result(brain_task)
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

    @staticmethod
    def _where_it_ran(brain_task: BrainTask) -> str:
        """One line on where the task ran and who paid, for the voice to say truthfully."""
        place = {
            "pc": "on the user's PC",
            "cloud": "in the cloud (Jarvis's work thread on the server)",
            "here": "inside this call (not on the PC's work thread, nor in the cloud)",
        }.get(brain_task.ran_on, "")
        if not place:
            return ""
        paid = {
            "account": ", on their Claude account (their plan, nothing per token)",
            "api": f", on the API key ({brain_task.cost_usd:.2f} $)",
        }.get(brain_task.paid_by, "")
        asked = (
            f" The user asked for {'the PC' if brain_task.where == 'pc' else 'the cloud'}."
            if brain_task.where
            else ""
        )
        return f"Where it ran: {place}{paid}.{asked}"

    def _spoken(self, brain_task: BrainTask) -> str:
        who = " y ".join(brain_task.models) or "el cerebro"
        result = brain_task.result
        if len(result) > SPOKEN_RESULT_CHARS:
            result = result[:SPOKEN_RESULT_CHARS] + " [...]"
        where = self._where_it_ran(brain_task)
        where_note = (
            f"\n{where} Say where it ran only when the user asked where, or how it was paid."
            if where
            else ""
        )
        until = out_until(self.claude_plan)
        if brain_task.paid_by == "api" and until is not None and not self._told_plan_out:
            # Once per call: the plan running out changes what the user pays.
            self._told_plan_out = True
            where_note += (
                f"\nAlso tell the user, once and briefly: {out_words(until, self._zone())} "
                "Until then the API key pays."
            )
        if brain_task.status != "done":
            return (
                f"A delegated task failed. Task: {brain_task.title}. What happened: {result}\n"
                "Tell the user briefly what it says, without guessing a cause it does not "
                "give, and offer to retry or try another model."
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
            f"A delegated task just finished (task id {brain_task.id}), answered by {who}."
            f"{council}{where_note}\n"
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
                "the page is open in a clean Chrome window on their computer, and that once "
                "they are in they close that window and you carry on by yourself. You never "
                "see what they type."
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
            return
        await self._journal(brain_task)

    async def _journal(self, brain_task: BrainTask) -> None:
        """One line in long-term memory for the work just told: what, when, its result.

        The result reaches the user in the voice's words, which memory does not
        extract from, so a finished job left nothing to recall the next day.
        A result kept for the greeting is journaled when the greeting tells it.
        """
        memory = getattr(self._voice(), "_memory", None)
        write = getattr(memory, "add_work_done", None)
        if not callable(write) or brain_task.status != "done":
            return
        try:
            await write(
                brain_task.title,
                brain_task.result,
                brain_task.finished_at,
                language=self.language,
            )
        except Exception:
            logger.debug("Could not journal %s", brain_task.id, exc_info=True)

    def _remember(self, brain_task: BrainTask) -> None:
        """Keep the result in the voice's history, not only in the reply's instructions.

        Instructions last one reply. Without this the next turn saw an answer with no
        source, and on a real call the voice took it for a guess of its own and apologised
        for it, right answers included. It also lets the voice answer a follow-up. Only
        `_chat_ctx`, which is what generation reads: a realtime model keeps its own.
        """
        try:
            history = getattr(self.session.current_agent, "_chat_ctx", None)
            if history is None or brain_task.status not in ("done", "failed"):
                return
            result = brain_task.result[:SPOKEN_RESULT_CHARS]
            # A failure too: asked "why?" a turn later, the voice had lost the reason and
            # said the call "returned nothing" (2026-11-16).
            ended = "finished" if brain_task.status == "done" else "failed"
            where = self._where_it_ran(brain_task)
            history.add_message(
                role="system",
                content=(
                    f"Delegated task {brain_task.id} ({brain_task.title}) {ended}. "
                    f"{where + ' ' if where else ''}What it returned, which you tell the "
                    f"user next:\n{result}"
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
