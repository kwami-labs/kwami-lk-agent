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

from ..adapters.supabase import as_user_id, direct_supabase
from ..factories.availability import voice_provider_keys
from ..settings import Settings, get_settings
from ..utils.logging import get_logger
from .config import BrainConfig, BrainModel, default_brain, executor_in_cloud
from .executor import CloudWorkspace, run_executor
from .ledger import Ledger, PendingResult, create_ledger
from .planner import PlanStep, final_step, plan_task, waves
from .providers import BrainAnswer, BrainRequest, run_anthropic, run_gemini, run_openai
from .router import (
    PROVIDER_KEYS,
    can_delegate,
    council_members,
    find_model,
    pick,
    provider_available,
    should_convene,
    tavily_writer,
    usable,
)

logger = get_logger("brain.bridge")

#: How long a model that answered "out of quota" is left alone. A daily quota
#: does not come back sooner; a per-minute one will have by then.
QUOTA_REST_SECONDS = 15 * 60

#: How long a finished result waits for the user to stop talking.
PAUSE_WAIT_SECONDS = 120.0
#: Results longer than this are summarised in speech; the full text stays in
#: `check_tasks` and in the app.
SPOKEN_RESULT_CHARS = 3500
MAX_TASKS_KEPT = 30
#: How long the greeting waits for the ledger to load from Supabase.
LEDGER_READY_SECONDS = 1.5
#: How long the end of a call waits for ledger writes to land.
LEDGER_FLUSH_SECONDS = 4.0

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
    delivered: bool = False
    #: When the planner split the task: what each part was and who did it.
    steps: list[dict[str, Any]] = field(default_factory=list)
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
            "steps": self.steps,
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
        self._tavily_key = tavily_key
        self.ledger = ledger if ledger is not None else create_ledger(settings)
        #: The user's Supabase when the agent writes it directly. Where the
        #: cloud executor uploads its files.
        self._store = direct_supabase(settings)
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
        stages = voice_provider_keys(self._keys)
        return {**stages["realtime"], **stages}

    def status(self) -> dict[str, Any]:
        spend = self.ledger.month_spend()
        return {
            "type": "brain_status",
            "enabled": self.config.enabled,
            "providers": self.providers(),
            "voiceProviders": self.voice_providers(),
            "tavily": bool(self._tavily_key),
            "usableModels": [m.id for m in usable(self.config, self._keys)],
            "spend": {**spend, "budgetUsd": self.config.monthly_budget_usd},
            "tasks": [t.public() for t in list(self.tasks.values())[-10:]],
        }

    async def publish_status(self) -> None:
        if self.publisher is None:
            return
        try:
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
    ) -> BrainTask | str:
        """Queue a task. Returns it, or a sentence saying why it cannot run."""
        if not self.config.enabled:
            return "The brain is switched off in the Models panel."
        if not usable(self.config, self._keys):
            return "No brain model is usable: none is enabled with its API key present."

        spend = self.ledger.month_spend()["usd"]
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
            return "No usable model fits this task."

        brain_task = BrainTask(
            id=f"t{next(self._ids)}",
            title=(title or task)[:120],
            task=task if not context else f"{task}\n\nContext from the conversation:\n{context}",
            kind=kind,
            difficulty=difficulty,
            council=convene,
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
        self._schedule(self.publish_status())
        return brain_task

    def cancel(self, task_id: str) -> bool:
        brain_task = self.tasks.get(task_id)
        if brain_task is None or brain_task.status != "running" or brain_task.handle is None:
            return False
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

    async def _ask(self, model: BrainModel, request: BrainRequest) -> BrainAnswer:
        if model.provider == "anthropic":
            answer = await run_anthropic(
                self._keys["ANTHROPIC_API_KEY"], model.id, model.model, request
            )
        elif model.provider == "google":
            answer = await run_gemini(self._keys["GOOGLE_API_KEY"], model.id, model.model, request)
        elif model.provider == "openai":
            answer = await run_openai(self._keys["OPENAI_API_KEY"], model.id, model.model, request)
        else:
            result = await run_executor(
                self._keys["ANTHROPIC_API_KEY"],
                model.model,
                request.task,
                self.config.executor,
                cloud=self._cloud_workspace(),
            )
            answer = BrainAnswer(result.text, model.id, model.model, result.cost_usd, ok=result.ok)
        self.ledger.add_spend(model.id, answer.cost_usd)
        return answer

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

    async def _single(self, brain_task: BrainTask, prefer: str) -> BrainAnswer:
        request = self._request(brain_task.task, brain_task.difficulty)
        answer, _ = await self._ask_role(
            brain_task, brain_task.kind, brain_task.difficulty, request, prefer
        )
        return answer

    async def _staged(self, brain_task: BrainTask, prefer: str) -> BrainAnswer:
        """Plan, run each part on its own model, combine (planner.py)."""
        if (
            prefer
            or not self.config.split
            or brain_task.kind == "code"
            or brain_task.difficulty == "simple"
        ):
            return await self._single(brain_task, prefer)
        planner = pick(self.config, self._keys, "analysis", "normal")
        if planner is None or planner.provider != "anthropic":
            return await self._single(brain_task, prefer)

        plan = await plan_task(self._keys["ANTHROPIC_API_KEY"], planner.model, brain_task.task)
        if plan is not None:
            brain_task.cost_usd += plan.cost_usd
            self.ledger.add_spend(planner.id, plan.cost_usd)
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
        try:
            answer = await (
                self._council(brain_task)
                if brain_task.council
                else self._staged(brain_task, prefer)
            )
            brain_task.result = answer.text
            brain_task.sources = answer.sources
            brain_task.status = "done" if answer.ok else "failed"
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
        logger.info(
            "Brain task %s %s on %s (%.4f $)",
            brain_task.id,
            brain_task.status,
            ", ".join(brain_task.models),
            brain_task.cost_usd,
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
            quiet = session.user_state != "speaking" and session.agent_state in (
                "listening",
                "idle",
            )
            now = loop.time()
            if quiet:
                quiet_since = quiet_since or now
                if now - quiet_since >= 1.0:
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
        try:
            self.session.generate_reply(
                instructions=self._spoken(brain_task), allow_interruptions=True
            )
            brain_task.delivered = True
        except Exception:
            logger.exception("Could not speak the result of %s", brain_task.id)
            self._store_pending(brain_task)

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

    def find(self, wanted: str) -> BrainModel | None:
        return find_model(self.config.models, wanted)
