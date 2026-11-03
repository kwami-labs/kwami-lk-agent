"""The mind: several jobs per model, a brief for each, and Claude on the user's account."""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import respx
from cryptography.fernet import Fernet

from src.adapters.supabase import SupabaseClient
from src.brain import account as account_module
from src.brain import bridge as bridge_module
from src.brain import providers as providers_module
from src.brain.account import account_login, read_login, sign_in
from src.brain.bridge import BrainTask, Bridge
from src.brain.config import BrainConfig, BrainModel, default_brain, parse_brain
from src.brain.executor import ExecutorResult
from src.brain.jobs import BrainJob, JobStore
from src.brain.ledger import Ledger
from src.brain.planner import plan_from_text
from src.brain.providers import BrainAnswer, BrainRequest, run_compatible, run_xai
from src.brain.router import (
    ACCOUNT_MARK,
    director_of,
    model_available,
    pick,
    provider_available,
    role_for,
    should_convene,
    usable,
)
from src.brain.user_keys import load_user_keys
from src.brain.worker import THINK_SLOTS, Worker
from src.domain import KwamiConfig
from src.domain.prompt import BRIDGE_GUIDANCE, RELAY_GUIDANCE, build_system_prompt
from src.settings import Settings

USER = "11111111-1111-1111-1111-111111111111"
KWAMI = "22222222-2222-2222-2222-222222222222"
OPUS = BrainModel("opus", "anthropic", "claude-opus-5-5", "expert", "Opus")


# -- config ----------------------------------------------------------------------


def test_a_model_holds_several_jobs_and_a_brief() -> None:
    config = parse_brain(
        {
            "director": "grok",
            "relay": "work_only",
            "instructions": "  Sé breve.  ",
            "models": [
                {
                    "id": "grok",
                    "provider": "xai",
                    "model": "grok-4.7",
                    "role": "search",
                    "roles": ["search", "x_search", "expert", "code", 7],
                    "access": "account",
                    "instructions": " Mira X primero. ",
                }
            ],
        }
    )
    grok = config.models[0]
    assert grok.roles == ["search", "x_search", "expert"]  # no provider-less code, no junk
    assert grok.role == "search"
    assert grok.has_role("x_search") and not grok.has_role("fast")
    assert grok.instructions == "Mira X primero."
    assert (config.director, config.relay, config.instructions) == (
        "grok",
        "work_only",
        "Sé breve.",
    )


def test_a_saved_model_with_only_the_old_role_still_loads() -> None:
    model = parse_brain(
        {"models": [{"provider": "openai", "model": "gpt-5.2", "role": "standard"}]}
    ).models[0]
    assert (model.roles, model.access, model.instructions) == (["standard"], "auto", "")


def test_a_job_the_provider_cannot_do_is_replaced_by_one_it_can() -> None:
    model = parse_brain(
        {"models": [{"provider": "groq", "model": "llama", "role": "search"}]}
    ).models[0]
    assert model.roles == ["fast"]


def test_a_model_with_neither_role_nor_roles_is_dropped() -> None:
    assert parse_brain({"models": [{"provider": "groq", "model": "llama"}]}).models == []


def test_a_brief_is_capped() -> None:
    config = parse_brain(
        {
            "instructions": "x" * 5000,
            "models": [
                {"provider": "openai", "model": "m", "role": "fast", "instructions": "y" * 5000}
            ],
        }
    )
    assert len(config.instructions) == len(config.models[0].instructions) == 2000


# -- router ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("access", "keys", "expected"),
    [
        ("auto", {ACCOUNT_MARK: "1"}, True),
        ("auto", {"ANTHROPIC_API_KEY": "k"}, True),
        ("auto", {}, False),
        ("account", {"ANTHROPIC_API_KEY": "k"}, False),
        ("account", {ACCOUNT_MARK: "1"}, True),
        ("api", {ACCOUNT_MARK: "1"}, False),
        ("api", {"ANTHROPIC_API_KEY": "k"}, True),
    ],
)
def test_how_a_claude_model_may_be_reached(access: str, keys: dict, expected: bool) -> None:
    model = BrainModel("c", "anthropic", "m", "fast", access=access)  # type: ignore[arg-type]
    assert model_available(model, keys) is expected


def test_the_account_counts_for_claude_only() -> None:
    assert provider_available("anthropic", {ACCOUNT_MARK: "1"})
    assert not provider_available("openai", {ACCOUNT_MARK: "1"})
    assert model_available(BrainModel("g", "google", "m", "fast"), {"GOOGLE_API_KEY": "k"})


def test_only_a_model_with_the_job_reads_x() -> None:
    config = BrainConfig(
        models=[
            BrainModel("opus", "anthropic", "o", "expert"),
            BrainModel("grok", "xai", "g", "expert", roles=["expert", "x_search"]),
        ]
    )
    keys = {"ANTHROPIC_API_KEY": "a", "XAI_API_KEY": "x"}
    assert role_for("x_search", "hard") == "x_search"
    assert pick(config, keys, "x_search", "normal").id == "grok"  # type: ignore[union-attr]
    assert pick(config, {"ANTHROPIC_API_KEY": "a"}, "x_search", "normal") is None


def test_manual_routing_does_not_send_x_to_the_default_model() -> None:
    config = BrainConfig(
        routing="manual",
        default_model="opus",
        models=[
            BrainModel("opus", "anthropic", "o", "expert"),
            BrainModel("grok", "xai", "g", "x_search"),
        ],
    )
    keys = {"ANTHROPIC_API_KEY": "a", "XAI_API_KEY": "x"}
    assert pick(config, keys, "x_search", "normal").id == "grok"  # type: ignore[union-attr]
    assert pick(config, keys, "question", "normal").id == "opus"  # type: ignore[union-attr]


def test_research_avoids_models_with_no_web() -> None:
    config = BrainConfig(
        models=[
            BrainModel("groq", "groq", "l", "standard"),
            BrainModel("opus", "anthropic", "o", "expert"),
        ]
    )
    keys = {"GROQ_API_KEY": "g", "ANTHROPIC_API_KEY": "a"}
    assert pick(config, keys, "research", "normal").id == "opus"  # type: ignore[union-attr]
    # Rather than nothing, when it is all there is.
    assert pick(config, {"GROQ_API_KEY": "g"}, "research", "normal").id == "groq"  # type: ignore[union-attr]


def test_the_director_is_the_users_choice_when_it_can_run() -> None:
    config = default_brain()
    config.director = "claude-opus"
    keys = {"ANTHROPIC_API_KEY": "a"}
    assert director_of(config, keys).id == "claude-opus"  # type: ignore[union-attr]
    config.director = "claude-code"  # it writes code, it does not plan
    assert director_of(config, keys).id == "claude-sonnet"  # type: ignore[union-attr]
    config.director = "nobody"
    assert director_of(config, keys).id == "claude-sonnet"  # type: ignore[union-attr]
    assert director_of(config, {}) is None


def test_plan_from_text_reads_through_a_fence() -> None:
    text = (
        'Plan:\n```json\n{"split": true, "steps": ['
        '{"id": "a", "goal": "uno", "difficulty": "simple", "web": false, "after": []},'
        '{"id": "b", "goal": "dos", "difficulty": "hard", "web": true, "after": ["a"]}]}\n```'
    )
    assert [s.id for s in plan_from_text(text).steps] == ["a", "b"]
    assert plan_from_text("no hay plan").steps == []
    assert plan_from_text("{no es json}").steps == []


# -- the voice's prompt ------------------------------------------------------------


def test_the_voice_passes_everything_on_only_when_asked() -> None:
    soul = KwamiConfig().soul
    relaying = build_system_prompt(soul, relay_all=True)
    assert RELAY_GUIDANCE in relaying
    # Replaced, not added to: the other guidance tells it to answer a capital itself.
    assert BRIDGE_GUIDANCE not in relaying
    assert "only makes the user wait" not in relaying
    assert BRIDGE_GUIDANCE in build_system_prompt(soul)
    # Seen on a real call: with results delivered as one-off instructions, the voice took
    # its own right answers for guesses and apologised for them.
    assert "Never apologise for an earlier answer" in BRIDGE_GUIDANCE
    assert "Never apologise for an earlier answer" in RELAY_GUIDANCE
    # With no delegate_task there is nothing to pass it on with.
    withheld = build_system_prompt(soul, unavailable_tools={"delegate_task"}, relay_all=True)
    assert RELAY_GUIDANCE not in withheld


# -- the bridge --------------------------------------------------------------------


class FakeJobs:
    """Answers a thought the way the PC's worker would."""

    def __init__(self, status: str = "done", result: str = "Pensado.") -> None:
        self.status = status
        self.result = result
        self.enqueued: list[BrainJob] = []
        self.cancelled: list[str] = []
        self.delivered: list[str] = []
        self.online = True
        self.ready = True
        self.accept = True

    async def runner_online(self, user_id: str) -> bool:
        return self.online

    async def account_ready(self, user_id: str) -> bool:
        return self.ready

    async def enqueue(self, job: BrainJob) -> bool:
        job.id = job.id or "j1"
        self.enqueued.append(job)
        return self.accept

    async def get(self, job_id: str) -> BrainJob | None:
        job = self.enqueued[-1]
        job.status = self.status
        job.result = self.result
        return job

    async def mark_delivered(self, job_id: str) -> bool:
        self.delivered.append(job_id)
        return True

    async def request_cancel(self, job_id: str) -> bool:
        self.cancelled.append(job_id)
        return True


def _bridge(tmp_path: Path, keys: dict[str, str] | None = None, jobs: Any = None) -> Bridge:
    bridge = Bridge(
        keys=keys if keys is not None else {},
        ledger=Ledger(tmp_path),
        settings=Settings(executor_mode="local"),
    )
    bridge.jobs = jobs
    bridge.user_id, bridge.kwami_id = USER, KWAMI
    return bridge


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge_module, "THINK_POLL_SECONDS", 0.0)
    monkeypatch.setattr(bridge_module, "THINK_POLL_FAST_SECONDS", 0.0)


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> list[BrainRequest]:
    seen: list[BrainRequest] = []

    async def run(key: str, model_id: str, model: str, request: BrainRequest) -> BrainAnswer:
        seen.append(request)
        return BrainAnswer("Por la API.", model_id, model, cost_usd=0.5)

    monkeypatch.setattr(bridge_module, "run_anthropic", run)
    return seen


async def test_claude_thinks_on_the_account_and_costs_nothing(
    tmp_path: Path, api: list[BrainRequest]
) -> None:
    jobs = FakeJobs()
    bridge = _bridge(tmp_path, {"ANTHROPIC_API_KEY": "a"}, jobs)
    await bridge._refresh_account(force=True)

    answer = await bridge._ask(OPUS, BrainRequest("¿Y esto?", "Sistema", native_search=False))

    assert answer.text == "Pensado." and answer.via_account and answer.cost_usd == 0
    assert api == []  # the key was not touched
    job = jobs.enqueued[0]
    assert (job.mode, job.model, job.user_id) == ("think", "claude-opus-5-5", USER)
    assert "Sistema" in job.task and "¿Y esto?" in job.task
    assert jobs.delivered == [job.id]
    assert bridge.ledger.month_spend()["usd"] == 0


async def test_research_on_the_account_may_use_the_web(tmp_path: Path) -> None:
    jobs = FakeJobs()
    bridge = _bridge(tmp_path, {}, jobs)
    await bridge._refresh_account(force=True)
    await bridge._ask(OPUS, BrainRequest("Noticias", "S", native_search=True))
    assert jobs.enqueued[0].mode == "research"


async def test_on_auto_a_failed_account_falls_to_the_key(
    tmp_path: Path, api: list[BrainRequest]
) -> None:
    jobs = FakeJobs(status="failed", result="Límite de uso")
    bridge = _bridge(tmp_path, {"ANTHROPIC_API_KEY": "a"}, jobs)
    await bridge._refresh_account(force=True)
    answer = await bridge._ask(OPUS, BrainRequest("T", "S"))
    assert answer.text == "Por la API." and not answer.via_account
    assert bridge.ledger.month_spend()["usd"] == pytest.approx(0.5)


async def test_account_only_never_touches_the_key(tmp_path: Path, api: list[BrainRequest]) -> None:
    jobs = FakeJobs(status="failed", result="Límite de uso")
    bridge = _bridge(tmp_path, {"ANTHROPIC_API_KEY": "a"}, jobs)
    await bridge._refresh_account(force=True)
    model = BrainModel("o", "anthropic", "m", "expert", access="account")

    answer = await bridge._ask(model, BrainRequest("T", "S"))
    assert answer.text == "Límite de uso" and not answer.ok and api == []

    jobs.online = False
    await bridge._refresh_account(force=True)
    answer = await bridge._ask(model, BrainRequest("T", "S"))
    assert not answer.ok and "no está disponible" in answer.text and api == []


async def test_api_only_never_touches_the_account(tmp_path: Path, api: list[BrainRequest]) -> None:
    jobs = FakeJobs()
    bridge = _bridge(tmp_path, {"ANTHROPIC_API_KEY": "a"}, jobs)
    await bridge._refresh_account(force=True)
    model = BrainModel("o", "anthropic", "m", "expert", access="api")
    answer = await bridge._ask(model, BrainRequest("T", "S"))
    assert answer.text == "Por la API." and jobs.enqueued == []


async def test_a_full_queue_gives_up_and_withdraws_the_thought(
    tmp_path: Path, api: list[BrainRequest], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bridge_module, "THINK_QUEUE_PATIENCE_SECONDS", 0.0)
    jobs = FakeJobs(status="queued")
    bridge = _bridge(tmp_path, {"ANTHROPIC_API_KEY": "a"}, jobs)
    await bridge._refresh_account(force=True)
    answer = await bridge._ask(OPUS, BrainRequest("T", "S"))
    assert answer.text == "Por la API." and jobs.cancelled == [jobs.enqueued[0].id]


async def test_a_queue_that_will_not_take_it_goes_to_the_key(
    tmp_path: Path, api: list[BrainRequest]
) -> None:
    jobs = FakeJobs()
    jobs.accept = False  # no migration 024 yet
    bridge = _bridge(tmp_path, {"ANTHROPIC_API_KEY": "a"}, jobs)
    await bridge._refresh_account(force=True)
    assert (await bridge._ask(OPUS, BrainRequest("T", "S"))).text == "Por la API."


async def test_a_pc_that_goes_away_mid_thought_is_noticed(
    tmp_path: Path, api: list[BrainRequest], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bridge_module, "THINK_PC_CHECK_SECONDS", 0.0)
    jobs = FakeJobs(status="running")
    bridge = _bridge(tmp_path, {"ANTHROPIC_API_KEY": "a"}, jobs)
    await bridge._refresh_account(force=True)
    jobs.online = False
    answer = await asyncio.wait_for(bridge._ask(OPUS, BrainRequest("T", "S")), 2)
    assert answer.text == "Por la API." and jobs.cancelled == [jobs.enqueued[0].id]


async def test_a_cancelled_task_withdraws_its_thought(tmp_path: Path) -> None:
    jobs = FakeJobs(status="running")
    bridge = _bridge(tmp_path, {}, jobs)
    await bridge._refresh_account(force=True)
    running = asyncio.create_task(bridge._ask(OPUS, BrainRequest("T", "S")))
    await asyncio.sleep(0.01)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    await asyncio.sleep(0)
    assert jobs.cancelled


async def test_without_a_login_on_the_pc_there_is_no_account(tmp_path: Path) -> None:
    jobs = FakeJobs()
    jobs.ready = False
    bridge = _bridge(tmp_path, {}, jobs)
    await bridge._refresh_account(force=True)
    assert bridge.pc_online and not bridge.status()["account"]["loggedIn"]
    assert usable(default_brain(), bridge._keys) == []


async def test_a_phone_caller_never_reaches_the_account(tmp_path: Path) -> None:
    jobs = FakeJobs()
    bridge = _bridge(tmp_path, {}, jobs)
    bridge.user_id = ""
    await bridge._refresh_account(force=True)
    assert ACCOUNT_MARK not in bridge._keys


async def test_the_account_check_is_not_repeated_within_its_window(tmp_path: Path) -> None:
    jobs = FakeJobs()
    bridge = _bridge(tmp_path, {}, jobs)
    await bridge._refresh_account()
    jobs.ready = False
    await bridge._refresh_account()
    assert ACCOUNT_MARK in bridge._keys
    await bridge._refresh_account(force=True)
    assert ACCOUNT_MARK not in bridge._keys


async def test_every_worker_gets_the_users_briefs(tmp_path: Path, api: list[BrainRequest]) -> None:
    bridge = _bridge(tmp_path, {"ANTHROPIC_API_KEY": "a"})
    bridge.config.instructions = "Siempre en euros."
    model = BrainModel("o", "anthropic", "m", "expert", instructions="Sé escéptico.")
    await bridge._ask(model, BrainRequest("T", "S"))
    assert "Siempre en euros." in api[0].system and "Sé escéptico." in api[0].system
    assert api[0].system.startswith("S")


async def test_grok_and_the_chat_providers_are_reached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    async def xai(key: str, model_id: str, model: str, request: BrainRequest) -> BrainAnswer:
        calls.append(f"xai:{key}")
        return BrainAnswer("x", model_id, model)

    async def chat(
        provider: str, key: str, model_id: str, model: str, request: BrainRequest
    ) -> BrainAnswer:
        calls.append(f"{provider}:{key}")
        return BrainAnswer("y", model_id, model)

    monkeypatch.setattr(bridge_module, "run_xai", xai)
    monkeypatch.setattr(bridge_module, "run_compatible", chat)
    bridge = _bridge(tmp_path, {"XAI_API_KEY": "kx", "GROQ_API_KEY": "kg"})
    await bridge._ask(BrainModel("g", "xai", "m", "x_search"), BrainRequest("T", "S"))
    await bridge._ask(BrainModel("l", "groq", "m", "fast"), BrainRequest("T", "S"))
    assert calls == ["xai:kx", "groq:kg"]


async def test_asking_x_with_no_model_for_it_says_what_is_missing(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path, {"ANTHROPIC_API_KEY": "a"})
    started = bridge.start(task="q", title="t", kind="x_search", difficulty="normal")
    assert isinstance(started, str) and "Grok" in started


async def test_a_task_on_x_goes_to_grok_whole_and_with_its_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[BrainRequest] = []

    async def xai(key: str, model_id: str, model: str, request: BrainRequest) -> BrainAnswer:
        seen.append(request)
        return BrainAnswer("En X dicen", model_id, model)

    async def no_plan(*args: Any) -> Any:
        raise AssertionError("a task on X is not split")

    monkeypatch.setattr(bridge_module, "run_xai", xai)
    monkeypatch.setattr(bridge_module, "plan_task", no_plan)
    bridge = Bridge(
        keys={"ANTHROPIC_API_KEY": "a", "XAI_API_KEY": "x"},
        tavily_key="t",
        ledger=Ledger(tmp_path),
        settings=Settings(executor_mode="local"),
    )
    bridge.config = default_brain()
    bridge.config.search = "tavily"  # it must not take the search away from Grok
    bridge.config.models.append(BrainModel("grok", "xai", "grok-4.7", "x_search"))

    task = bridge.start(
        task="¿Qué se dice?",
        title="X",
        kind="x_search",
        difficulty="hard",
        important=True,
        long_running=True,
        model="opus",
    )
    assert not isinstance(task, str) and task.council is False
    assert task.handle is not None
    await task.handle

    assert task.result == "En X dicen"
    assert seen[0].native_search is True and seen[0].tavily_key == ""


def test_only_a_model_that_reads_x_may_be_named_for_it() -> None:
    config = BrainConfig(
        models=[
            BrainModel("opus", "anthropic", "o", "expert"),
            BrainModel("grok", "xai", "g", "x_search"),
        ]
    )
    keys = {"ANTHROPIC_API_KEY": "a", "XAI_API_KEY": "x"}
    assert pick(config, keys, "x_search", "normal", prefer="opus").id == "grok"  # type: ignore[union-attr]
    assert should_convene(config, "x_search", True, True) is False


def test_a_payload_that_is_not_text_is_dropped_not_raised() -> None:
    config = parse_brain(
        {
            "models": [
                {"provider": ["x"], "model": "m", "role": "fast"},
                {"provider": "openai", "model": "m", "role": ["fast"]},
                {"provider": "openai", "model": "n", "role": {}, "roles": [["fast"], "expert"]},
            ]
        }
    )
    assert [m.model for m in config.models] == ["n"]
    assert config.models[0].roles == ["expert"]


async def test_a_director_that_is_not_claude_plans_in_plain_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def gemini(key: str, model_id: str, model: str, request: BrainRequest) -> BrainAnswer:
        assert "JSON" in request.system and request.native_search is False
        text = '{"split": false, "steps": [{"id": "a", "goal": "g", "difficulty": "hard", "web": false, "after": []}]}'
        return BrainAnswer(text, model_id, model, cost_usd=0.01)

    monkeypatch.setattr(bridge_module, "run_gemini", gemini)
    bridge = _bridge(tmp_path, {"GOOGLE_API_KEY": "g"})
    plan = await bridge._plan(BrainModel("f", "google", "m", "fast"), "tarea")
    assert plan is not None and plan.steps[0].difficulty == "hard" and plan.cost_usd == 0.01


async def test_a_director_that_failed_gives_no_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def gemini(key: str, model_id: str, model: str, request: BrainRequest) -> BrainAnswer:
        return BrainAnswer("mal", model_id, model, ok=False)

    monkeypatch.setattr(bridge_module, "run_gemini", gemini)
    bridge = _bridge(tmp_path, {"GOOGLE_API_KEY": "g"})
    assert await bridge._plan(BrainModel("f", "google", "m", "fast"), "tarea") is None


async def test_a_claude_director_on_the_key_uses_structured_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def plan(key: str, model: str, task: str) -> Any:
        return SimpleNamespace(cost_usd=0.2, steps=[])

    monkeypatch.setattr(bridge_module, "plan_task", plan)
    bridge = _bridge(tmp_path, {"ANTHROPIC_API_KEY": "a"})
    await bridge._plan(OPUS, "tarea")
    assert bridge.ledger.month_spend()["usd"] == pytest.approx(0.2)


async def test_a_users_key_reaches_the_brain_not_the_voice(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path, {})
    bridge._keys["ELEVEN_API_KEY"] = "from-a-user"
    assert bridge.voice_providers()["tts"]["elevenlabs"] is False


# -- the worker's thinking line ------------------------------------------------------


class WorkerJobs:
    def __init__(self) -> None:
        self.queue: list[BrainJob] = []
        self.finished: list[tuple[str, str, str, float]] = []
        self.claims: list[bool] = []
        self.cancel = False
        self.thinking_ready = True

    async def claim(self, user: str, runner: str, thinking: bool = False) -> BrainJob | None:
        self.claims.append(thinking)
        return self.queue.pop(0) if self.queue else None

    async def finish(self, job_id: str, status: str, text: str, cost: float, session: str) -> bool:
        self.finished.append((job_id, status, text, cost))
        return True

    async def update_running(self, job_id: str, progress: str) -> bool:
        return True

    async def get(self, job_id: str) -> Any:
        return SimpleNamespace(cancel_requested=self.cancel)


def _thinking_job(mode: str = "think", waited_for: bool = True) -> BrainJob:
    seen = datetime.now(UTC) - timedelta(seconds=0 if waited_for else 600)
    return BrainJob(
        user_id=USER,
        kwami_id=KWAMI,
        title="t",
        task="Piensa",
        id="j",
        mode=mode,
        watcher_seen_at=seen.isoformat(),
    )


def _worker(jobs: WorkerJobs, run: Any, tmp_path: Path) -> Worker:
    return Worker(jobs, None, "api-key", USER, run=run, folder=tmp_path)  # type: ignore[arg-type]


@pytest.mark.parametrize(("mode", "permissions"), [("think", "think"), ("research", "think_web")])
async def test_a_thought_runs_on_the_login_with_no_hands(
    tmp_path: Path, mode: str, permissions: str
) -> None:
    seen: dict[str, Any] = {}

    async def run(api_key: str, model: str, task: str, config: Any, **kwargs: Any) -> Any:
        seen.update(api_key=api_key, config=config)
        return ExecutorResult("Hecho", 9.0, True)

    jobs = WorkerJobs()
    await _worker(jobs, run, tmp_path).think(_thinking_job(mode))
    assert seen["api_key"] == ""  # the PC's login, whatever key the worker holds
    assert seen["config"].permissions == permissions
    assert jobs.finished == [("j", "done", "Hecho", 0.0)]  # the notional cost is not spend


async def test_a_thought_nobody_waits_for_is_not_thought(tmp_path: Path) -> None:
    async def run(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("it must not run")

    jobs = WorkerJobs()
    await _worker(jobs, run, tmp_path).think(_thinking_job(waited_for=False))
    assert jobs.finished == [("j", "cancelled", "Caducada.", 0.0)]


async def test_a_thought_that_crashes_is_recorded_failed(tmp_path: Path) -> None:
    async def run(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom")

    jobs = WorkerJobs()
    await _worker(jobs, run, tmp_path).think(_thinking_job())
    assert jobs.finished[0][1] == "failed" and "boom" in jobs.finished[0][2]


async def test_a_thought_can_be_cancelled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.brain.worker.RUNNING_POLL_SECONDS", 0.0)

    async def run(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(30)

    jobs = WorkerJobs()
    jobs.cancel = True
    await _worker(jobs, run, tmp_path).think(_thinking_job())
    assert jobs.finished == [("j", "cancelled", "Cancelada.", 0.0)]


async def test_the_thinking_line_takes_only_thoughts_and_only_so_many(tmp_path: Path) -> None:
    release = asyncio.Event()

    async def run(*args: Any, **kwargs: Any) -> Any:
        await release.wait()
        return ExecutorResult("Hecho", 0.0, True)

    jobs = WorkerJobs()
    jobs.queue = [_thinking_job() for _ in range(THINK_SLOTS + 1)]
    worker = _worker(jobs, run, tmp_path)
    for _ in range(THINK_SLOTS):
        assert await worker.think_tick() is True
    assert await worker.think_tick() is False  # full: nothing more is claimed
    assert jobs.claims == [True] * THINK_SLOTS
    release.set()
    await asyncio.gather(*worker._thinking)
    assert len(jobs.finished) == THINK_SLOTS


async def test_before_the_migration_the_thinking_line_claims_nothing(tmp_path: Path) -> None:
    jobs = WorkerJobs()
    jobs.thinking_ready = False
    assert await _worker(jobs, None, tmp_path).think_tick() is False
    assert jobs.claims == []


async def test_the_login_is_checked_now_and_then(tmp_path: Path) -> None:
    calls = 0

    async def check() -> bool:
        nonlocal calls
        calls += 1
        return calls == 1

    worker = Worker(WorkerJobs(), None, "", USER, folder=tmp_path, check_login=check)  # type: ignore[arg-type]
    await worker._note_login()
    await worker._note_login()
    assert calls == 1 and worker.claude_login is True


# -- the job store -----------------------------------------------------------------


@pytest.fixture
def store() -> JobStore:
    return JobStore(SupabaseClient("https://db.test", "secret"))


@respx.mock
async def test_only_a_thought_names_its_mode(store: JobStore) -> None:
    route = respx.post("https://db.test/rest/v1/brain_jobs").mock(return_value=httpx.Response(201))
    await store.enqueue(BrainJob(USER, KWAMI, "t", "x"))
    await store.enqueue(BrainJob(USER, KWAMI, "t", "x", mode="think"))
    bodies = [call.request.content for call in route.calls]
    assert b'"mode"' not in bodies[0] and b'"mode": "think"' in bodies[1].replace(b'":"', b'": "')


@respx.mock
async def test_a_heartbeat_survives_a_database_without_the_login_column(store: JobStore) -> None:
    route = respx.post("https://db.test/rest/v1/brain_runners").mock(
        side_effect=[httpx.Response(400), httpx.Response(201)]
    )
    assert await store.beat(USER, "r", "host", True) is True
    assert store.thinking_ready is False
    assert b"claude_login" in route.calls[0].request.content
    assert b"claude_login" not in route.calls[1].request.content


@respx.mock
async def test_the_thinking_claim_asks_for_thoughts(store: JobStore) -> None:
    route = respx.post("https://db.test/rest/v1/rpc/brain_claim_job").mock(
        return_value=httpx.Response(200, json=[{"id": "j", "user_id": USER, "mode": "think"}])
    )
    work = await store.claim(USER, "r")
    await store.claim(USER, "r", thinking=True)
    assert work is not None and work.mode == "think"
    assert b"p_thinking" not in route.calls[0].request.content
    assert b"p_thinking" in route.calls[1].request.content


@respx.mock
async def test_carrying_on_means_the_last_work_not_the_last_thought(store: JobStore) -> None:
    route = respx.get("https://db.test/rest/v1/brain_jobs").mock(
        return_value=httpx.Response(200, json=[{"thread_id": "t9"}])
    )
    assert await store.last_thread(KWAMI) == "t9"
    assert route.calls[0].request.url.params["mode"] == "eq.work"


@respx.mock
async def test_account_ready_needs_a_fresh_login(store: JobStore) -> None:
    route = respx.get("https://db.test/rest/v1/brain_runners").mock(
        side_effect=[httpx.Response(200, json=[{"runner_id": "r"}]), httpx.Response(400)]
    )
    assert await store.account_ready(USER) is True
    assert route.calls[0].request.url.params["claude_login"] == "is.true"
    assert await store.account_ready(USER) is False  # no such column yet


# -- the login ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"loggedIn": true, "authMethod": "claude.ai", "email": "x"}', True),
        ('{"loggedIn": true, "authMethod": "apiKey"}', False),
        ('{"loggedIn": false}', False),
        ("[]", False),
        ("no es json", False),
    ],
)
def test_only_a_claude_account_counts(raw: str, expected: bool) -> None:
    assert read_login(raw) is expected


class FakeProcess:
    def __init__(self, stdout: bytes, hang: bool = False) -> None:
        self.stdout = stdout
        self.hang = hang
        self.killed = False
        self.returncode: int | None = None

    async def communicate(self) -> tuple[bytes, bytes]:
        if self.hang:
            await asyncio.sleep(30)
        self.returncode = 0
        return self.stdout, b""

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        return 0


async def test_account_login_asks_the_binary_without_the_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    seen: dict[str, Any] = {}

    async def spawn(*args: Any, **kwargs: Any) -> FakeProcess:
        seen.update(args=args, env=kwargs["env"])
        return FakeProcess(b'{"loggedIn": true, "authMethod": "claude.ai"}')

    monkeypatch.setattr(account_module, "_binary", lambda: "claude")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    assert await account_login() is True
    assert seen["args"] == ("claude", "auth", "status")
    assert "ANTHROPIC_API_KEY" not in seen["env"]


async def test_signing_in_is_claude_codes_own(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[Any, ...]] = []

    async def spawn(*args: Any, **kwargs: Any) -> FakeProcess:
        seen.append(args)
        return FakeProcess(b"")

    monkeypatch.setattr(account_module, "_binary", lambda: "claude")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    assert await sign_in() == 0
    assert seen == [("claude", "auth", "login")]
    monkeypatch.setattr(account_module, "_binary", lambda: None)
    assert await sign_in() == 2


async def test_account_login_with_no_binary_is_no(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(account_module, "_binary", lambda: None)
    assert await account_login() is False


async def test_account_login_that_cannot_start_is_no(monkeypatch: pytest.MonkeyPatch) -> None:
    async def spawn(*args: Any, **kwargs: Any) -> Any:
        raise OSError("nope")

    monkeypatch.setattr(account_module, "_binary", lambda: "claude")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    assert await account_login() is False


async def test_account_login_that_hangs_is_no(monkeypatch: pytest.MonkeyPatch) -> None:
    process = FakeProcess(b"", hang=True)

    async def spawn(*args: Any, **kwargs: Any) -> FakeProcess:
        return process

    monkeypatch.setattr(account_module, "_binary", lambda: "claude")
    monkeypatch.setattr(account_module, "STATUS_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    assert await account_login() is False and process.killed


def test_the_binary_is_the_one_the_sdk_ships(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import claude_agent_sdk

    bundled = tmp_path / "_bundled"
    bundled.mkdir()
    monkeypatch.setattr(claude_agent_sdk, "__file__", str(tmp_path / "__init__.py"))
    monkeypatch.setattr(account_module.shutil, "which", lambda name: "/on/path/claude")
    assert account_module._binary() == "/on/path/claude"
    (bundled / "claude").write_text("")
    assert account_module._binary() == str(bundled / "claude")


# -- users' keys -------------------------------------------------------------------


class KeyStore:
    def __init__(self, rows: list[dict[str, str]] | None) -> None:
        self.rows = rows

    async def select(self, table: str, params: dict[str, str], *, what: str = "") -> Any:
        assert params["user_id"] == f"eq.{USER}"
        return self.rows


async def test_a_users_keys_are_opened_with_the_secret() -> None:
    secret = Fernet.generate_key().decode()
    box = Fernet(secret.encode())
    rows = [
        {"provider": "xai", "encrypted_key": box.encrypt(b"xai-key").decode()},
        {"provider": "unknown", "encrypted_key": "whatever"},
        {"provider": "local", "encrypted_key": box.encrypt(b"no").decode()},
        {"provider": "openai", "encrypted_key": "not-a-token"},
    ]
    keys = await load_user_keys(KeyStore(rows), USER, secret)  # type: ignore[arg-type]
    assert keys == {"XAI_API_KEY": "xai-key"}


async def test_with_no_secret_or_a_bad_one_no_keys_are_read() -> None:
    assert await load_user_keys(KeyStore(None), USER, "") == {}  # type: ignore[arg-type]
    assert await load_user_keys(KeyStore(None), USER, "short") == {}  # type: ignore[arg-type]
    good = Fernet.generate_key().decode()
    assert await load_user_keys(KeyStore(None), USER, good) == {}  # type: ignore[arg-type]


async def test_knowing_the_user_brings_in_their_keys_and_their_pc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def load(store: Any, user_id: str, secret: str, *_: Any) -> dict[str, str]:
        return {"XAI_API_KEY": f"for-{user_id}"}

    monkeypatch.setattr(bridge_module, "load_user_keys", load)
    jobs = FakeJobs()
    bridge = _bridge(tmp_path, {}, jobs)
    bridge._keys_store = object()  # type: ignore[assignment]
    await bridge._on_user_known()
    assert bridge._keys["XAI_API_KEY"] == f"for-{USER}" and ACCOUNT_MARK in bridge._keys


# -- the new providers ---------------------------------------------------------------


class FakeOpenAI:
    def __init__(self, **kwargs: Any) -> None:
        FakeOpenAI.kwargs = kwargs
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.responses = SimpleNamespace(create=self._responses)

    kwargs: dict[str, Any] = {}
    sent: dict[str, Any] = {}
    error: Exception | None = None

    async def _chat(self, **kwargs: Any) -> Any:
        FakeOpenAI.sent = kwargs
        if FakeOpenAI.error:
            raise FakeOpenAI.error
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=1000),
            choices=[SimpleNamespace(message=SimpleNamespace(content=" Hola "))],
        )

    async def _responses(self, **kwargs: Any) -> Any:
        FakeOpenAI.sent = kwargs
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=1000, output_tokens=1000),
            output=[SimpleNamespace(type="x_search_call")],
            output_text="En X dicen",
        )


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> type[FakeOpenAI]:
    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeOpenAI)
    FakeOpenAI.error = None
    return FakeOpenAI


async def test_grok_is_called_at_xai_with_x_search(fake_openai: type[FakeOpenAI]) -> None:
    answer = await run_xai("k", "g", "grok-4.7", BrainRequest("T", "S"))
    assert fake_openai.kwargs["base_url"] == providers_module.XAI_URL
    assert {"type": "x_search"} in fake_openai.sent["tools"]
    assert answer.text == "En X dicen" and answer.searches == 1 and answer.cost_usd > 0


async def test_a_chat_provider_is_called_at_its_address(fake_openai: type[FakeOpenAI]) -> None:
    answer = await run_compatible("groq", "k", "l", "llama", BrainRequest("T", "S"))
    assert fake_openai.kwargs["base_url"] == providers_module.COMPATIBLE_URLS["groq"]
    assert fake_openai.sent["messages"][0] == {"role": "system", "content": "S"}
    assert answer.text == "Hola" and answer.cost_usd > 0 and answer.ok


@pytest.mark.parametrize(
    ("error", "expected", "limited"),
    [("AuthenticationError", "no es válida", False), ("RateLimitError", "limitando", True)],
)
async def test_a_chat_provider_that_refuses_says_so(
    fake_openai: type[FakeOpenAI], error: str, expected: str, limited: bool
) -> None:
    import openai

    response = httpx.Response(401, request=httpx.Request("POST", "https://x.test"))
    fake_openai.error = getattr(openai, error)("no", response=response, body=None)
    answer = await run_compatible("mistral", "k", "l", "m", BrainRequest("T", "S"))
    assert expected in answer.text and "Mistral" in answer.text
    assert not answer.ok and answer.rate_limited is limited


async def test_a_chat_provider_that_breaks_says_so(fake_openai: type[FakeOpenAI]) -> None:
    import openai

    fake_openai.error = openai.APIError("no", httpx.Request("POST", "https://x.test"), body=None)
    answer = await run_compatible("deepseek", "k", "l", "m", BrainRequest("T", "S"))
    assert "DeepSeek devolvió un error" in answer.text and not answer.ok


async def test_a_chat_provider_with_no_text_is_not_an_answer(
    fake_openai: type[FakeOpenAI], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def empty(self: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(usage=None, choices=[])

    monkeypatch.setattr(FakeOpenAI, "_chat", empty)
    answer = await run_compatible("openrouter", "k", "l", "m", BrainRequest("T", "S"))
    assert not answer.ok and answer.cost_usd == 0


# -- the corners ---------------------------------------------------------------------


def test_without_the_sdk_the_binary_is_the_one_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    monkeypatch.setattr(account_module.shutil, "which", lambda name: "/on/path/claude")
    assert account_module._binary() == "/on/path/claude"


async def test_a_kwami_that_is_no_uuid_cannot_queue_a_thought(
    tmp_path: Path, api: list[BrainRequest]
) -> None:
    jobs = FakeJobs()
    bridge = _bridge(tmp_path, {"ANTHROPIC_API_KEY": "a"}, jobs)
    await bridge._refresh_account(force=True)
    bridge.kwami_id = "not-a-uuid"
    assert (await bridge._ask(OPUS, BrainRequest("T", "S"))).text == "Por la API."
    assert jobs.enqueued == []


async def test_a_blip_reading_the_job_does_not_end_the_wait(tmp_path: Path) -> None:
    jobs = FakeJobs()
    real_get = jobs.get
    calls = 0

    async def get(job_id: str) -> BrainJob | None:
        nonlocal calls
        calls += 1
        return None if calls == 1 else await real_get(job_id)

    jobs.get = get  # type: ignore[method-assign]
    bridge = _bridge(tmp_path, {}, jobs)
    await bridge._refresh_account(force=True)
    assert (await bridge._ask(OPUS, BrainRequest("T", "S"))).text == "Pensado."
    assert calls == 2


async def test_a_thought_that_never_ends_is_given_up_on(
    tmp_path: Path, api: list[BrainRequest], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bridge_module, "THINK_TIMEOUT_SECONDS", 0.0)
    jobs = FakeJobs(status="running")
    bridge = _bridge(tmp_path, {"ANTHROPIC_API_KEY": "a"}, jobs)
    await bridge._refresh_account(force=True)
    assert (await bridge._ask(OPUS, BrainRequest("T", "S"))).text == "Por la API."
    assert jobs.cancelled == [jobs.enqueued[0].id]


async def test_the_worker_forgets_a_thought_once_it_ends_and_keeps_going_after_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("src.brain.worker.THINK_POLL_SECONDS", 0.0)
    monkeypatch.setattr("src.brain.worker.THINK_POLL_FAST_SECONDS", 0.0)

    async def run(*args: Any, **kwargs: Any) -> Any:
        return ExecutorResult("Hecho", 0.0, True)

    jobs = WorkerJobs()
    worker = _worker(jobs, run, tmp_path)
    stop = asyncio.Event()
    calls = 0
    real = worker.think_tick

    async def tick() -> bool:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("down")
        if calls == 2:
            jobs.queue.append(_thinking_job())
        if calls >= 3:
            stop.set()
        return await real()

    worker.think_tick = tick  # type: ignore[method-assign]
    await asyncio.wait_for(worker._think_forever(stop), 2)
    await asyncio.gather(*worker._thinking)
    assert "thinking line's tick failed" in caplog.text
    assert jobs.finished == [("j", "done", "Hecho", 0.0)]
    assert not worker._thinking


async def test_a_thought_that_cannot_be_recorded_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    async def run(*args: Any, **kwargs: Any) -> Any:
        return ExecutorResult("Hecho", 0.0, True)

    jobs = WorkerJobs()

    async def finish(*args: Any) -> bool:
        return False

    jobs.finish = finish  # type: ignore[method-assign]
    await _worker(jobs, run, tmp_path).think(_thinking_job())
    assert "could not be recorded" in caplog.text


async def test_a_pc_with_no_login_says_how_to_sign_in(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    async def no() -> bool:
        return False

    worker = Worker(WorkerJobs(), None, "", USER, folder=tmp_path, check_login=no)  # type: ignore[arg-type]
    await worker._note_login()
    assert worker.claude_login is False and "--login" in caplog.text


async def test_the_heartbeat_says_whether_the_pc_is_signed_in(tmp_path: Path) -> None:
    beats: list[tuple[Any, ...]] = []
    stop = asyncio.Event()

    class Beating(WorkerJobs):
        async def beat(self, *args: Any) -> bool:
            beats.append(args)
            stop.set()
            return True

    async def yes() -> bool:
        return True

    worker = Worker(Beating(), None, "", USER, folder=tmp_path, check_login=yes)  # type: ignore[arg-type]
    await asyncio.wait_for(worker._beat_forever(stop), 2)
    assert beats[0][0] == USER and beats[0][-1] is True


async def test_a_pc_still_there_keeps_being_waited_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bridge_module, "THINK_PC_CHECK_SECONDS", 0.0)
    jobs = FakeJobs()
    bridge = _bridge(tmp_path, {}, jobs)
    await bridge._refresh_account(force=True)
    assert (await bridge._ask(OPUS, BrainRequest("T", "S"))).text == "Pensado."


async def test_with_no_director_the_task_runs_whole(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path, {})
    bridge.config = BrainConfig(models=[])
    task = BrainTask("t1", "T", "tarea", "analysis", "normal")
    answer = await bridge._staged(task, "")
    assert answer.ok is False


class _History:
    def __init__(self, fails: bool = False) -> None:
        self.messages: list[tuple[str, str]] = []
        self.fails = fails

    def add_message(self, *, role: str, content: str) -> None:
        if self.fails:
            raise RuntimeError("no")
        self.messages.append((role, content))


class _Speaking:
    user_state = "listening"
    agent_state = "listening"

    def __init__(self, agent: Any) -> None:
        self.current_agent = agent
        self.replies: list[str] = []

    def generate_reply(self, *, instructions: str, allow_interruptions: bool = True) -> None:
        self.replies.append(instructions)


async def test_a_delivered_result_stays_in_the_voices_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def now(self: Any) -> bool:
        return True

    monkeypatch.setattr(Bridge, "_wait_for_pause", now)
    history = _History()
    session = _Speaking(SimpleNamespace(_chat_ctx=history))
    bridge = _bridge(tmp_path, {})
    bridge.attach(session, None)
    task = BrainTask("t1", "Capital", "?", "question", "simple", status="done", result="Ulán Bator")
    await bridge._deliver(task)
    assert history.messages[0][0] == "system" and "Ulán Bator" in history.messages[0][1]
    assert task.delivered and session.replies


@pytest.mark.parametrize(
    "agent", [SimpleNamespace(), SimpleNamespace(_chat_ctx=_History(fails=True))]
)
async def test_a_history_that_cannot_keep_it_does_not_stop_the_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent: Any
) -> None:
    async def now(self: Any) -> bool:
        return True

    monkeypatch.setattr(Bridge, "_wait_for_pause", now)
    session = _Speaking(agent)
    bridge = _bridge(tmp_path, {})
    bridge.attach(session, None)
    await bridge._deliver(
        BrainTask("t1", "T", "?", "question", "simple", status="done", result="R")
    )
    assert session.replies


async def test_a_failed_task_is_told_but_not_kept_as_a_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def now(self: Any) -> bool:
        return True

    monkeypatch.setattr(Bridge, "_wait_for_pause", now)
    history = _History()
    session = _Speaking(SimpleNamespace(_chat_ctx=history))
    bridge = _bridge(tmp_path, {})
    bridge.attach(session, None)
    await bridge._deliver(
        BrainTask("t1", "T", "?", "question", "simple", status="failed", result="No pude")
    )
    assert history.messages == [] and session.replies


def test_whose_repositories_is_the_deployments_to_say() -> None:
    from src.brain.selfwork import REPOS
    from src.settings import set_settings

    set_settings(Settings(self_repo_owner="alguien"))
    try:
        assert REPOS["agent"].slug == "alguien/kwami-lk-agent"
        assert REPOS["app"].url == "https://github.com/alguien/kwami-app.git"
    finally:
        set_settings(None)


async def test_with_no_owner_there_is_no_self_improvement(tmp_path: Path) -> None:
    from src.brain.selfwork import SelfWorkError, prepare
    from src.settings import set_settings

    set_settings(Settings(self_repo_owner=""))
    try:
        with pytest.raises(SelfWorkError, match="KWAMI_SELF_REPO_OWNER"):
            await prepare("agent", "t1", tmp_path, tmp_path)
    finally:
        set_settings(None)


async def test_a_login_check_cancelled_leaves_no_process_behind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = FakeProcess(b"", hang=True)

    async def spawn(*args: Any, **kwargs: Any) -> FakeProcess:
        return process

    monkeypatch.setattr(account_module, "_binary", lambda: "claude")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    running = asyncio.create_task(account_login())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert process.killed
