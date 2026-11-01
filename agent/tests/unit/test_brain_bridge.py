"""The bridge at its edges: wiring, budget, fallbacks, delivery and hang-up.

As in `test_brain.py`, models are replaced at our own boundary -- `Bridge._ask`
or the runner functions the bridge imports -- never by mocking a vendor SDK.
Most tests deliver with ``interrupt`` so they do not sit through the
one-second pause the default delivery waits for; the pause itself has its own
tests below.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest

from src.brain import bridge as bridge_module
from src.brain.bridge import BrainTask, Bridge
from src.brain.config import BrainConfig, BrainModel, default_brain
from src.brain.executor import CloudWorkspace, ExecutorResult
from src.brain.ledger import Ledger
from src.brain.planner import Plan, PlanStep
from src.brain.providers import BrainAnswer, BrainRequest
from src.settings import Settings

KEYS = {"ANTHROPIC_API_KEY": "a", "GOOGLE_API_KEY": "g", "OPENAI_API_KEY": "o"}
USER = "11111111-1111-4111-8111-111111111111"


class FakeSession:
    """The slice of `AgentSession` the bridge uses."""

    def __init__(self, agent: Any = None) -> None:
        self.user_state = "listening"
        self.agent_state = "listening"
        self.replies: list[str] = []
        self.interrupts = 0
        self.agent = agent
        self.fail_reply = False
        self.fail_interrupt = False

    @property
    def current_agent(self) -> Any:
        if isinstance(self.agent, Exception):
            raise self.agent
        return self.agent

    def generate_reply(self, *, instructions: str, allow_interruptions: bool = True) -> None:
        if self.fail_reply:
            raise RuntimeError("session closing")
        self.replies.append(instructions)

    async def interrupt(self) -> None:
        self.interrupts += 1
        if self.fail_interrupt:
            raise RuntimeError("nothing to interrupt")


class FakePublisher:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.messages: list[dict[str, Any]] = []

    async def publish(self, payload: dict[str, Any], *, topic: str | None = None) -> bool:
        if self.fail:
            raise RuntimeError("room gone")
        self.messages.append(payload)
        return True


class RecordingLedger(Ledger):
    """The disk ledger, plus a record of what was bound and a controllable bind."""

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir)
        self.binds: list[tuple[str, str | None]] = []
        self.bind_error: Exception | None = None
        self.bind_gate: asyncio.Event | None = None

    async def bind(self, kwami_id: str, user_id: str | None = None) -> None:
        self.binds.append((kwami_id, user_id))
        if self.bind_gate is not None:
            await self.bind_gate.wait()
        if self.bind_error is not None:
            raise self.bind_error


def _bridge(
    tmp_path: Path,
    answers: dict[str, BrainAnswer] | None = None,
    *,
    keys: dict[str, str] | None = None,
    config: BrainConfig | None = None,
    cloud: bool = False,
) -> Bridge:
    settings = Settings(executor_mode="cloud" if cloud else "local")
    bridge = Bridge(
        keys=KEYS if keys is None else keys, ledger=RecordingLedger(tmp_path), settings=settings
    )
    bridge.config = config or default_brain()
    bridge.config.split = False
    bridge.config.delivery = "interrupt"
    bridge.kwami_id = "k1"
    answers = answers or {}

    async def fake_ask(model: BrainModel, request: BrainRequest) -> BrainAnswer:
        answer = answers.get(model.id)
        if isinstance(answer, Exception):
            raise answer
        return answer or BrainAnswer(f"respuesta de {model.id}", model.id, model.model, 0.01)

    bridge._ask = fake_ask  # type: ignore[method-assign]
    return bridge


def _failed(model_id: str) -> BrainAnswer:
    return BrainAnswer(f"{model_id} caído", model_id, model_id, 0.001, ok=False)


async def _finish(bridge: Bridge) -> None:
    for task in list(bridge.tasks.values()):
        if task.handle is not None:
            await asyncio.gather(task.handle, return_exceptions=True)
    await asyncio.gather(*bridge._background, return_exceptions=True)


def _run(bridge: Bridge, **kwargs: Any) -> BrainTask:
    values: dict[str, Any] = {"task": "t", "title": "T", "kind": "question", "difficulty": "normal"}
    values.update(kwargs)
    task = bridge.start(**values)
    assert isinstance(task, BrainTask), task
    return task


# -- wiring ---------------------------------------------------------------------


class GatedAgent:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.refreshed = 0

    async def refresh_brain_gating(self) -> bool:
        self.refreshed += 1
        if self.fail:
            raise RuntimeError("tools locked")
        return True


async def test_configure_publishes_and_regates_the_voice(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    agent = GatedAgent()
    publisher = FakePublisher()
    bridge.attach(FakeSession(agent), publisher)

    bridge.configure(default_brain())
    await _finish(bridge)

    assert agent.refreshed == 1
    status = publisher.messages[-1]
    assert status["type"] == "brain_status"
    assert status["providers"] == {"anthropic": True, "google": True, "openai": True, "local": True}
    assert "realtime" in status["voiceProviders"]


@pytest.mark.parametrize(
    "agent", [GatedAgent(fail=True), RuntimeError("no agent yet"), object(), None]
)
async def test_regating_never_breaks_a_config_change(tmp_path: Path, agent: Any) -> None:
    bridge = _bridge(tmp_path)
    bridge.attach(FakeSession(agent), FakePublisher(fail=True))
    bridge.configure(default_brain())
    await _finish(bridge)


async def test_regating_without_a_session_does_nothing(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    bridge.configure(default_brain())
    await _finish(bridge)


def test_work_scheduled_with_no_event_loop_is_dropped(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    bridge.configure(default_brain())  # would publish and regate; there is no loop
    assert bridge._background == set()


async def test_context_and_user_bind_the_ledger(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    ledger = bridge.ledger
    assert isinstance(ledger, RecordingLedger)

    bridge.set_context(user_profile="x" * 5000)
    assert len(bridge.user_profile) == 3000
    assert bridge.assistant_name == "Jarvis"

    bridge.set_context(kwami_id="k2", name="Viernes", language="English", timezone="UTC")
    bridge.set_user("sip_+34600000000")  # not a user id
    bridge.set_user(USER)
    bridge.set_user(USER)  # already known
    await _finish(bridge)

    assert (bridge.kwami_id, bridge.assistant_name, bridge.language) == ("k2", "Viernes", "English")
    assert ledger.binds == [("k2", None), ("k2", USER)]


async def test_ready_waits_briefly_and_never_raises(tmp_path: Path, caplog) -> None:
    bridge = _bridge(tmp_path)
    ledger = bridge.ledger
    assert isinstance(ledger, RecordingLedger)

    await bridge.ready("")
    assert ledger.binds == []
    await bridge.ready("k1")

    ledger.bind_gate = asyncio.Event()
    with caplog.at_level(logging.WARNING):
        await bridge.ready("k1", timeout=0.01)
    assert "not loaded after" in caplog.text

    ledger.bind_gate = None
    ledger.bind_error = RuntimeError("supabase down")
    await bridge.ready("k1")
    assert "Brain ledger failed to load" in caplog.text


def test_the_cloud_workspace_belongs_to_the_owner(tmp_path: Path) -> None:
    local = _bridge(tmp_path)
    assert local._cloud_workspace() is None

    cloud = _bridge(tmp_path, cloud=True)
    assert cloud._cloud_workspace() == CloudWorkspace(store=None, user_id="", kwami_id="k1")
    cloud.user_id = USER
    assert cloud._cloud_workspace().user_id == USER  # type: ignore[union-attr]


def test_delegating_and_finding_models(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    assert bridge.delegates()
    assert bridge.find("opus").id == "claude-opus"  # type: ignore[union-attr]
    assert not _bridge(tmp_path, keys={}).delegates()


# -- starting work ----------------------------------------------------------------


def test_a_switched_off_brain_takes_nothing(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    bridge.config.enabled = False
    assert "switched off" in str(
        bridge.start(task="t", title="t", kind="question", difficulty="simple")
    )


def test_a_spent_budget_stops_delegation_when_asked_to(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    bridge.ledger.add_spend("claude-opus", 60.0)
    bridge.config.budget_policy = "stop"
    out = bridge.start(task="t", title="t", kind="analysis", difficulty="hard")
    assert isinstance(out, str) and "budget" in out


async def test_a_spent_budget_downgrades_when_asked_to(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    bridge.ledger.add_spend("claude-opus", 60.0)
    bridge.config.budget_policy = "downgrade"
    task = _run(bridge, kind="analysis", difficulty="hard", important=True, model="opus")
    await _finish(bridge)
    assert not task.council
    assert task.models == ["Claude Haiku 5.5"]


async def test_a_council_of_one_answers_alone(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path, keys={"ANTHROPIC_API_KEY": "a"})
    bridge.config.models = [m for m in bridge.config.models if m.id == "claude-sonnet"]
    task = _run(bridge, council=True)
    await _finish(bridge)
    assert not task.council and task.models == ["Claude Sonnet 5.5"]


def test_no_model_for_the_task_is_said_plainly(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    bridge.config.models = [m for m in bridge.config.models if m.id == "claude-code"]
    out = bridge.start(task="t", title="t", kind="question", difficulty="simple")
    assert out == "No usable model fits this task."
    bridge.config.models = [m for m in default_brain().models if m.id == "claude-haiku"]
    out = bridge.start(task="t", title="t", kind="code", difficulty="simple")
    assert isinstance(out, str) and "executor" in out


async def test_old_finished_tasks_make_room(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bridge_module, "MAX_TASKS_KEPT", 2)
    bridge = _bridge(tmp_path)
    first = _run(bridge, title="uno", context="lo que se dijo")
    assert "Context from the conversation" in first.task
    await _finish(bridge)
    _run(bridge, title="dos")
    _run(bridge, title="tres")
    assert [t.title for t in bridge.tasks.values()] == ["dos", "tres"]
    # The oldest is still running: nothing more is dropped.
    _run(bridge, title="cuatro")
    assert [t.title for t in bridge.tasks.values()] == ["dos", "tres", "cuatro"]
    await _finish(bridge)


async def test_a_running_task_can_be_cancelled_once(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    gate = asyncio.Event()
    publisher = FakePublisher()
    bridge.attach(FakeSession(), publisher)

    async def stuck(model: BrainModel, request: BrainRequest) -> BrainAnswer:
        await gate.wait()
        raise AssertionError("never released")

    bridge._ask = stuck  # type: ignore[method-assign]
    task = _run(bridge)
    await asyncio.sleep(0)
    assert bridge.cancel(task.id)
    await _finish(bridge)

    assert task.status == "cancelled" and task.result == "Cancelada."
    assert not bridge.cancel(task.id)
    assert not bridge.cancel("nope")
    assert publisher.messages[-1]["tasks"][-1]["status"] == "cancelled"
    # A cancelled task is never spoken.
    await bridge._deliver(task)
    assert bridge.session.replies == []


async def test_a_crash_inside_a_task_is_a_failed_task(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path, {"claude-sonnet": RuntimeError("boom")})  # type: ignore[dict-item]
    session = FakeSession()
    bridge.attach(session, FakePublisher())
    task = _run(bridge)
    await _finish(bridge)
    assert task.status == "failed" and task.result == "La tarea falló: boom"
    assert "A delegated task failed" in session.replies[0]


# -- the request a worker gets -------------------------------------------------------


def test_an_unknown_timezone_falls_back_to_utc(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    bridge.timezone = "Mars/Olympus"
    bridge.user_profile = "Vive en Madrid."
    system = bridge._system()
    assert "(Mars/Olympus)" in system
    assert "Vive en Madrid." in system


@pytest.mark.parametrize(
    ("search", "tavily", "native", "sent_key"),
    [("native", "tv", True, ""), ("tavily", "tv", False, "tv"), ("tavily", "", True, "")],
)
def test_the_search_setting_shapes_the_request(
    tmp_path: Path, search: str, tavily: str, native: bool, sent_key: str
) -> None:
    bridge = _bridge(tmp_path)
    bridge._tavily_key = tavily
    bridge.config.search = search  # type: ignore[assignment]
    request = bridge._request("t", "normal")
    assert (request.native_search, request.tavily_key) == (native, sent_key)


async def test_each_provider_is_called_with_its_own_key(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []

    def runner(name: str):
        async def run(
            api_key: str, model_id: str, model: str, request: BrainRequest
        ) -> BrainAnswer:
            calls.append((name, api_key, model))
            return BrainAnswer(name, model_id, model, 0.5)

        return run

    executor_calls: list[dict[str, Any]] = []

    async def fake_executor(
        api_key: str, model: str, task: str, config: Any, *, cloud: Any
    ) -> ExecutorResult:
        executor_calls.append({"key": api_key, "task": task, "cloud": cloud})
        return ExecutorResult("hecho", 0.25, True)

    monkeypatch.setattr(bridge_module, "run_anthropic", runner("anthropic"))
    monkeypatch.setattr(bridge_module, "run_gemini", runner("google"))
    monkeypatch.setattr(bridge_module, "run_openai", runner("openai"))
    monkeypatch.setattr(bridge_module, "run_executor", fake_executor)

    bridge = Bridge(keys=KEYS, ledger=Ledger(tmp_path), settings=Settings(executor_mode="cloud"))
    bridge.kwami_id = "k1"
    request = BrainRequest(task="haz algo", system="s")
    models = [
        BrainModel("c", "anthropic", "claude-x", "fast"),
        BrainModel("g", "google", "gemini-x", "search"),
        BrainModel("o", "openai", "gpt-x", "expert"),
        BrainModel("cc", "local", "claude-y", "code"),
    ]
    answers = [await bridge._ask(m, request) for m in models]

    assert calls == [
        ("anthropic", "a", "claude-x"),
        ("google", "g", "gemini-x"),
        ("openai", "o", "gpt-x"),
    ]
    assert executor_calls[0]["task"] == "haz algo" and executor_calls[0]["key"] == "a"
    assert executor_calls[0]["cloud"].kwami_id == "k1"
    assert answers[-1].text == "hecho" and answers[-1].cost_usd == 0.25
    assert bridge.ledger.month_spend()["byModel"] == {"c": 0.5, "g": 0.5, "o": 0.5, "cc": 0.25}


# -- fallbacks -------------------------------------------------------------------------


async def test_no_model_at_all_is_a_failed_answer(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    task = BrainTask("t1", "T", "t", "code", "normal")
    bridge.config.models = []
    answer, model = await bridge._ask_role(task, "code", "normal", bridge._request("t", "normal"))
    assert not answer.ok and model is None
    assert answer.text == "No usable model could take the task."


async def test_a_failed_gemini_search_is_redone_on_tavily(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path, {"gemini-flash": _failed("gemini-flash")})
    bridge._tavily_key = "tv"
    task = _run(bridge, kind="research", difficulty="simple")
    await _finish(bridge)
    assert task.status == "done"
    assert task.models == ["Gemini 3.5 Flash", "Claude Sonnet 5.5 + Tavily"]
    assert task.result == "respuesta de claude-sonnet"


async def test_when_tavily_fails_too_the_normal_fallback_goes_on(tmp_path: Path) -> None:
    bridge = _bridge(
        tmp_path,
        {"gemini-flash": _failed("gemini-flash"), "claude-sonnet": _failed("claude-sonnet")},
    )
    bridge._tavily_key = "tv"
    task = _run(bridge, kind="research", difficulty="simple")
    await _finish(bridge)
    assert task.models == ["Gemini 3.5 Flash", "Claude Sonnet 5.5 + Tavily", "Claude Haiku 5.5"]
    assert task.result == "respuesta de claude-haiku"
    assert task.cost_usd == pytest.approx(0.012)


async def test_without_a_writer_for_tavily_gemini_fails_alone(tmp_path: Path) -> None:
    bridge = _bridge(
        tmp_path, {"gemini-flash": _failed("gemini-flash")}, keys={"GOOGLE_API_KEY": "g"}
    )
    bridge._tavily_key = "tv"
    task = _run(bridge, kind="research", difficulty="simple")
    await _finish(bridge)
    assert task.status == "failed" and task.models == ["Gemini 3.5 Flash"]


def _out_of_quota(model_id: str) -> BrainAnswer:
    answer = _failed(model_id)
    answer.rate_limited = True
    return answer


async def test_a_model_out_of_quota_is_not_tried_again_for_a_while(tmp_path: Path) -> None:
    """A spent free-tier quota used to cost a failed round trip on every task."""
    bridge = _bridge(tmp_path, {"gemini-flash": _out_of_quota("gemini-flash")})
    bridge._tavily_key = "tv"
    first = _run(bridge, kind="research", difficulty="simple")
    await _finish(bridge)
    second = _run(bridge, kind="research", difficulty="simple")
    await _finish(bridge)

    assert first.models == ["Gemini 3.5 Flash", "Claude Sonnet 5.5 + Tavily"]
    # Straight to the cheap rescue, Gemini left alone.
    assert second.models == ["Claude Sonnet 5.5 + Tavily"]
    assert second.status == "done"


async def test_a_resting_model_is_skipped_for_tasks_without_a_search(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path, {"gemini-flash": _out_of_quota("gemini-flash")})
    _run(bridge, kind="research", difficulty="simple")
    await _finish(bridge)
    task = _run(bridge, kind="research", difficulty="simple")
    await _finish(bridge)
    assert "Gemini 3.5 Flash" not in task.models
    assert task.status == "done"


async def test_a_rest_ends(tmp_path: Path, monkeypatch) -> None:
    import src.brain.bridge as bridge_module

    bridge = _bridge(tmp_path, {"gemini-flash": _out_of_quota("gemini-flash")})
    _run(bridge, kind="research", difficulty="simple")
    await _finish(bridge)
    assert bridge._resting_ids() == {"gemini-flash"}
    later = bridge_module.time.monotonic() + bridge_module.QUOTA_REST_SECONDS + 1
    monkeypatch.setattr(bridge_module.time, "monotonic", lambda: later)
    assert bridge._resting_ids() == set()


async def test_an_ordinary_failure_does_not_rest_the_model(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path, {"gemini-flash": _failed("gemini-flash")})
    _run(bridge, kind="research", difficulty="simple")
    await _finish(bridge)
    assert bridge._resting_ids() == set()


# -- splitting ----------------------------------------------------------------------------


def _split_bridge(tmp_path: Path, monkeypatch, plan: Plan | None, **kw: Any) -> Bridge:
    bridge = _bridge(tmp_path, **kw)
    bridge.config.split = True

    async def fake_plan(api_key: str, model: str, task: str) -> Plan | None:
        return plan

    monkeypatch.setattr(bridge_module, "plan_task", fake_plan)
    return bridge


async def test_planning_needs_a_claude_planner(tmp_path: Path, monkeypatch) -> None:
    bridge = _split_bridge(tmp_path, monkeypatch, None)
    bridge.config.models = [m for m in bridge.config.models if m.provider == "google"]
    task = _run(bridge, kind="analysis")
    await _finish(bridge)
    assert task.models == ["Gemini 3.5 Flash"]


async def test_a_failed_plan_runs_the_task_whole(tmp_path: Path, monkeypatch) -> None:
    bridge = _split_bridge(tmp_path, monkeypatch, None)
    task = _run(bridge, kind="analysis")
    await _finish(bridge)
    assert task.models == ["Claude Sonnet 5.5"]


@pytest.mark.parametrize(
    ("kind", "web", "planned_kind", "model"),
    [
        ("analysis", True, "research", "Gemini 3.5 Flash"),
        ("research", False, "analysis", "Claude Haiku 5.5"),
        ("writing", False, "writing", "Claude Haiku 5.5"),
    ],
)
async def test_an_indivisible_plan_corrects_the_kind(
    tmp_path: Path, monkeypatch, kind: str, web: bool, planned_kind: str, model: str
) -> None:
    plan = Plan(steps=[PlanStep("s1", "", "simple", web)], cost_usd=0.002)
    bridge = _split_bridge(tmp_path, monkeypatch, plan)
    task = _run(bridge, kind=kind)
    await _finish(bridge)
    assert task.kind == planned_kind and task.difficulty == "simple"
    assert task.models == [model]
    assert task.cost_usd == pytest.approx(0.012)


async def test_independent_parts_are_combined_by_one_more_call(tmp_path: Path, monkeypatch) -> None:
    plan = Plan(
        steps=[
            PlanStep("s1", "dato uno", "normal", False),
            PlanStep("s2", "dato dos", "normal", False, ["s1"]),
            PlanStep("s3", "aparte", "hard", True),
        ]
    )
    seen: list[str] = []
    bridge = _split_bridge(tmp_path, monkeypatch, plan)
    inner = bridge._ask

    async def record(model: BrainModel, request: BrainRequest) -> BrainAnswer:
        seen.append(request.task)
        if request.task.endswith("Your part:\naparte"):
            return _failed(model.id)  # on the chosen model and on its fallback
        return await inner(model, request)

    bridge._ask = record  # type: ignore[method-assign]
    task = _run(bridge, kind="analysis", difficulty="hard")
    await _finish(bridge)

    # s2 was told what s1 found; the combine call saw every part, failed or not.
    assert any("What earlier parts found" in t and "respuesta de claude-sonnet" in t for t in seen)
    assert "combine them into one answer" in seen[-1]
    assert "(this part failed:" in seen[-1]
    assert [s["ok"] for s in task.steps] == [True, False, True]
    # Sonnet did s1, s2 and nothing else; Opus combined (the plan has a hard part).
    assert task.models.count("Claude Sonnet 5.5") == 1
    assert task.result == "respuesta de claude-opus"


async def test_a_combine_of_normal_parts_stays_normal(tmp_path: Path, monkeypatch) -> None:
    plan = Plan(steps=[PlanStep("a", "x", "simple", False), PlanStep("b", "y", "simple", False)])
    bridge = _split_bridge(tmp_path, monkeypatch, plan)
    task = _run(bridge, kind="analysis")
    await _finish(bridge)
    assert task.result == "respuesta de claude-sonnet"


# -- the council ----------------------------------------------------------------------------


async def test_a_council_with_one_good_answer_returns_it(tmp_path: Path) -> None:
    bridge = _bridge(
        tmp_path, {"claude-opus": _failed("claude-opus"), "gemini-flash": _failed("gemini-flash")}
    )
    task = _run(bridge, council=True)
    await _finish(bridge)
    assert task.result == "respuesta de claude-sonnet"


async def test_a_council_where_everyone_failed_reports_the_first_failure(tmp_path: Path) -> None:
    bridge = _bridge(
        tmp_path,
        {m: _failed(m) for m in ("claude-sonnet", "claude-opus", "gemini-flash")},
    )
    task = _run(bridge, council=True)
    await _finish(bridge)
    assert task.status == "failed" and task.result == "claude-sonnet caído"


async def test_when_the_chair_fails_the_first_good_answer_stands(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    calls = 0
    inner = bridge._ask

    async def chair_fails(model: BrainModel, request: BrainRequest) -> BrainAnswer:
        nonlocal calls
        calls += 1
        if calls == 4:  # three members, then the chair
            return _failed(model.id)
        return await inner(model, request)

    bridge._ask = chair_fails  # type: ignore[method-assign]
    task = _run(bridge, council=True)
    await _finish(bridge)
    assert task.status == "done" and task.result == "respuesta de claude-sonnet"


# -- delivery -------------------------------------------------------------------------------


def _done(title: str = "Oro", result: str = "Sube.\n\nDetalle: mucho", **kw: Any) -> BrainTask:
    kw.setdefault("status", "done")
    task = BrainTask("t1", title, "t", "question", "normal", result=result, **kw)
    task.models = ["Claude Opus 5.5"]
    return task


def test_what_the_voice_is_told(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bridge_module, "SPOKEN_RESULT_CHARS", 10)
    bridge = _bridge(tmp_path)
    spoken = bridge._spoken(_done(result="x" * 50, council=True))
    assert "x" * 10 + " [...]" in spoken
    assert "The council met" in spoken

    bridge.config.delivery = "on_request"
    on_request = bridge._spoken(_done())
    assert "ask whether they want it now" in on_request and "Sube" not in on_request

    nobody = _done(status="failed")
    nobody.models = []
    assert bridge._spoken(nobody).startswith("A delegated task failed")


async def test_interrupting_is_best_effort(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    session = FakeSession()
    session.fail_interrupt = True
    bridge.attach(session, FakePublisher())
    task = _done()
    await bridge._deliver(task)
    assert session.interrupts == 1 and task.delivered
    assert "Sube." in session.replies[0]


async def test_a_result_that_cannot_be_spoken_is_kept_for_later(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    session = FakeSession()
    session.fail_reply = True
    bridge.attach(session, FakePublisher())
    await bridge._deliver(_done())
    owed = bridge.take_pending("k1")
    assert [(p.title, p.summary) for p in owed] == [("Oro", "Sube.")]


def test_nothing_is_kept_without_a_kwami(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    bridge.kwami_id = ""
    task = _done()
    bridge._store_pending(task)
    assert not task.delivered
    assert bridge.take_pending("") == []


async def test_the_next_pause_waits_for_the_user_to_stop(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    bridge.config.delivery = "next_pause"
    session = FakeSession()
    session.user_state = "speaking"
    bridge.attach(session, FakePublisher())

    delivering = asyncio.create_task(bridge._deliver(_done()))
    await asyncio.sleep(0.3)
    assert session.replies == []
    session.user_state = "listening"
    await asyncio.wait_for(delivering, timeout=5)
    assert len(session.replies) == 1


async def test_a_user_who_never_pauses_still_hears_it(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bridge_module, "PAUSE_WAIT_SECONDS", 0.3)
    bridge = _bridge(tmp_path)
    bridge.config.delivery = "next_pause"
    session = FakeSession()
    session.agent_state = "speaking"
    bridge.attach(session, FakePublisher())
    await bridge._deliver(_done())
    assert len(session.replies) == 1


async def test_a_call_that_ends_during_the_wait_keeps_the_result(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    bridge.config.delivery = "next_pause"
    session = FakeSession()
    session.user_state = "speaking"
    bridge.attach(session, FakePublisher())

    delivering = asyncio.create_task(bridge._deliver(_done()))
    await asyncio.sleep(0.1)
    bridge.session = None
    await asyncio.wait_for(delivering, timeout=5)
    assert session.replies == []
    assert [p.title for p in bridge.take_pending("k1")] == ["Oro"]


# -- hang-up -----------------------------------------------------------------------------------


async def test_cancel_on_hangup_drops_running_work(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    bridge.config.on_hangup = "cancel"
    gate = asyncio.Event()

    async def stuck(model: BrainModel, request: BrainRequest) -> BrainAnswer:
        await gate.wait()
        raise AssertionError("never released")

    bridge._ask = stuck  # type: ignore[method-assign]
    task = _run(bridge)
    await asyncio.sleep(0)
    await bridge.close(timeout=1)
    await _finish(bridge)
    assert task.status == "cancelled"
    assert bridge.take_pending("k1") == []


async def test_work_that_outlives_the_grace_period_is_cancelled(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    gate = asyncio.Event()

    async def stuck(model: BrainModel, request: BrainRequest) -> BrainAnswer:
        await gate.wait()
        raise AssertionError("never released")

    bridge._ask = stuck  # type: ignore[method-assign]
    task = _run(bridge)
    await asyncio.sleep(0)
    await bridge.close(timeout=0.01)
    await _finish(bridge)
    assert task.status == "cancelled"


async def test_a_finished_but_unspoken_result_is_kept_at_hangup(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    bridge.tasks["t1"] = _done()
    bridge.tasks["t2"] = _done(title="Ya dicho", delivered=True)
    await bridge.close(timeout=1)
    assert [p.title for p in bridge.take_pending("k1")] == ["Oro"]
