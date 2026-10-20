"""The brain: who gets a task, what the executor may touch, what is owed.

The bridge is tested with the providers replaced at our own boundary
(`Bridge._ask`), never by mocking the vendor SDKs: what matters here is the
routing, the council, delivery and the pending list, not an HTTP call.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from src.brain import Bridge, parse_brain
from src.brain.config import EXECUTOR_ROOT, BrainModel, default_brain, resolve_executor_folder
from src.brain.executor import check_tool_call
from src.brain.ledger import Ledger, token_cost
from src.brain.planner import Plan, PlanStep, final_step, parse_plan, waves
from src.brain.providers import BrainAnswer
from src.brain.router import council_members, pick, should_convene
from src.domain.greeting import GreetingFacts, build_greeting_instructions

ALL_KEYS = {"ANTHROPIC_API_KEY": "a", "GOOGLE_API_KEY": "g"}

# -- config ---------------------------------------------------------------------


def test_defaults_are_the_roster_the_user_chose() -> None:
    config = default_brain()
    roles = {m.id: m.role for m in config.models}
    assert roles == {
        "claude-haiku": "fast",
        "claude-sonnet": "standard",
        "claude-opus": "expert",
        "gemini-flash": "search",
        "claude-code": "code",
    }
    assert config.council == "important"
    assert config.budget_policy == "inform"
    assert config.on_hangup == "finish"


def test_a_bad_model_entry_is_dropped_not_the_message() -> None:
    config = parse_brain(
        {
            "council": "on_request",
            "models": [
                {"id": "x", "provider": "nope", "model": "m", "role": "fast"},
                {
                    "id": "gpt",
                    "provider": "openai",
                    "model": "gpt-5.2",
                    "role": "expert",
                    "council": True,
                },
            ],
        }
    )
    assert [m.id for m in config.models] == ["gpt"]
    assert config.models[0].council is True
    assert config.council == "on_request"


def test_executor_folder_cannot_leave_the_root(tmp_path: Path) -> None:
    assert resolve_executor_folder("C:\\Windows") == EXECUTOR_ROOT.resolve()
    inside = EXECUTOR_ROOT / "proyecto"
    assert resolve_executor_folder(str(inside)) == inside.resolve()


# -- routing ------------------------------------------------------------------


def test_difficulty_picks_the_model() -> None:
    config = default_brain()
    assert pick(config, ALL_KEYS, "question", "simple").id == "claude-haiku"
    assert pick(config, ALL_KEYS, "analysis", "normal").id == "claude-sonnet"
    assert pick(config, ALL_KEYS, "analysis", "hard").id == "claude-opus"
    assert pick(config, ALL_KEYS, "research", "simple").id == "gemini-flash"
    assert pick(config, ALL_KEYS, "code", "normal").id == "claude-code"


def test_a_model_without_its_key_is_skipped() -> None:
    config = default_brain()
    only_anthropic = {"ANTHROPIC_API_KEY": "a"}
    assert pick(config, only_anthropic, "research", "simple").id == "claude-sonnet"


def test_a_named_model_wins_but_never_crosses_into_code() -> None:
    config = default_brain()
    assert pick(config, ALL_KEYS, "question", "simple", prefer="opus").id == "claude-opus"
    assert pick(config, ALL_KEYS, "question", "simple", prefer="claude code").id == "claude-haiku"


def test_the_council_meets_for_important_things_only() -> None:
    config = default_brain()
    assert should_convene(config, "analysis", important=True, requested=False)
    assert not should_convene(config, "analysis", important=False, requested=False)
    assert not should_convene(config, "code", important=True, requested=True)
    members = {m.id for m in council_members(config, ALL_KEYS)}
    assert members == {"claude-sonnet", "claude-opus", "gemini-flash"}


# -- executor guard -------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "tool_input"),
    [
        ("Bash", {"command": "git push origin main"}),
        ("Bash", {"command": "cat .env"}),
        ("Bash", {"command": "rm -rf ../otro"}),
        ("Bash", {"command": "ls C:\\Users\\Usuario\\Documents\\Phyton.Code"}),
        ("Bash", {"command": "ls /c/Users/Usuario/Documents"}),
        ("Read", {"file_path": "C:\\Users\\Usuario\\.ssh\\id_rsa"}),
        ("Edit", {"file_path": ".env.local"}),
    ],
)
def test_executor_refuses_what_the_user_ruled_out(tool: str, tool_input: dict[str, Any]) -> None:
    assert check_tool_call(EXECUTOR_ROOT.resolve(), tool, tool_input) is not None


@pytest.mark.parametrize(
    ("tool", "tool_input"),
    [
        ("Bash", {"command": "git commit -m 'informe'"}),
        ("Bash", {"command": "python script.py"}),
        ("Write", {"file_path": "informes/hoy.md"}),
        ("Read", {"file_path": str(EXECUTOR_ROOT / "notas.txt")}),
    ],
)
def test_executor_works_freely_inside(tool: str, tool_input: dict[str, Any]) -> None:
    assert check_tool_call(EXECUTOR_ROOT.resolve(), tool, tool_input) is None


# -- ledger -----------------------------------------------------------------------


def test_spend_accumulates_per_month_and_model(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path)
    ledger.add_spend("claude-opus", 0.5)
    ledger.add_spend("claude-opus", 0.25)
    spend = ledger.month_spend()
    assert spend["usd"] == pytest.approx(0.75)
    assert spend["byModel"] == {"claude-opus": 0.75}


def test_unknown_models_are_never_free() -> None:
    assert token_cost("google", "gemini-99", 1_000_000, 0) > 0


# -- greeting ---------------------------------------------------------------------


def test_results_owed_from_the_last_call_are_greeted_even_with_no_greeting() -> None:
    owed = ["Precio del oro: sube un 2 %"]
    with_charter = build_greeting_instructions(
        GreetingFacts(agent_name="Jarvis", charter_greeting="none", brain_results=owed)
    )
    assert "Precio del oro" in with_charter
    plain = build_greeting_instructions(GreetingFacts(agent_name="Jarvis"))
    assert "brain finished" not in plain


# -- bridge -------------------------------------------------------------------------


class FakeSession:
    user_state = "listening"
    agent_state = "listening"

    def __init__(self) -> None:
        self.replies: list[str] = []

    def generate_reply(self, *, instructions: str, allow_interruptions: bool = True) -> None:
        self.replies.append(instructions)


class FakePublisher:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def publish(self, payload: dict[str, Any], *, topic: str | None = None) -> bool:
        self.messages.append(payload)
        return True


def _bridge(tmp_path: Path, answers: dict[str, BrainAnswer]) -> Bridge:
    bridge = Bridge(keys=ALL_KEYS, ledger=Ledger(tmp_path))
    # Planning is a real API call; the staged tests turn it back on with a fake.
    bridge.config.split = False
    bridge.set_context(kwami_id="k1", name="Jarvis", language="es-ES", timezone="Europe/Madrid")

    async def fake_ask(model: BrainModel, request: Any) -> BrainAnswer:
        answer = answers.get(model.id) or BrainAnswer(
            f"respuesta de {model.id}", model.id, model.model, 0.01
        )
        bridge.ledger.add_spend(model.id, answer.cost_usd)
        return answer

    bridge._ask = fake_ask  # type: ignore[method-assign]
    return bridge


async def _finish(bridge: Bridge) -> None:
    for task in list(bridge.tasks.values()):
        if task.handle is not None:
            await asyncio.wait_for(task.handle, timeout=5)


async def test_a_task_is_spoken_at_the_next_pause(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path, {})
    session, publisher = FakeSession(), FakePublisher()
    bridge.attach(session, publisher)

    task = bridge.start(
        task="¿Qué hora es en Tokio?", title="Hora en Tokio", kind="question", difficulty="simple"
    )
    assert not isinstance(task, str)
    await _finish(bridge)

    assert task.status == "done"
    assert task.models == ["Claude Haiku 5.5"]
    assert len(session.replies) == 1 and "respuesta de claude-haiku" in session.replies[0]
    assert any(m["type"] == "brain_status" for m in publisher.messages)


async def test_a_failed_model_falls_back_once(tmp_path: Path) -> None:
    bridge = _bridge(
        tmp_path, {"claude-sonnet": BrainAnswer("caído", "claude-sonnet", "x", ok=False)}
    )
    bridge.attach(FakeSession(), FakePublisher())
    task = bridge.start(task="analiza", title="Análisis", kind="analysis", difficulty="normal")
    await _finish(bridge)
    assert task.status == "done"
    assert task.models == ["Claude Sonnet 5.5", "Claude Opus 5.5"]


async def test_the_council_compares_and_the_chair_answers(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path, {})
    bridge.attach(FakeSession(), FakePublisher())
    task = bridge.start(
        task="¿Vendo mis acciones?",
        title="Vender acciones",
        kind="analysis",
        difficulty="hard",
        important=True,
    )
    await _finish(bridge)
    assert task.council is True
    assert set(task.models) == {"Claude Sonnet 5.5", "Claude Opus 5.5", "Gemini 3.5 Flash"}
    # Three members plus the chair's synthesis (Opus).
    assert task.result == "respuesta de claude-opus"
    assert task.cost_usd == pytest.approx(0.04)


async def test_work_finished_after_hangup_waits_for_the_next_greeting(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path, {})
    gate = asyncio.Event()
    original = bridge._ask

    async def slow_ask(model: BrainModel, request: Any) -> BrainAnswer:
        await gate.wait()
        return await original(model, request)

    bridge._ask = slow_ask  # type: ignore[method-assign]
    bridge.attach(FakeSession(), FakePublisher())
    bridge.start(
        task="investiga", title="Investigación larga", kind="research", difficulty="normal"
    )

    closing = asyncio.create_task(bridge.close(timeout=5))
    await asyncio.sleep(0)
    gate.set()
    await closing

    owed = bridge.take_pending("k1")
    assert [p.title for p in owed] == ["Investigación larga"]
    assert bridge.take_pending("k1") == []


async def test_nothing_runs_without_a_usable_model(tmp_path: Path) -> None:
    bridge = Bridge(keys={}, ledger=Ledger(tmp_path))
    assert isinstance(bridge.start(task="x", title="x", kind="question", difficulty="simple"), str)


# -- splitting --------------------------------------------------------------------


def test_a_plan_keeps_only_backward_dependencies() -> None:
    steps = parse_plan(
        '{"split": true, "steps": ['
        '{"id": "a", "goal": "dato", "difficulty": "simple", "web": true, "after": ["b"]},'
        '{"id": "b", "goal": "decide", "difficulty": "hard", "web": false, "after": ["a"]}]}'
    )
    assert [s.after for s in steps] == [[], ["a"]]
    assert [[s.id for s in w] for w in waves(steps)] == [["a"], ["b"]]


def test_an_indivisible_plan_keeps_its_hardest_difficulty() -> None:
    steps = parse_plan(
        '{"split": false, "steps": ['
        '{"id": "a", "goal": "x", "difficulty": "normal", "web": false, "after": []},'
        '{"id": "b", "goal": "y", "difficulty": "hard", "web": true, "after": []}]}'
    )
    assert len(steps) == 1 and steps[0].difficulty == "hard" and steps[0].web


async def test_split_tasks_send_each_part_to_its_model(tmp_path: Path, monkeypatch) -> None:
    bridge = _bridge(tmp_path, {})
    bridge.config.split = True
    plan = Plan(
        steps=[
            PlanStep("s1", "precio actual del oro", "simple", True),
            PlanStep("s2", "resumir la normativa", "normal", False),
            PlanStep("s3", "decidir si comprar", "hard", False, ["s1", "s2"]),
        ],
        cost_usd=0.001,
    )

    async def fake_plan(api_key: str, model: str, task: str) -> Plan:
        return plan

    monkeypatch.setattr("src.brain.bridge.plan_task", fake_plan)
    bridge.attach(FakeSession(), FakePublisher())
    task = bridge.start(task="¿Compro oro?", title="Oro", kind="analysis", difficulty="hard")
    await _finish(bridge)

    assert [s["model"] for s in task.steps] == [
        "Gemini 3.5 Flash",  # quick web lookup: Gemini is the efficient one
        "Claude Sonnet 5.5",  # everyday reasoning
        "Claude Opus 5.5",  # the decision
    ]
    # The decision builds on both lookups, so it answers directly: Opus runs
    # once, with no separate combine call. Haiku was never needed.
    assert task.result == "respuesta de claude-opus"
    assert bridge.ledger.month_spend()["byModel"]["claude-opus"] == pytest.approx(0.01)
    assert "Claude Haiku 5.5" not in task.models


def test_independent_parts_still_need_combining() -> None:
    parallel = [PlanStep("a", "x", "simple", True), PlanStep("b", "y", "normal", False)]
    assert final_step(parallel) is None
    chained = [*parallel, PlanStep("c", "z", "hard", False, ["b"])]
    assert final_step(chained) is None  # c never sees a
    assert final_step([*parallel, PlanStep("c", "z", "hard", False, ["a", "b"])]) == "c"


async def test_an_indivisible_task_moves_to_the_planned_difficulty(
    tmp_path: Path, monkeypatch
) -> None:
    bridge = _bridge(tmp_path, {})
    bridge.config.split = True

    async def fake_plan(api_key: str, model: str, task: str) -> Plan:
        return Plan(steps=[PlanStep("s1", "", "hard", False)])

    monkeypatch.setattr("src.brain.bridge.plan_task", fake_plan)
    bridge.attach(FakeSession(), FakePublisher())
    task = bridge.start(task="analiza", title="Análisis", kind="analysis", difficulty="normal")
    await _finish(bridge)
    assert task.models == ["Claude Opus 5.5"]


async def test_simple_tasks_skip_the_planner(tmp_path: Path, monkeypatch) -> None:
    bridge = _bridge(tmp_path, {})
    bridge.config.split = True

    async def no_plan(*args: object) -> Plan:
        raise AssertionError("the planner must not run for a simple task")

    monkeypatch.setattr("src.brain.bridge.plan_task", no_plan)
    bridge.attach(FakeSession(), FakePublisher())
    task = bridge.start(task="hora", title="Hora", kind="question", difficulty="simple")
    await _finish(bridge)
    assert task.models == ["Claude Haiku 5.5"]
