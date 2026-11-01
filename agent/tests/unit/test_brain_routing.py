"""The brain's roster, routing and planner, at their edges.

`test_brain.py` covers the everyday paths; these pin what happens with a
hostile or half-filled config from the app, a roster with gaps, and a planner
call that fails or answers badly. The planner's SDK client is replaced at
`anthropic.AsyncAnthropic`, with the SDK's own error classes.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest

from src.brain.config import (
    EXECUTOR_ROOT,
    BrainConfig,
    BrainModel,
    default_brain,
    parse_brain,
    resolve_executor_folder,
)
from src.brain.planner import PLAN_SCHEMA, PlanStep, final_step, parse_plan, plan_task, waves
from src.brain.router import (
    council_members,
    find_model,
    pick,
    should_convene,
    tavily_writer,
)

KEYS = {"ANTHROPIC_API_KEY": "a", "GOOGLE_API_KEY": "g", "OPENAI_API_KEY": "o"}

# -- config -------------------------------------------------------------------


def test_a_non_object_payload_keeps_the_defaults() -> None:
    assert parse_brain(["not", "a", "dict"]) == default_brain()


def test_every_option_is_read_and_clamped() -> None:
    config = parse_brain(
        {
            "enabled": False,
            "routing": "manual",
            "defaultModel": "gpt" * 40,
            "council": "research",
            "search": "tavily",
            "monthlyBudgetUsd": 99_999,
            "budgetPolicy": "stop",
            "delivery": "interrupt",
            "onHangup": "cancel",
            "splitTasks": False,
            "executor": {
                "folder": str(EXECUTOR_ROOT / "proyecto"),
                "permissions": "read_only",
                "maxTurns": 0,
                "maxTaskUsd": 1000,
            },
        }
    )
    assert config.enabled is False
    assert config.routing == "manual"
    assert config.default_model == ("gpt" * 40)[:64]
    assert (config.council, config.search) == ("research", "tavily")
    assert config.monthly_budget_usd == 10_000.0
    assert (config.budget_policy, config.delivery, config.on_hangup) == (
        "stop",
        "interrupt",
        "cancel",
    )
    assert config.split is False
    assert config.executor.folder == str((EXECUTOR_ROOT / "proyecto").resolve())
    assert config.executor.permissions == "read_only"
    assert config.executor.max_turns == 1
    assert config.executor.max_task_usd == 50.0


def test_bad_values_fall_back_rather_than_fail() -> None:
    config = parse_brain(
        {
            "monthlyBudgetUsd": True,  # a bool is not a number here
            "models": "not a list",
            "executor": {"folder": 42, "permissions": "root", "maxTurns": "many"},
        }
    )
    defaults = default_brain()
    assert config.monthly_budget_usd == defaults.monthly_budget_usd
    assert [m.id for m in config.models] == [m.id for m in defaults.models]
    assert config.executor.folder == str(EXECUTOR_ROOT.resolve())
    assert config.executor.permissions == "full_no_push"
    assert config.executor.max_turns == defaults.executor.max_turns


def test_model_entries_are_validated_and_deduplicated() -> None:
    config = parse_brain(
        {
            "models": [
                "not an object",
                {"provider": "openai", "role": "expert", "model": "   "},
                {"provider": "openai", "role": "expert", "model": "gpt-5.2", "label": 7},
                {"provider": "google", "role": "search", "model": "gpt-5.2", "enabled": False},
                {"id": "flash", "provider": "google", "role": "search", "model": " gemini "},
            ]
        }
    )
    # No id: the model name is the id, so the second gpt-5.2 is a duplicate.
    assert [(m.id, m.model, m.label) for m in config.models] == [
        ("gpt-5.2", "gpt-5.2", ""),
        ("flash", "gemini", ""),
    ]


def test_an_id_taken_from_the_model_name_is_stripped_too() -> None:
    config = parse_brain(
        {
            "models": [
                {"provider": "openai", "role": "expert", "model": " gpt-5.2 "},
                {"provider": "openai", "role": "expert", "model": "gpt-5.2"},
            ]
        }
    )
    # The padded and the plain spelling are the same model, so one entry.
    assert [(m.id, m.model) for m in config.models] == [("gpt-5.2", "gpt-5.2")]


def test_a_folder_that_cannot_be_resolved_is_the_root() -> None:
    assert resolve_executor_folder("bad\x00path") == EXECUTOR_ROOT.resolve()
    assert resolve_executor_folder("") == EXECUTOR_ROOT.resolve()


# -- routing --------------------------------------------------------------------


def _model(mid: str, provider: str, role: str, **kw: Any) -> BrainModel:
    return BrainModel(mid, provider, f"{mid}-model", role, label=kw.pop("label", ""), **kw)  # type: ignore[arg-type]


def test_find_model_matches_exactly_then_loosely() -> None:
    models = default_brain().models
    assert find_model(models, "  ") is None
    assert find_model(models, "Claude Opus 5.5").id == "claude-opus"  # type: ignore[union-attr]
    assert find_model(models, "gemini").id == "gemini-flash"  # type: ignore[union-attr]
    assert find_model(models, "llama") is None


def test_manual_routing_sends_everything_but_code_to_the_default() -> None:
    config = default_brain()
    config.routing = "manual"
    config.default_model = "opus"
    assert pick(config, KEYS, "question", "simple").id == "claude-opus"  # type: ignore[union-attr]
    assert pick(config, KEYS, "code", "normal").id == "claude-code"  # type: ignore[union-attr]
    # A default naming the executor never takes non-code work.
    config.default_model = "claude code"
    assert pick(config, KEYS, "question", "simple").id == "claude-haiku"  # type: ignore[union-attr]


def test_no_fitting_model_is_none() -> None:
    config = BrainConfig(models=[_model("haiku", "anthropic", "fast")])
    assert pick(config, KEYS, "code", "normal") is None


def test_the_tavily_writer_prefers_the_tasks_role_then_anyone() -> None:
    config = BrainConfig(
        models=[
            _model("flash", "google", "search"),
            _model("opus", "anthropic", "expert"),
            _model("gpt", "openai", "fast"),
        ]
    )
    assert tavily_writer(config, KEYS, "research", "hard").id == "opus"  # type: ignore[union-attr]
    # A role nobody fills: the first Claude or GPT model that is left.
    only_code = BrainConfig(models=[_model("cc", "local", "code"), _model("gpt", "openai", "fast")])
    assert tavily_writer(only_code, KEYS, "code", "normal").id == "gpt"  # type: ignore[union-attr]
    assert tavily_writer(config, {"GOOGLE_API_KEY": "g"}, "research", "simple") is None


def test_a_council_without_flags_is_drawn_from_different_providers() -> None:
    config = BrainConfig(
        models=[
            _model("haiku", "anthropic", "fast"),
            _model("sonnet", "anthropic", "standard"),
            _model("opus", "anthropic", "expert"),
            _model("flash", "google", "search"),
            _model("gpt", "openai", "standard"),
            _model("cc", "local", "code"),
        ]
    )
    assert [m.id for m in council_members(config, KEYS)] == ["opus", "gpt", "flash"]


def test_a_council_of_one_provider_stays_small() -> None:
    config = BrainConfig(
        models=[_model("haiku", "anthropic", "fast"), _model("sonnet", "anthropic", "standard")]
    )
    assert [m.id for m in council_members(config, KEYS)] == ["sonnet"]


@pytest.mark.parametrize(
    ("policy", "kind", "important", "requested", "convenes"),
    [
        ("on_request", "analysis", False, True, True),
        ("on_request", "analysis", True, False, False),
        ("research", "research", False, False, True),
        ("research", "analysis", True, False, True),
        ("research", "analysis", False, False, False),
    ],
)
def test_when_the_council_meets(policy, kind, important, requested, convenes) -> None:
    config = default_brain()
    config.council = policy
    assert should_convene(config, kind, important, requested) is convenes


# -- planner ----------------------------------------------------------------------


def test_steps_without_a_goal_or_with_a_repeated_id_are_dropped() -> None:
    steps = parse_plan(
        '{"split": true, "steps": ['
        '{"id": "a", "goal": "uno", "difficulty": "extreme"},'
        '{"id": "a", "goal": "repetido"},'
        '{"id": "b", "goal": "  "},'
        '{"goal": "sin id"}]}'
    )
    assert [(s.id, s.goal, s.difficulty) for s in steps] == [
        ("a", "uno", "normal"),
        ("s2", "sin id", "normal"),
    ]


def test_waves_never_loop_on_a_cycle() -> None:
    cycle = [PlanStep("a", "x", "simple", False, ["b"]), PlanStep("b", "y", "simple", False, ["a"])]
    assert [[s.id for s in w] for w in waves(cycle)] == [["a", "b"]]


def test_a_single_step_needs_no_final_step() -> None:
    assert final_step([PlanStep("a", "x", "simple", False)]) is None


class FakePlanner:
    """Stands in for `anthropic.AsyncAnthropic` for one planning call."""

    reply: Any = None
    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs: Any) -> Any:
        FakePlanner.calls.append(kwargs)
        if isinstance(FakePlanner.reply, Exception):
            raise FakePlanner.reply
        return FakePlanner.reply


@pytest.fixture
def planner(monkeypatch: pytest.MonkeyPatch) -> type[FakePlanner]:
    FakePlanner.reply = None
    FakePlanner.calls = []
    monkeypatch.setattr(anthropic, "AsyncAnthropic", FakePlanner)
    return FakePlanner


def _planned(stop: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop,
        usage=SimpleNamespace(input_tokens=1000, output_tokens=None),
        content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=text)],
    )


async def test_the_planner_asks_for_structured_output(planner) -> None:
    planner.reply = _planned(
        "end_turn",
        '{"split": true, "steps": [{"id": "a", "goal": "dato", "difficulty": "simple",'
        ' "web": true, "after": []}, {"id": "b", "goal": "decide", "difficulty": "hard",'
        ' "web": false, "after": ["a"]}]}',
    )
    plan = await plan_task("k", "claude-sonnet-5-5", "¿Compro oro?")

    call = planner.calls[0]
    assert call["output_config"]["format"] == {"type": "json_schema", "schema": PLAN_SCHEMA}
    assert plan is not None and plan.split
    assert [s.id for s in plan.steps] == ["a", "b"]
    assert plan.cost_usd > 0


@pytest.mark.parametrize(
    ("stop", "text"),
    [("max_tokens", '{"split": true'), ("end_turn", "not json"), ("end_turn", "[]")],
)
async def test_an_unusable_plan_runs_the_task_whole(planner, stop: str, text: str) -> None:
    planner.reply = _planned(stop, text)
    plan = await plan_task("k", "claude-sonnet-5-5", "t")
    assert plan is not None and plan.steps == [] and plan.cost_usd > 0


async def test_a_failed_planner_call_is_no_plan(planner) -> None:
    planner.reply = anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))
    assert await plan_task("k", "claude-sonnet-5-5", "t") is None
