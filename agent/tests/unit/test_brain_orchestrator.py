"""Claude directs, the other models work (brain/orchestrator.py), and the bridge's side."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import pytest

from src.brain import orchestrator as orch
from src.brain.bridge import BrainTask, Bridge
from src.brain.config import BrainModel, default_brain
from src.brain.ledger import Ledger, anthropic_cost
from src.brain.orchestrator import delegate_tool, describe_workers, orchestrate
from src.brain.providers import BrainAnswer, BrainRequest
from src.settings import Settings

from .test_brain_providers import FakeAnthropic, _http_error, _message, _text, _usage


@pytest.fixture
def fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> type[FakeAnthropic]:
    FakeAnthropic.script = []
    FakeAnthropic.calls = []
    FakeAnthropic.built = []
    monkeypatch.setattr(anthropic, "AsyncAnthropic", FakeAnthropic)
    return FakeAnthropic


OPUS = BrainModel("claude-opus", "anthropic", "claude-opus-5-5", "expert", "Claude Opus 5.5")
HAIKU = BrainModel("claude-haiku", "anthropic", "claude-haiku-5-5", "fast", "Claude Haiku 5.5")
FLASH = BrainModel("gemini-flash", "google", "gemini-3.5-flash", "search", "Gemini 3.5 Flash")
DEEP = BrainModel("deepseek", "deepseek", "deepseek-chat", "standard", "DeepSeek")


def _call(block_id: str, worker: str, instructions: str, web: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        type="tool_use",
        id=block_id,
        name="delegate",
        input={"worker": worker, "instructions": instructions, "web": web},
    )


class Workers:
    """Records each piece handed out, and answers it."""

    def __init__(self, answers: dict[str, Any] | None = None) -> None:
        self.answers = answers or {}
        self.asked: list[tuple[str, str, bool]] = []

    async def __call__(self, model: BrainModel, instructions: str, web: bool) -> BrainAnswer:
        self.asked.append((model.id, instructions, web))
        answer = self.answers.get(model.id)
        if isinstance(answer, BaseException):
            raise answer
        if callable(answer):
            return await answer()
        return answer or BrainAnswer(
            f"{model.id}: hecho", model.id, model.model, 0.002, sources=[f"https://{model.id}"]
        )


def _searches(model: BrainModel) -> bool:
    return model.provider in ("google", "anthropic")


async def _run(workers: Workers, team: list[BrainModel] | None = None) -> Any:
    return await orchestrate(
        "k", OPUS, team or [HAIKU, FLASH], "¿Compro oro?", "SYS", "hard", workers, _searches
    )


# -- the tool ---------------------------------------------------------------------------


def test_the_workers_are_described_cheapest_first() -> None:
    text = describe_workers([FLASH, HAIKU, DEEP], _searches)
    lines = text.splitlines()
    assert lines[0].startswith("- claude-haiku:")  # 0.10 / 0.50
    assert "searches the web" in lines[0]
    assert "no web search" in next(line for line in lines if line.startswith("- deepseek"))


def test_the_tool_only_names_real_workers() -> None:
    tool = delegate_tool([HAIKU, FLASH], _searches)
    schema = tool["input_schema"]
    assert schema["properties"]["worker"]["enum"] == ["claude-haiku", "gemini-flash"]
    assert tool["strict"] is True and schema["additionalProperties"] is False


# -- the loop ---------------------------------------------------------------------------


async def test_the_director_hands_pieces_out_at_once_and_writes_the_answer(
    fake_anthropic,
) -> None:
    first = _message(
        "tool_use",
        [
            _text("Reparto."),
            _call("t1", "gemini-flash", "precio del oro", True),
            _call("t2", "claude-haiku", "resume"),
        ],
        _usage(500, 80),
    )
    last = _message("end_turn", [_text("Compre poco.")], _usage(900, 120))
    fake_anthropic.script = [first, last]
    workers = Workers()

    result = await _run(workers)

    assert sorted(workers.asked) == [
        ("claude-haiku", "resume", False),
        ("gemini-flash", "precio del oro", True),
    ]
    second = fake_anthropic.calls[1]["messages"]
    # Both results go back in one message, in the order the calls were made.
    results = second[-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert all(r["is_error"] is False for r in results)
    # Opus takes the refusal fallback and the prompt cache.
    assert fake_anthropic.calls[0]["_beta"] is True
    assert fake_anthropic.calls[0]["cache_control"] == {"type": "ephemeral"}
    assert fake_anthropic.calls[0]["output_config"] == {"effort": "high"}
    answer = result.answer
    assert answer.ok and answer.text == "Compre poco."
    director = anthropic_cost("claude-opus-5-5", _usage(500, 80)) + anthropic_cost(
        "claude-opus-5-5", _usage(900, 120)
    )
    assert answer.cost_usd == pytest.approx(director + 0.004)
    assert [d.worker for d in result.delegations] == ["gemini-flash", "claude-haiku"]
    assert set(answer.sources) == {"https://gemini-flash", "https://claude-haiku"}


async def test_a_simple_task_needs_no_team(fake_anthropic) -> None:
    fake_anthropic.script = [_message("end_turn", [_text("Sí.")])]
    workers = Workers()
    result = await orchestrate("k", HAIKU, [FLASH], "¿2+2?", "SYS", "normal", workers, _searches)
    assert result.answer.text == "Sí." and workers.asked == []
    # Haiku takes no server-side fallback.
    assert "_beta" not in fake_anthropic.calls[0]


async def test_bad_calls_and_failed_workers_are_told_to_the_director(fake_anthropic) -> None:
    fake_anthropic.script = [
        _message(
            "tool_use",
            [
                _call("t1", "nobody", "x"),
                _call("t2", "claude-haiku", "   "),
                _call("t3", "gemini-flash", "falla"),
            ],
        ),
        _message("end_turn", [_text("Hecho igual.")]),
    ]
    failed = BrainAnswer("Gemini caído", "gemini-flash", "g", 0.0, ok=False)
    workers = Workers({"gemini-flash": failed})

    result = await _run(workers)

    results = fake_anthropic.calls[1]["messages"][-1]["content"]
    assert [r["is_error"] for r in results] == [True, True, True]
    assert results[2]["content"] == "Gemini caído"
    assert [d.ok for d in result.delegations] == [False]


async def test_a_non_object_input_is_a_bad_call(fake_anthropic) -> None:
    call = _call("t1", "claude-haiku", "x")
    call.input = "worker=claude-haiku"
    fake_anthropic.script = [
        _message("tool_use", [call]),
        _message("end_turn", [_text("Ok.")]),
    ]
    workers = Workers()
    await _run(workers)
    assert workers.asked == []


async def test_a_worker_that_hangs_is_reported_as_failed(fake_anthropic, monkeypatch) -> None:
    monkeypatch.setattr(orch, "WORKER_SECONDS", 0.05)

    async def hang() -> BrainAnswer:
        await asyncio.sleep(60)
        raise AssertionError("never")

    fake_anthropic.script = [
        _message("tool_use", [_call("t1", "claude-haiku", "x")]),
        _message("end_turn", [_text("Sin él.")]),
    ]
    result = await _run(Workers({"claude-haiku": hang}))
    results = fake_anthropic.calls[1]["messages"][-1]["content"]
    assert results[0]["is_error"] is True and "in time" in results[0]["content"]
    assert result.answer.ok


async def test_there_is_a_ceiling_on_pieces(fake_anthropic, monkeypatch) -> None:
    monkeypatch.setattr(orch, "MAX_DELEGATIONS", 1)
    fake_anthropic.script = [
        _message("tool_use", [_call("t1", "claude-haiku", "a"), _call("t2", "claude-haiku", "b")]),
        _message("end_turn", [_text("Ok.")]),
    ]
    workers = Workers()
    await _run(workers)
    assert len(workers.asked) == 1
    results = fake_anthropic.calls[1]["messages"][-1]["content"]
    assert "finish now" in results[1]["content"]


async def test_a_paused_turn_is_resumed(fake_anthropic) -> None:
    fake_anthropic.script = [
        _message("pause_turn", [_text("…")]),
        _message("end_turn", [_text("Listo.")]),
    ]
    result = await _run(Workers())
    assert result.answer.text == "Listo."
    assert len(fake_anthropic.calls) == 2


async def test_out_of_rounds_says_so(fake_anthropic, monkeypatch) -> None:
    monkeypatch.setattr(orch, "MAX_ROUNDS", 1)
    fake_anthropic.script = [_message("tool_use", [_call("t1", "claude-haiku", "a")])]
    result = await _run(Workers())
    assert "número de pasos" in result.answer.text
    assert result.answer.ok


async def test_a_refusal_is_a_failed_answer(fake_anthropic) -> None:
    fake_anthropic.script = [_message("refusal", [])]
    result = await _run(Workers())
    assert not result.answer.ok and "declinó" in result.answer.text


async def test_an_empty_answer_is_a_failed_one(fake_anthropic) -> None:
    fake_anthropic.script = [_message("end_turn", [])]
    result = await _run(Workers())
    assert not result.answer.ok and result.answer.text == "Sin contenido."


async def test_an_unreachable_director_lets_the_caller_choose(fake_anthropic) -> None:
    fake_anthropic.script = [
        _http_error(anthropic.InternalServerError, 503, "https://api.anthropic.com")
    ]
    assert await _run(Workers()) is None


async def test_no_round_at_all_is_a_failed_answer(fake_anthropic, monkeypatch) -> None:
    monkeypatch.setattr(orch, "MAX_ROUNDS", 0)
    result = await _run(Workers())
    assert not result.answer.ok


# -- the bridge -------------------------------------------------------------------------


KEYS = {"ANTHROPIC_API_KEY": "a", "GOOGLE_API_KEY": "g"}


def _bridge(tmp_path: Path, keys: dict[str, str] | None = None) -> Bridge:
    bridge = Bridge(
        keys=KEYS if keys is None else keys,
        ledger=Ledger(tmp_path),
        settings=Settings(executor_mode="local"),
    )
    bridge.config = default_brain()
    bridge.kwami_id = "k1"
    return bridge


def _task(difficulty: str = "hard", kind: str = "analysis") -> BrainTask:
    return BrainTask("t1", "Oro", "¿Compro oro?", kind, difficulty)


async def test_opus_directs_and_the_rest_of_the_roster_works(tmp_path, monkeypatch) -> None:
    bridge = _bridge(tmp_path)
    seen: dict[str, Any] = {}

    async def fake(api_key, director, workers, task, system, difficulty, delegate, searches, **kw):
        seen.update(director=director.id, workers=[w.id for w in workers])
        seen["web"] = {w.id: searches(w) for w in workers}
        piece = await delegate(workers[0], "parte", True)
        answer = BrainAnswer("Hecho.", director.id, director.model, 0.05 + piece.cost_usd)
        return orch.Orchestrated(answer, [orch.Delegation(workers[0].id, "parte", True, 0.01)])

    asked: list[BrainRequest] = []

    async def ask(model: BrainModel, request: BrainRequest) -> BrainAnswer:
        asked.append(request)
        return BrainAnswer("ok", model.id, model.model, 0.01)

    monkeypatch.setattr("src.brain.bridge.orchestrate", fake)
    bridge._ask = ask  # type: ignore[method-assign]
    task = _task()

    answer = await bridge._staged(task, "")

    assert answer.text == "Hecho."
    assert seen["director"] == "claude-opus"
    # Not the director itself, not the PC's executor.
    assert seen["workers"] == ["claude-haiku", "claude-sonnet", "gemini-flash"]
    assert seen["web"]["gemini-flash"] is True
    assert asked[0].native_search is True
    assert task.models == ["Claude Opus 5.5", "Claude Haiku 5.5"]
    assert task.steps[0]["model"] == "Claude Haiku 5.5"
    assert task.cost_usd == pytest.approx(0.06)
    # The director's own share goes to it; the worker's went through ask.
    assert bridge.ledger.month_spend()["byModel"]["claude-opus"] == pytest.approx(0.05)


async def test_a_piece_without_the_web_gets_no_search(tmp_path, monkeypatch) -> None:
    bridge = _bridge(tmp_path)
    bridge._tavily_key = "tv"
    asked: list[BrainRequest] = []

    async def ask(model: BrainModel, request: BrainRequest) -> BrainAnswer:
        asked.append(request)
        answer = BrainAnswer("lleno", model.id, model.model, 0.0, ok=False)
        answer.rate_limited = True
        return answer

    async def fake(api_key, director, workers, task, system, difficulty, delegate, searches, **kw):
        await delegate(workers[0], "parte", False)
        return orch.Orchestrated(BrainAnswer("x", director.id, director.model), [])

    monkeypatch.setattr("src.brain.bridge.orchestrate", fake)
    bridge._ask = ask  # type: ignore[method-assign]
    await bridge._staged(_task(), "")
    assert asked[0].native_search is False and asked[0].tavily_key == ""
    # A worker out of quota rests, as anywhere else.
    assert bridge._resting_ids() == {"claude-haiku"}


@pytest.mark.parametrize(
    ("difficulty", "kind", "prefer"),
    [
        ("simple", "analysis", ""),
        ("hard", "code", ""),
        ("hard", "x_search", ""),
        ("hard", "analysis", "opus"),
        # Writing and reading go straight to Sonnet: a director only adds its tokens.
        ("hard", "writing", ""),
        ("normal", "reading", ""),
    ],
)
async def test_what_the_orchestrator_does_not_take(
    tmp_path, monkeypatch, difficulty, kind, prefer
) -> None:
    bridge = _bridge(tmp_path)
    called = False

    async def fake(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True

    async def single(brain_task: BrainTask, prefer: str) -> BrainAnswer:
        return BrainAnswer("solo", "", "")

    monkeypatch.setattr("src.brain.bridge.orchestrate", fake)
    bridge._single = single  # type: ignore[method-assign]
    bridge.config.split = False
    await bridge._staged(_task(difficulty, kind), prefer)
    assert called is False


async def test_it_steps_aside_when_it_cannot_run(tmp_path, monkeypatch) -> None:
    async def fake(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not run")

    monkeypatch.setattr("src.brain.bridge.orchestrate", fake)
    # No Anthropic key: the director is not Claude on a key.
    assert await _bridge(tmp_path, {"GOOGLE_API_KEY": "g"})._orchestrated(_task()) is None
    # Nobody to hand work to.
    lonely = _bridge(tmp_path)
    lonely.config.models = [OPUS]
    assert await lonely._orchestrated(_task()) is None
    # The PC's account answers for Opus: it has no tool loop.
    on_account = _bridge(tmp_path, {**KEYS, "@claude-account": "1"})
    assert await on_account._orchestrated(_task()) is None


async def test_a_director_that_cannot_be_reached_hands_back_to_the_plan(
    tmp_path, monkeypatch
) -> None:
    bridge = _bridge(tmp_path)

    async def unreachable(*args: Any, **kwargs: Any) -> None:
        return None

    async def single(brain_task: BrainTask, prefer: str) -> BrainAnswer:
        return BrainAnswer("por el plan", "", "")

    monkeypatch.setattr("src.brain.bridge.orchestrate", unreachable)
    bridge._single = single  # type: ignore[method-assign]
    bridge.config.split = False
    assert (await bridge._staged(_task(), "")).text == "por el plan"


def test_a_worker_searches_on_its_own_or_through_tavily(tmp_path) -> None:
    bridge = _bridge(tmp_path)
    assert bridge._searches(FLASH) is True
    assert bridge._searches(DEEP) is False
    bridge._tavily_key = "tv"
    assert bridge._searches(DEEP) is True
    bridge.config.search = "tavily"
    assert bridge._searches(FLASH) is True
    bridge._tavily_key = ""
    assert bridge._searches(FLASH) is False


def test_the_app_can_switch_it_off() -> None:
    from src.brain.config import parse_brain

    assert parse_brain({}).orchestrate is True
    assert parse_brain({"orchestrate": False}).orchestrate is False


async def test_past_the_tasks_budget_no_more_pieces_go_out(fake_anthropic) -> None:
    fake_anthropic.script = [
        _message("tool_use", [_call("t1", "claude-haiku", "a")]),
        _message("tool_use", [_call("t2", "claude-haiku", "b")]),
        _message("end_turn", [_text("Con lo que hay.")]),
    ]
    workers = Workers({"claude-haiku": BrainAnswer("caro", "claude-haiku", "h", 1.5)})

    result = await orchestrate(
        "k", OPUS, [HAIKU], "T", "SYS", "hard", workers, _searches, budget_usd=1.0
    )

    assert len(workers.asked) == 1
    refused = fake_anthropic.calls[2]["messages"][-1]["content"][0]
    assert refused["is_error"] is True and "budget" in refused["content"]
    assert result.answer.text == "Con lo que hay."


async def test_the_bridge_gives_the_director_the_tasks_ceiling(tmp_path, monkeypatch) -> None:
    bridge = _bridge(tmp_path)
    bridge.config.executor.max_task_usd = 0.75
    seen: dict[str, Any] = {}

    async def fake(*args: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return orch.Orchestrated(BrainAnswer("x", "claude-opus", "o"), [])

    monkeypatch.setattr("src.brain.bridge.orchestrate", fake)
    await bridge._staged(_task(), "")
    assert seen["budget_usd"] == 0.75


def test_each_worker_says_what_it_is_best_at() -> None:
    from src.brain.orchestrator import specialty

    sonnet = BrainModel("claude-sonnet", "anthropic", "claude-sonnet-5-5", "standard")
    assert "lookups" in specialty(HAIKU)
    assert "writing" in specialty(sonnet) and "reading" in specialty(sonnet)
    assert "reasoning" in specialty(OPUS)
    assert "cheap drafts" in specialty(DEEP)
    unknown = BrainModel("x", "openrouter", "someone/new-model", "fast", roles=["fast"])
    assert specialty(unknown) == "fast"
    text = describe_workers([HAIKU, sonnet], _searches)
    assert "best for web lookups" in text and "best for writing and drafting" in text


# -- the user's instructions reach the orchestrator's director and workers -----------


def test_the_director_reads_what_the_user_asked_of_each_worker() -> None:
    briefed = BrainModel(
        "gemini-flash",
        "google",
        "gemini-3.5-flash",
        "search",
        "Gemini 3.5 Flash",
        instructions="Cita  siempre\nla fuente.",
    )
    text = describe_workers([briefed, HAIKU], _searches)
    assert "The user's instructions for it: Cita siempre la fuente." in text
    haiku_line = next(line for line in text.splitlines() if "claude-haiku" in line)
    assert "instructions" not in haiku_line


async def test_an_orchestrated_task_lists_who_got_the_instructions(tmp_path, monkeypatch) -> None:
    bridge = _bridge(tmp_path)
    bridge.config.instructions = "En euros."

    async def fake(api_key, director, workers, task, system, difficulty, delegate, searches, **kw):
        assert "En euros." in system
        await delegate(workers[0], "parte", False)
        return orch.Orchestrated(BrainAnswer("x", director.id, director.model), [])

    async def ask(model: BrainModel, request: BrainRequest) -> BrainAnswer:
        answer = BrainAnswer("ok", model.id, model.model)
        answer.briefed = True
        return answer

    monkeypatch.setattr("src.brain.bridge.orchestrate", fake)
    bridge._ask = ask  # type: ignore[method-assign]
    task = _task()

    await bridge._staged(task, "")

    assert task.briefed == ["Claude Opus 5.5", "Claude Haiku 5.5"]
