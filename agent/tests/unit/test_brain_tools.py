"""The voice-side brain tools: what they hand the bridge and what they say back."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from src.agent import KwamiAgent
from src.brain.config import default_brain
from src.runtime.container import AgentDeps


class Ctx:
    def __init__(self, deps: AgentDeps) -> None:
        self.userdata = deps


@dataclass
class FakeTask:
    id: str
    council: bool = False

    def public(self, full: bool = False) -> dict[str, Any]:
        return {"id": self.id, "full": full}


@dataclass
class FakeBridge:
    refuse: str = ""
    started: list[dict[str, Any]] = field(default_factory=list)
    tasks: dict[str, FakeTask] = field(default_factory=dict)
    config: Any = field(default_factory=default_brain)

    def start(self, **kwargs: Any) -> FakeTask | str:
        self.started.append(kwargs)
        if self.refuse:
            return self.refuse
        return FakeTask("t1", council=kwargs["council"])

    def cancel(self, task_id: str) -> bool:
        return task_id in self.tasks

    def status(self) -> dict[str, Any]:
        return {"type": "brain_status", "spentUsd": 1.5}


@pytest.fixture
def agent() -> KwamiAgent:
    return KwamiAgent()


@pytest.fixture
def bridge() -> FakeBridge:
    return FakeBridge(tasks={"t1": FakeTask("t1"), "t2": FakeTask("t2")})


def _ctx(bridge: FakeBridge | None) -> Ctx:
    return Ctx(AgentDeps(bridge=bridge))


async def test_every_tool_degrades_without_a_bridge(agent: KwamiAgent) -> None:
    ctx = _ctx(None)
    assert "error" in await agent.delegate_task(ctx, "task", "title")
    assert "error" in await agent.check_tasks(ctx)
    assert "not connected" in await agent.cancel_task(ctx, "t1")
    assert "error" in await agent.brain_status(ctx)


async def test_delegate_normalises_its_arguments(agent: KwamiAgent, bridge: FakeBridge) -> None:
    result = await agent.delegate_task(
        _ctx(bridge),
        "  busca vuelos  ",
        " Vuelos a Roma ",
        kind="nonsense",
        difficulty="extreme",
        council=True,
        model=" opus ",
        context_notes=" en marzo ",
    )

    assert result["taskId"] == "t1"
    assert result["status"] == "running"
    assert result["council"] is True
    sent = bridge.started[0]
    assert sent["task"] == "busca vuelos"
    assert sent["title"] == "Vuelos a Roma"
    assert sent["kind"] == "question"
    assert sent["difficulty"] == "normal"
    assert sent["model"] == "opus"
    assert sent["context"] == "en marzo"


async def test_delegate_relays_a_refusal(agent: KwamiAgent) -> None:
    result = await agent.delegate_task(_ctx(FakeBridge(refuse="Sin presupuesto.")), "t", "t")
    assert result == {"error": "Sin presupuesto."}


async def test_check_tasks_lists_or_details(agent: KwamiAgent, bridge: FakeBridge) -> None:
    ctx = _ctx(bridge)
    assert (await agent.check_tasks(ctx))["tasks"] == [
        {"id": "t1", "full": False},
        {"id": "t2", "full": False},
    ]
    assert await agent.check_tasks(ctx, " t2 ") == {"id": "t2", "full": True}
    assert "error" in await agent.check_tasks(ctx, "missing")


async def test_cancel_task(agent: KwamiAgent, bridge: FakeBridge) -> None:
    ctx = _ctx(bridge)
    assert await agent.cancel_task(ctx, " t1 ") == "Cancelled."
    assert "No running task" in await agent.cancel_task(ctx, "missing")


async def test_brain_status_adds_the_roster(agent: KwamiAgent, bridge: FakeBridge) -> None:
    status = await agent.brain_status(_ctx(bridge))

    assert "type" not in status
    assert status["spentUsd"] == 1.5
    assert {m["id"] for m in status["roster"]} == {m.id for m in bridge.config.models}
    assert status["note"]
