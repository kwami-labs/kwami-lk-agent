"""The voice-side brain tools: what they hand the bridge and what they say back."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from livekit.agents import StopResponse

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
    edits: list[tuple[str, str, str]] = field(default_factory=list)

    def start(self, **kwargs: Any) -> FakeTask | str:
        self.started.append(kwargs)
        if self.refuse:
            return self.refuse
        return FakeTask("t1", council=kwargs["council"])

    async def runner_online(self) -> bool:
        return True

    def cancel(self, task_id: str) -> bool:
        return task_id in self.tasks

    def status(self) -> dict[str, Any]:
        return {"type": "brain_status", "spentUsd": 1.5}

    def instructions(self) -> dict[str, Any]:
        return {"brain": "en euros", "models": {}}

    async def set_instructions(self, target: str, text: str, mode: str = "replace") -> str:
        self.edits.append((target, text, mode))
        return f"saved {target}"


@pytest.fixture
def agent() -> KwamiAgent:
    realtime = KwamiAgent()
    realtime.kwami_config.voice.pipeline_type = "realtime"
    return realtime


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
    assert status["instructions"] == {"brain": "en euros", "models": {}}
    assert status["note"]


async def test_brain_status_changes_instructions_when_asked(
    agent: KwamiAgent, bridge: FakeBridge
) -> None:
    status = await agent.brain_status(_ctx(bridge), "Gemini", "cita la fuente", "append")
    assert status["changed"] == "saved Gemini"
    assert bridge.edits == [("Gemini", "cita la fuente", "append")]


async def test_brain_status_only_reads_without_a_target(
    agent: KwamiAgent, bridge: FakeBridge
) -> None:
    status = await agent.brain_status(_ctx(bridge), "  ", "ignored")
    assert "changed" not in status
    assert bridge.edits == []


async def test_on_a_text_llm_the_tool_asks_for_no_second_reply(
    agent: KwamiAgent, bridge: FakeBridge
) -> None:
    agent.kwami_config.voice.pipeline_type = "standard"
    with pytest.raises(StopResponse):
        await agent.delegate_task(_ctx(bridge), "task", "title")
    assert bridge.started


async def test_a_realtime_model_is_still_told_to_say_it(
    agent: KwamiAgent, bridge: FakeBridge
) -> None:
    assert "say" in await agent.delegate_task(_ctx(bridge), "task", "title")


async def test_a_refusal_is_answered_on_a_text_llm_too(
    agent: KwamiAgent, bridge: FakeBridge
) -> None:
    agent.kwami_config.voice.pipeline_type = "standard"
    bridge.refuse = "No"
    assert await agent.delegate_task(_ctx(bridge), "task", "title") == {"error": "No"}


async def test_an_agent_named_by_the_user_goes_to_the_bridge(
    agent: KwamiAgent, bridge: FakeBridge
) -> None:
    await agent.delegate_task(_ctx(bridge), "revisa octubre", "Facturas", agent=" Facturas ")
    await agent.delegate_task(_ctx(bridge), "otra cosa", "Otra")
    assert bridge.started[0]["agent"] == "facturas"
    assert bridge.started[1]["agent"] == ""
