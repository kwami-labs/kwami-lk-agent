"""The user's agents, managed by voice: what may be saved, and what Supabase receives.

Until this, Jarvis could only hand work to an agent whose name it had to guess,
and every agent was made in the app. Now it lists them and saves them, with the
app's own rules -- and Bash left to the app, where the user writes it down.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import respx

from src.adapters.supabase import SupabaseClient
from src.brain.bridge import Bridge
from src.brain.jobs import MAX_USER_AGENTS, JobStore
from src.brain.team import HAIKU, agent_fields_from_voice
from src.settings import Settings
from src.tools import brain as brain_tools
from src.tools.brain import BrainToolsMixin

URL = "https://proj.supabase.co"
REST = f"{URL}/rest/v1"
USER = "11111111-1111-4111-8111-111111111111"


def _new(**extra: Any) -> dict[str, Any]:
    return {"name": "analista", "description": "Mercados", "prompt": "Analiza.", **extra}


# -- the rules --------------------------------------------------------------------


def test_a_complete_brief_becomes_the_columns() -> None:
    fields = agent_fields_from_voice(
        _new(
            name=" Analista ",
            model=HAIKU,
            tools=["Read", "WebSearch", "Read"],
            effort="high",
            target="cloud",
            enabled=False,
        )
    )
    assert fields == {
        "name": "analista",
        "description": "Mercados",
        "prompt": "Analiza.",
        "model": HAIKU,
        "tools": ["Read", "WebSearch"],
        "effort": "high",
        "target": "cloud",
        "enabled": False,
    }


def test_empty_values_leave_a_column_as_it_is() -> None:
    assert agent_fields_from_voice({"name": "analista", "tools": [], "enabled": "yes"}) == {
        "name": "analista"
    }


@pytest.mark.parametrize(
    ("raw", "said"),
    [
        ({"name": "Con Espacios"}, "lowercase"),
        ({"name": "x"}, "lowercase"),
        (_new(description="d" * 501), "longer than 500"),
        (_new(model="gpt-9"), "model must be"),
        (_new(tools=["Read", "Bash"]), "in the app"),
        (_new(tools=["Telepathy"]), "Unknown tools: Telepathy"),
        (_new(effort="max"), "effort must be"),
        (_new(target="moon"), "auto, pc or cloud"),
    ],
)
def test_what_cannot_be_saved_is_said(raw: dict[str, Any], said: str) -> None:
    refused = agent_fields_from_voice(raw)
    assert isinstance(refused, str) and said in refused


# -- on the wire --------------------------------------------------------------------


@pytest.fixture
def jobs() -> JobStore:
    return JobStore(SupabaseClient(URL, "sb_secret_test"))


@respx.mock
async def test_listing_reads_every_agent_of_the_user(jobs: JobStore) -> None:
    route = respx.get(f"{REST}/brain_agents").mock(
        return_value=httpx.Response(200, json=[{"name": "a", "enabled": False}])
    )
    assert await jobs.all_agents(USER) == [{"name": "a", "enabled": False}]
    params = route.calls[0].request.url.params
    assert params["user_id"] == f"eq.{USER}"
    # Switched-off agents too: the user asks about those as well.
    assert "enabled" not in params
    # No prompt: what the voice reads aloud is what each agent is for.
    assert "prompt" not in params["select"]


@respx.mock
async def test_a_new_agent_is_inserted_for_the_user(jobs: JobStore) -> None:
    respx.get(f"{REST}/brain_agents").mock(return_value=httpx.Response(200, json=[]))
    route = respx.post(f"{REST}/brain_agents").mock(return_value=httpx.Response(201))

    said = await jobs.save_agent(USER, {"name": "analista", "description": "d", "prompt": "p"})

    assert said == "Agent analista created."
    body = json.loads(route.calls[0].request.content)
    assert body["user_id"] == USER and body["name"] == "analista"
    assert body["created_at"] == body["updated_at"]


@respx.mock
async def test_a_new_agent_needs_its_brief(jobs: JobStore) -> None:
    respx.get(f"{REST}/brain_agents").mock(return_value=httpx.Response(200, json=[]))
    said = await jobs.save_agent(USER, {"name": "analista", "description": "d"})
    assert "needs a description" in said


@respx.mock
async def test_an_agent_that_exists_is_changed_in_place(jobs: JobStore) -> None:
    respx.get(f"{REST}/brain_agents").mock(
        return_value=httpx.Response(200, json=[{"id": "a1", "name": "analista"}])
    )
    route = respx.patch(f"{REST}/brain_agents").mock(return_value=httpx.Response(204))

    said = await jobs.save_agent(USER, {"name": "analista", "enabled": False})

    assert said == "Agent analista updated."
    params = route.calls[0].request.url.params
    # The owner in the filter as well as the id: never someone else's row.
    assert params["id"] == "eq.a1" and params["user_id"] == f"eq.{USER}"
    body = json.loads(route.calls[0].request.content)
    assert body["enabled"] is False and "name" not in body and body["updated_at"]


@respx.mock
async def test_nothing_to_change_writes_nothing(jobs: JobStore) -> None:
    respx.get(f"{REST}/brain_agents").mock(
        return_value=httpx.Response(200, json=[{"id": "a1", "name": "analista"}])
    )
    assert await jobs.save_agent(USER, {"name": "analista"}) == "Nothing to change in analista."


@respx.mock
async def test_the_ceiling_holds(jobs: JobStore) -> None:
    rows = [{"id": str(i), "name": f"a{i}x"} for i in range(MAX_USER_AGENTS)]
    respx.get(f"{REST}/brain_agents").mock(return_value=httpx.Response(200, json=rows))
    said = await jobs.save_agent(USER, {"name": "otro", "description": "d", "prompt": "p"})
    assert "already have" in said


@respx.mock
async def test_a_store_that_fails_is_said(jobs: JobStore) -> None:
    respx.get(f"{REST}/brain_agents").mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(200, json=[]),
            httpx.Response(200, json=[{"id": "a1", "name": "analista"}]),
        ]
    )
    respx.post(f"{REST}/brain_agents").mock(return_value=httpx.Response(500))
    respx.patch(f"{REST}/brain_agents").mock(return_value=httpx.Response(500))
    new = {"name": "analista", "description": "d", "prompt": "p"}
    assert "cannot be reached" in await jobs.save_agent(USER, new)
    assert await jobs.save_agent(USER, new) == "The agent was not saved."
    assert await jobs.save_agent(USER, {"name": "analista", "effort": "low"}) == (
        "The agent was not saved."
    )


# -- the bridge ----------------------------------------------------------------------


class _Jobs:
    def __init__(self, rows: list[dict[str, Any]] | None) -> None:
        self.rows = rows
        self.saved: list[tuple[str, dict[str, Any]]] = []

    async def all_agents(self, user_id: str) -> list[dict[str, Any]] | None:
        return self.rows

    async def save_agent(self, user_id: str, fields: dict[str, Any]) -> str:
        self.saved.append((user_id, fields))
        return "ok"


def _bridge(jobs: Any, user: str = USER) -> Bridge:
    bridge = Bridge(keys={}, settings=Settings())
    bridge.jobs = jobs
    bridge.user_id = user
    return bridge


async def test_the_bridge_lists_what_the_voice_may_read() -> None:
    rows = [{"name": "a", "description": "d", "prompt": "secret", "enabled": True, "id": "1"}]
    listed = await _bridge(_Jobs(rows)).list_agents()
    assert listed == {
        "agents": [
            {
                "name": "a",
                "description": "d",
                "model": None,
                "tools": None,
                "target": None,
                "enabled": True,
            }
        ]
    }


async def test_the_bridge_says_when_the_agents_are_out_of_reach() -> None:
    assert "error" in await _bridge(_Jobs(None)).list_agents()
    assert "error" in await _bridge(None).list_agents()
    assert "error" in await _bridge(_Jobs([]), user="").list_agents()


async def test_the_bridge_saves_for_the_rooms_user_only() -> None:
    jobs = _Jobs([])
    assert await _bridge(jobs).save_agent(_new()) == "ok"
    assert jobs.saved[0][0] == USER
    # On a phone line anyone can talk.
    assert "not by phone" in await _bridge(jobs, user="").save_agent(_new())
    assert "in the app" in await _bridge(jobs).save_agent(_new(tools=["Bash"]))
    assert len(jobs.saved) == 1


# -- the voice's tools ----------------------------------------------------------------


class _Bridge:
    def __init__(self) -> None:
        self.fields: dict[str, Any] = {}

    async def list_agents(self) -> dict[str, Any]:
        return {"agents": []}

    async def save_agent(self, fields: dict[str, Any]) -> str:
        self.fields = fields
        return "Agent analista created."


def _plain(tool: Any) -> Any:
    return getattr(tool, "__wrapped__", tool)


async def test_the_tools_go_through_the_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _Bridge()
    monkeypatch.setattr(brain_tools, "_bridge", lambda _context: fake)
    ctx = SimpleNamespace()
    assert await _plain(BrainToolsMixin.list_my_agents)(ctx, ctx) == {"agents": []}
    save = _plain(BrainToolsMixin.save_my_agent)
    assert await save(ctx, ctx, "analista", "d", "p") == "Agent analista created."
    assert "enabled" not in fake.fields and fake.fields["tools"] == []
    await save(ctx, ctx, "analista", enabled=False)
    assert fake.fields["enabled"] is False


async def test_without_a_brain_the_tools_say_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(brain_tools, "_bridge", lambda _context: None)
    ctx = SimpleNamespace()
    assert "error" in await _plain(BrainToolsMixin.list_my_agents)(ctx, ctx)
    assert "not connected" in await _plain(BrainToolsMixin.save_my_agent)(ctx, ctx, "a")
