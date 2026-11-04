"""What the work thread may do in the Kwami database, and what it may not."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.brain import supabase_tool
from src.brain.executor import _Lanes, check_tool_call
from src.brain.supabase_tool import ALLOWED_TOOL, call_database, supabase_available


class _Store:
    def __init__(self, *, works: bool = True) -> None:
        self.works = works
        self.calls: list[tuple[Any, ...]] = []

    async def select(self, table: str, params: dict[str, str], **_: Any) -> Any:
        self.calls.append(("select", table, params))
        return [{"id": 1}] if self.works else None

    async def insert(self, table: str, rows: Any, **_: Any) -> bool:
        self.calls.append(("insert", table, rows))
        return self.works

    async def update(self, table: str, values: Any, params: dict[str, str], **_: Any) -> bool:
        self.calls.append(("update", table, values, params))
        return self.works


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> _Store:
    held = _Store()
    monkeypatch.setattr(supabase_tool, "get_supabase", lambda _settings: held)
    return held


async def test_reading_caps_the_rows(store: _Store) -> None:
    answer = await call_database("select", {"table": "profiles", "params": {"limit": "9999"}})
    assert json.loads(answer) == [{"id": 1}]
    assert store.calls[0][2]["limit"] == "500"


async def test_reading_has_a_default_limit(store: _Store) -> None:
    await call_database("select", {"table": "profiles"})
    assert store.calls[0][2]["limit"] == "100"


async def test_writing_works_on_ordinary_tables(store: _Store) -> None:
    assert await call_database("insert", {"table": "kwami_programs", "rows": [{"a": 1}]}) == "Done."
    said = await call_database(
        "update", {"table": "profiles", "values": {"a": 1}, "params": {"id": "eq.1"}}
    )
    assert said == "Done."


async def test_an_update_needs_a_filter(store: _Store) -> None:
    said = await call_database("update", {"table": "profiles", "values": {"a": 1}})
    assert "filter" in said and store.calls == []


@pytest.mark.parametrize("table", ["kwami_google_accounts", "transcript_turns", "kwamis"])
async def test_keys_and_what_people_wrote_stay_shut(store: _Store, table: str) -> None:
    assert "shut" in await call_database("select", {"table": table})
    assert store.calls == []


@pytest.mark.parametrize(
    "select",
    [
        "*,kwami_provider_keys(*)",
        "id,keys:kwami_provider_keys(value)",
        "id,kwami_secrets!inner(*)",
        "...kwami_secrets(*)",
    ],
)
async def test_a_shut_table_cannot_be_read_through_a_related_one(
    store: _Store, select: str
) -> None:
    said = await call_database("select", {"table": "user_kwamis", "params": {"select": select}})
    assert said.startswith("Refused") and store.calls == []


@pytest.mark.parametrize(
    "table", ["user_credits", "brain_spend", "brain_jobs", "wallet_identities"]
)
async def test_money_and_limits_are_read_only(store: _Store, table: str) -> None:
    assert "only be read" in await call_database("insert", {"table": table, "rows": []})
    assert "only be read" in await call_database(
        "update", {"table": table, "values": {}, "params": {"id": "eq.1"}}
    )
    assert store.calls == []
    assert await call_database("select", {"table": table})


@pytest.mark.parametrize("action", ["delete", "rpc", ""])
async def test_nothing_deletes(store: _Store, action: str) -> None:
    assert "Refused" in await call_database(action, {"table": "profiles"})
    assert store.calls == []


@pytest.mark.parametrize("table", ["", "auth.users", "a;drop", "../x", "UPPER"])
async def test_only_plain_table_names(store: _Store, table: str) -> None:
    assert "not a table" in await call_database("select", {"table": table})


async def test_a_refusal_from_the_database_is_said(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(supabase_tool, "get_supabase", lambda _settings: _Store(works=False))
    assert "refused" in await call_database("select", {"table": "profiles"})
    assert "nothing changed" in await call_database("insert", {"table": "profiles", "rows": []})


async def test_without_a_database_it_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(supabase_tool, "get_supabase", lambda _settings: None)
    assert "not configured" in await call_database("select", {"table": "profiles"})


def test_it_is_off_unless_switched_on_the_pc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(supabase_tool, "get_supabase", lambda _settings: object())
    monkeypatch.setattr(supabase_tool, "executor_in_cloud", lambda _mode: False)
    monkeypatch.delenv(supabase_tool.SWITCH, raising=False)
    assert not supabase_available("u")
    monkeypatch.setenv(supabase_tool.SWITCH, "on")
    assert supabase_available("u")
    assert not supabase_available("")
    monkeypatch.setattr(supabase_tool, "executor_in_cloud", lambda _mode: True)
    assert not supabase_available("u")


def test_the_lanes_keep_it_away_from_the_open_world() -> None:
    lane = _Lanes(resuming=False)
    assert lane.refuse(ALLOWED_TOOL) is None
    assert "closed" in (lane.refuse("Bash") or "")
    assert "closed" in (lane.refuse("mcp__browser__open") or "")
    assert lane.closed_session
    opened = _Lanes(resuming=False)
    assert opened.refuse("Bash") is None
    assert "database" in (opened.refuse(ALLOWED_TOOL) or "")


def test_the_guard_lets_it_through(tmp_path: Path) -> None:
    assert check_tool_call(tmp_path, ALLOWED_TOOL, {"action": "select"}) is None


def test_the_server_offers_the_tool() -> None:
    held: dict[str, Any] = {}

    def tool(name: str, _description: str, _schema: Any) -> Any:
        def wrap(function: Any) -> Any:
            held[name] = function
            return function

        return wrap

    server = supabase_tool.supabase_server(tool, lambda **kwargs: kwargs)
    assert server["name"] == supabase_tool.SERVER_NAME and held


async def test_a_bad_limit_is_a_refusal_not_a_crash(store: _Store) -> None:
    held: dict[str, Any] = {}

    def tool(name: str, _description: str, _schema: Any) -> Any:
        def wrap(function: Any) -> Any:
            held[name] = function
            return function

        return wrap

    supabase_tool.supabase_server(tool, lambda **kwargs: kwargs)
    answer = await held[supabase_tool.TOOL_NAME](
        {"action": "select", "args": {"table": "profiles", "params": {"limit": "x"}}}
    )
    assert "Refused" in answer["content"][0]["text"]
    answer = await held[supabase_tool.TOOL_NAME](
        {"action": "select", "args": {"table": "profiles"}}
    )
    assert "id" in answer["content"][0]["text"]
