"""The brain's ledger when things go wrong: bad files, a slow or failing store.

The disk ledger is exercised on real files under tmp_path. The Supabase
ledger runs against `FakeStore`, a stand-in for our own `SupabaseClient` port
(its wire format is pinned with respx in `test_supabase_backends.py`); here
what matters is the cache, the ordering of loads and writes, and that nothing
raises into a live call.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from src.brain.ledger import Ledger, PendingResult, SupabaseLedger

USER = "11111111-1111-4111-8111-111111111111"
KWAMI = "22222222-2222-4222-8222-222222222222"


def _pending(task_id: str = "t1") -> PendingResult:
    return PendingResult(task_id, "Oro", "Sube", "2026-10-19T10:00:00Z", ["Opus"])


# -- disk -----------------------------------------------------------------------


def test_an_unreadable_spend_file_starts_afresh(tmp_path: Path, caplog) -> None:
    (tmp_path / "spend.json").write_text("{not json", encoding="utf-8")
    ledger = Ledger(tmp_path)
    with caplog.at_level(logging.WARNING):
        assert ledger.month_spend()["usd"] == 0.0
    assert "Unreadable brain ledger file" in caplog.text

    ledger.add_spend("claude-opus", 0.5)
    assert ledger.month_spend()["byModel"] == {"claude-opus": 0.5}


def test_writes_that_fail_are_logged_not_raised(tmp_path: Path, caplog) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    ledger = Ledger(blocker)
    with caplog.at_level(logging.ERROR):
        ledger.add_spend("claude-opus", 0.5)
        ledger.add_pending(KWAMI, _pending())
    assert "Could not record brain spend" in caplog.text
    assert "Could not store a pending brain result" in caplog.text


def test_a_pending_file_that_cannot_be_cleared_is_reported(tmp_path: Path, caplog) -> None:
    ledger = Ledger(tmp_path)
    # A directory where the file should be: unreadable and not unlinkable.
    ledger._pending_path(KWAMI).mkdir(parents=True)
    with caplog.at_level(logging.WARNING):
        assert ledger.take_pending(KWAMI) == []
    assert "Could not clear pending brain results" in caplog.text


def test_malformed_pending_entries_are_skipped(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path)
    path = ledger._pending_path("odd/kwami id")
    path.parent.mkdir(parents=True)
    good = {
        "task_id": "t1",
        "title": "Oro",
        "summary": "Sube",
        "finished_at": "",
        "models": [],
    }
    path.write_text(json.dumps(["text", {"unexpected": 1}, good]), encoding="utf-8")
    assert path.name == "odd_kwami_id.json"
    assert [p.task_id for p in ledger.take_pending("odd/kwami id")] == ["t1"]


# -- Supabase -------------------------------------------------------------------


class FakeStore:
    """Our `SupabaseClient` port: canned selects, recorded writes."""

    def __init__(self) -> None:
        self.rows: dict[str, list[dict[str, Any]] | None] = {
            "user_kwamis": [{"user_id": USER}],
            "brain_spend": [],
            "brain_pending": [],
        }
        self.gates: dict[str, asyncio.Event] = {}
        self.fail: set[str] = set()
        #: Calls that have started, whether or not a gate holds them.
        self.reached: defaultdict[str, asyncio.Event] = defaultdict(asyncio.Event)
        self.writes: list[tuple[str, str, Any]] = []

    async def _gate(self, name: str) -> None:
        self.reached[name].set()
        if name in self.gates:
            await self.gates[name].wait()
        if name in self.fail:
            raise RuntimeError(f"{name} exploded")

    async def select(self, table: str, params: dict[str, str], *, what: str = "") -> Any:
        await self._gate(table)
        return self.rows.get(table)

    async def insert(self, table: str, rows: Any, *, on_conflict: str = "", what: str = "") -> bool:
        self.writes.append(("insert", table, rows))
        return True

    async def update(
        self, table: str, values: dict[str, Any], params: dict[str, str], *, what: str = ""
    ) -> bool:
        self.writes.append(("update", table, params))
        return True

    async def rpc(self, function: str, args: dict[str, Any], *, what: str = "") -> Any:
        await self._gate(function)
        self.writes.append(("rpc", function, args))
        return 1


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


def _ledger(store: FakeStore) -> SupabaseLedger:
    return SupabaseLedger(store)  # type: ignore[arg-type]


async def test_spend_made_while_loading_is_not_lost(store: FakeStore) -> None:
    store.rows["brain_spend"] = [{"model_id": "claude-opus", "usd": 1.5}, {"usd": None}]
    ledger = _ledger(store)
    ledger.add_spend("claude-opus", 2.0)
    ledger.add_spend("claude-haiku", 0.1)

    await ledger.bind(KWAMI)
    await ledger.flush()

    # This process spent 2.0 on Opus; the read said 1.5. The larger one stays.
    assert ledger.month_spend()["byModel"] == {"claude-opus": 2.0, "": 0.0, "claude-haiku": 0.1}
    assert [w[2]["p_model_id"] for w in store.writes] == ["claude-opus", "claude-haiku"]


async def test_a_kwami_is_loaded_once(store: FakeStore) -> None:
    ledger = _ledger(store)
    await ledger.bind(KWAMI)
    store.rows["brain_pending"] = [{"id": "late", "title": "never read"}]
    await ledger.bind(KWAMI)
    assert ledger.take_pending(KWAMI) == []


async def test_an_owner_learned_later_has_their_spend_read(store: FakeStore) -> None:
    store.rows["user_kwamis"] = []
    store.rows["brain_spend"] = [{"model_id": "claude-opus", "usd": 3.0}]
    ledger = _ledger(store)
    await ledger.bind(KWAMI)
    assert ledger.user_id is None and ledger.month_spend()["usd"] == 0.0

    await ledger.bind("", user_id=USER)
    await ledger.flush()
    assert ledger.month_spend()["usd"] == 3.0


async def test_the_owner_arriving_mid_load_is_not_set_twice(store: FakeStore) -> None:
    store.gates["user_kwamis"] = asyncio.Event()
    ledger = _ledger(store)
    loading = asyncio.create_task(ledger.bind(KWAMI))
    # Until the load is parked on the owner lookup, having seen no owner yet.
    await store.reached["user_kwamis"].wait()

    await ledger.bind("", user_id=USER)
    store.gates["user_kwamis"].set()
    await loading
    await ledger.flush()
    assert ledger.user_id == USER


async def test_a_store_that_raises_leaves_the_ledger_usable(store: FakeStore, caplog) -> None:
    store.fail.add("brain_pending")
    ledger = _ledger(store)
    with caplog.at_level(logging.ERROR):
        await ledger.bind(KWAMI)
    assert "Could not load the brain ledger" in caplog.text
    assert ledger.user_id == USER


async def test_unreadable_spend_keeps_what_is_in_memory(store: FakeStore) -> None:
    store.rows["brain_spend"] = None
    ledger = _ledger(store)
    await ledger.bind(KWAMI)
    ledger.add_spend("claude-opus", 0.5)
    assert ledger.month_spend()["usd"] == 0.5


async def test_owed_rows_already_known_are_not_repeated(store: FakeStore) -> None:
    ledger = _ledger(store)
    ledger.add_pending(KWAMI, _pending("t1"))
    known = ledger._pending[KWAMI][0][0]
    store.rows["brain_pending"] = [
        {"id": known, "task_id": "t1"},
        {"task_id": "no id"},
        {"id": "row-2", "task_id": "t2", "models": None},
    ]
    await ledger.bind(KWAMI)
    assert [p.task_id for p in ledger.take_pending(KWAMI)] == ["t2", "t1"]


async def test_an_owed_result_without_an_owner_is_stored_without_one(store: FakeStore) -> None:
    store.rows["user_kwamis"] = []
    ledger = _ledger(store)
    await ledger.bind(KWAMI)
    ledger.add_pending(KWAMI, _pending())
    await ledger.flush()
    inserted = next(w[2] for w in store.writes if w[0] == "insert")
    assert "user_id" not in inserted


async def test_flush_with_nothing_in_flight_returns_at_once(store: FakeStore) -> None:
    await _ledger(store).flush(timeout=0)


async def test_a_write_that_does_not_land_in_time_is_reported(store: FakeStore, caplog) -> None:
    store.gates["brain_add_spend"] = asyncio.Event()
    ledger = _ledger(store)
    await ledger.bind(KWAMI)
    ledger.add_spend("claude-opus", 0.5)
    with caplog.at_level(logging.WARNING):
        await ledger.flush(timeout=0.01)
    assert "did not reach Supabase in time" in caplog.text
    store.gates["brain_add_spend"].set()
    await ledger.flush()


def test_a_write_with_no_event_loop_is_dropped(store: FakeStore, caplog) -> None:
    ledger = _ledger(store)
    with caplog.at_level(logging.WARNING):
        ledger.add_pending(KWAMI, _pending())
    assert "a brain ledger write was dropped" in caplog.text
    # The cache still has it: the greeting in this process can read it out.
    assert ledger._pending[KWAMI]


async def test_spend_resets_when_the_month_turns(store: FakeStore) -> None:
    ledger = _ledger(store)
    ledger.add_spend("claude-opus", 1.0)
    ledger._month = "1999-12"
    assert ledger.month_spend()["usd"] == 0.0
    assert ledger.month_spend()["month"] != "1999-12"
