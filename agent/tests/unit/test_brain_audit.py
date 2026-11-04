"""The record of what Jarvis does, the copies it leaves, and the agents it saves."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.brain import audit, team
from src.brain import worker as worker_module
from src.brain.jobs import BrainJob
from src.brain.worker import Worker


@pytest.fixture(autouse=True)
def places(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit, "AUDIT_DIR", tmp_path / "logs")
    monkeypatch.setattr(audit, "BACKUP_DIR", tmp_path / "backups")


def _lines(tmp_path: Path) -> list[dict[str, Any]]:
    log = next((tmp_path / "logs").glob("actions-*.jsonl"))
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def test_every_act_is_written_down(tmp_path: Path) -> None:
    audit.record("Bash", {"command": "npm test"}, None)
    audit.record("Read", {"file_path": "C:/x"}, "outside")
    audit.record("mcp__browser__read", {}, None)
    lines = _lines(tmp_path)
    assert lines[0]["tool"] == "Bash" and lines[0]["detail"] == "npm test"
    assert lines[1]["refused"] == "outside"
    assert lines[2]["detail"] == ""


def test_a_record_that_cannot_be_written_does_not_stop_the_work(tmp_path, monkeypatch) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setattr(audit, "AUDIT_DIR", blocker / "logs")
    audit.record("Bash", {"command": "x"}, None)


def test_a_file_is_copied_before_it_changes(tmp_path: Path) -> None:
    source = tmp_path / "web" / "index.html"
    source.parent.mkdir()
    source.write_text("antes")
    copy = audit.back_up(tmp_path / "root", "Edit", {"file_path": str(source)})
    assert copy is not None and copy.read_text() == "antes"


def test_nothing_is_copied_when_nothing_is_lost(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.txt").write_text("x")
    assert audit.back_up(root, "Edit", {"file_path": str(root / "a.txt")}) is None
    assert audit.back_up(root, "Read", {"file_path": str(tmp_path)}) is None
    assert audit.back_up(root, "Write", {"file_path": str(tmp_path / "new.txt")}) is None
    assert audit.back_up(root, "Write", {}) is None


def test_a_huge_file_is_left_out(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "big.bin"
    source.write_text("xx")
    monkeypatch.setattr(audit, "MAX_BACKUP_BYTES", 1)
    assert audit.back_up(tmp_path / "root", "Edit", {"file_path": str(source)}) is None


def test_a_copy_that_fails_does_not_stop_the_work(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "a.txt"
    source.write_text("x")
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setattr(audit, "BACKUP_DIR", blocker)
    assert audit.back_up(tmp_path / "root", "Edit", {"file_path": str(source)}) is None


def _save(root: Path, name: str, content: Any) -> None:
    folder = root / team.AGENTS_FOLDER
    folder.mkdir(exist_ok=True)
    text = content if isinstance(content, str) else json.dumps(content)
    (folder / f"{name}.json").write_text(text, encoding="utf-8")


def test_a_saved_agent_joins_the_team(tmp_path: Path) -> None:
    _save(
        tmp_path,
        "traductor",
        {
            "description": "Traduce",
            "prompt": "Traduce bien",
            "model": team.HAIKU,
            "tools": ["Read", "Write", "mcp__google__call", "Agent"],
            "effort": "low",
        },
    )
    member = team.own_members(tmp_path)["traductor"]
    assert member.model == team.HAIKU and member.effort == "low"
    assert member.tools == ["Read", "Write"]


def test_what_a_saved_agent_leaves_out_has_a_safe_default(tmp_path: Path) -> None:
    _save(tmp_path, "lector", {"description": "d", "prompt": "p", "model": "gpt", "effort": "x"})
    member = team.own_members(tmp_path)["lector"]
    assert member.model == team.SONNET and member.tools == team._READ
    assert member.effort == "medium"


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("reviewer", {"description": "d", "prompt": "p"}),
        ("mal", "{no es json"),
        ("lista", ["x"]),
        ("sinprompt", {"description": "d"}),
        ("x y", {"description": "d", "prompt": "p"}),
    ],
)
def test_a_bad_file_does_not_join(tmp_path: Path, name: str, content: Any) -> None:
    _save(tmp_path, name, content)
    assert team.own_members(tmp_path) == {}


def test_the_team_definitions_take_the_saved_ones(tmp_path: Path) -> None:
    _save(tmp_path, "lector", {"description": "d", "prompt": "p"})
    names = team.agent_definitions(lambda **kwargs: kwargs, tmp_path)  # type: ignore[arg-type]
    assert "lector" in names and "reviewer" in names
    assert "lector" not in team.agent_definitions(lambda **kwargs: kwargs)  # type: ignore[arg-type]


class _Jobs:
    thinking_ready = False

    def __init__(self, jobs: list[BrainJob]) -> None:
        self.jobs = jobs

    async def stale_jobs(self, user_id: str) -> list[BrainJob]:
        return []

    async def month_spend(self, user_id: str) -> float | None:
        return 0.0

    async def claim(self, user_id: str, runner_id: str, thinking: bool = False) -> BrainJob | None:
        return self.jobs.pop(0) if self.jobs else None


def _worker(jobs: list[BrainJob], tmp_path: Path) -> tuple[Worker, list[str], asyncio.Event]:
    started: list[str] = []
    release = asyncio.Event()
    worker = Worker(_Jobs(jobs), SimpleNamespace(), "", "user", folder=tmp_path)  # type: ignore[arg-type]

    async def process(job: BrainJob, spent: float) -> None:
        started.append(job.id)
        await release.wait()

    worker.process = process  # type: ignore[method-assign]
    return worker, started, release


async def test_a_slow_job_does_not_hold_back_the_next(tmp_path: Path) -> None:
    jobs = [BrainJob("u", "k", "t", "x", id=f"j{i}") for i in range(5)]
    worker, started, release = _worker(jobs, tmp_path)
    for _ in range(5):
        await worker.start_next()
    await asyncio.sleep(0)
    assert started == ["j0", "j1", "j2"], "only as many as there are slots"
    release.set()
    await asyncio.gather(*worker._working)
    assert await worker.start_next() is True


async def test_with_nothing_queued_nothing_starts(tmp_path: Path) -> None:
    worker, started, _ = _worker([], tmp_path)
    assert await worker.start_next() is False and started == []
    assert worker_module.WORK_SLOTS == 3
