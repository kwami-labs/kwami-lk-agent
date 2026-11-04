"""Merging on the user's own words, and Claude's sign-in in a window of its own."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.brain import orders
from src.brain.jobs import BrainJob
from src.brain.selfwork import SelfWorkError
from src.brain.worker import Worker


@pytest.mark.parametrize(
    "words",
    ["ya lo he probado, fusiónalo", "Lo probé y funciona, haz el merge", "funciona, fusiona"],
)
def test_having_tried_it_and_saying_merge_is_enough(words: str) -> None:
    assert orders.merge_approved(words, f"Vale. {words}. Gracias")


@pytest.mark.parametrize(
    ("words", "turn"),
    [
        ("fusiónalo", "fusiónalo"),
        ("lo he probado", "lo he probado"),
        ("no lo he probado, fusiona", "no lo he probado, fusiona"),
        ("fusiona sin probar", "fusiona sin probar"),
        ("lo probé, no lo fusiones", "lo probé, no lo fusiones"),
        ("lo probé, fusiona", "qué tiempo hace"),
        ("", "lo probé, fusiona"),
    ],
)
def test_anything_else_is_not_approval(words: str, turn: str) -> None:
    assert not orders.merge_approved(words, turn)


class _Gh:
    """The `gh` the orders run: what it was asked, and the pull request it reports."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.pull: dict[str, Any] = {
            "headRefName": "jarvis/abc",
            "baseRefName": "main",
            "state": "OPEN",
            "isDraft": False,
        }

    async def must(self, step: str, *args: str, **kwargs: Any) -> str:
        self.calls.append(args)
        return json.dumps(self.pull) if args[:3] == ("gh", "pr", "view") else ""


@pytest.fixture
def gh(monkeypatch: pytest.MonkeyPatch) -> _Gh:
    fake = _Gh()
    monkeypatch.setattr(orders, "_must", fake.must)
    monkeypatch.setattr(orders, "REPOS", {"agent": SimpleNamespace(slug="me/agent")})
    return fake


async def test_it_merges_its_own_pull_request(gh) -> None:
    text = await orders.run_order(orders.MERGE, "agent#7")
    assert "Fusionado" in text
    assert gh.calls[-1] == ("gh", "pr", "merge", "7", "--repo", "me/agent", "--merge")


@pytest.mark.parametrize(
    "extra",
    [
        {"headRefName": "feat/other"},
        {"baseRefName": "dev"},
        {"state": "MERGED"},
        {"isDraft": True},
    ],
)
async def test_it_merges_nothing_else(gh, extra: dict[str, Any]) -> None:
    gh.pull.update(extra)
    with pytest.raises(SelfWorkError):
        await orders.merge_pull_request("agent#7")
    assert not any(call[:3] == ("gh", "pr", "merge") for call in gh.calls)


@pytest.mark.parametrize("order", ["web#7", "agent#x", "agent"])
async def test_an_unknown_order_stops_there(gh, order: str) -> None:
    with pytest.raises(SelfWorkError):
        await orders.merge_pull_request(order)
    assert gh.calls == []


async def test_signing_in_opens_the_official_flow_without_the_api_key(monkeypatch) -> None:
    opened: list[Any] = []
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(orders, "binary", lambda: "claude.exe")
    monkeypatch.setattr(
        subprocess, "Popen", lambda args, env, creationflags: opened.append((args, env))
    )
    assert "Claude" in await orders.run_order(orders.LOGIN, "login")
    assert opened[0][0] == ["claude.exe", "auth", "login"]
    assert "ANTHROPIC_API_KEY" not in opened[0][1]


def test_signing_in_without_claude_code_says_so(monkeypatch) -> None:
    monkeypatch.setattr(orders, "binary", lambda: None)
    with pytest.raises(SelfWorkError):
        orders.open_login_window()


class _Jobs:
    thinking_ready = True

    def __init__(self) -> None:
        self.finished: list[tuple[str, str]] = []

    async def finish(self, job_id, status, text, cost, session) -> bool:
        self.finished.append((status, text))
        return True


def _worker(jobs: _Jobs, tmp_path: Path) -> Worker:
    return Worker(jobs, SimpleNamespace(), "", "user", folder=tmp_path)  # type: ignore[arg-type]


def _order(watched: bool) -> BrainJob:
    seen = "2999-01-01T00:00:00+00:00" if watched else ""
    return BrainJob("u", "k", "t", "agent#1", id="j", mode=orders.MERGE, watcher_seen_at=seen)


@pytest.mark.parametrize(
    ("outcome", "status", "said"),
    [
        (lambda: "Hecho", "done", "Hecho"),
        (lambda: (_ for _ in ()).throw(SelfWorkError("No")), "failed", "No"),
        (lambda: (_ for _ in ()).throw(ValueError("x")), "failed", "ValueError"),
    ],
)
async def test_the_worker_carries_out_an_order(
    monkeypatch, tmp_path, outcome, status, said
) -> None:
    async def fake(mode: str, order: str, profile: str = "", note: Any = None) -> str:
        return outcome()

    monkeypatch.setattr("src.brain.worker.run_order", fake)
    jobs = _Jobs()
    await _worker(jobs, tmp_path).think(_order(True))
    assert jobs.finished[0][0] == status and said in jobs.finished[0][1]


async def test_an_order_from_a_call_that_is_gone_does_nothing(monkeypatch, tmp_path) -> None:
    async def fake(mode: str, order: str, profile: str = "", note: Any = None) -> str:
        raise AssertionError("it must not run")

    monkeypatch.setattr("src.brain.worker.run_order", fake)
    jobs = _Jobs()
    await _worker(jobs, tmp_path).think(_order(False))
    assert jobs.finished[0][0] == "cancelled"


async def test_a_site_opens_in_the_given_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    shown: list[tuple[str, str]] = []

    async def show(url: str, profile: str, _note: Any = None) -> str:
        shown.append((url, profile))
        return "Listo"

    monkeypatch.setattr(orders, "browser_available", lambda _profile: True)
    monkeypatch.setattr(orders, "show_site", show)
    assert await orders.run_order(orders.SITE, "https://github.com", "k1") == "Listo"
    assert shown == [("https://github.com", "k1")]


async def test_without_the_browser_a_site_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orders, "browser_available", lambda _profile: False)
    with pytest.raises(SelfWorkError, match="navegador"):
        await orders.run_order(orders.SITE, "https://github.com", "k1")


async def test_the_worker_can_stop_a_site_left_open(monkeypatch, tmp_path) -> None:
    profiles: list[str] = []
    stopped = asyncio.Event()

    async def fake(mode: str, order: str, profile: str = "", note: Any = None) -> str:
        profiles.append(profile)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            stopped.set()
            raise
        return ""

    async def cancelled(_job: Any, _running: Any, _progress: Any) -> bool:
        await asyncio.sleep(0)
        _running.cancel()
        await asyncio.gather(_running, return_exceptions=True)
        return True

    monkeypatch.setattr("src.brain.worker.run_order", fake)
    jobs = _Jobs()
    worker = _worker(jobs, tmp_path)
    worker._watch = cancelled  # type: ignore[method-assign]
    await worker.think(_order(True))
    assert jobs.finished[0][0] == "cancelled" and stopped.is_set()
    assert profiles
