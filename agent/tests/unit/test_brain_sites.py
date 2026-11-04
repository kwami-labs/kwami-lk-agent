"""Signing in to sites from the app, and looking after those sessions."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.brain import browser_tool, sites, worker
from src.brain.bridge import Bridge
from src.brain.browser_tool import CONNECTED, REFUSED, SIGN_IN_TOOL, SIGNED_OUT
from src.brain.jobs import BrainJob, JobStore
from src.brain.worker import Worker
from src.browser import local_chrome


class _Window:
    def __init__(self) -> None:
        self.active = [True, True, False]
        self.signing_in = [True, False, False]
        self.refused = False
        self.closed: list[bool] = []
        self.opened: list[str] = []
        self.fronted = False
        self.broken = False

    @property
    def is_active(self) -> bool:
        return self.active.pop(0) if len(self.active) > 1 else self.active[0]

    async def start(self, _profile: str, url: str) -> str:
        self.opened.append(url)
        return ""

    async def read_page(self) -> str:
        return "page"

    async def on_sign_in_page(self) -> bool:
        if self.broken:
            raise RuntimeError("gone")
        return self.signing_in.pop(0) if len(self.signing_in) > 1 else self.signing_in[0]

    async def sign_in_refused(self) -> bool:
        return self.refused

    async def bring_to_front(self) -> None:
        self.fronted = True

    async def close(self, *, quit_browser: bool = False) -> None:
        self.closed.append(quit_browser)


@pytest.fixture
def window(monkeypatch: pytest.MonkeyPatch) -> _Window:
    held = _Window()
    monkeypatch.setattr(browser_tool, "SIGN_IN_POLL_SECONDS", 0.001)
    monkeypatch.setattr(sites, "REDIRECT_WAIT_SECONDS", 0.001)
    monkeypatch.setattr(sites, "browser_available", lambda _profile: True)
    monkeypatch.setattr(sites, "window_open", _answer(False))
    original = browser_tool.BrowserTools.__init__

    def init(self: browser_tool.BrowserTools, user_id: str) -> None:
        original(self, user_id)
        self._session = held  # type: ignore[assignment]

    monkeypatch.setattr(browser_tool.BrowserTools, "__init__", init)
    return held


def _answer(value: Any) -> Any:
    async def answer(*_: Any, **__: Any) -> Any:
        return value

    return answer


async def test_a_site_stays_until_the_user_closes_it_and_tells_how_it_ended(
    window: _Window,
) -> None:
    states: list[str] = []

    async def note(state: str) -> None:
        states.append(state)

    said = await sites.show_site("https://github.com", "k1", note)
    assert "guardado" in said and states == [CONNECTED]
    assert window.opened == ["https://github.com"] and window.fronted
    assert window.closed == [False]


async def test_a_site_that_refuses_says_so(window: _Window) -> None:
    window.refused = True
    assert "no deja" in await sites.show_site("https://accounts.google.com", "k1")


async def test_a_site_closed_on_its_sign_in_page_is_not_connected(window: _Window) -> None:
    window.signing_in = [True]
    assert "sigue pidiendo" in await sites.show_site("https://a.b", "k1")


async def test_a_window_closed_between_two_looks_is_fine(window: _Window) -> None:
    window.signing_in = [True]
    window.active = [True]
    tools = browser_tool.BrowserTools("k1")
    looks = 0
    original = window.on_sign_in_page

    async def twice() -> bool:
        nonlocal looks
        looks += 1
        if looks > 1:
            raise RuntimeError("gone")
        return await original()

    window.on_sign_in_page = twice  # type: ignore[method-assign]
    assert await tools.wait_until_closed() == SIGNED_OUT


async def test_a_forgotten_window_is_let_go(
    window: _Window, monkeypatch: pytest.MonkeyPatch
) -> None:
    window.active = [True]
    monkeypatch.setattr(browser_tool, "SIGN_IN_WAIT_SECONDS", 0.005)
    await sites.show_site("https://github.com", "k1")
    assert window.closed == [False]


async def test_a_check_looks_and_closes_the_window_it_opened(window: _Window) -> None:
    window.signing_in = [False]
    assert await sites.check_site("https://a.b", "k1") == CONNECTED
    assert window.closed == [True]


async def test_a_check_notices_a_session_gone(window: _Window) -> None:
    window.signing_in = [True]
    assert await sites.check_site("https://a.b", "k1") == SIGNED_OUT


async def test_a_check_leaves_a_chrome_in_use_alone(
    window: _Window, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sites, "window_open", _answer(True))
    assert await sites.check_site("https://a.b", "k1") is None
    assert window.opened == []


def test_the_host_of_an_address() -> None:
    assert sites.host_of("https://GitHub.com/x") == "github.com"


async def test_whether_chrome_is_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_chrome, "profile_dir_for", lambda _user: tmp_path)
    assert await local_chrome.window_open("u") is False
    (tmp_path / "DevToolsActivePort").write_text("1234\n/x")
    monkeypatch.setattr(local_chrome, "_version_info", _answer({"Browser": "x"}))
    assert await local_chrome.window_open("u") is True


class _Jobs:
    thinking_ready = True

    def __init__(self, due: list[dict[str, str]] | None = None) -> None:
        self.due = due or []
        self.noted: list[tuple[str, str]] = []
        self.finished: list[str] = []

    async def due_sites(self, _user: str, _age: float) -> list[dict[str, str]]:
        return self.due

    async def note_site(self, _user: str, _kwami: str, url: str, state: str) -> bool:
        self.noted.append((url, state))
        return True

    async def update_running(self, _id: str, _progress: str) -> bool:
        return True

    async def get(self, _id: str) -> None:
        return None

    async def finish(self, _id: str, status: str, *_: Any) -> bool:
        self.finished.append(status)
        return True


def _worker(jobs: _Jobs, tmp_path: Path) -> Worker:
    return Worker(jobs, SimpleNamespace(), "", "user", folder=tmp_path)  # type: ignore[arg-type]


async def test_the_daily_round_records_each_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = _Jobs([{"url": "https://a.b", "kwami_id": "k"}])
    monkeypatch.setattr(worker, "check_site", _answer(SIGNED_OUT))
    await _worker(jobs, tmp_path)._check_sites()
    assert jobs.noted == [("https://a.b", SIGNED_OUT)]


async def test_the_round_stops_while_chrome_is_in_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = _Jobs([{"url": "https://a.b", "kwami_id": "k"}])
    monkeypatch.setattr(worker, "check_site", _answer(None))
    await _worker(jobs, tmp_path)._check_sites()
    assert jobs.noted == []


async def test_the_round_survives_a_failure_and_stops_with_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker, "SITE_POLL_SECONDS", 0.001)
    stop = asyncio.Event()
    rounds = 0

    async def check() -> None:
        nonlocal rounds
        rounds += 1
        if rounds == 2:
            stop.set()
        raise RuntimeError("down")

    w = _worker(_Jobs(), tmp_path)
    w._check_sites = check  # type: ignore[method-assign]
    await w._check_sites_forever(stop)
    assert rounds == 2


async def test_a_site_order_records_how_it_ended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def order(_mode: str, _task: str, _profile: str, note: Any) -> str:
        await note(REFUSED)
        return "No deja"

    monkeypatch.setattr(worker, "run_order", order)
    jobs = _Jobs()
    job = BrainJob("u", "k", "t", "https://a.b", mode="site")
    job.watcher_seen_at = "2999-01-01T00:00:00+00:00"
    await _worker(jobs, tmp_path)._carry_out(job)
    assert jobs.noted == [("https://a.b", REFUSED)] and jobs.finished == ["done"]


async def test_a_task_asked_to_sign_in_adds_the_site_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker, "RUNNING_POLL_SECONDS", 0.001)
    jobs = _Jobs()
    done = asyncio.Event()
    progress = [f"{SIGN_IN_TOOL}: github.com"]

    async def running() -> None:
        await done.wait()

    task = asyncio.create_task(running())
    asyncio.get_running_loop().call_later(0.02, done.set)
    job = BrainJob("u", "k", "t", "x")
    assert await _worker(jobs, tmp_path)._watch(job, task, progress) is False
    assert jobs.noted == [("https://github.com", SIGNED_OUT)]


class _Store:
    def __init__(self, rows: list[dict[str, str]] | None = None) -> None:
        self.rows = rows
        self.selected: list[tuple[str, dict[str, str]]] = []
        self.inserted: list[dict[str, str]] = []

    async def select(self, table: str, params: dict[str, str], **_: Any) -> Any:
        self.selected.append((table, params))
        return self.rows

    async def insert(self, _table: str, row: dict[str, str], **_: Any) -> bool:
        self.inserted.append(row)
        return True


async def test_the_store_reads_and_writes_the_list() -> None:
    store = _Store([{"host": "a.b"}])
    jobs = JobStore(store)  # type: ignore[arg-type]
    assert await jobs.sites("u") == [{"host": "a.b"}]
    assert await jobs.due_sites("u", 60.0) == [{"host": "a.b"}]
    assert store.selected[1][1]["state"] == "in.(connected,signed_out)"
    assert await jobs.note_site("u", "k", "https://GitHub.com/x", CONNECTED)
    assert store.inserted[0]["host"] == "github.com"
    assert store.inserted[0]["state"] == CONNECTED


async def test_the_store_survives_a_database_without_the_list() -> None:
    jobs = JobStore(_Store(None))  # type: ignore[arg-type]
    assert await jobs.sites("u") == []
    assert await jobs.due_sites("u", 60.0) == []


async def test_the_voice_hears_how_each_site_stands() -> None:
    site = {"host": "a.b", "url": "https://a.b", "state": CONNECTED}

    class Jobs:
        rows = [site]

        async def sites(self, _user: str) -> list[dict[str, str]]:
            return self.rows

    bridge = SimpleNamespace(jobs=None, user_id="u")
    ask = Bridge.sites_to_connect
    assert "from the app" in await ask(bridge)  # type: ignore[arg-type]
    bridge.jobs = Jobs()
    assert "a.b (https://a.b): connected" in await ask(bridge)  # type: ignore[arg-type]
    Jobs.rows = []
    assert "no sites" in await ask(bridge)  # type: ignore[arg-type]


async def test_stopping_during_the_wait_ends_the_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker, "SITE_POLL_SECONDS", 60.0)
    stop = asyncio.Event()
    running = asyncio.create_task(_worker(_Jobs(), tmp_path)._check_sites_forever(stop))
    await asyncio.sleep(0)
    stop.set()
    await asyncio.wait_for(running, 1.0)
