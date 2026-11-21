"""What the app's HUD is told as Jarvis works, and the clean window for signing in.

Two things the user asked for on 2026-11-16: to see Jarvis work (each tool call, each
step of a task, the whole report at the end) and to be able to sign in to sites that
refuse the window Jarvis drives ("this browser or app may not be secure").
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.brain import bridge as bridge_module
from src.brain import executor
from src.brain.bridge import ACTIVITY_KEPT, ACTIVITY_SHOWN, BrainTask
from src.browser import browser_session, local_chrome
from src.browser.browser_session import CloudBrowserSession
from src.browser.launched import LOCAL, BrowserVendorError
from src.browser.local_chrome import LocalChromeProvider, SignInWindowOpenError
from src.runtime import dispatch
from src.runtime.action_feed import ActionFeed, detail_of
from src.settings import Settings
from src.tools import builtin

from .test_brain_bridge import FakePublisher, _bridge, _finish
from .test_builtin_navigation_tools import _tools
from .test_runtime_dispatch import FakeState, _router

# -- a task's live log --------------------------------------------------------------


def _task() -> BrainTask:
    return BrainTask(id="t1", title="Informe", task="t", kind="research", difficulty="normal")


def test_a_task_keeps_each_new_line_once() -> None:
    task = _task()
    assert task.note("  Bash:   npm   test ") is True
    assert task.note("Bash: npm test") is False, "the same line twice is one line"
    assert task.note("   ") is False
    assert [line["text"] for line in task.activity] == ["Bash: npm test"]
    assert task.activity[0]["at"]


def test_a_task_keeps_its_latest_lines_and_shows_fewer() -> None:
    task = _task()
    for n in range(ACTIVITY_KEPT + 5):
        task.note(f"Read: file{n}.py")
    assert len(task.activity) == ACTIVITY_KEPT
    shown = task.public()["activity"]
    assert len(shown) == ACTIVITY_SHOWN and shown[-1]["text"].endswith(f"{ACTIVITY_KEPT + 4}.py")


async def test_a_new_line_is_sent_to_the_app(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    publisher = FakePublisher()
    bridge.attach(None, publisher)
    task = _task()
    bridge.tasks[task.id] = task
    bridge._log(task, "model: Opus")
    bridge._log(task, "model: Opus")
    await _finish(bridge)
    statuses = [m for m in publisher.messages if m["type"] == "brain_status"]
    assert len(statuses) == 1
    assert statuses[0]["tasks"][-1]["activity"][-1]["text"] == "model: Opus"


async def test_a_model_joining_a_task_is_logged_once(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    task = _task()
    bridge._note_model(task, "Sonnet")
    bridge._note_model(task, "Sonnet")
    assert [line["text"] for line in task.activity] == ["model: Sonnet"]


async def test_the_whole_result_goes_to_the_app(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    task = _task()
    task.status, task.result = "done", "# Informe\n" + "x" * 3000
    await bridge.publish_result(task)  # no app yet: nothing to do

    publisher = FakePublisher()
    bridge.attach(None, publisher)
    await bridge.publish_result(task)
    sent = publisher.messages[-1]
    assert sent["type"] == "brain_result" and sent["result"] == task.result


@pytest.mark.parametrize(("status", "result"), [("running", "x"), ("failed", "x"), ("done", " ")])
async def test_only_a_finished_answer_is_sent(tmp_path: Path, status: str, result: str) -> None:
    bridge = _bridge(tmp_path)
    publisher = FakePublisher()
    bridge.attach(None, publisher)
    task = _task()
    task.status, task.result = status, result
    await bridge.publish_result(task)
    assert publisher.messages == []


async def test_a_room_gone_does_not_break_the_task(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    bridge.attach(None, FakePublisher(fail=True))
    task = _task()
    task.status, task.result = "done", "Hecho"
    await bridge.publish_result(task)


async def test_a_finished_task_sends_its_report(tmp_path: Path, monkeypatch) -> None:
    bridge = _bridge(tmp_path)
    publisher = FakePublisher()
    bridge.attach(None, publisher)
    sent: list[str] = []
    original = bridge.publish_result

    async def spy(task: BrainTask) -> None:
        sent.append(task.status)
        await original(task)

    monkeypatch.setattr(bridge, "publish_result", spy)
    monkeypatch.setattr(bridge_module, "QUICK_ANSWER_SECONDS", 0)
    bridge.start(task="t", title="T", kind="analysis", difficulty="normal")
    await _finish(bridge)
    assert sent == ["done"]
    assert any(m["type"] == "brain_result" for m in publisher.messages)


# -- what the PC's run is doing, in words -------------------------------------------


def _block(name: str, **tool_input: Any) -> Any:
    return SimpleNamespace(name=name, input=tool_input)


@pytest.mark.parametrize(
    ("block", "line"),
    [
        (_block("Bash", command="npm   test"), "Bash: npm test"),
        (
            _block("Read", file_path=r"C:\Users\x\Documents\Jarvis\notes.md"),
            "Read: Jarvis/notes.md",
        ),
        (_block("Grep", pattern="TODO"), "Grep: TODO"),
        (_block("WebSearch", query="precio del oro"), "WebSearch: precio del oro"),
        (
            _block("mcp__browser__open", url="https://github.com"),
            "mcp__browser__open: https://github.com",
        ),
        (_block("mcp__google__call", action="list"), "mcp__google__call: list"),
        (_block("mcp__kwami_db__call"), "mcp__kwami_db__call"),
        (_block("TodoWrite", todos=[]), "TodoWrite"),
        (_block("Bash"), "Bash"),
    ],
)
def test_a_tool_call_says_what_it_does(block: Any, line: str) -> None:
    assert executor._progress_line(block) == line


def test_what_looks_like_a_key_is_not_shown() -> None:
    # Made up, and only as long as it takes to look like a key.
    made_up = "sk-ant-api03-AbCdEf0123456789xyz"  # gitleaks:allow
    line = executor._progress_line(
        _block("Bash", command=f'curl -H "Authorization: Bearer {made_up}" x')
    )
    assert "sk-ant" not in line and "•••" in line
    assert "AbCdEf" not in line


def test_a_long_command_is_cut() -> None:
    line = executor._progress_line(_block("Bash", command="echo " + "palabra " * 60))
    assert len(line) <= len("Bash: ") + executor.PROGRESS_DETAIL_CHARS


def test_a_tool_input_that_is_not_a_mapping_is_only_named() -> None:
    assert executor._progress_line(SimpleNamespace(name="Bash", input="ls")) == "Bash"


# -- each tool call of the voice, as it happens --------------------------------------


class _Feed(FakePublisher):
    pass


def _started(name: str, arguments: str) -> Any:
    call = SimpleNamespace(call_id="c1", name=name, arguments=arguments)
    return SimpleNamespace(update=SimpleNamespace(type="tool_call_started", function_call=call))


async def test_a_tool_call_is_announced_when_it_starts_and_when_it_ends() -> None:
    publisher = _Feed()
    feed = ActionFeed(publisher)
    started = feed.on_tool_execution_updated(
        _started("navigate_to", '{"url": "https://github.com", "sign_in": false}')
    )
    assert started == {
        "type": "agent_action",
        "id": "c1",
        "name": "navigate_to",
        "phase": "start",
        "detail": "https://github.com",
    }
    update = SimpleNamespace(
        type="tool_call_updated", call_id="c1", message="  Abriendo  ", silent=False
    )
    assert feed.on_tool_execution_updated(SimpleNamespace(update=update))["detail"] == "Abriendo"
    ended = SimpleNamespace(type="tool_call_ended", call_id="c1", status="done")
    assert feed.on_tool_execution_updated(SimpleNamespace(update=ended))["status"] == "done"
    await asyncio.sleep(0)
    await asyncio.gather(*feed._tasks)
    assert [m["phase"] for m in publisher.messages] == ["start", "update", "end"]


async def test_what_the_user_does_not_hear_or_need_is_not_sent() -> None:
    feed = ActionFeed(_Feed())
    silent = SimpleNamespace(type="tool_call_updated", call_id="c1", message="x", silent=True)
    assert feed.on_tool_execution_updated(SimpleNamespace(update=silent)) is None
    reply = SimpleNamespace(type="tool_reply_updated")
    assert feed.on_tool_execution_updated(SimpleNamespace(update=reply)) is None
    assert feed.on_tool_execution_updated(SimpleNamespace()) is None


def test_outside_a_loop_there_is_no_room_to_send_to() -> None:
    feed = ActionFeed(_Feed())
    assert feed.on_tool_execution_updated(_started("web_search", '{"query": "x"}')) is None


@pytest.mark.parametrize(
    ("arguments", "detail"),
    [
        ('{"query": "  noticias  de   hoy "}', "noticias de hoy"),
        ({"title": "Informe", "url": ""}, "Informe"),
        ('{"body": "un correo entero", "to": "a@b.c"}', ""),
        ("not json", ""),
        ("", ""),
        ("[1, 2]", ""),
        (None, ""),
        ({"task": "x" * 400}, "x" * 140),
    ],
)
def test_the_detail_is_a_few_telling_words(arguments: Any, detail: str) -> None:
    assert detail_of(arguments) == detail


# -- the clean window --------------------------------------------------------------


@pytest.fixture
def profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(local_chrome, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(local_chrome, "_clean_windows", {})
    monkeypatch.setattr(local_chrome, "_POLL_SECONDS", 0.001)
    monkeypatch.setattr(local_chrome, "CLEAN_WINDOW_POLL_SECONDS", 0.001)
    monkeypatch.setattr(local_chrome, "RELEASE_TIMEOUT_SECONDS", 0.05)
    return tmp_path


class _Process:
    def __init__(self, code: int | None = None) -> None:
        self.code = code

    def poll(self) -> int | None:
        return self.code


def test_a_profile_is_in_use_while_chrome_holds_its_lock(tmp_path: Path) -> None:
    assert local_chrome._profile_in_use(tmp_path, windows=True) is False
    (tmp_path / "lockfile").write_text("")
    assert local_chrome._profile_in_use(tmp_path, windows=True) is False, "a lock nobody holds"
    assert not (tmp_path / "lockfile").exists()
    (tmp_path / "lockfile").mkdir()  # cannot be deleted, as when Chrome holds it
    assert local_chrome._profile_in_use(tmp_path, windows=True) is True


def test_elsewhere_the_singleton_lock_says_it(tmp_path: Path) -> None:
    assert local_chrome._profile_in_use(tmp_path, windows=False) is False
    (tmp_path / "SingletonLock").write_text("")
    assert local_chrome._profile_in_use(tmp_path, windows=False) is True
    assert local_chrome._profile_in_use(tmp_path) in (True, False)


def test_a_clean_window_is_open_until_its_chrome_lets_go(profiles: Path, monkeypatch) -> None:
    assert local_chrome.clean_window_open("u") is False
    profile = local_chrome.profile_dir_for("u")
    process = _Process(None)
    local_chrome._clean_windows[profile.name] = process  # type: ignore[assignment]
    assert local_chrome.clean_window_open("u") is True
    process.code = 0
    monkeypatch.setattr(local_chrome, "_profile_in_use", lambda _p: True)
    assert local_chrome.clean_window_open("u") is True, "handed to a Chrome still running"
    monkeypatch.setattr(local_chrome, "_profile_in_use", lambda _p: False)
    assert local_chrome.clean_window_open("u") is False
    assert profile.name not in local_chrome._clean_windows


async def test_closing_the_driven_window_asks_chrome_and_waits(profiles: Path, monkeypatch) -> None:
    profile = profiles / "p"
    profile.mkdir()
    (profile / "DevToolsActivePort").write_text("4321\n/x")
    answers = [{"webSocketDebuggerUrl": "ws://x"}, {"Browser": "y"}, None]
    monkeypatch.setattr(
        local_chrome, "_version_info", AsyncMock(side_effect=lambda _port: answers.pop(0))
    )
    sent: list[str] = []

    class Connection:
        async def __aenter__(self) -> Connection:
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        async def send(self, text: str) -> None:
            sent.append(text)

    import websockets

    monkeypatch.setattr(websockets, "connect", lambda *_a, **_k: Connection())
    await local_chrome._close_debug_window(profile)
    assert "Browser.close" in sent[0]
    assert not (profile / "DevToolsActivePort").exists()


async def test_a_chrome_that_will_not_close_is_not_waited_on_forever(
    profiles: Path, monkeypatch
) -> None:
    profile = profiles / "p"
    profile.mkdir()
    (profile / "DevToolsActivePort").write_text("4321\n/x")
    monkeypatch.setattr(
        local_chrome, "_version_info", AsyncMock(return_value={"webSocketDebuggerUrl": "ws://x"})
    )

    def refuse(*_a: Any, **_k: Any) -> Any:
        raise OSError("refused")

    import websockets

    monkeypatch.setattr(websockets, "connect", refuse)
    await local_chrome._close_debug_window(profile)  # gives up at the deadline


async def test_with_no_driven_window_there_is_nothing_to_close(profiles: Path) -> None:
    profile = profiles / "p"
    profile.mkdir()
    await local_chrome._close_debug_window(profile)


async def test_the_clean_window_is_a_chrome_with_no_debugging_port(
    profiles: Path, monkeypatch
) -> None:
    launched: list[list[str]] = []
    closed: list[Path] = []

    def popen(args: list[str], **_: Any) -> _Process:
        launched.append(args)
        return _Process(None)

    async def close(profile: Path) -> None:
        closed.append(profile)

    monkeypatch.setattr(local_chrome.subprocess, "Popen", popen)
    monkeypatch.setattr(local_chrome, "_close_debug_window", close)
    await local_chrome.open_clean_window("u", "https://github.com/login", chrome="chrome.exe")
    args = launched[0]
    assert args[0] == "chrome.exe" and args[-1] == "https://github.com/login"
    assert not any(a.startswith("--remote-debugging") for a in args)
    assert any(a.startswith("--user-data-dir=") for a in args)
    assert closed == [local_chrome.profile_dir_for("u")]
    assert local_chrome.clean_window_open("u") is True

    # A second page goes to the same window: nothing is closed and it stays the one tracked.
    first = local_chrome._clean_windows[local_chrome.profile_dir_for("u").name]
    await local_chrome.open_clean_window("u", "https://a.b", chrome="chrome.exe")
    assert closed == [local_chrome.profile_dir_for("u")]
    assert local_chrome._clean_windows[local_chrome.profile_dir_for("u").name] is first


async def test_a_clean_window_needs_a_user_and_a_chrome(profiles: Path, monkeypatch) -> None:
    with pytest.raises(ValueError):
        await local_chrome.open_clean_window(" ", "https://a.b", chrome="chrome.exe")
    monkeypatch.setattr(local_chrome, "find_chrome", lambda: "")
    with pytest.raises(BrowserVendorError):
        await local_chrome.open_clean_window("u", "https://a.b")


async def test_waiting_for_the_user_to_close_it(profiles: Path, monkeypatch) -> None:
    looks = [True, True, False]
    monkeypatch.setattr(local_chrome, "clean_window_open", lambda _u: looks.pop(0))
    assert await local_chrome.wait_clean_window_closed("u", 5) is True
    monkeypatch.setattr(local_chrome, "clean_window_open", lambda _u: True)
    assert await local_chrome.wait_clean_window_closed("u", 0.01) is False


async def test_nothing_drives_the_profile_while_the_user_signs_in(
    profiles: Path, monkeypatch
) -> None:
    monkeypatch.setattr(local_chrome, "clean_window_open", lambda _u: True)
    assert await local_chrome.window_open("u") is True
    with pytest.raises(SignInWindowOpenError):
        await LocalChromeProvider(chrome_path="chrome.exe").launch("u")


# -- the voice opens it -------------------------------------------------------------


@pytest.fixture
def local(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(builtin, "get_settings", lambda: Settings(browser_provider=LOCAL))

    async def public(url: str) -> str:
        return url

    monkeypatch.setattr(builtin, "validate_url_async", public)
    opened: list[tuple[str, str]] = []

    async def open_window(user_id: str, url: str) -> None:
        opened.append((user_id, url))

    monkeypatch.setattr(builtin, "open_clean_window", open_window)
    monkeypatch.setattr(builtin, "wait_sign_in", AsyncMock(return_value=local_chrome.CLOSED))
    return opened


async def test_the_voice_opens_a_clean_window_to_sign_in(local: Any) -> None:
    tools, session = _tools()
    said = await tools.navigate_to(None, "https://github.com/login", sign_in=True)
    assert local == [("kwami-1", "https://github.com/login")]
    assert session.closed == 1, "off the driven window first"
    assert "clean Chrome window" in said and "closes by itself" in said
    await asyncio.gather(*tools._background_tasks)
    actions = [m.get("action") for m in tools.publisher.published]
    assert actions == ["sign_in", "sign_in_done"]


async def test_a_window_never_closed_is_not_announced(local: Any, monkeypatch) -> None:
    monkeypatch.setattr(builtin, "wait_sign_in", AsyncMock(return_value=local_chrome.STILL_OPEN))
    tools, _ = _tools()
    await tools.navigate_to(None, "https://github.com/login", sign_in=True)
    await asyncio.gather(*tools._background_tasks)
    assert [m.get("action") for m in tools.publisher.published] == ["sign_in"]


async def test_once_the_user_is_in_jarvis_says_so_and_carries_on(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "wait_sign_in", AsyncMock(return_value=local_chrome.SIGNED_IN))
    publisher = FakePublisher()
    voice = MagicMock()
    await builtin.AgentToolsMixin._announce_sign_in_done(
        publisher, "u", "https://github.com/login", voice
    )
    assert [m.get("action") for m in publisher.messages] == ["sign_in_done"]
    instructions = voice.generate_reply.call_args.kwargs["instructions"]
    assert "github.com" in instructions and "carry on" in instructions


async def test_a_window_the_user_closed_moves_the_panel_on_quietly(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "wait_sign_in", AsyncMock(return_value=local_chrome.CLOSED))
    publisher = FakePublisher()
    voice = MagicMock()
    await builtin.AgentToolsMixin._announce_sign_in_done(publisher, "u", "https://a.b", voice)
    assert [m.get("action") for m in publisher.messages] == ["sign_in_done"]
    voice.generate_reply.assert_not_called()


async def test_a_conversation_gone_does_not_break_the_sign_in(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "wait_sign_in", AsyncMock(return_value=local_chrome.SIGNED_IN))
    publisher = FakePublisher()
    voice = MagicMock()
    voice.generate_reply.side_effect = RuntimeError("session closed")
    await builtin.AgentToolsMixin._announce_sign_in_done(publisher, "u", "https://a.b", voice)
    await builtin.AgentToolsMixin._announce_sign_in_done(publisher, "u", "https://a.b", None)
    assert [m.get("action") for m in publisher.messages] == ["sign_in_done", "sign_in_done"]


async def test_the_conversation_is_handed_to_the_wait(local: Any, monkeypatch) -> None:
    seen: list[Any] = []

    async def announce(_publisher: Any, _user: str, _url: str, voice: Any = None) -> None:
        seen.append(voice)

    tools, _ = _tools()
    monkeypatch.setattr(tools, "_announce_sign_in_done", announce)
    voice = object()

    class Context:
        session = voice

    monkeypatch.setattr(tools, "_site_refusal", AsyncMock(return_value=None))
    monkeypatch.setattr(tools, "_remember_site", AsyncMock(return_value=None))
    await tools.navigate_to(Context(), "https://github.com/login", sign_in=True)
    await asyncio.gather(*tools._background_tasks)
    assert seen == [voice]

    # The HUD's button comes with no context: the agent's own conversation is told.
    tools.session = agent_voice = object()
    await tools.navigate_to(None, "https://github.com/login", sign_in=True)
    await asyncio.gather(*tools._background_tasks)
    assert seen[-1] is agent_voice


async def test_an_agent_out_of_its_conversation_is_not_told(local: Any, monkeypatch) -> None:
    seen: list[Any] = []

    async def announce(_publisher: Any, _user: str, _url: str, voice: Any = None) -> None:
        seen.append(voice)

    class Ended(type(_tools()[0])):  # type: ignore[misc]
        @property
        def session(self) -> Any:
            raise RuntimeError("the agent is not running")

        @session.setter
        def session(self, _value: Any) -> None:
            return None

    tools = Ended(session=_tools()[1])
    monkeypatch.setattr(tools, "_announce_sign_in_done", announce)
    await tools.navigate_to(None, "https://github.com/login", sign_in=True)
    await asyncio.gather(*tools._background_tasks)
    assert seen == [None]


async def test_a_second_sign_in_reuses_the_task_set(local: Any) -> None:
    tools, _ = _tools()
    await tools.navigate_to(None, "https://a.b", sign_in=True)
    await tools.navigate_to(None, "https://c.d", sign_in=True)
    assert len(local) == 2
    await asyncio.gather(*tools._background_tasks)


async def test_a_window_that_cannot_open_is_said(local: Any, monkeypatch) -> None:
    async def fail(*_: Any) -> None:
        raise BrowserVendorError(LOCAL, "missing", "Neither Chrome nor Edge is installed.")

    monkeypatch.setattr(builtin, "open_clean_window", fail)
    tools, _ = _tools()
    said = await tools.navigate_to(None, "https://a.b", sign_in=True)
    assert "can't open a window to sign in" in said and "Edge" in said


async def test_away_from_the_users_computer_there_is_no_clean_window(monkeypatch) -> None:
    async def public(url: str) -> str:
        return url

    monkeypatch.setattr(builtin, "validate_url_async", public)
    monkeypatch.setattr(builtin, "get_settings", lambda: Settings(browser_provider="browser_use"))
    tools, session = _tools()
    said = await tools.navigate_to(None, "https://a.b", sign_in=True)
    assert "user's computer" in said and session.closed == 0


def test_each_browser_gets_its_own_sign_in_words() -> None:
    assert builtin._sign_in_hint(SimpleNamespace(is_builtin=True)) == builtin.NO_SIGN_IN_HINT
    local_session = SimpleNamespace(is_builtin=False, runs_locally=True)
    assert builtin._sign_in_hint(local_session) == builtin.LOCAL_SIGN_IN_HINT
    cloud = SimpleNamespace(is_builtin=False, runs_locally=False)
    assert builtin._sign_in_hint(cloud) == builtin.SIGN_IN_HINT


# -- the app's buttons ----------------------------------------------------------------


class _Agent:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def navigate_to(self, _ctx: Any, url: str, **kwargs: Any) -> str:
        self.calls.append((url, kwargs))
        return "ok"


def test_the_sign_in_button_goes_through_navigate_to() -> None:
    state = FakeState(agent=_Agent())
    assert _router(state).handle({"type": "browser_sign_in_request", "url": " https://a.b "})
    assert state.spawned == ["browser_sign_in_request"]


@pytest.mark.parametrize("message", [{"url": ""}, {"url": 3}, {}])
def test_a_sign_in_request_without_an_address_is_dropped(message: dict) -> None:
    state = FakeState(agent=_Agent())
    _router(state).handle({"type": "browser_sign_in_request", **message})
    assert state.spawned == []


def test_a_sign_in_request_before_any_agent_is_dropped() -> None:
    state = FakeState(agent=None)
    _router(state).handle({"type": "browser_sign_in_request", "url": "https://a.b"})
    assert state.spawned == []


async def test_the_sign_in_request_asks_for_the_clean_window() -> None:
    agent = _Agent()
    await dispatch.DataMessageRouter._open_and_report(agent, None, "https://a.b", sign_in=True)
    await dispatch.DataMessageRouter._open_and_report(agent, None, "https://c.d")
    assert agent.calls == [("https://a.b", {"sign_in": True}), ("https://c.d", {})]


def test_the_stop_button_cancels_the_task() -> None:
    cancelled: list[str] = []
    bridge = SimpleNamespace(cancel=lambda task_id: cancelled.append(task_id) or True)
    router = _router(FakeState())
    router.deps = SimpleNamespace(bridge=bridge)
    assert router.handle({"type": "brain_cancel", "id": "t1"}) == "brain_cancel"
    assert cancelled == ["t1"]
    bridge.cancel = lambda _task_id: False
    router.handle({"type": "brain_cancel", "id": "t2"})


@pytest.mark.parametrize("deps", [None, SimpleNamespace(bridge=None)])
def test_a_stop_with_no_brain_or_no_id_does_nothing(deps: Any) -> None:
    router = _router(FakeState())
    router.deps = deps
    router.handle({"type": "brain_cancel", "id": "t1"})
    router.deps = SimpleNamespace(bridge=SimpleNamespace(cancel=MagicMock()))
    router.handle({"type": "brain_cancel", "id": 7})
    router.deps.bridge.cancel.assert_not_called()


# -- the window on the PC, mirrored in the app ---------------------------------------


def _local_session() -> tuple[CloudBrowserSession, MagicMock]:
    room = MagicMock()
    room.local_participant.stream_bytes = AsyncMock(return_value=AsyncMock())
    room.local_participant.publish_data = AsyncMock()
    session = CloudBrowserSession(room=room)
    session._vendor = LOCAL
    session._browser_id = "local-1"
    session._cdp = MagicMock(is_connected=True)
    return session, room


async def test_a_page_the_user_opened_on_their_pc_is_left_alone(monkeypatch) -> None:
    session, room = _local_session()
    session._cdp.evaluate = AsyncMock(return_value="http://localhost:5173/")
    session._cdp.navigate = AsyncMock()
    session._cdp.screenshot = AsyncMock(return_value=base64.b64encode(b"jpeg").decode())

    async def sleep(_seconds: float) -> None:
        session._cdp.is_connected = False

    monkeypatch.setattr(browser_session.asyncio, "sleep", sleep)
    await session._stream_frames()
    session._cdp.screenshot.assert_not_awaited()
    session._cdp.navigate.assert_not_awaited()


async def test_the_window_on_the_pc_sends_frames_too(monkeypatch) -> None:
    from src.browser.launched import LaunchedBrowser

    provider = LocalChromeProvider(chrome_path="chrome.exe")
    launched = LaunchedBrowser(browser_id="local-1", vendor=LOCAL, cdp_http_url="http://x")
    provider.launch = AsyncMock(return_value=launched)  # type: ignore[method-assign]
    cdp = MagicMock(is_connected=True, connect=AsyncMock(), send=AsyncMock(), close=AsyncMock())
    cdp.screenshot = AsyncMock(return_value="")
    monkeypatch.setattr(browser_session, "CDPConnection", lambda: cdp)
    session = CloudBrowserSession(room=None, provider=provider)
    await session.start("user-1")
    assert session._frame_task is not None
    await session.close()
