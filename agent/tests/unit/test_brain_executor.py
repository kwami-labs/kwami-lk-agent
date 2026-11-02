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

from src.brain import config as brain_config
from src.brain import executor
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
