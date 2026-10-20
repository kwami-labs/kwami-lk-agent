"""The local executor: Claude Code, working on the user's PC, in one folder.

It runs through the Claude Agent SDK, which ships its own Claude Code binary
and bills to ANTHROPIC_API_KEY -- so nothing has to be installed on PATH and
the user's personal Claude Code settings (CLAUDE.md, hooks, MCP servers) are
not loaded: `setting_sources` is left empty on purpose.

The user asked for "full control, without push". That is enforced here, by a
PreToolUse hook that runs on every tool call whatever the permission mode, not
by asking the model nicely:

* every file path must resolve inside the executor folder;
* nothing named ``.env*`` is read or written;
* no ``git push``, and no shell command that names a path outside the folder
  or climbs out of it with ``..``.

The shell checks are a tripwire, not a sandbox: a determined program could
still reach outside. What they stop is the realistic failure -- a model that
helpfully pushes, or "cleans up" a sibling folder.

On LiveKit Cloud there is no user PC to work on. There the executor works in a
temporary folder, the same guard confined to it, and what it leaves behind is
uploaded to the user's Storage (`jarvis-files/<user>/<kwami>/<task>/`) before
the folder is deleted.
"""

from __future__ import annotations

import mimetypes
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any

from ..utils.logging import get_logger
from .config import ExecutorConfig, resolve_executor_folder

if TYPE_CHECKING:
    from ..adapters.supabase import SupabaseClient

logger = get_logger("brain.executor")

#: The Storage bucket the cloud executor leaves its work in.
FILES_BUCKET = "jarvis-files"
#: Bounds on what one task uploads. The free plan caps a file at 50 MB, and a
#: task that produced thousands of files built something, not a deliverable.
MAX_UPLOAD_FILES = 100
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
#: Never worth keeping: regenerable, huge, or both.
_SKIP_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv", "venv", ".cache"})

_PATH_KEYS = ("file_path", "path", "notebook_path")
_READ_ONLY_TOOLS = ["Read", "Glob", "Grep", "WebSearch", "WebFetch", "TodoWrite"]
#: Where a path that starts with "/" may begin in a command: its start, or after
#: whitespace, a quote, `=`, a bracket or a shell operator -- not mid-word, where
#: the "/a/" of "/home/u/a/b.txt" or "src/a/b.py" is no path of its own.
_PATH_START = r"(?<![^\s\"'=(<>|;&])"
_PATH_REST = r"[^\s\"'|;&]*"
#: On Windows: drive (C:\, C:/), Git Bash (/c/...) and UNC (\\server) paths.
_WINDOWS_PATH = re.compile(rf"(?:[A-Za-z]:[\\/]|{_PATH_START}/[a-z]/|\\\\){_PATH_REST}")
#: On Linux (the cloud executor): POSIX absolute paths, and the Windows forms,
#: which `_inside` refuses there.
_POSIX_PATH = re.compile(rf"(?:[A-Za-z]:[\\/]|\\\\|{_PATH_START}/){_PATH_REST}")
#: Devices a command may name without leaving the folder ("2> /dev/null").
_DEVICES = frozenset({"/dev/null", "/dev/stdin", "/dev/stdout", "/dev/stderr"})
_GIT_PUSH = re.compile(r"\bgit\b[^\n|;&]*\bpush\b", re.IGNORECASE)
_ENV_FILE = re.compile(r"(^|[\\/\s\"'])\.env(\.|\b)", re.IGNORECASE)
_PARENT = re.compile(r"(^|[\s\"'\\/])\.\.([\\/]|$|\s)")

SYSTEM_APPEND = """
You are the local executor of Jarvis, the user's voice assistant. A voice model
handed you this task and will read your final answer aloud in Spanish (Spain).

- Work only inside the current folder. Never push, never touch .env files.
- Finish with a short spoken-style summary in Spanish: what you did, where it
  is, and anything the user must decide. No markdown tables, no code in it.
"""


@dataclass
class ExecutorResult:
    text: str
    cost_usd: float
    ok: bool
    #: Storage prefix the files went to, when the executor ran in the cloud.
    uploaded_to: str = ""
    uploaded: int = 0


@dataclass
class CloudWorkspace:
    """Where a cloud run's results go: `jarvis-files/<user>/<kwami>/<task>/`.

    `store` is None when the agent has no Supabase; the work then happens in a
    throwaway folder and only its spoken summary survives.
    """

    store: SupabaseClient | None
    user_id: str = ""
    kwami_id: str = ""

    def prefix(self, folder: str) -> str:
        return "/".join((self.user_id, self.kwami_id or "default", folder))


def _task_folder() -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def _files_to_upload(root: Path) -> list[Path]:
    out: list[Path] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in _SKIP_DIRS for part in rel.parts[:-1]) or not path.is_file():
            continue
        if _ENV_FILE.search(f"/{rel.as_posix()}"):
            continue
        out.append(path)
    return out


async def upload_results(root: Path, workspace: CloudWorkspace, folder: str) -> tuple[str, int]:
    """Copy what the executor left in `root` to Storage. (prefix, files stored)."""
    if workspace.store is None or not workspace.user_id:
        return "", 0
    prefix = workspace.prefix(folder)
    stored = 0
    skipped = 0
    for path in _files_to_upload(root):
        if stored >= MAX_UPLOAD_FILES:
            skipped += 1
            continue
        try:
            size = path.stat().st_size
            if size > MAX_UPLOAD_BYTES:
                skipped += 1
                continue
            data = path.read_bytes()
        except OSError:
            skipped += 1
            continue
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        rel = path.relative_to(root).as_posix()
        if await workspace.store.upload(
            FILES_BUCKET, f"{prefix}/{rel}", data, content_type=content_type
        ):
            stored += 1
        else:
            skipped += 1
    if skipped:
        logger.warning("Executor upload left %d file(s) out of %s", skipped, prefix)
    return prefix, stored


def _inside(root: Path, raw: str) -> bool:
    # A drive-letter or UNC path is absolute only on Windows. On the Linux cloud
    # executor Path() reads "C:\Users\..." as a relative name under root and
    # would wave it through.
    if os.name != "nt" and PureWindowsPath(raw).drive:
        return False
    try:
        target = (root / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    except (OSError, ValueError):
        return False
    return target == root or root in target.parents


def _command_paths(command: str, *, windows: bool) -> list[tuple[str, str]]:
    """The absolute paths a shell command names, as (written, to check).

    Git Bash's /c/Users/... is C:/Users/... only on Windows; on Linux it is an
    ordinary POSIX path and is checked as one.
    """
    if not windows:
        return [(m, m) for m in _POSIX_PATH.findall(command) if m not in _DEVICES]
    paths = []
    for match in _WINDOWS_PATH.findall(command):
        candidate = match
        if re.match(r"/[a-z]/", candidate):  # Git Bash style /c/Users/...
            candidate = f"{candidate[1].upper()}:{candidate[2:]}"
        paths.append((match, candidate))
    return paths


def check_tool_call(root: Path, tool_name: str, tool_input: dict[str, Any]) -> str | None:
    """Why this call is refused, or None when it may run."""
    for key in _PATH_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            if _ENV_FILE.search(value):
                return ".env files are off limits."
            if not _inside(root, value):
                return f"{value} is outside the Jarvis folder."
    if tool_name == "Bash":
        command = str(tool_input.get("command", ""))
        if _GIT_PUSH.search(command):
            return "Pushing is not allowed. Commit locally and tell the user."
        if _ENV_FILE.search(command):
            return ".env files are off limits."
        if _PARENT.search(command):
            return "Stay inside the Jarvis folder: no '..' paths."
        for match, candidate in _command_paths(command, windows=os.name == "nt"):
            if not _inside(root, candidate):
                return f"{match} is outside the Jarvis folder."
    return None


async def run_executor(
    api_key: str,
    model: str,
    task: str,
    config: ExecutorConfig,
    *,
    cloud: CloudWorkspace | None = None,
) -> ExecutorResult:
    """Run the task. Locally in the Jarvis folder; with `cloud`, in a temporary
    folder whose files are uploaded to Storage and then deleted."""
    if cloud is None:
        root = resolve_executor_folder(config.folder)
        root.mkdir(parents=True, exist_ok=True)
        return await _run_in(root, api_key, model, task, config)

    root = Path(tempfile.mkdtemp(prefix="jarvis-"))
    try:
        result = await _run_in(root, api_key, model, task, config)
        prefix, stored = await upload_results(root, cloud, _task_folder())
    finally:
        shutil.rmtree(root, ignore_errors=True)
    if stored:
        result.uploaded_to = f"{FILES_BUCKET}/{prefix}"
        result.uploaded = stored
        result.text += (
            f"\n\nDetalle: {stored} archivo(s) guardado(s) en tu almacenamiento de Jarvis, "
            f"en {result.uploaded_to}."
        )
    return result


async def _run_in(
    root: Path, api_key: str, model: str, task: str, config: ExecutorConfig
) -> ExecutorResult:
    try:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ClaudeSDKClient,
            HookContext,
            HookInput,
            HookJSONOutput,
            HookMatcher,
            ResultMessage,
            TextBlock,
        )
    except ImportError:
        return ExecutorResult(
            "El ejecutor local no está instalado (falta claude-agent-sdk).", 0.0, False
        )

    async def guard(
        input_data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        tool_input = input_data.get("tool_input")
        reason = check_tool_call(
            root,
            str(input_data.get("tool_name", "")),
            tool_input if isinstance(tool_input, dict) else {},
        )
        if reason is None:
            return {}
        logger.info("Executor refused %s: %s", input_data.get("tool_name"), reason)
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }

    options = ClaudeAgentOptions(
        cwd=str(root),
        model=model,
        system_prompt={"type": "preset", "preset": "claude_code", "append": SYSTEM_APPEND},
        max_turns=config.max_turns,
        max_budget_usd=config.max_task_usd,
        env={"ANTHROPIC_API_KEY": api_key},
        hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[guard])]},
    )
    if config.permissions == "read_only":
        options.allowed_tools = list(_READ_ONLY_TOOLS)
        options.disallowed_tools = ["Write", "Edit", "MultiEdit", "NotebookEdit", "Bash"]
        options.permission_mode = "dontAsk"
    elif config.permissions == "edit_only":
        options.disallowed_tools = ["Bash"]
        options.permission_mode = "acceptEdits"
    else:
        options.permission_mode = "bypassPermissions"

    texts: list[str] = []
    cost = 0.0
    ok = True
    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(task)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    texts += [b.text for b in message.content if isinstance(b, TextBlock)]
                elif isinstance(message, ResultMessage):
                    cost = message.total_cost_usd or 0.0
                    ok = not message.is_error
                    if message.result:
                        texts.append(message.result)
    except Exception as e:
        logger.exception("Local executor failed")
        return ExecutorResult(f"El ejecutor local falló: {e}", cost, False)

    final = texts[-1].strip() if texts else "El ejecutor terminó sin decir nada."
    return ExecutorResult(final, cost, ok)
