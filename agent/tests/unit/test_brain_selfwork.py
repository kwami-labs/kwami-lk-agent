"""Self-improvement against real git and a local "GitHub"; only `gh` is stood in for.

What matters is pinned on the real thing: only the branch reaches the remote, `main`
never moves, and what must not leave the PC does not.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.brain import selfwork
from src.brain import worker as worker_module
from src.brain.executor import ExecutorResult, check_tool_call, child_env
from src.brain.jobs import BrainJob
from src.brain.selfwork import (
    SelfRepo,
    Workspace,
    branch_for,
    outcome_text,
    prepare,
    publish,
    pull_request_body,
    refusal,
    run_tests,
    sensitive_files,
)
from src.brain.worker import Worker

USER = "11111111-1111-4111-8111-111111111111"
KWAMI = "22222222-2222-4222-8222-222222222222"


def git(cwd: Path, *args: str) -> str:
    identity = ["-c", "user.name=T", "-c", "user.email=t@example.com"]
    return subprocess.run(  # noqa: S603
        ["git", *identity, *args],  # noqa: S607
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A bare repository with one commit on main, standing in for GitHub."""
    bare = tmp_path / "github.git"
    git(tmp_path, "init", "--bare", "-b", "main", str(bare))
    seed = tmp_path / "seed"
    git(tmp_path, "clone", str(bare), str(seed))
    git(seed, "switch", "-c", "main")
    (seed / "README.md").write_text("hola\n", encoding="utf-8")
    git(seed, "add", ".")
    git(seed, "commit", "-m", "first")
    git(seed, "push", "origin", "main")
    monkeypatch.setattr(
        selfwork,
        "REPOS",
        {"agent": SelfRepo("Octane-666/kwami-lk-agent", remote=str(bare))},
    )
    return bare


@pytest.fixture
def gh(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, bool]]:
    opened: list[tuple[str, str, bool]] = []

    async def fake(workspace: Workspace, title: str, body: str, draft: bool) -> str:
        opened.append((title, body, draft))
        return "https://github.com/Octane-666/kwami-lk-agent/pull/99"

    monkeypatch.setattr(selfwork, "_open_pull_request", fake)
    return opened


async def _workspace(tmp_path: Path, thread: str = "abcd1234ef") -> Workspace:
    return await prepare("agent", thread, tmp_path / "jarvis", tmp_path / "mirrors")


def _commit(workspace: Workspace, name: str, content: str) -> None:
    path = workspace.path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    git(workspace.path, "add", ".")
    git(workspace.path, "commit", "-m", "change")


PASSED = selfwork.TestRun(ran=True, passed=True)


async def test_a_workspace_starts_on_its_branch_from_main(tmp_path: Path, remote: Path) -> None:
    workspace = await _workspace(tmp_path)
    assert workspace.path == tmp_path / "jarvis" / "self" / "agent"
    assert git(workspace.path, "branch", "--show-current") == "jarvis/abcd1234"
    assert (workspace.path / "README.md").is_file()


async def test_a_follow_up_finds_its_work(tmp_path: Path, remote: Path) -> None:
    first = await _workspace(tmp_path)
    _commit(first, "a.txt", "uno\n")
    await _workspace(tmp_path, "ffff0000")
    again = await _workspace(tmp_path)
    assert (again.path / "a.txt").read_text(encoding="utf-8") == "uno\n"


async def test_only_the_branch_reaches_the_remote(tmp_path: Path, remote: Path, gh: list) -> None:
    workspace = await _workspace(tmp_path)
    main_before = git(remote, "rev-parse", "refs/heads/main")
    _commit(workspace, "a.txt", "uno\n")
    git(workspace.path, "branch", "otra")

    published = await publish(workspace, "Mejora", "Hecho", PASSED)

    assert published.url.endswith("/pull/99") and published.commits == 1
    assert git(remote, "rev-parse", "refs/heads/main") == main_before
    refs = git(remote, "for-each-ref", "--format=%(refname)").splitlines()
    assert sorted(refs) == ["refs/heads/jarvis/abcd1234", "refs/heads/main"]
    assert gh[0][2] is False


async def test_it_ignores_a_remote_the_model_rewrote(
    tmp_path: Path, remote: Path, gh: list
) -> None:
    workspace = await _workspace(tmp_path)
    elsewhere = tmp_path / "elsewhere.git"
    git(tmp_path, "init", "--bare", str(elsewhere))
    git(workspace.path, "remote", "set-url", "origin", str(elsewhere))
    _commit(workspace, "a.txt", "uno\n")

    await publish(workspace, "Mejora", "Hecho", PASSED)

    assert git(elsewhere, "for-each-ref") == ""
    assert "jarvis/abcd1234" in git(remote, "for-each-ref", "--format=%(refname)")


async def test_a_commit_on_main_in_the_clone_goes_nowhere(
    tmp_path: Path, remote: Path, gh: list
) -> None:
    workspace = await _workspace(tmp_path)
    main_before = git(remote, "rev-parse", "refs/heads/main")
    git(workspace.path, "switch", "main")
    _commit(workspace, "malo.txt", "x\n")
    git(workspace.path, "switch", "jarvis/abcd1234")

    published = await publish(workspace, "Mejora", "Hecho", PASSED)

    assert published.url == "" and gh == []
    assert git(remote, "rev-parse", "refs/heads/main") == main_before


async def test_nothing_new_opens_no_pull_request(tmp_path: Path, remote: Path, gh: list) -> None:
    published = await publish(await _workspace(tmp_path), "Mejora", "Hecho", PASSED)
    assert published.url == "" and gh == []
    assert "No hay commits" in outcome_text(published, PASSED)


@pytest.mark.parametrize(
    ("name", "content", "why"),
    [
        (".env", "A=1\n", "prohibidos"),
        ("agent/.env.local", "A=1\n", "prohibidos"),
        ("certs/server.pem", "x\n", "prohibidos"),
        ("config/credentials.json", "{}\n", "prohibidos"),
        ("a.py", 'KEY = "sk-ant-abcdefghijkl"\n', "clave"),
        ("b.py", 'KEY = "sb_secret_abcdefghijkl"\n', "clave"),
    ],
)
async def test_what_must_not_leave_does_not(
    tmp_path: Path, remote: Path, gh: list, name: str, content: str, why: str
) -> None:
    workspace = await _workspace(tmp_path)
    _commit(workspace, name, content)

    published = await publish(workspace, "Mejora", "Hecho", PASSED)

    assert why in published.refused and gh == []
    assert "jarvis" not in git(remote, "for-each-ref", "--format=%(refname)")
    assert "No he subido nada" in outcome_text(published, PASSED)


async def test_failing_tests_make_a_draft(tmp_path: Path, remote: Path, gh: list) -> None:
    workspace = await _workspace(tmp_path)
    _commit(workspace, "a.txt", "uno\n")
    tests = selfwork.TestRun(ran=True, passed=False, tail="1 failed")

    published = await publish(workspace, "Mejora", "Hecho", tests)

    assert published.draft and gh[0][2] is True
    assert "**Fallan.**" in gh[0][1] and "1 failed" in gh[0][1]
    assert "borrador" in outcome_text(published, tests)


async def test_touching_its_own_limits_is_flagged(tmp_path: Path, remote: Path, gh: list) -> None:
    workspace = await _workspace(tmp_path)
    _commit(workspace, "agent/src/brain/executor.py", "x = 1\n")

    published = await publish(workspace, "Mejora", "Hecho", PASSED)

    assert published.sensitive == ["agent/src/brain/executor.py"]
    assert "[!WARNING]" in gh[0][1]
    assert "límites" in outcome_text(published, PASSED)


@pytest.mark.parametrize(
    "name",
    [
        ".github/dependabot.yml",
        "migrations/022_x.sql",
        "agent/src/brain/worker.py",
        "agent/src/domain/charter.py",
        "Dockerfile",
    ],
)
def test_sensitive_paths(name: str) -> None:
    assert sensitive_files([name]) == [name]


@pytest.mark.parametrize(
    "name", [".github/workflows/ci.yml", ".GitHub/Workflows/x.yaml", ".github/actions/a/action.yml"]
)
def test_what_runs_with_the_repos_secrets_is_never_published(name: str) -> None:
    assert "prohibidos" in refusal([name], "+x = 1\n")


def test_ordinary_paths_pass() -> None:
    assert sensitive_files(["src/lib/brain.ts"]) == []
    assert refusal(["src/environment.ts", "docs/secret-sauce.md"], "+x = 1\n") == ""


async def test_tests_run_without_secrets(
    tmp_path: Path, remote: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "plantado")
    workspace = await _workspace(tmp_path)
    show = "import os; print('[' + os.environ.get('SUPABASE_SECRET_KEY', '') + ']')"
    workspace.repo = SelfRepo("x", tests=f'"{sys.executable}" -c "{show}"')
    run = await run_tests(workspace)
    assert run.ran and run.passed
    assert "plantado" not in run.tail


async def test_a_failing_command_fails_and_no_command_is_said(tmp_path: Path, remote: Path) -> None:
    workspace = await _workspace(tmp_path)
    workspace.repo = SelfRepo("x", tests="git no-such-command")
    assert (await run_tests(workspace)).passed is False
    workspace.repo = SelfRepo("x")
    assert (await run_tests(workspace)).ran is False


async def test_slow_tests_are_stopped(
    tmp_path: Path, remote: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(selfwork, "TEST_SECONDS", 0.01)
    workspace = await _workspace(tmp_path)
    workspace.repo = SelfRepo("x", tests=f'"{sys.executable}" -c "import time; time.sleep(30)"')
    run = await run_tests(workspace)
    assert run.ran and not run.passed and "a tiempo" in run.tail


def test_branch_names_are_safe() -> None:
    assert branch_for("abc-def/../12345") == "jarvis/abcdef12"
    assert branch_for("///") == "jarvis/work"


def test_the_body_says_when_tests_could_not_run() -> None:
    assert "No hay forma" in pull_request_body("Hecho", selfwork.TestRun(ran=False), [])


# -- the executor cannot publish ------------------------------------------------


def test_the_executor_has_no_git_or_gh_login() -> None:
    env = child_env("")
    assert env["ANTHROPIC_API_KEY"] == ""
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert Path(env["GH_CONFIG_DIR"]).is_dir()
    assert list(Path(env["GH_CONFIG_DIR"]).iterdir()) == []
    assert env["GIT_AUTHOR_NAME"] == "Jarvis"


@pytest.mark.parametrize(
    "command",
    ["gh pr create", "git status && gh auth token", "gh.exe pr merge 1", "(gh repo list)"],
)
def test_the_guard_refuses_the_github_cli(tmp_path: Path, command: str) -> None:
    assert "GitHub CLI" in str(check_tool_call(tmp_path, "Bash", {"command": command}))


def test_the_guard_leaves_the_word_alone(tmp_path: Path) -> None:
    assert check_tool_call(tmp_path, "Bash", {"command": "git commit -m 'fix gh wrapper'"}) is None


# -- the worker -----------------------------------------------------------------


class Jobs:
    def __init__(self) -> None:
        self.rows: dict[str, BrainJob] = {}
        self.finished: list[tuple[str, str, str, float, str]] = []

    async def session_of_thread(self, thread_id: str) -> str:
        return ""

    async def agents(self, user_id: str) -> list[dict[str, Any]]:
        return []

    async def thread_cost(self, thread_id: str) -> float:
        raise AssertionError("a self-improvement job reports its own cost")

    async def update_running(self, job_id: str, progress: str) -> bool:
        return True

    async def get(self, job_id: str) -> BrainJob | None:
        return self.rows.get(job_id)

    async def finish(
        self, job_id: str, status: str, result: str, cost: float, session: str
    ) -> bool:
        self.finished.append((job_id, status, result, cost, session))
        return True

    async def mark_delivered(self, job_id: str) -> bool:
        return True


class Ledger:
    spend: list[float] = []

    def __init__(self, store: Any) -> None:
        pass

    async def bind(self, kwami_id: str, user_id: str | None = None) -> None:
        return None

    def add_spend(self, model_id: str, usd: float) -> None:
        Ledger.spend.append(usd)

    def add_pending(self, kwami_id: str, item: Any) -> None:
        return None

    async def flush(self, timeout: float = 5.0) -> None:
        return None


@pytest.fixture
def worker_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote: Path, gh: list) -> Any:
    Ledger.spend = []
    monkeypatch.setattr(worker_module, "SupabaseLedger", Ledger)
    monkeypatch.setattr(worker_module, "RUNNING_POLL_SECONDS", 0.01)

    def build(run: Any) -> tuple[Worker, Jobs, BrainJob]:
        jobs = Jobs()
        job = BrainJob(USER, KWAMI, "Mejora", "Haz X", id="j1", thread_id="abcd1234", repo="agent")
        jobs.rows[job.id] = job
        worker = Worker(
            jobs,  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            "ak",
            USER,
            runner_id="r",
            run=run,
            folder=tmp_path / "jarvis",
            mirrors=tmp_path / "mirrors",
        )
        return worker, jobs, job

    return build


def _committing_run(calls: list[str], outcomes: list[ExecutorResult]) -> Any:
    async def run(
        api_key: str, model: str, task: str, config: Any, **kwargs: Any
    ) -> ExecutorResult:
        calls.append(api_key)
        result = outcomes.pop(0)
        if result.ok:
            folder = Path(config.folder)
            (folder / "cambio.txt").write_text("x\n", encoding="utf-8")
            git(folder, "add", ".")
            git(folder, "commit", "-m", "feat: cambio")
        return result

    return run


async def test_changing_its_own_code_never_opens_google(worker_setup: Any) -> None:
    seen: list[str] = []

    async def run(api_key: str, model: str, task: str, config: Any, **kwargs: Any) -> Any:
        seen.append(config.google_user_id)
        return ExecutorResult("No he sabido hacerlo.", 0.0, False)

    worker, _, job = worker_setup(run)
    await worker.process(job, 0.0)
    assert seen == [""]


async def test_on_the_login_nothing_is_billed(worker_setup: Any, gh: list) -> None:
    calls: list[str] = []
    run = _committing_run(calls, [ExecutorResult("Hecho.", 1.5, True, session_id="s1")])
    worker, jobs, job = worker_setup(run)

    await worker.process(job, 0.0)

    assert calls == [""]
    assert Ledger.spend == []
    _, status, text, cost, session = jobs.finished[0]
    assert (status, cost, session) == ("done", 0.0, "s1")
    assert text.startswith("PR abierto") and gh


async def test_when_the_login_cannot_the_key_takes_over_and_is_billed(worker_setup: Any) -> None:
    calls: list[str] = []
    run = _committing_run(
        calls,
        [
            ExecutorResult("Invalid API key · Please run /login", 0.0, False),
            ExecutorResult("Hecho.", 0.4, True, session_id="s2"),
        ],
    )
    worker, jobs, job = worker_setup(run)

    await worker.process(job, 0.0)

    assert calls == ["", "ak"]
    assert Ledger.spend == [0.4]
    assert jobs.finished[0][1] == "done" and jobs.finished[0][3] == 0.4


async def test_real_failures_do_not_spend_the_key(worker_setup: Any, gh: list) -> None:
    calls: list[str] = []
    run = _committing_run(calls, [ExecutorResult("No he sabido hacerlo.", 3.0, False)])
    worker, jobs, job = worker_setup(run)

    await worker.process(job, 0.0)

    assert calls == [""]
    assert Ledger.spend == [] and gh == []
    assert jobs.finished[0][1] == "failed"


async def test_a_refused_branch_fails_the_job(worker_setup: Any, gh: list) -> None:
    async def run(
        api_key: str, model: str, task: str, config: Any, **kwargs: Any
    ) -> ExecutorResult:
        folder = Path(config.folder)
        (folder / ".env").write_text("A=1\n", encoding="utf-8")
        git(folder, "add", "-f", ".env")
        git(folder, "commit", "-m", "oops")
        return ExecutorResult("Hecho.", 0.0, True)

    worker, jobs, job = worker_setup(run)
    await worker.process(job, 0.0)
    assert jobs.finished[0][1] == "failed" and "No he subido nada" in jobs.finished[0][2]
    assert gh == []


async def test_an_unknown_repo_and_a_broken_step_are_said(
    worker_setup: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, jobs, job = worker_setup(_committing_run([], []))
    job.repo = "kwami-labs"
    await worker.process(job, 0.0)
    assert "No conozco" in jobs.finished[0][2]

    async def broken(*args: Any) -> Any:
        raise selfwork.SelfWorkError("traer main: sin red")

    monkeypatch.setattr(worker_module, "prepare", broken)
    worker, jobs, job = worker_setup(_committing_run([], []))
    await worker.process(job, 0.0)
    assert "sin red" in jobs.finished[0][2]


async def test_the_run_gets_the_limits_in_its_brief(worker_setup: Any) -> None:
    seen: list[str] = []

    async def run(
        api_key: str, model: str, task: str, config: Any, **kwargs: Any
    ) -> ExecutorResult:
        seen.append(task)
        return ExecutorResult("Nada que hacer.", 0.0, True)

    worker, _, job = worker_setup(run)
    await worker.process(job, 0.0)
    assert "jarvis/abcd1234" in seen[0] and "Haz X" in seen[0] and "cannot push" in seen[0]


# -- real gh arguments, without GitHub -------------------------------------------


async def test_the_pull_request_is_opened_against_main(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []

    async def fake_run(*args: str, **kwargs: Any) -> tuple[int, str]:
        calls.append(args)
        if "list" in args:
            return 0, ""
        return 0, "https://github.com/x/y/pull/7"

    monkeypatch.setattr(selfwork, "_run", fake_run)
    workspace = Workspace("agent", SelfRepo("x/y"), Path("."), Path("."), "jarvis/ab")
    url = await selfwork._open_pull_request(workspace, "T", "B", True)
    assert url.endswith("/7")
    create = calls[1]
    assert create[create.index("--base") + 1] == "main"
    assert create[create.index("--head") + 1] == "jarvis/ab"
    assert "--draft" in create


async def test_an_open_pull_request_is_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(*args: str, **kwargs: Any) -> tuple[int, str]:
        assert "create" not in args
        return 0, "https://github.com/x/y/pull/3"

    monkeypatch.setattr(selfwork, "_run", fake_run)
    workspace = Workspace("agent", SelfRepo("x/y"), Path("."), Path("."), "jarvis/ab")
    assert (await selfwork._open_pull_request(workspace, "T", "B", False)).endswith("/3")


async def test_git_errors_become_a_sentence(tmp_path: Path) -> None:
    with pytest.raises(selfwork.SelfWorkError, match="paso"):
        await selfwork._must("paso", "git", "-C", str(tmp_path), "no-such-command")


async def test_a_cancelled_command_is_killed(tmp_path: Path) -> None:
    task = asyncio.create_task(
        selfwork._run(sys.executable, "-c", "import time; time.sleep(30)", cwd=tmp_path, seconds=60)
    )
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_a_url_is_not_a_drive_letter(tmp_path: Path) -> None:
    command = "git ls-remote https://github.com/x/y HEAD"
    assert check_tool_call(tmp_path, "Bash", {"command": command}) is None
    outside = {"command": "type C:\\Windows\\win.ini"}
    if os.name == "nt":
        assert "outside" in str(check_tool_call(tmp_path, "Bash", outside))


async def test_the_pull_request_is_titled_by_its_first_commit(
    tmp_path: Path, remote: Path, gh: list
) -> None:
    workspace = await _workspace(tmp_path)
    _commit(workspace, "a.txt", "uno\n")
    await publish(workspace, "La frase larga del encargo", "Hecho", PASSED)
    assert gh[0][0] == "change"


@pytest.mark.parametrize("tool", ["Agent", "Task"])
def test_team_members_are_waited_for(tmp_path: Path, tool: str) -> None:
    background = {"subagent_type": "reviewer", "run_in_background": True}
    assert "wait" in str(check_tool_call(tmp_path, tool, background))
    assert check_tool_call(tmp_path, tool, {"subagent_type": "reviewer"}) is None
    assert check_tool_call(tmp_path, "Bash", {"command": "ls", "run_in_background": True}) is None
