"""Whether the Claude account of this PC can do the brain's thinking.

The work thread (`worker.py`) runs on the user's PC, under the user's own login to
Claude Code, so a task there draws on the user's plan instead of an API key. The
sign-in itself always happens in Anthropic's flow (`claude auth login`). This
module never reads the credentials: it asks the unmodified binary whether there
is a login, and passes on a yes or a no. Anything more it prints (the email, the
organisation) stops here.

That is also why this is not offered as "log in with Claude" for other users: a
deployment may not collect or route anyone's Claude login. Someone else can have
the same thing only by running a work thread on their own PC, signed in there.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path

from ..utils.logging import get_logger

logger = get_logger("brain.account")

STATUS_TIMEOUT_SECONDS = 20.0


def _binary() -> str | None:
    """The Claude Code the executor itself runs: the SDK's, else the one on PATH."""
    try:
        import claude_agent_sdk
    except ImportError:
        return shutil.which("claude")
    folder = Path(claude_agent_sdk.__file__).parent / "_bundled"
    for name in ("claude.exe", "claude"):
        if (folder / name).is_file():
            return str(folder / name)
    return shutil.which("claude")


async def sign_in() -> int:
    """Run Claude Code's own sign-in on this terminal. Its exit code.

    The same binary the worker runs, which may be the only one on this PC: it is
    usually not on PATH.
    """
    binary = _binary()
    if binary is None:
        logger.error("Claude Code is not installed here")
        return 2
    process = await asyncio.create_subprocess_exec(binary, "auth", "login")
    return await process.wait()


def read_login(raw: str) -> bool:
    """Whether `claude auth status` reports a Claude account, not an API key."""
    try:
        data = json.loads(raw)
    except ValueError:
        return False
    return (
        isinstance(data, dict)
        and data.get("loggedIn") is True
        and data.get("authMethod") == "claude.ai"
    )


async def account_login() -> bool:
    """Whether Claude Code on this PC is signed in to a Claude account. Never raises."""
    binary = _binary()
    if binary is None:
        return False
    # An API key in the environment would be reported instead of the login.
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    try:
        process = await asyncio.create_subprocess_exec(
            binary,
            "auth",
            "status",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), STATUS_TIMEOUT_SECONDS)
        except TimeoutError:
            return False
        finally:
            # Also when cancelled: the worker stopping must not leave it behind.
            if process.returncode is None:
                process.kill()
                await process.wait()
    except OSError as e:
        logger.warning("Could not ask Claude Code about its login: %s", e)
        return False
    return read_login(stdout.decode("utf-8", errors="replace"))
