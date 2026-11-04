"""Orders for the PC that need no model: merging Jarvis's own work, signing in to Claude.

Both run in the work thread, which holds what they need, and neither reaches a model,
so nothing a web page or a mail said can talk one into running.

The user's rule for merging (2026-11-03): Jarvis asks first and the user tries the
change by hand. So it takes their own last words, saying both things, and only for a
pull request Jarvis opened itself (`jarvis/*` into `main`). Deploying stays theirs.

Signing in is Anthropic's own flow in a window of its own: the user's Claude account
for the PC's work thread. No credential passes through here (see `account.py`), which
is also why this is only for a PC of the user's, never something to offer to others.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import unicodedata
from collections.abc import Awaitable, Callable

from ..utils.logging import get_logger
from .account import binary
from .browser_tool import browser_available
from .selfwork import REPOS, SelfWorkError, _must
from .sites import show_site

logger = get_logger("brain.orders")

MERGE = "merge"
LOGIN = "login"
SITE = "site"
MODES = frozenset({MERGE, LOGIN, SITE})

_TRIED = re.compile(r"\bprob(e|ado|ada|ando|amos)\b|\bfunciona\b|\bfunciono\b|\bva bien\b")
_MERGE = re.compile(r"\bfusion\w*|\bmerge\w*")
_DENIED = re.compile(r"\bno\s+(lo\s+|la\s+)?(he\s+)?(fusion|merge|prob)\w*|\bsin\s+prob")


def _plain(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(c for c in decomposed if not unicodedata.combining(c)).split())


def merge_approved(user_words: str, last_user_turn: str) -> bool:
    """Whether the user's own last turn holds these words, and they tried it and said merge."""
    words = _plain(user_words)
    if not words or words not in _plain(last_user_turn):
        return False
    return bool(_TRIED.search(words) and _MERGE.search(words) and not _DENIED.search(words))


async def merge_pull_request(order: str) -> str:
    """Merge `<repo>#<number>`. Raises SelfWorkError with what stopped it."""
    key, _, raw_number = order.partition("#")
    repo = REPOS.get(key)
    if repo is None or not repo.slug or not raw_number.isdigit():
        raise SelfWorkError(f"No puedo fusionar {order}.")
    view = await _must(
        "leer el PR",
        "gh",
        "pr",
        "view",
        raw_number,
        "--repo",
        repo.slug,
        "--json",
        "headRefName,baseRefName,state,isDraft",
    )
    pull = json.loads(view)
    if not str(pull.get("headRefName", "")).startswith("jarvis/"):
        raise SelfWorkError("Ese PR no lo abrí yo, señor: ese lo fusiona usted.")
    if pull.get("baseRefName") != "main" or pull.get("state") != "OPEN" or pull.get("isDraft"):
        raise SelfWorkError("Ese PR no está abierto y listo contra main.")
    await _must("fusionar el PR", "gh", "pr", "merge", raw_number, "--repo", repo.slug, "--merge")
    return f"Fusionado el PR {raw_number} de {key}. El despliegue sigue siendo cosa suya, señor."


def open_login_window() -> str:
    """Claude Code's own sign-in, in a window the user sees. It ends on their side."""
    claude = binary()
    if claude is None:
        raise SelfWorkError("Claude Code no está instalado en este ordenador.")
    # An API key in the environment would stand in for the account being signed in to.
    env = {name: value for name, value in os.environ.items() if name != "ANTHROPIC_API_KEY"}
    flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    subprocess.Popen([claude, "auth", "login"], env=env, creationflags=flags)  # noqa: S603
    return (
        "Le he abierto la ventana de inicio de sesión de Claude, señor. Elija allí su cuenta: "
        "yo no veo nada de lo que escriba, y en cuanto termine trabajaré con ella."
    )


async def run_order(
    mode: str,
    order: str,
    profile: str = "",
    note_site: Callable[[str], Awaitable[object]] | None = None,
) -> str:
    """Carry out one order. Raises SelfWorkError with what stopped it."""
    if mode == MERGE:
        return await merge_pull_request(order)
    if mode == SITE:
        if not browser_available(profile):
            raise SelfWorkError("El navegador de Jarvis no está en este ordenador.")
        return await show_site(order, profile, note_site)
    return await asyncio.to_thread(open_login_window)
