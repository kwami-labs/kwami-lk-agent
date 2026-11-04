"""What stands in for being asked: a record of every act, and a way back from an edit.

The user's rule (2026-11-03): Jarvis stops asking across their PC, and in exchange all
it does is written down and a file it changes can be recovered. Both live in
`~/.jarvis`, which the executor's guard keeps the model out of, so it cannot tidy them.

The copy is taken before Write and Edit only. What a shell command changes has no copy:
it cannot be told from the command, which is why the summary says so to whoever reads it.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..utils.logging import get_logger

logger = get_logger("brain.audit")

AUDIT_DIR = Path.home() / ".jarvis" / "logs"
BACKUP_DIR = Path.home() / ".jarvis" / "backups"
MAX_DETAIL_CHARS = 500
MAX_BACKUP_BYTES = 50 * 1024 * 1024

_EDITS = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit"})
_DETAIL_KEYS = ("command", "file_path", "notebook_path", "path", "url", "pattern", "description")


def _detail(tool_input: dict[str, Any]) -> str:
    for key in _DETAIL_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value[:MAX_DETAIL_CHARS]
    return ""


def record(tool_name: str, tool_input: dict[str, Any], refused: str | None) -> None:
    """Add one line to this month's record. Never raises: it must not stop the work."""
    now = datetime.now(UTC)
    line = {
        "at": now.isoformat(timespec="seconds"),
        "tool": tool_name,
        "detail": _detail(tool_input),
        "refused": refused or "",
    }
    try:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        with (AUDIT_DIR / f"actions-{now:%Y-%m}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.warning("Could not write the action record: %s", e)


def back_up(root: Path, tool_name: str, tool_input: dict[str, Any]) -> Path | None:
    """Copy a file outside the Jarvis folder before it changes. The copy, if one was made."""
    if tool_name not in _EDITS:
        return None
    raw = tool_input.get("file_path") or tool_input.get("notebook_path")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        source = Path(raw).resolve()
        if source == root or root in source.parents or not source.is_file():
            return None
        if source.stat().st_size > MAX_BACKUP_BYTES:
            logger.warning("%s is too large to copy before editing", source)
            return None
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        target = BACKUP_DIR / stamp / source.relative_to(source.anchor)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    except (OSError, ValueError) as e:
        logger.warning("Could not copy %s before editing: %s", raw, e)
        return None
    return target
