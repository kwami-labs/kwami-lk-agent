"""Load a dotenv file, skipping one that git-crypt has not unlocked.

CI and a fresh clone see the ciphertext. ``load_dotenv`` decodes as UTF-8 and
raises ``UnicodeDecodeError``, which aborts test collection before any test
runs. Secrets in those environments already come from the process environment.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_unlocked_dotenv(path: Path) -> bool:
    """Load ``path`` when it is UTF-8 text. A missing or locked file is skipped."""
    if not path.is_file():
        return False
    try:
        path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    load_dotenv(path)
    return True
