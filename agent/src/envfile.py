"""Load the repo-root `.env` without tripping over git-crypt.

The tracked `.env` is encrypted with git-crypt. On a checkout that has not run
`git-crypt unlock` -- CI, a fresh worktree, a contributor without the key --
the file on disk is the encrypted blob, and `load_dotenv` dies on it with a
`UnicodeDecodeError`. That crash happened at import time of `src.main`, so it
took down every test that imports the entry point, not just the worker.

A locked file holds no usable values anyway, so it is skipped: the process
then runs on whatever the real environment provides, exactly as if the file
were absent.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# Every git-crypt encrypted file starts with this header.
GIT_CRYPT_MAGIC = b"\x00GITCRYPT\x00"


def is_git_crypt_locked(path: Path) -> bool:
    """True when `path` is still a git-crypt encrypted blob."""
    with path.open("rb") as handle:
        return handle.read(len(GIT_CRYPT_MAGIC)) == GIT_CRYPT_MAGIC


def load_env_file(path: Path) -> bool:
    """Load `path` into `os.environ` if it is a readable dotenv file.

    Returns False, loading nothing, when the file is missing or still
    encrypted. Existing environment variables are never overridden.
    """
    if not path.is_file() or is_git_crypt_locked(path):
        return False
    return load_dotenv(path)


#: The name kwami-labs gave the same loader (c9124ea).
load_unlocked_dotenv = load_env_file
