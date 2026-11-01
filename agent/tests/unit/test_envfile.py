"""A git-crypt locked `.env` is skipped instead of crashing the import."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.envfile import GIT_CRYPT_MAGIC, is_git_crypt_locked, load_env_file


def test_locked_file_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    locked = tmp_path / ".env"
    # The bytes after the header are what made load_dotenv raise UnicodeDecodeError.
    locked.write_bytes(GIT_CRYPT_MAGIC + b"\xbb\xff\x00KWAMI_ENVFILE_TEST=1")
    monkeypatch.delenv("KWAMI_ENVFILE_TEST", raising=False)

    assert is_git_crypt_locked(locked)
    assert load_env_file(locked) is False
    assert "KWAMI_ENVFILE_TEST" not in os.environ


def test_plain_file_is_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plain = tmp_path / ".env"
    plain.write_text("KWAMI_ENVFILE_TEST=loaded\n", encoding="utf-8")
    monkeypatch.delenv("KWAMI_ENVFILE_TEST", raising=False)

    assert not is_git_crypt_locked(plain)
    assert load_env_file(plain) is True
    assert os.environ["KWAMI_ENVFILE_TEST"] == "loaded"


def test_missing_file_is_skipped(tmp_path: Path) -> None:
    assert load_env_file(tmp_path / ".env") is False
