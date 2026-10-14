"""A git-crypt locked .env must not crash import or collection."""

from __future__ import annotations

import os

from src.envfile import load_unlocked_dotenv

_PROBE = "KWAMI_ENVFILE_PROBE"


def test_a_missing_env_file_is_skipped(tmp_path) -> None:
    assert load_unlocked_dotenv(tmp_path / ".env") is False


def test_a_git_crypt_locked_env_is_skipped(tmp_path, monkeypatch) -> None:
    path = tmp_path / ".env"
    path.write_bytes(b"\x00GITCRYPT\xbb\xffnot utf8")
    monkeypatch.delenv(_PROBE, raising=False)

    assert load_unlocked_dotenv(path) is False
    assert _PROBE not in os.environ


def test_a_utf8_env_file_is_loaded(tmp_path, monkeypatch) -> None:
    path = tmp_path / ".env"
    path.write_text(f"{_PROBE}=loaded\n", encoding="utf-8")
    monkeypatch.delenv(_PROBE, raising=False)

    assert load_unlocked_dotenv(path) is True
    assert os.environ[_PROBE] == "loaded"
