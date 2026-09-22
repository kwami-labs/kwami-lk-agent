"""`get_zep_imports` is the lazy-import seam in front of zep_cloud.

The happy path asserts against the *real* installed SDK, as this repo requires.
The failure path cannot be reached with zep_cloud installed, so it is provoked
by making the import itself fail -- which is testing the fallback, not stubbing
the SDK to avoid it.
"""

from __future__ import annotations

import builtins
import logging

import pytest

from src.memory.utils import get_zep_imports


def test_the_real_zep_symbols_are_returned() -> None:
    from zep_cloud.client import AsyncZep as RealAsyncZep
    from zep_cloud.types import Message as RealMessage

    async_zep, zep_message, role_type = get_zep_imports()

    assert async_zep is RealAsyncZep
    assert zep_message is RealMessage
    assert role_type is not None


def test_the_returned_message_type_is_the_one_the_manager_constructs() -> None:
    """Pins the contract the memory manager depends on: a Zep message carries
    `content` and `role`, so a wrong symbol here fails loudly rather than at
    the first write."""
    _, zep_message, _ = get_zep_imports()

    message = zep_message(content="hello", role="user")

    assert message.content == "hello"


def test_a_missing_zep_cloud_degrades_to_a_triple_of_none(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    real_import = builtins.__import__

    def refuse_zep(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("zep_cloud"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", refuse_zep)

    with caplog.at_level(logging.ERROR):
        assert get_zep_imports() == (None, None, None)

    assert "Failed to import zep_cloud" in caplog.text
    # The operator needs to be told how to fix it, not just that it broke.
    assert "pip install zep-cloud" in caplog.text
