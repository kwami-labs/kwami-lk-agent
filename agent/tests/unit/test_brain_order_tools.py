"""The voice's side of merging and signing in."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.tools import brain as brain_tools
from src.tools.brain import BrainToolsMixin


class _Bridge:
    def __init__(self) -> None:
        self.merged: list[Any] = []

    async def merge_pull_request(self, repo: str, number: int, user_words: str) -> str:
        self.merged.append((repo, number, user_words))
        return "Fusionado"

    async def connect_claude_account(self) -> str:
        return "Abierta"

    async def sites_to_connect(self) -> str:
        return "github.com: connected"


def _plain(tool: Any) -> Any:
    return getattr(tool, "__wrapped__", tool)


@pytest.fixture
def bridge(monkeypatch: pytest.MonkeyPatch) -> _Bridge:
    fake = _Bridge()
    monkeypatch.setattr(brain_tools, "_bridge", lambda _context: fake)
    return fake


async def test_merging_passes_on_the_users_words(bridge: _Bridge) -> None:
    said = await _plain(BrainToolsMixin.merge_own_pull_request)(
        SimpleNamespace(), SimpleNamespace(), " Agent ", 7, "lo probé, fusiona"
    )
    assert said == "Fusionado"
    assert bridge.merged == [("agent", 7, "lo probé, fusiona")]


async def test_signing_in_goes_through_the_bridge(bridge: _Bridge) -> None:
    tool = _plain(BrainToolsMixin.connect_claude_account)
    assert await tool(SimpleNamespace(), SimpleNamespace()) == "Abierta"


async def test_without_a_brain_they_say_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(brain_tools, "_bridge", lambda _context: None)
    merge = _plain(BrainToolsMixin.merge_own_pull_request)
    login = _plain(BrainToolsMixin.connect_claude_account)
    assert "not connected" in await merge(SimpleNamespace(), SimpleNamespace(), "agent", 1, "x")
    assert "not connected" in await login(SimpleNamespace(), SimpleNamespace())


async def test_the_sites_come_from_the_bridge(bridge: _Bridge) -> None:
    tool = _plain(BrainToolsMixin.sites_to_connect)
    assert "github" in await tool(SimpleNamespace(), SimpleNamespace())


async def test_without_a_brain_there_are_no_sites(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(brain_tools, "_bridge", lambda _context: None)
    tool = _plain(BrainToolsMixin.sites_to_connect)
    assert "not connected" in await tool(SimpleNamespace(), SimpleNamespace())
