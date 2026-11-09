"""The agent asks for the app's config when it never arrived.

The app's config used to be published before the agent was in the room and
was lost; the session then ran on the placeholder agent, answering as "Kwami"
with the default voice. The SDK now holds messages until the agent joins, and
as a second line the agent asks again after a few seconds without one.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import src.main as main


class FakePublisher:
    sent: list[dict[str, Any]] = []

    def __init__(self, room: Any) -> None:
        pass

    async def publish(self, payload: dict[str, Any], *, topic: str | None = None) -> bool:
        FakePublisher.sent.append(payload)
        return True


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    FakePublisher.sent = []
    monkeypatch.setattr(main, "LiveKitRoomPublisher", FakePublisher)
    monkeypatch.setattr(main, "CONFIG_REQUEST_AFTER_SECONDS", 0)


async def test_asks_until_the_config_arrives() -> None:
    state = SimpleNamespace(config_received=False)
    await main._request_config_if_missing(state, room=None)
    assert FakePublisher.sent == [{"type": "config_request"}] * main.CONFIG_REQUEST_ATTEMPTS


async def test_does_not_ask_when_it_arrived() -> None:
    state = SimpleNamespace(config_received=True)
    await main._request_config_if_missing(state, room=None)
    assert FakePublisher.sent == []
