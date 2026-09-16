"""Tools publish through one adapter, not three hand-rolled trimming blocks."""

from __future__ import annotations

import json

from src.agent import KwamiAgent
from src.domain import KwamiConfig
from src.ports.publisher import MAX_DATA_PACKET_BYTES


class RecordingParticipant:
    def __init__(self) -> None:
        self.sent: list[bytes] = []

    async def publish_data(self, payload: bytes, *, reliable: bool = True) -> None:
        self.sent.append(payload)


class FakeRoom:
    def __init__(self) -> None:
        self.local_participant = RecordingParticipant()

    def messages(self) -> list[dict]:
        return [json.loads(p) for p in self.local_participant.sent]


def agent_with_room(room) -> KwamiAgent:
    agent = KwamiAgent(config=KwamiConfig(kwami_id="tenant-1"))
    agent.room = room
    return agent


async def test_dismiss_publishes_a_remove_result_message() -> None:
    room = FakeRoom()
    agent = agent_with_room(room)

    result = await agent.dismiss_search_result(None, 2)

    assert room.messages() == [{"type": "remove_result", "index": 2}]
    assert "Removed" in result


async def test_a_negative_index_is_clamped() -> None:
    room = FakeRoom()
    agent = agent_with_room(room)

    await agent.dismiss_search_result(None, -5)

    assert room.messages()[0]["index"] == 0


async def test_dismiss_reports_failure_when_there_is_no_room() -> None:
    """Previously this returned a success string regardless."""
    agent = agent_with_room(None)

    result = await agent.dismiss_search_result(None, 0)

    assert "Could not remove" in result


async def test_published_payloads_respect_the_channel_limit() -> None:
    """The limit lives in the adapter, so every tool inherits it."""
    from src.adapters.publisher import LiveKitRoomPublisher

    room = FakeRoom()
    huge = [
        {"title": "t", "url": "u", "content": "x" * 2000, "features": ["a"], "image": "data:img"}
        for _ in range(40)
    ]

    await LiveKitRoomPublisher(room).publish(
        {"type": "search_results", "query": "q", "results": huge, "answer": "a" * 1000}
    )

    assert len(room.local_participant.sent[0]) <= MAX_DATA_PACKET_BYTES


async def test_the_publisher_follows_the_room_across_an_agent_swap() -> None:
    """Room resolution goes through AgentDeps now, not a ContextVar."""
    room = FakeRoom()
    agent = KwamiAgent(config=KwamiConfig(kwami_id="tenant-1"))

    # No room yet: the tool must still answer rather than raise.
    assert "Could not remove" in await agent.dismiss_search_result(None, 0)

    agent.room = room
    assert "Removed" in await agent.dismiss_search_result(None, 0)
