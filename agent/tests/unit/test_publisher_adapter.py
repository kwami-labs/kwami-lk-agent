"""The publisher owns the data-channel size limit, so callers don't reinvent it."""

from __future__ import annotations

import json

from src.adapters.publisher import LiveKitRoomPublisher, NullRoomPublisher
from src.ports.publisher import MAX_DATA_PACKET_BYTES


class RecordingParticipant:
    def __init__(self, fail: bool = False) -> None:
        self.sent: list[bytes] = []
        self._fail = fail

    async def publish_data(self, payload: bytes, *, reliable: bool = True) -> None:
        if self._fail:
            raise RuntimeError("channel closed")
        self.sent.append(payload)


class FakeRoom:
    def __init__(self, fail: bool = False) -> None:
        self.local_participant = RecordingParticipant(fail=fail)


def _result(content: str = "x" * 400, image: str | None = "data:" + "i" * 2000) -> dict:
    return {
        "title": "A product",
        "url": "https://example.com/p",
        "content": content,
        "features": ["a", "b", "c", "d", "e"],
        "image": image,
    }


async def test_a_small_payload_goes_out_untouched() -> None:
    room = FakeRoom()
    payload = {
        "type": "search_results",
        "query": "q",
        "results": [_result(content="short", image=None)],
    }

    assert await LiveKitRoomPublisher(room).publish(payload) is True

    sent = json.loads(room.local_participant.sent[0])
    assert sent["results"][0]["content"] == "short"


async def test_an_oversized_payload_is_trimmed_to_fit() -> None:
    room = FakeRoom()
    payload = {"type": "search_results", "query": "q", "results": [_result() for _ in range(30)]}

    assert await LiveKitRoomPublisher(room).publish(payload) is True

    sent = room.local_participant.sent[0]
    assert len(sent) <= MAX_DATA_PACKET_BYTES


async def test_trimming_sheds_prose_before_images() -> None:
    """A thumbnail is the point of a product card; prose is in the spoken answer."""
    room = FakeRoom()
    payload = {
        "type": "search_results",
        "query": "q",
        # Comfortably over the limit on prose alone, so trimming must engage.
        "results": [_result(content="y" * 3000, image="data:small") for _ in range(12)],
    }

    await LiveKitRoomPublisher(room).publish(payload)

    sent = json.loads(room.local_participant.sent[0])
    assert all(r["image"] == "data:small" for r in sent["results"]), "images shed too early"
    assert all(len(r["content"]) <= 120 for r in sent["results"])


async def test_the_callers_payload_is_never_mutated() -> None:
    """Trimming used to edit the dict in place, corrupting what the tool returned."""
    room = FakeRoom()
    payload = {"type": "search_results", "results": [_result() for _ in range(30)]}
    original = json.dumps(payload)

    await LiveKitRoomPublisher(room).publish(payload)

    assert json.dumps(payload) == original


async def test_no_room_is_not_an_error() -> None:
    """A tool must still answer the user when the UI is absent."""
    assert await LiveKitRoomPublisher(None).publish({"type": "x"}) is False
    assert await LiveKitRoomPublisher(object()).publish({"type": "x"}) is False


async def test_a_publish_failure_is_reported_not_raised() -> None:
    room = FakeRoom(fail=True)
    assert await LiveKitRoomPublisher(room).publish({"type": "x"}) is False


async def test_an_untrimmable_payload_is_refused_rather_than_truncated_blindly() -> None:
    room = FakeRoom()
    payload = {"type": "status", "blob": "z" * (MAX_DATA_PACKET_BYTES * 2)}

    assert await LiveKitRoomPublisher(room).publish(payload) is False
    assert room.local_participant.sent == []


async def test_set_room_lets_a_publisher_outlive_an_agent_swap() -> None:
    publisher = LiveKitRoomPublisher(None)
    room = FakeRoom()
    publisher.set_room(room)

    assert await publisher.publish({"type": "x"}) is True


async def test_the_null_publisher_records_instead_of_sending() -> None:
    publisher = NullRoomPublisher()
    assert await publisher.publish({"type": "search_results"}) is False
    assert publisher.published == [{"type": "search_results"}]
