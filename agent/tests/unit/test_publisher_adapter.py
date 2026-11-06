"""The publisher owns the data-channel size limit, so callers don't reinvent it."""

from __future__ import annotations

import json

from src.adapters.publisher import (
    MESSAGE_STREAM_ATTRIBUTE,
    MESSAGE_STREAM_TOPIC,
    MESSAGE_STREAM_VALUE,
    LiveKitRoomPublisher,
    NullRoomPublisher,
    stream_readers,
)
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


# =============================================================================
# Oversized payloads, whole, to an app that reads the `kwami.message` stream
# =============================================================================


class StreamingParticipant(RecordingParticipant):
    def __init__(self, fail_stream: bool = False) -> None:
        super().__init__()
        self.streamed: list[dict] = []
        self._fail_stream = fail_stream

    async def send_text(self, text: str, *, topic: str, destination_identities: list[str]):
        if self._fail_stream:
            raise RuntimeError("stream refused")
        self.streamed.append({"text": text, "topic": topic, "to": destination_identities})


class StreamingRoom:
    def __init__(self, *readers: str, fail_stream: bool = False, others: tuple = ()) -> None:
        self.local_participant = StreamingParticipant(fail_stream=fail_stream)
        people = [_person(identity, stream=True) for identity in readers]
        people += [_person(identity, stream=False) for identity in others]
        self.remote_participants = {p.identity: p for p in people}


def _person(identity: str, *, stream: bool):
    from types import SimpleNamespace

    attributes = {MESSAGE_STREAM_ATTRIBUTE: MESSAGE_STREAM_VALUE} if stream else {}
    return SimpleNamespace(identity=identity, attributes=attributes)


def _big() -> dict:
    return {"type": "search_results", "results": [_result() for _ in range(20)]}


async def test_an_oversized_payload_goes_whole_to_an_app_that_streams() -> None:
    room = StreamingRoom("user-1", others=("agent-2",))
    payload = _big()

    assert await LiveKitRoomPublisher(room).publish(payload) is True

    [sent] = room.local_participant.streamed
    assert sent["topic"] == MESSAGE_STREAM_TOPIC
    assert sent["to"] == ["user-1"]
    assert json.loads(sent["text"]) == payload  # nothing trimmed, images included
    assert room.local_participant.sent == []


async def test_a_small_payload_stays_on_the_data_channel_even_for_a_streaming_app() -> None:
    room = StreamingRoom("user-1")

    assert await LiveKitRoomPublisher(room).publish({"type": "ping"}) is True

    assert room.local_participant.streamed == []
    assert len(room.local_participant.sent) == 1


async def test_without_a_streaming_app_an_oversized_payload_is_trimmed_as_before() -> None:
    room = StreamingRoom(others=("user-1",))

    assert await LiveKitRoomPublisher(room).publish(_big()) is True

    assert room.local_participant.streamed == []
    assert len(room.local_participant.sent[0]) <= MAX_DATA_PACKET_BYTES


async def test_a_stream_that_fails_falls_back_to_trimming(caplog) -> None:
    room = StreamingRoom("user-1", fail_stream=True)

    assert await LiveKitRoomPublisher(room).publish(_big()) is True

    assert len(room.local_participant.sent) == 1
    assert "trimming it" in caplog.text


async def test_a_room_whose_participant_cannot_stream_trims() -> None:
    room = FakeRoom()
    room.remote_participants = {"u": _person("user-1", stream=True)}  # type: ignore[attr-defined]

    assert await LiveKitRoomPublisher(room).publish(_big()) is True

    assert len(room.local_participant.sent) == 1


def test_stream_readers_needs_an_identity_and_the_attribute() -> None:
    from types import SimpleNamespace

    room = SimpleNamespace(
        remote_participants={
            "a": _person("user-1", stream=True),
            "b": _person("user-2", stream=False),
            "c": SimpleNamespace(identity="", attributes={MESSAGE_STREAM_ATTRIBUTE: "stream"}),
        }
    )
    assert stream_readers(room) == ["user-1"]
    assert stream_readers(SimpleNamespace()) == []
