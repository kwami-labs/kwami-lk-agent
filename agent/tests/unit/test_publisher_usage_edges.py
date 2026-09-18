"""The last branches in the publish trimmer and the usage summary.

Search results arrive from providers, not from us, so the trimmer has to cope
with rows that are not shaped the way the happy path assumes -- a missing
field, a scalar where a list was expected, a row that is not an object at all.
Any of those raising would drop the whole result set rather than shrinking it.
"""

from __future__ import annotations

import json

from src.adapters.publisher import LiveKitRoomPublisher
from src.domain import UsageTracker


class RecordingParticipant:
    def __init__(self) -> None:
        self.sent: list[bytes] = []

    async def publish_data(self, payload: bytes, *, reliable: bool = True) -> None:
        self.sent.append(payload)


class FakeRoom:
    def __init__(self) -> None:
        self.local_participant = RecordingParticipant()


# =============================================================================
# Trimming rows that are not the expected shape
# =============================================================================


async def test_a_row_without_the_trim_field_is_left_alone() -> None:
    """`content` missing entirely: the row survives, it just cannot shrink."""
    room = FakeRoom()
    publisher = LiveKitRoomPublisher(room, max_bytes=400)

    sent = await publisher.publish(
        {
            "type": "search_results",
            "results": [
                {"title": "no content field", "url": "u", "features": ["a", "b", "c", "d"]},
                {"content": "x" * 500, "features": ["a", "b", "c", "d"]},
            ],
        }
    )

    assert sent is True
    payload = json.loads(room.local_participant.sent[0].decode())
    assert payload["results"][0]["title"] == "no content field"


async def test_a_non_string_trim_field_is_left_alone() -> None:
    """A provider that returns `content` as a number must not be sliced."""
    room = FakeRoom()
    publisher = LiveKitRoomPublisher(room, max_bytes=300)

    sent = await publisher.publish(
        {
            "type": "search_results",
            "results": [{"content": 12345, "features": ["a", "b", "c", "d"]}, {"content": "y" * 400}],
        }
    )

    assert sent is True
    payload = json.loads(room.local_participant.sent[0].decode())
    assert payload["results"][0]["content"] == 12345


async def test_a_row_whose_features_are_not_a_list_is_left_alone() -> None:
    room = FakeRoom()
    publisher = LiveKitRoomPublisher(room, max_bytes=300)

    sent = await publisher.publish(
        {
            "type": "search_results",
            "results": [{"content": "z" * 400, "features": "not a list"}],
        }
    )

    assert sent is True
    payload = json.loads(room.local_participant.sent[0].decode())
    assert payload["results"][0]["features"] == "not a list"


async def test_a_row_that_is_not_an_object_survives_the_last_resort_pass() -> None:
    """The image-dropping pass iterates the same rows; a bare string among them
    must not raise, or an oversized payload is dropped instead of shrunk."""
    room = FakeRoom()
    publisher = LiveKitRoomPublisher(room, max_bytes=260)

    sent = await publisher.publish(
        {
            "type": "search_results",
            "results": [
                "a bare string row",
                {"content": "q" * 300, "image": "data:" + "i" * 3000},
            ],
        }
    )

    assert sent is True
    payload = json.loads(room.local_participant.sent[0].decode())
    assert "a bare string row" in payload["results"]
    assert "image" not in payload["results"][1]


# =============================================================================
# Usage summary
# =============================================================================


def test_a_non_billable_entry_is_left_out_of_the_summary() -> None:
    """Only billable entries reach the API; a recorded-but-zero entry would
    otherwise charge the user for nothing."""
    tracker = UsageTracker()
    tracker.record_external_usage("search", "tavily/search", units_used=1.0)
    tracker.record_external_usage("search", "free/thing", units_used=0, request_count=0)

    model_ids = {item["model_id"] for item in tracker.get_usage_summary()}

    assert model_ids == {"tavily/search"}


def test_a_model_with_metadata_but_no_name_falls_back_to_the_label() -> None:
    """`if name:` after `if provider and name:` -- metadata that carries a
    provider but no model name is not enough to identify the model."""
    from src.domain.usage import _get_model_id

    metrics = type(
        "M",
        (),
        {
            "metadata": type("Meta", (), {"model_provider": "openai", "model_name": ""})(),
            "label": "fallback-label",
        },
    )()

    assert _get_model_id(metrics) == "fallback-label"
