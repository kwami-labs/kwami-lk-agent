"""Reading the user's timezone off their LiveKit participant.

The `config` message is the good source, but it only exists for app sessions
that are new enough to send it. Telephony has no config message at all, and an
older app build has one without the field. Both still know where the user is,
because LiveKit participants carry attributes and metadata -- so this is the
fallback that makes "what time is it?" correct for them without a protocol
change on either side.

Deliberately mirrors how `runtime_bootstrap.resolve_kwami_id` resolves a kwami
id (attributes, then JSON metadata, several spellings), so a client that already
follows one convention does not have to learn a second.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from livekit.rtc import ParticipantKind

from src.utils.room import participant_timezone


def _human(**kwargs: Any) -> SimpleNamespace:
    kwargs.setdefault("attributes", {})
    kwargs.setdefault("metadata", None)
    kwargs.setdefault("identity", "user-1")
    kwargs.setdefault("kind", ParticipantKind.PARTICIPANT_KIND_STANDARD)
    return SimpleNamespace(**kwargs)


def _agent(**kwargs: Any) -> SimpleNamespace:
    kwargs["kind"] = ParticipantKind.PARTICIPANT_KIND_AGENT
    return _human(**kwargs)


class FakeRoom:
    def __init__(self, *participants: Any) -> None:
        self.remote_participants = {str(i): p for i, p in enumerate(participants)}


def test_no_room_yields_nothing() -> None:
    assert participant_timezone(None) is None


def test_an_empty_room_yields_nothing() -> None:
    assert participant_timezone(FakeRoom()) is None


def test_an_attribute_is_read() -> None:
    room = FakeRoom(_human(attributes={"timezone": "Europe/Madrid"}))

    assert participant_timezone(room) == "Europe/Madrid"


def test_the_camel_case_spelling_is_accepted() -> None:
    """The SDK, the app and SIP each pick their own; accept all three."""
    room = FakeRoom(_human(attributes={"timeZone": "Asia/Tokyo"}))

    assert participant_timezone(room) == "Asia/Tokyo"


def test_the_short_spelling_is_accepted() -> None:
    room = FakeRoom(_human(attributes={"tz": "America/New_York"}))

    assert participant_timezone(room) == "America/New_York"


def test_json_metadata_is_read_when_there_is_no_attribute() -> None:
    """SIP participants carry metadata rather than attributes."""
    room = FakeRoom(_human(metadata=json.dumps({"timezone": "Europe/Lisbon"})))

    assert participant_timezone(room) == "Europe/Lisbon"


def test_an_attribute_wins_over_metadata() -> None:
    room = FakeRoom(
        _human(
            attributes={"timezone": "Europe/Madrid"},
            metadata=json.dumps({"timezone": "Asia/Tokyo"}),
        )
    )

    assert participant_timezone(room) == "Europe/Madrid"


def test_the_agent_is_not_asked_where_it_lives() -> None:
    """The agent is a participant too, and its attributes are not the user's."""
    room = FakeRoom(
        _agent(attributes={"timezone": "UTC"}),
        _human(attributes={"timezone": "Europe/Madrid"}),
    )

    assert participant_timezone(room) == "Europe/Madrid"


def test_malformed_metadata_is_skipped_rather_than_raising() -> None:
    """This runs on a voice path; a bad packet costs a labelled-UTC answer."""
    room = FakeRoom(_human(metadata="not json at all"))

    assert participant_timezone(room) is None


def test_metadata_that_is_not_an_object_is_skipped() -> None:
    room = FakeRoom(_human(metadata=json.dumps(["Europe/Madrid"])))

    assert participant_timezone(room) is None


def test_blank_values_are_not_treated_as_a_zone() -> None:
    room = FakeRoom(_human(attributes={"timezone": "   "}))

    assert participant_timezone(room) is None


def test_a_non_string_attribute_is_ignored() -> None:
    room = FakeRoom(_human(attributes={"timezone": 3600}))

    assert participant_timezone(room) is None


def test_non_dict_attributes_do_not_raise() -> None:
    room = FakeRoom(_human(attributes="Europe/Madrid"))

    assert participant_timezone(room) is None


def test_a_room_that_raises_on_access_yields_nothing() -> None:
    """`remote_participants` is a property on the real Room and can throw while
    the connection is tearing down."""

    class ExplodingRoom:
        @property
        def remote_participants(self) -> dict[str, Any]:
            raise RuntimeError("disconnected")

    assert participant_timezone(ExplodingRoom()) is None


def test_the_first_participant_with_a_zone_wins() -> None:
    room = FakeRoom(
        _human(identity="a", attributes={}),
        _human(identity="b", attributes={"timezone": "Europe/Madrid"}),
    )

    assert participant_timezone(room) == "Europe/Madrid"


def test_metadata_without_a_timezone_key_moves_on_to_the_next_participant() -> None:
    """Exhausting the key list must fall through to the next participant, not
    stop the search at the first one that merely *has* metadata."""
    room = FakeRoom(
        _human(identity="a", metadata=json.dumps({"kwami_id": "kwami_1", "locale": "en-GB"})),
        _human(identity="b", attributes={"timezone": "Europe/Madrid"}),
    )

    assert participant_timezone(room) == "Europe/Madrid"


def test_every_participant_lacking_a_zone_yields_nothing() -> None:
    room = FakeRoom(
        _human(identity="a", attributes={"role": "caller"}),
        _human(identity="b", metadata=json.dumps({"kwami_id": "kwami_1"})),
    )

    assert participant_timezone(room) is None
