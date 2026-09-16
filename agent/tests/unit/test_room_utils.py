"""Participant classification — the input to usage reporting.

`user_identity` decides whether a session's usage is billed at all. The
entrypoint used to scan `ctx.room.remote_participants` before the room was
connected, so the roster was always empty and the identity always None; usage
was then silently dropped. These cover the resolution itself.
"""

from __future__ import annotations

from livekit import rtc

from src.utils.room import is_agent_participant, resolve_user_identity

AGENT = rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
STANDARD = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
SIP = rtc.ParticipantKind.PARTICIPANT_KIND_SIP


class FakeParticipant:
    def __init__(self, identity: str, kind: int = STANDARD) -> None:
        self.identity = identity
        self.kind = kind


class FakeRoom:
    def __init__(self, *participants: FakeParticipant) -> None:
        self.remote_participants = {p.identity: p for p in participants}


def test_agents_are_recognised_by_kind_not_by_name() -> None:
    """The old check was `identity.startswith("agent")`, a string heuristic."""
    assert is_agent_participant(FakeParticipant("kwami-agent", AGENT)) is True
    assert is_agent_participant(FakeParticipant("agent-smith", STANDARD)) is False


def test_a_telephony_caller_counts_as_human() -> None:
    """SIP callers are the billing case that matters most."""
    assert is_agent_participant(FakeParticipant("sip_+34600000000", SIP)) is False


def test_resolve_user_identity_skips_agents() -> None:
    room = FakeRoom(
        FakeParticipant("kwami-agent", AGENT),
        FakeParticipant("user-42", STANDARD),
    )
    assert resolve_user_identity(room) == "user-42"


def test_resolve_user_identity_finds_a_sip_caller() -> None:
    room = FakeRoom(FakeParticipant("sip_+34600000000", SIP))
    assert resolve_user_identity(room) == "sip_+34600000000"


def test_resolve_user_identity_is_none_when_only_agents_are_present() -> None:
    room = FakeRoom(FakeParticipant("kwami-agent", AGENT))
    assert resolve_user_identity(room) is None


def test_resolve_user_identity_on_an_empty_room() -> None:
    """The pre-connect state the entrypoint used to read."""
    assert resolve_user_identity(FakeRoom()) is None


def test_blank_identities_are_ignored() -> None:
    room = FakeRoom(FakeParticipant("", STANDARD), FakeParticipant("user-7", STANDARD))
    assert resolve_user_identity(room) == "user-7"


def test_missing_kind_does_not_raise() -> None:
    class Bare:
        identity = "user-9"

    assert is_agent_participant(Bare()) is False
