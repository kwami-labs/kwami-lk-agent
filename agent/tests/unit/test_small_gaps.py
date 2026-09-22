"""The remaining edges: realtime voice resolution, the persona alias, falsy-safe
parsing, and duplicate-agent priority.

Each of these is one branch away from a real failure -- Gemini Live rejects a
lower-cased voice name, `persona` is still what older clients send, and the
duplicate guard decides which of two agents hangs up.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from src.constants import RealtimeVoices, resolve_realtime_voice
from src.domain import KwamiConfig, KwamiSoulConfig
from src.domain.parsing import boolean, section
from src.utils.room import should_disconnect_as_duplicate

# =============================================================================
# resolve_realtime_voice
# =============================================================================


def test_a_known_voice_keeps_the_providers_capitalisation() -> None:
    """Gemini Live rejects `puck` for `Puck`, so matching is case-insensitive
    but the provider's own spelling is what comes back."""
    provider, voices = next(iter(RealtimeVoices.BY_PROVIDER.items()))
    expected = sorted(voices)[0]

    assert resolve_realtime_voice(provider, expected.lower()) == expected


def test_an_unknown_provider_resolves_nothing() -> None:
    assert resolve_realtime_voice("nonesuch", "Puck") is None


@pytest.mark.parametrize("voice", ["", "   ", None])
def test_an_empty_voice_resolves_nothing(voice: str | None) -> None:
    provider = next(iter(RealtimeVoices.BY_PROVIDER))

    assert resolve_realtime_voice(provider, voice) is None


def test_a_voice_from_another_provider_resolves_nothing() -> None:
    """An OpenAI voice left over from a pipeline switch must not be sent to
    Gemini, where it is a hard rejection."""
    provider = next(iter(RealtimeVoices.BY_PROVIDER))

    assert resolve_realtime_voice(provider, "definitely-not-a-voice") is None


def test_a_null_provider_resolves_nothing() -> None:
    assert resolve_realtime_voice(None, "Puck") is None


# =============================================================================
# The persona alias
# =============================================================================


def test_persona_reads_through_to_soul() -> None:
    """Older clients and modules still say `persona`."""
    config = KwamiConfig(soul=KwamiSoulConfig(name="Ada"))

    assert config.persona is config.soul
    assert config.persona.name == "Ada"


def test_assigning_persona_writes_through_to_soul() -> None:
    config = KwamiConfig()

    config.persona = KwamiSoulConfig(name="Grace")

    assert config.soul.name == "Grace"


# =============================================================================
# Falsy-safe parsing
# =============================================================================


def test_a_section_stops_at_a_non_mapping() -> None:
    """Chaining must not raise when an intermediate key is a scalar; the whole
    config message used to be dropped by exactly this."""
    assert section({"voice": "not a dict"}, "voice", "tts") == {}


def test_a_nested_section_is_returned() -> None:
    assert section({"voice": {"tts": {"speed": 1.0}}}, "voice", "tts") == {"speed": 1.0}


def test_a_missing_nested_section_is_an_empty_dict() -> None:
    assert section({"voice": {}}, "voice", "tts") == {}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        (1.5, True),
        (0.0, False),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("1", True),
    ],
)
def test_booleans_are_read_from_every_wire_spelling(value: object, expected: bool) -> None:
    assert boolean({"flag": value}, "flag") is expected


# =============================================================================
# Duplicate-agent priority
# =============================================================================


def agent_participant(identity: str, *, disconnect_reason: object = None, kind: int = 4):
    return SimpleNamespace(
        identity=identity, disconnect_reason=disconnect_reason, kind=kind, name=identity
    )


class RoomWith:
    def __init__(self, *participants) -> None:
        self.remote_participants = {p.identity: p for p in participants}


async def test_the_later_identity_disconnects(caplog) -> None:
    """Two agents in one room both answer, and the user hears doubled audio.
    Ordering by identity gives both sides the same answer without coordination."""
    room = RoomWith(agent_participant("agent-aaa"))

    with caplog.at_level(logging.WARNING):
        assert await should_disconnect_as_duplicate(room, "agent-zzz") is True

    assert "has priority" in caplog.text


async def test_the_earlier_identity_stays(caplog) -> None:
    room = RoomWith(agent_participant("agent-zzz"))

    with caplog.at_level(logging.INFO):
        assert await should_disconnect_as_duplicate(room, "agent-aaa") is False

    assert "has priority over" in caplog.text


async def test_an_agent_on_its_way_out_does_not_count(caplog) -> None:
    """Membership of remote_participants already means connected; the only
    thing worth filtering is one that is leaving."""
    room = RoomWith(agent_participant("agent-aaa", disconnect_reason="CLIENT_INITIATED"))

    with caplog.at_level(logging.DEBUG):
        assert await should_disconnect_as_duplicate(room, "agent-zzz") is False

    assert "none are actively connected" in caplog.text


async def test_an_empty_room_is_not_a_duplicate() -> None:
    assert await should_disconnect_as_duplicate(RoomWith(), "agent-1") is False
