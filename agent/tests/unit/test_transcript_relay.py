"""`runtime/transcript_relay.py`: each turn's text reaches the browser with its role."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from livekit.agents.llm import ChatMessage

from src.adapters.publisher import NullRoomPublisher
from src.runtime.transcript_relay import TranscriptRelay


def _added(role: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(item=ChatMessage(role=role, content=[text]))


async def _settle() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


async def test_the_users_turn_is_sent_as_the_player() -> None:
    publisher = NullRoomPublisher()
    relay = TranscriptRelay(publisher)

    frame = relay.on_conversation_item_added(_added("user", "  hola  "))
    await _settle()

    assert frame == {"type": "transcript", "role": "player", "text": "hola"}
    assert publisher.published == [frame]


async def test_the_assistants_turn_is_sent_as_the_kwami() -> None:
    publisher = NullRoomPublisher()
    relay = TranscriptRelay(publisher)

    frame = relay.on_conversation_item_added(_added("assistant", "Buenas."))
    await _settle()

    assert frame is not None and frame["role"] == "kwami"
    assert publisher.published == [frame]


async def test_other_roles_empty_text_and_missing_items_are_skipped() -> None:
    publisher = NullRoomPublisher()
    relay = TranscriptRelay(publisher)

    assert relay.on_conversation_item_added(_added("system", "reglas")) is None
    assert relay.on_conversation_item_added(_added("user", "   ")) is None
    assert relay.on_conversation_item_added(SimpleNamespace()) is None
    await _settle()

    assert publisher.published == []


def test_outside_a_running_loop_nothing_is_published() -> None:
    publisher = NullRoomPublisher()
    relay = TranscriptRelay(publisher)

    assert relay.on_conversation_item_added(_added("user", "hola")) is None
    assert publisher.published == []
