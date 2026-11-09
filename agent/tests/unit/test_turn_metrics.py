"""`runtime/turn_metrics.py`: each reply's latency reaches the app's Metrics panel.

Built on real `ChatMessage` items, so the field names are the framework's and a
rename in livekit-agents fails here rather than silently emptying the panel.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from livekit.agents.llm import ChatMessage

from src.adapters.publisher import NullRoomPublisher
from src.runtime.turn_metrics import TurnMetricsRelay


def _added(role: str, **metrics: Any) -> SimpleNamespace:
    return SimpleNamespace(item=ChatMessage(role=role, content=["hola"], metrics=metrics))


async def _relay() -> tuple[TurnMetricsRelay, NullRoomPublisher]:
    publisher = NullRoomPublisher()
    return TurnMetricsRelay(publisher), publisher


async def _settle() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


class TestATurn:
    async def test_both_halves_are_published_in_milliseconds(self) -> None:
        relay, publisher = await _relay()

        assert (
            relay.on_conversation_item_added(
                _added("user", transcription_delay=0.21, end_of_turn_delay=0.8)
            )
            is None
        )
        payload = relay.on_conversation_item_added(
            _added("assistant", llm_node_ttft=0.4567, tts_node_ttfb=0.25, e2e_latency=1.5)
        )
        await _settle()

        expected = {
            "type": "metrics",
            "metrics": {"stt": 210, "endOfTurn": 800, "llm": 457, "tts": 250, "overall": 1500},
        }
        assert payload == expected
        assert publisher.published == [expected]

    async def test_a_stage_that_never_ran_is_left_out_not_zeroed(self) -> None:
        # A realtime voice has no TTS of ours, and no transcript delay either.
        relay, publisher = await _relay()

        relay.on_conversation_item_added(_added("user"))
        payload = relay.on_conversation_item_added(_added("assistant", e2e_latency=0.9))
        await _settle()

        assert payload == {"type": "metrics", "metrics": {"overall": 900}}
        assert publisher.published == [payload]

    async def test_the_users_half_belongs_to_the_first_reply_only(self) -> None:
        relay, _ = await _relay()

        relay.on_conversation_item_added(_added("user", transcription_delay=0.3))
        relay.on_conversation_item_added(_added("assistant", e2e_latency=1.0))
        after_tool = relay.on_conversation_item_added(_added("assistant", llm_node_ttft=0.5))

        assert after_tool == {"type": "metrics", "metrics": {"llm": 500}}

    async def test_a_new_user_turn_replaces_one_never_answered(self) -> None:
        relay, _ = await _relay()

        relay.on_conversation_item_added(_added("user", transcription_delay=2.0))
        relay.on_conversation_item_added(_added("user", transcription_delay=0.1))
        payload = relay.on_conversation_item_added(_added("assistant", e2e_latency=1.0))

        assert payload == {"type": "metrics", "metrics": {"stt": 100, "overall": 1000}}


class TestWhatIsNotSent:
    async def test_a_reply_with_no_timings_publishes_nothing(self) -> None:
        relay, publisher = await _relay()

        assert relay.on_conversation_item_added(_added("assistant")) is None
        await _settle()

        assert publisher.published == []

    async def test_other_items_are_ignored(self) -> None:
        relay, _ = await _relay()

        assert relay.on_conversation_item_added(SimpleNamespace(item=None)) is None
        assert relay.on_conversation_item_added(_added("system", e2e_latency=1.0)) is None

    async def test_flags_negatives_and_text_are_not_durations(self) -> None:
        relay, _ = await _relay()
        item = SimpleNamespace(
            role="assistant",
            metrics={"llm_node_ttft": True, "tts_node_ttfb": -0.1, "e2e_latency": "1.2"},
        )

        assert relay.on_conversation_item_added(SimpleNamespace(item=item)) is None

    async def test_an_item_without_a_metrics_mapping_counts_as_unmeasured(self) -> None:
        relay, _ = await _relay()
        item = SimpleNamespace(role="assistant", metrics=None)

        assert relay.on_conversation_item_added(SimpleNamespace(item=item)) is None


class TestOutsideTheLoop:
    def test_no_running_loop_means_nothing_to_publish_to(self) -> None:
        publisher = NullRoomPublisher()
        relay = TurnMetricsRelay(publisher)

        payload = relay.on_conversation_item_added(_added("assistant", e2e_latency=1.0))

        # Measured and returned, but there is no loop to send it on.
        assert payload == {"type": "metrics", "metrics": {"overall": 1000}}
        assert publisher.published == []
