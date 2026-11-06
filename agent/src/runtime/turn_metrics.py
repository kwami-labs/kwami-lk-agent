"""How long each reply kept the user waiting, sent to the app's Metrics panel.

The SDK has always understood a `metrics` data message: `LiveKitAdapter` hands
it to `VoiceSession.updateLatency`, and the panel shows it as STT, end of turn,
LLM, TTS and overall latency. Nothing on this side ever sent one, so the panel
showed dashes or the client's own guesses -- and the one number a voice product
lives or dies by was only readable by grepping worker logs.

livekit-agents already measures every turn and stores the timings on the chat
items it commits. The two halves of a turn arrive separately: the user's
message carries how long the transcript and the end-of-turn decision took, the
assistant's carries the LLM's first token, the TTS's first audio and the
end-to-end wait (on a realtime model too, since 1.8.7, when the VAD is local).
This keeps the user's half until the reply lands, then publishes both in the
milliseconds the SDK expects, and logs the same line for `lk agent logs`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from ..ports import RoomPublisherPort
from ..utils.logging import get_logger

logger = get_logger("turn_metrics")

#: Framework timing (seconds, on the user's message) -> SDK field (milliseconds).
USER_FIELDS: tuple[tuple[str, str], ...] = (
    ("transcription_delay", "stt"),
    ("end_of_turn_delay", "endOfTurn"),
)

#: Framework timing (seconds, on the assistant's message) -> SDK field (milliseconds).
ASSISTANT_FIELDS: tuple[tuple[str, str], ...] = (
    ("llm_node_ttft", "llm"),
    ("tts_node_ttfb", "tts"),
    ("e2e_latency", "overall"),
)


def _milliseconds(
    metrics: Mapping[str, Any], fields: tuple[tuple[str, str], ...]
) -> dict[str, int]:
    """The fields that were measured, in whole milliseconds; the rest left out.

    A turn that never reached a stage (no TTS on a realtime voice, an
    interruption before the first token) has no timing for it, and a zero would
    read on the panel as an instant stage rather than an absent one.
    """
    out: dict[str, int] = {}
    for source, target in fields:
        value = metrics.get(source)
        # bool is an int; a flag is never a duration.
        if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
            continue
        out[target] = round(value * 1000)
    return out


class TurnMetricsRelay:
    """Joins a turn's two halves and publishes them as one `metrics` message."""

    def __init__(self, publisher: RoomPublisherPort) -> None:
        self._publisher = publisher
        self._user: dict[str, int] = {}
        self._tasks: set[asyncio.Task[bool]] = set()

    def on_conversation_item_added(self, event: Any) -> dict[str, Any] | None:
        """Session listener. Returns the payload it published, for tests and logs."""
        item = getattr(event, "item", None)
        role = getattr(item, "role", None)
        metrics = getattr(item, "metrics", None)
        if not isinstance(metrics, Mapping):
            metrics = {}

        if role == "user":
            # A new turn starts; whatever the previous one left is stale.
            self._user = _milliseconds(metrics, USER_FIELDS)
            return None
        if role != "assistant":
            return None

        latency = {**self._user, **_milliseconds(metrics, ASSISTANT_FIELDS)}
        # Only the first reply to a turn owns the user's half: a reply after a
        # tool call, or the greeting, did not wait on that transcript.
        self._user = {}
        if not latency:
            return None

        payload = {"type": "metrics", "metrics": latency}
        logger.info(
            "Turn latency: %s",
            ", ".join(f"{name}={value}ms" for name, value in latency.items()),
        )
        self._publish(payload)
        return payload

    def _publish(self, payload: dict[str, Any]) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # The framework emits from inside its loop; outside one there is no
            # room to publish to either.
            return
        task = loop.create_task(self._publisher.publish(payload), name="turn_metrics")
        # Keep a strong reference: the loop holds only a weak one.
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
