"""Each turn's text, sent to the browser as an explicit `transcript` frame.

The worker owns the microphone on this path, so only it knows what was said;
the studio renders the transcript and checks a live guess against the phrase
there (the worker must never decide who won). `lk.transcription` carries interim
fragments without a trusted role, so this sends the frame the Kwami client
parses: `{"type": "transcript", "role": "player" | "kwami", "text": ...}`.

It used to be a closure inside `main.entrypoint`, where no test reached it and
mypy could not see that an event without an item was already handled.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..ports import RoomPublisherPort

#: The framework's chat roles -> the roles the Kwami client renders.
ROLES = {"user": "player", "assistant": "kwami"}


class TranscriptRelay:
    def __init__(self, publisher: RoomPublisherPort) -> None:
        self._publisher = publisher
        self._tasks: set[asyncio.Task[bool]] = set()

    def on_conversation_item_added(self, event: Any) -> dict[str, Any] | None:
        """Session listener. Returns the frame it published, for tests and logs."""
        item = getattr(event, "item", None)
        role = ROLES.get(getattr(item, "role", None) or "")
        text = (getattr(item, "text_content", None) or "").strip()
        if role is None or not text:
            return None
        frame = {"type": "transcript", "role": role, "text": text}
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # The framework emits from inside its loop; outside one there is no
            # room to publish to either.
            return None
        task = loop.create_task(self._publisher.publish(frame), name="transcript")
        # Keep a strong reference: the loop holds only a weak one.
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return frame
