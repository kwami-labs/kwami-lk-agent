"""What the voice is doing, tool by tool, sent to the app as it happens.

LiveKit reports every tool call when it starts, while it reports progress, and when
it ends (`tool_execution_updated`, livekit-agents 1.8). The app draws each call in
its action log, so the user sees Jarvis work instead of waiting on silence. One
frame per change:

    {"type": "agent_action", "id": <call id>, "name": <tool>, "phase": "start",
     "detail": <what it is about>}
    {"type": "agent_action", "id": <call id>, "phase": "update", "detail": <progress>}
    {"type": "agent_action", "id": <call id>, "phase": "end", "status": "done" | ...}

`detail` is a few words picked from the arguments (an address, a search, a title),
never the arguments whole: the body of an email or a message has no place in a log.
Session-wide, like the transcript relay: it outlives agent swaps.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ..ports import RoomPublisherPort

#: Arguments that say what a call is about, most telling first. Bodies of messages
#: (`text`, `body`, `message`, `content`) are left out on purpose.
DETAIL_KEYS = (
    "url",
    "query",
    "title",
    "task",
    "topic",
    "name",
    "description",
    "path",
    "location",
)
DETAIL_CHARS = 140


def detail_of(arguments: Any) -> str:
    """The few words of a call's arguments worth showing. Empty when none are."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except ValueError:
            return ""
    if not isinstance(arguments, dict):
        return ""
    for key in DETAIL_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())[:DETAIL_CHARS]
    return ""


class ActionFeed:
    def __init__(self, publisher: RoomPublisherPort) -> None:
        self._publisher = publisher
        self._tasks: set[asyncio.Task[bool]] = set()

    def on_tool_execution_updated(self, event: Any) -> dict[str, Any] | None:
        """Session listener. Returns the frame it published, for tests and logs."""
        update: Any = getattr(event, "update", None)
        kind = getattr(update, "type", "")
        if kind == "tool_call_started":
            call = update.function_call
            frame: dict[str, Any] = {
                "type": "agent_action",
                "id": call.call_id,
                "name": call.name,
                "phase": "start",
                "detail": detail_of(call.arguments),
            }
        elif kind == "tool_call_updated" and not update.silent:
            frame = {
                "type": "agent_action",
                "id": update.call_id,
                "phase": "update",
                "detail": " ".join(str(update.message).split())[:DETAIL_CHARS],
            }
        elif kind == "tool_call_ended":
            frame = {
                "type": "agent_action",
                "id": update.call_id,
                "phase": "end",
                "status": update.status,
            }
        else:
            return None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # The framework emits from inside its loop; outside one there is no
            # room to publish to either.
            return None
        task = loop.create_task(self._publisher.publish(frame), name="agent_action")
        # Keep a strong reference: the loop holds only a weak one.
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return frame
