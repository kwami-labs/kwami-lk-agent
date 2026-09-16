"""Current LiveKit room for the running job. Set in entrypoint so tools can publish data."""

from contextvars import ContextVar
from typing import Any

_current_room: ContextVar[Any | None] = ContextVar("kwami_current_room", default=None)


def set_current_room(room: Any) -> None:
    """Set the room for the current async context (call from entrypoint)."""
    _current_room.set(room)


def get_current_room() -> Any | None:
    """Get the room for the current async context (e.g. from a tool)."""
    return _current_room.get()
