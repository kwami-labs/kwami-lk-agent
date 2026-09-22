"""AgentDeps: everything a tool or handler needs, resolved once per job.

Before this existed, a tool reached for the room through a chain of fallbacks --
`get_current_room() or context.room or self.room` -- because none of the three
was reliable on its own. The ContextVar was set in the entrypoint coroutine, so
any task created outside it (including LiveKit's own data-channel dispatch)
inherited a snapshot where it was None; `self.room` was nulled on every agent
entry by a bad `on_enter` signature; and `context.room` only exists on a real
RunContext. The triple-fallback was a symptom, not a design.

The container replaces all three. It is attached to `AgentSession.userdata`,
which the framework threads through to every tool's `RunContext`, so a tool asks
for what it needs instead of hunting for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..settings import Settings, get_settings

if TYPE_CHECKING:
    from ..ports import MemoryPort, RoomPublisherPort, UsageReporterPort


@dataclass
class AgentDeps:
    """Per-job dependencies, built once in the entrypoint and passed down.

    Mutable by design: `room` and `memory` are not known at construction and are
    filled in as the job starts up, and `memory` is replaced when the frontend
    reconfigures the agent.
    """

    settings: Settings = field(default_factory=get_settings)
    room: Any = None
    memory: MemoryPort | None = None
    publisher: RoomPublisherPort | None = None
    usage_tracker: Any = None
    usage_reporter: UsageReporterPort | None = None
    http: Any = None
    # The handle a tool uses to rebuild its own pipeline. Optional because a
    # bare `AgentDeps()` (tests, telephony bootstrap before the session exists)
    # is still a valid container; tools degrade to "not available here" rather
    # than raising.
    reconfigure: Any = None


def deps_from_context(context: Any) -> AgentDeps | None:
    """Pull the container out of a tool's RunContext.

    Tolerates being handed something that is not a RunContext at all: several
    call sites synthesise a stand-in object, and a tool must degrade rather than
    raise when the wiring is incomplete.
    """
    if context is None:
        return None
    userdata = getattr(context, "userdata", None)
    if isinstance(userdata, AgentDeps):
        return userdata
    # A RunContext may expose the session rather than userdata directly.
    session = getattr(context, "session", None)
    session_userdata = getattr(session, "userdata", None)
    if isinstance(session_userdata, AgentDeps):
        return session_userdata
    return None


def room_from_context(context: Any, fallback: Any = None) -> Any:
    """Resolve the active room for a tool call.

    Order: the container, then whatever the RunContext carries, then the
    caller's own reference. Unlike the old chain this has a single authoritative
    source -- the other two are only there for call sites that synthesise a
    context object.
    """
    deps = deps_from_context(context)
    if deps is not None and deps.room is not None:
        return deps.room
    direct = getattr(context, "room", None) if context is not None else None
    return direct if direct is not None else fallback


@dataclass(frozen=True)
class SyntheticRunContext:
    """A stand-in RunContext for work the framework did not originate.

    Two data-channel routes call agent tools directly -- `browser_open_request`
    and `search_similar` -- and there is no model turn behind either, so there is
    no real `RunContext` to pass. Both used to build one inline with
    `type("Ctx", (), {"room": self.room})()`, which carried a `room` and nothing
    else. Any tool reaching for `AgentDeps` off `context.userdata` therefore got
    None on exactly those paths, silently falling back or degrading, and the two
    ad-hoc classes had to be kept in step by hand.

    Carrying `userdata` makes these paths indistinguishable from a real tool call
    as far as `deps_from_context` and `room_from_context` are concerned.
    """

    room: Any = None
    userdata: AgentDeps | None = None
