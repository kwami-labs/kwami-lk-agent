"""AgentDeps replaces the room-resolution fallback chain with one source."""

from __future__ import annotations

from src.runtime.container import AgentDeps, deps_from_context, room_from_context
from src.settings import Settings


class FakeRunContext:
    """Shaped like a livekit RunContext for the attributes tools actually read."""

    def __init__(self, userdata=None, room=None, session=None) -> None:
        self.userdata = userdata
        self.room = room
        self.session = session


class FakeSession:
    def __init__(self, userdata=None) -> None:
        self.userdata = userdata


def test_deps_come_from_the_run_context() -> None:
    deps = AgentDeps(settings=Settings())
    assert deps_from_context(FakeRunContext(userdata=deps)) is deps


def test_deps_are_found_via_the_session_when_userdata_is_indirect() -> None:
    deps = AgentDeps(settings=Settings())
    ctx = FakeRunContext(userdata=None, session=FakeSession(userdata=deps))
    assert deps_from_context(ctx) is deps


def test_a_context_without_deps_is_tolerated() -> None:
    """Several call sites synthesise a stand-in context object.

    A tool must degrade rather than raise when the wiring is incomplete -- the
    old code hit this constantly, which is why it had three fallbacks.
    """
    assert deps_from_context(None) is None
    assert deps_from_context(FakeRunContext()) is None
    assert deps_from_context(FakeRunContext(userdata={"not": "deps"})) is None
    assert deps_from_context(object()) is None


def test_the_container_is_the_authoritative_room() -> None:
    room = object()
    other = object()
    deps = AgentDeps(settings=Settings(), room=room)
    ctx = FakeRunContext(userdata=deps, room=other)

    assert room_from_context(ctx, fallback=object()) is room


def test_room_falls_back_to_the_context_then_the_caller() -> None:
    ctx_room = object()
    assert room_from_context(FakeRunContext(room=ctx_room)) is ctx_room

    caller_room = object()
    assert room_from_context(FakeRunContext(), fallback=caller_room) is caller_room
    assert room_from_context(None, fallback=caller_room) is caller_room


def test_room_resolution_returns_none_when_there_is_nothing_to_find() -> None:
    assert room_from_context(None) is None
    assert room_from_context(FakeRunContext()) is None


def test_deps_with_empty_room_do_not_shadow_the_fallback() -> None:
    """A container present but not yet wired must not mask a usable room."""
    deps = AgentDeps(settings=Settings(), room=None)
    caller_room = object()
    assert room_from_context(FakeRunContext(userdata=deps), fallback=caller_room) is caller_room


def test_the_context_var_module_is_gone() -> None:
    """room_context.py was an ambient service locator; it must stay deleted.

    It was set in the entrypoint coroutine, so any task created outside it --
    including LiveKit's own data-channel dispatch -- inherited a snapshot where
    it was None. That is what the fallback chains were working around.
    """
    import importlib

    try:
        importlib.import_module("src.room_context")
    except ModuleNotFoundError:
        return
    raise AssertionError("src.room_context is back; tools should use AgentDeps instead")
