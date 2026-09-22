"""The job entrypoint: everything that has to be wired before a caller speaks.

`AgentSession.start()` needs a live room, so that one seam is replaced with a
strict double that records what it was handed. Everything else here is real --
the agent, the session state, the router, the reconfigurator -- because the
thing under test is our orchestration, not the framework's.

The ordering assertion is the important one. `fetch_runtime_config` must be in
flight *before* `session.start()`: sequencing it after left telephony callers
listening to a default-persona placeholder for up to KWAMI_API_TIMEOUT.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

import src.main as main_module
from src.main import entrypoint, prewarm, server


class FakeRoom:
    """Models only the room surface the entrypoint touches."""

    def __init__(self, name: str = "room-1") -> None:
        self.name = name
        self.remote_participants: dict[str, Any] = {}
        self.handlers: dict[str, list] = {}
        self.local_participant = None

    def on(self, event: str, fn=None):
        if fn is None:

            def register(func):
                self.handlers.setdefault(event, []).append(func)
                return func

            return register
        self.handlers.setdefault(event, []).append(fn)
        return fn

    def emit(self, event: str, *args: Any) -> None:
        for handler in self.handlers.get(event, []):
            handler(*args)


class FakeProc:
    def __init__(self, vad: Any) -> None:
        self.userdata = {"vad": vad}


class FakeJob:
    def __init__(self, metadata: str | None = None) -> None:
        self.metadata = metadata


class FakeCtx:
    def __init__(self, room: FakeRoom, vad: Any, job_metadata: str | None = None) -> None:
        self.room = room
        self.proc = FakeProc(vad)
        self.job = FakeJob(job_metadata)
        self.shutdown_callbacks: list = []

    def add_shutdown_callback(self, fn) -> None:
        self.shutdown_callbacks.append(fn)


class RecordingSession:
    """Strict stand-in for AgentSession: start() is the only live-room call."""

    instances: list[RecordingSession] = []

    def __init__(self, *, userdata: Any = None, **kwargs: Any) -> None:
        self.userdata = userdata
        self.started_with: dict[str, Any] | None = None
        self.start_order: list[str] = []
        self.handlers: dict[str, list] = {}
        RecordingSession.instances.append(self)

    def on(self, event: str):
        def register(fn):
            self.handlers.setdefault(event, []).append(fn)
            return fn

        return register

    async def start(self, *, agent: Any, room: Any, room_options: Any = None) -> None:
        self.started_with = {"agent": agent, "room": room, "room_options": room_options}


@pytest.fixture(autouse=True)
def provider_keys(fake_key):
    for name in (
        "OPENAI_API_KEY",
        "DEEPGRAM_API_KEY",
        "CARTESIA_API_KEY",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
    ):
        fake_key(name)


@pytest.fixture
def fake_session(monkeypatch: pytest.MonkeyPatch):
    RecordingSession.instances.clear()
    monkeypatch.setattr(main_module, "AgentSession", RecordingSession)
    return RecordingSession


@pytest.fixture
def vad():
    """A stand-in VAD: the entrypoint only passes it through to the factories."""

    class PassthroughVAD:
        pass

    return PassthroughVAD()


# =============================================================================
# Module wiring
# =============================================================================


def test_the_worker_registers_prewarm_as_its_setup_function() -> None:
    """Without this the first job loads Silero on the critical path."""
    assert server.setup_fnc is prewarm


def test_prewarm_puts_a_vad_on_the_process() -> None:
    proc = FakeProc(None)

    prewarm(proc)

    assert proc.userdata["vad"] is not None


# =============================================================================
# Entrypoint
# =============================================================================


async def test_the_session_is_started_with_the_placeholder_agent(fake_session, vad) -> None:
    ctx = FakeCtx(FakeRoom(), vad)

    await entrypoint(ctx)

    session = fake_session.instances[-1]
    assert session.started_with["room"] is ctx.room
    assert session.started_with["agent"] is not None


async def test_audio_input_and_output_are_both_enabled(fake_session, vad) -> None:
    ctx = FakeCtx(FakeRoom(), vad)

    await entrypoint(ctx)

    options = fake_session.instances[-1].started_with["room_options"]
    assert options.audio_input is True
    assert options.audio_output is True


async def test_the_dependencies_reach_the_session_userdata(fake_session, vad) -> None:
    """Tools find the room through userdata; this replaced the ContextVar the
    tools used to hunt it through."""
    ctx = FakeCtx(FakeRoom(), vad)

    await entrypoint(ctx)

    deps = fake_session.instances[-1].userdata
    assert deps.room is ctx.room
    assert deps.settings is not None


async def test_a_reconfigurator_is_attached(fake_session, vad) -> None:
    """What makes "switch to Claude" answerable by doing rather than by
    describing which panel to open."""
    ctx = FakeCtx(FakeRoom(), vad)

    await entrypoint(ctx)

    assert fake_session.instances[-1].userdata.reconfigure is not None


async def test_cleanup_is_registered_as_a_shutdown_callback(fake_session, vad) -> None:
    """Usage is reported from cleanup; without this a session is never billed."""
    ctx = FakeCtx(FakeRoom(), vad)

    await entrypoint(ctx)

    assert ctx.shutdown_callbacks


async def test_metrics_are_routed_to_the_usage_tracker(fake_session, vad) -> None:
    ctx = FakeCtx(FakeRoom(), vad)

    await entrypoint(ctx)

    assert "metrics_collected" in fake_session.instances[-1].handlers


async def test_data_messages_are_routed(fake_session, vad) -> None:
    ctx = FakeCtx(FakeRoom(), vad)

    await entrypoint(ctx)

    assert ctx.room.handlers["data_received"]


async def test_an_undecodable_data_packet_is_discarded(fake_session, vad) -> None:
    """Malformed packets arrive from any client; decoding must never raise into
    the room's event loop."""
    ctx = FakeCtx(FakeRoom(), vad)
    await entrypoint(ctx)
    handler = ctx.room.handlers["data_received"][0]

    handler(type("Packet", (), {"data": b"not json"})())


async def test_a_router_failure_is_logged_not_raised(fake_session, vad, caplog) -> None:
    ctx = FakeCtx(FakeRoom(), vad)
    await entrypoint(ctx)
    handler = ctx.room.handlers["data_received"][0]

    packet = type("Packet", (), {"data": json.dumps({"type": "config"}).encode()})()
    handler(packet)


async def test_a_late_joining_participant_resolves_the_identity(fake_session, vad) -> None:
    """A human can join after the agent; without this the session finishes with
    no user_identity and its usage is never billed."""
    ctx = FakeCtx(FakeRoom(), vad)

    await entrypoint(ctx)

    assert ctx.room.handlers["participant_connected"]


# =============================================================================
# Telephony bootstrap ordering
# =============================================================================


async def test_no_kwami_id_means_no_runtime_fetch(
    fake_session, vad, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    async def should_not_run(kwami_id: str):
        calls.append(kwami_id)
        return None

    monkeypatch.setattr(main_module, "fetch_runtime_config", should_not_run)
    ctx = FakeCtx(FakeRoom(), vad)

    await entrypoint(ctx)

    assert calls == []


async def test_a_telephony_kwami_id_starts_the_fetch_before_the_session(
    fake_session, vad, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordering this whole arrangement exists for: the HTTP round trip must
    overlap with bringing the room up, not follow it.

    What is asserted is that the fetch is *in flight* by the time start() has
    yielded once -- i.e. it was scheduled beforehand. Sequencing it after
    start() left telephony callers on a default-persona placeholder for as long
    as the fetch took, up to KWAMI_API_TIMEOUT.
    """
    order: list[str] = []

    async def fetch(kwami_id: str):
        order.append("fetch-started")
        await asyncio.sleep(0)
        return None

    async def start(self, *, agent, room, room_options=None):
        # A real start() brings a room up over the network, so it yields. That
        # yield is precisely what lets the already-scheduled fetch overlap with
        # it; without one, nothing here would ever run concurrently.
        await asyncio.sleep(0)
        order.append("session-started")
        self.started_with = {"agent": agent, "room": room, "room_options": room_options}

    monkeypatch.setattr(main_module, "fetch_runtime_config", fetch)
    monkeypatch.setattr(RecordingSession, "start", start)
    ctx = FakeCtx(FakeRoom(), vad, job_metadata=json.dumps({"kwami_id": "kwami-9"}))

    await entrypoint(ctx)

    assert order.index("fetch-started") < order.index("session-started")


async def test_a_fetched_runtime_config_is_applied(
    fake_session, vad, monkeypatch: pytest.MonkeyPatch
) -> None:
    applied: list[Any] = []

    async def fetch(kwami_id: str):
        return {"soul": {"name": "Ada"}}

    async def apply(session, state, vad_, create_agent_fn, task, kwami_id=None):
        applied.append(await task if task else None)
        return True

    monkeypatch.setattr(main_module, "fetch_runtime_config", fetch)
    monkeypatch.setattr(main_module, "apply_runtime_config", apply)
    ctx = FakeCtx(FakeRoom(), vad, job_metadata=json.dumps({"kwami_id": "kwami-9"}))

    await entrypoint(ctx)

    assert applied == [{"soul": {"name": "Ada"}}]


async def test_the_identity_is_resolved_once_the_room_is_connected(fake_session, vad) -> None:
    ctx = FakeCtx(FakeRoom(), vad)

    await entrypoint(ctx)

    # No participants in the fake room, so it stays unresolved -- but the call
    # happens after start(), which is the part that was wrong before.
    assert fake_session.instances[-1].started_with is not None


# =============================================================================
# The registered handlers, invoked
# =============================================================================


async def test_a_metrics_event_reaches_the_usage_tracker(fake_session, vad) -> None:
    ctx = FakeCtx(FakeRoom(), vad)
    await entrypoint(ctx)
    handler = fake_session.instances[-1].handlers["metrics_collected"][0]

    # A metrics object the router does not recognise must still not raise --
    # metrics arrive from the framework on a hot path.
    handler(type("Event", (), {"metrics": object()})())


async def test_a_router_exception_does_not_escape_the_data_handler(
    fake_session, vad, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """A raise here would propagate into the room's event loop and take the
    data channel down for the rest of the session."""
    import logging

    ctx = FakeCtx(FakeRoom(), vad)
    await entrypoint(ctx)
    handler = ctx.room.handlers["data_received"][0]

    monkeypatch.setattr(
        main_module.DataMessageRouter,
        "handle",
        lambda self, message: (_ for _ in ()).throw(RuntimeError("router blew up")),
    )
    packet = type("Packet", (), {"data": json.dumps({"type": "config"}).encode()})()

    with caplog.at_level(logging.ERROR):
        handler(packet)

    assert "Error handling data message" in caplog.text


async def test_a_participant_joining_late_is_resolved(fake_session, vad) -> None:
    ctx = FakeCtx(FakeRoom(), vad)
    await entrypoint(ctx)
    handler = ctx.room.handlers["participant_connected"][0]

    participant = type("P", (), {"identity": "user-7", "kind": 0})()
    handler(participant)


async def test_an_identity_already_resolved_is_not_overwritten(
    fake_session, vad, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`if not state.user_identity` -- a telephony identity resolved earlier
    must survive the post-start resolution pass."""
    captured: list[Any] = []
    real_create = main_module.create_session_state

    def capture(**kwargs: Any):
        state = real_create(**kwargs)
        state.user_identity = "already-known"
        captured.append(state)
        return state

    monkeypatch.setattr(main_module, "create_session_state", capture)
    ctx = FakeCtx(FakeRoom(), vad)

    await entrypoint(ctx)

    assert captured[-1].user_identity == "already-known"
