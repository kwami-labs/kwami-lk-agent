"""Kwami Agent - Entry point for LiveKit Cloud agent sessions."""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the root directory
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from livekit import rtc
from livekit.agents import (
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    room_io,
)

from .domain import KwamiConfig
from .factories.vad import prewarm_vad
from .health import heartbeat_path, run_heartbeat, write_heartbeat
from .runtime import (
    AgentDeps,
    DataMessageRouter,
    Reconfigurator,
    apply_runtime_config,
    decode_data_message,
    resolve_identity_on_join,
    route_metrics,
)
from .runtime.pipeline import create_agent_from_config
from .runtime_bootstrap import fetch_runtime_config, resolve_kwami_id
from .session import create_session_state
from .settings import Settings, get_settings, set_settings
from .telemetry import configure_tracing
from .utils.logging import (
    bind_session_fields,
    configure_logging,
    get_logger,
    session_context,
)
from .utils.room import resolve_user_identity

logger = get_logger()

# Resolve credentials once, here, after load_dotenv has run. Everything
# downstream takes them from Settings rather than reading os.environ itself.
set_settings(Settings.from_env())
# Install the correlation filter (and JSON output when asked for) before the
# first line is written, so startup is correlated too.
configure_logging()
configure_tracing()
logger.info("Settings resolved: %s", get_settings().describe())

server = AgentServer()

# Strong reference to the heartbeat task: the loop holds only a weak one, and a
# collected heartbeat would read to the probe as a dead worker.
_heartbeat_task: asyncio.Task | None = None


def prewarm(proc: JobProcess) -> None:
    """Prewarm the VAD model for faster startup."""
    proc.userdata["vad"] = prewarm_vad()


server.setup_fnc = prewarm


@server.on("worker_registered")
def _on_worker_registered(*_args: object) -> None:
    """Start reporting healthy once LiveKit has actually accepted this worker.

    This is the fact the container's health probe could not previously see. It
    answered "ok" whenever its own HTTP thread was alive, so a worker that was
    running but unregistered -- a dropped websocket, an auth loop -- looked
    identical to one taking calls.

    `worker_registered` fires again after a reconnect, so the guard keeps a
    single heartbeat task rather than stacking one per reconnection.
    """
    global _heartbeat_task

    write_heartbeat("registered")
    if _heartbeat_task is not None and not _heartbeat_task.done():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # The SDK emits this from inside the event loop, so in practice there is
        # always one. Handled rather than assumed: a worker that cannot start
        # its heartbeat should say so and keep serving calls, not raise inside
        # an event handler where the traceback goes nowhere useful.
        logger.warning("worker_registered fired outside the event loop; no heartbeat started")
        return
    _heartbeat_task = loop.create_task(run_heartbeat(), name="health_heartbeat")
    logger.info("Health heartbeat started at %s", heartbeat_path())


@server.rtc_session(agent_name="kwami-agent")
async def entrypoint(ctx: JobContext) -> None:
    """Main entry point for Kwami agent sessions.

    Binds the room for the whole job before doing anything else. Every record
    emitted from here on carries it -- ours and the SDK's -- so one session can
    be pulled out of a worker serving many without grepping for a string that
    appeared in three lines out of several hundred.
    """
    with session_context(room=ctx.room.name):
        await _run_session(ctx)


async def _run_session(ctx: JobContext) -> None:
    """The session itself, inside the logging context bound above."""
    logger.info("Kwami session starting")

    # The room is NOT connected yet -- JobContext.room is populated by
    # session.start() below, so participants are resolved after that point.
    user_identity = None

    # Get prewarmed VAD
    vad = ctx.proc.userdata["vad"]

    # Create initial agent with default configuration.
    # Skip greeting -- this is a placeholder agent until the frontend sends
    # the real config via the "config" data message. The configured agent
    # will greet properly with the correct persona, voice, and memory.
    config = KwamiConfig()
    initial_agent = create_agent_from_config(config, vad, skip_greeting=True)

    # Create session and state
    # Dependencies live on the session's userdata, which the framework threads
    # into every tool's RunContext. This is what replaces the ContextVar the
    # tools used to hunt the room through.
    deps = AgentDeps(settings=get_settings(), room=ctx.room)
    session = AgentSession(userdata=deps)
    state = create_session_state(
        initial_agent=initial_agent,
        user_identity=user_identity,
        room_name=ctx.room.name,
        vad=vad,
    )
    state.room = ctx.room
    initial_agent.room = ctx.room
    initial_agent.usage_tracker = state.usage_tracker

    # Lets the agent rebuild its own pipeline from inside a tool call, which is
    # what makes "switch to Claude" or "use the Cedar voice" answerable by doing
    # rather than by describing which panel to open.
    deps.reconfigure = Reconfigurator(
        state=state,
        vad=vad,
        create_agent_fn=create_agent_from_config,
    )

    # Wire up metrics events for usage tracking
    @session.on("metrics_collected")
    def on_metrics(event):
        route_metrics(state.usage_tracker, event.metrics)

    # Routing lives in runtime.dispatch so each branch is reachable from a test.
    router = DataMessageRouter(
        session=session,
        state=state,
        vad=vad,
        create_agent_fn=create_agent_from_config,
        room=ctx.room,
        # So a tool invoked from a data message reaches the same AgentDeps a
        # tool invoked from a model turn does.
        deps=deps,
    )

    def handle_data(data: rtc.DataPacket) -> None:
        message = decode_data_message(data.data)
        if message is None:
            return
        try:
            router.handle(message)
        except Exception:
            logger.exception("Error handling data message")

    ctx.room.on("data_received", handle_data)

    # Register cleanup for when the session ends
    ctx.add_shutdown_callback(state.cleanup)

    # Telephony sessions get their persona from the Kwami API. Neither the id
    # resolution nor the fetch needs the session, so start the fetch NOW and
    # await it after the room is up: sequencing it after session.start() left
    # callers listening to a default-persona placeholder agent for as long as
    # the HTTP call took, up to KWAMI_API_TIMEOUT (30s by default).
    kwami_id = resolve_kwami_id(ctx)
    runtime_config_task: asyncio.Task | None = None
    if kwami_id:
        bind_session_fields(kwami_id=kwami_id)
        logger.info("Resolved telephony kwami_id: %s", kwami_id)
        runtime_config_task = state.spawn(
            fetch_runtime_config(kwami_id), name="fetch_runtime_config"
        )

    # Start the session
    await session.start(
        agent=initial_agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=True,
            audio_output=True,
        ),
    )

    # Now that the room is connected, the participant list is real.
    if not state.user_identity:
        state.user_identity = resolve_user_identity(ctx.room)
    logger.info(
        "Room connected with %d remote participant(s); user identity: %s",
        len(ctx.room.remote_participants),
        state.user_identity or "<unresolved>",
    )

    @ctx.room.on("participant_connected")
    def on_participant_connected(participant) -> None:
        # A human can join after the agent; without this the session would
        # finish with no user_identity and its usage would never be billed.
        resolve_identity_on_join(state, participant)

    await apply_runtime_config(
        session,
        state,
        vad,
        create_agent_from_config,
        runtime_config_task,
        kwami_id,
    )

    logger.info("Kwami session started")


if __name__ == "__main__":  # pragma: no cover - process entry point
    cli.run_app(server)
