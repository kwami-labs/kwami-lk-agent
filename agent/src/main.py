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
from livekit.plugins import silero

from .agent import KwamiAgent
from .config import KwamiConfig
from .factories import create_llm, create_realtime_model, create_stt, create_tts
from .handlers import handle_full_config
from .room_context import set_current_room
from .runtime import DataMessageRouter, decode_data_message, route_metrics
from .runtime_bootstrap import fetch_runtime_config, resolve_kwami_id
from .session import create_session_state
from .settings import Settings, get_settings, set_settings
from .utils.logging import get_logger
from .utils.room import is_agent_participant, resolve_user_identity

logger = get_logger()

# Resolve credentials once, here, after load_dotenv has run. Everything
# downstream takes them from Settings rather than reading os.environ itself.
set_settings(Settings.from_env())
logger.info("Settings resolved: %s", get_settings().describe())

server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    """Prewarm the VAD model for faster startup."""
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="kwami-agent")
async def entrypoint(ctx: JobContext) -> None:
    """Main entry point for Kwami agent sessions."""
    set_current_room(ctx.room)
    logger.info(f"Kwami session starting in room: {ctx.room.name}")

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
    session = AgentSession()
    state = create_session_state(
        initial_agent=initial_agent,
        user_identity=user_identity,
        room_name=ctx.room.name,
        vad=vad,
    )
    state.room = ctx.room
    initial_agent.room = ctx.room
    initial_agent.usage_tracker = state.usage_tracker

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
    )

    def handle_data(data: rtc.DataPacket) -> None:
        message = decode_data_message(data.data)
        if message is None:
            return
        try:
            router.handle(message)
        except Exception as e:
            logger.error(f"Error handling data message: {e}")

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
        if state.user_identity or is_agent_participant(participant):
            return
        if participant.identity:
            state.user_identity = participant.identity
            logger.info("Resolved user identity on join: %s", state.user_identity)

    if runtime_config_task is not None:
        try:
            runtime_config = await runtime_config_task
        except Exception as e:
            logger.error("Failed to fetch runtime config for %s: %s", kwami_id, e)
            runtime_config = None
        if runtime_config:
            await state.run_serialized(
                handle_full_config(
                    session,
                    state,
                    runtime_config,
                    vad,
                    create_agent_from_config,
                )
            )
        else:
            logger.warning(
                "No runtime config for kwami_id=%s; staying on the placeholder agent", kwami_id
            )

    logger.info(f"Kwami session started for room: {ctx.room.name}")


def create_agent_from_config(
    config: KwamiConfig,
    vad,
    memory=None,
    skip_greeting: bool = False,
) -> KwamiAgent:
    """Create a KwamiAgent instance from a configuration object.

    Args:
        config: The Kwami configuration.
        vad: Voice Activity Detection instance.
        memory: Optional memory instance.
        skip_greeting: If True, skip the initial greeting (for reconfigurations).

    Returns:
        Configured KwamiAgent instance.
    """
    voice_config = config.voice

    if voice_config.pipeline_type == "realtime":
        logger.info(
            f"Using realtime pipeline: "
            f"{voice_config.realtime_provider}/{voice_config.realtime_model}"
        )
        realtime_model = create_realtime_model(voice_config)
        return KwamiAgent(
            config,
            vad=vad,
            memory=memory,
            llm=realtime_model,
            skip_greeting=skip_greeting,
        )
    else:
        logger.info(
            f"Using standard pipeline: "
            f"STT={voice_config.stt_provider}/{voice_config.stt_model}, "
            f"LLM={voice_config.llm_provider}/{voice_config.llm_model}, "
            f"TTS={voice_config.tts_provider}/{voice_config.tts_model}"
        )
        stt = create_stt(voice_config)
        llm = create_llm(voice_config)
        tts = create_tts(voice_config)
        return KwamiAgent(
            config,
            vad=vad,
            memory=memory,
            stt=stt,
            llm=llm,
            tts=tts,
            skip_greeting=skip_greeting,
        )


if __name__ == "__main__":
    cli.run_app(server)
