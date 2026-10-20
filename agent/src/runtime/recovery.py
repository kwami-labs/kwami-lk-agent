"""Keep the call alive when the realtime model dies.

An unrecoverable realtime error (Gemini Live closing with 1008 on a bad model
name, 1011 on a spent quota, a dropped socket that would not come back) made
the framework close the whole AgentSession: the user was left in a room with a
silent agent until they hung up. There is no backup *inside* a realtime model
the way FallbackAdapter gives the standard pipeline one, so the rescue is one
level up: mark the error recoverable, which stops the framework from closing,
and rebuild the session on the standard pipeline, whose stages do fall back.

The app is told (`voice_fallback`), so the panel can say the engine changed
instead of leaving the user to wonder why the voice sounds different.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..utils.logging import get_logger

logger = get_logger("recovery")

REALTIME_PIPELINE = "realtime"
STANDARD_PIPELINE = "standard"


def install_realtime_rescue(
    session: Any,
    state: Any,
    vad: Any,
    create_agent_fn: Any,
    publisher: Any = None,
) -> Any:
    """Register the rescue on `session`. Returns the handler, for tests."""
    rescuing = {"active": False}

    def on_error(event: Any) -> None:
        error = getattr(event, "error", None)
        if getattr(error, "type", None) != "realtime_model_error":
            return
        if getattr(error, "recoverable", True):
            return
        agent = getattr(state, "current_agent", None)
        if agent is None or agent.kwami_config.voice.pipeline_type != REALTIME_PIPELINE:
            return
        # Read synchronously by the framework right after this handler returns.
        error.recoverable = True
        if rescuing["active"]:
            return
        rescuing["active"] = True
        reason = str(getattr(error, "error", "") or "realtime model failed")[:300]
        logger.error(
            "Realtime model failed for good (%s); moving the call to the standard pipeline", reason
        )
        state.spawn(
            state.run_serialized(
                _rescue(session, state, vad, create_agent_fn, publisher, reason, rescuing)
            ),
            name="realtime_rescue",
        )

    # Decorator form: the framework emitter and the test doubles both take it.
    session.on("error")(on_error)
    return on_error


def _has_spoken(agent: Any) -> bool:
    """Whether the failed agent got any reply out before it died."""
    items = getattr(getattr(agent, "_chat_ctx", None), "items", None) or []
    return any(getattr(item, "role", None) == "assistant" for item in items)


async def _rescue(
    session: Any,
    state: Any,
    vad: Any,
    create_agent_fn: Any,
    publisher: Any,
    reason: str,
    rescuing: dict,
) -> None:
    try:
        agent = state.current_agent
        if agent is None or agent.kwami_config.voice.pipeline_type != REALTIME_PIPELINE:
            return
        from ..domain import clone_config

        new_config = clone_config(agent.kwami_config)
        new_config.voice = replace(agent.kwami_config.voice, pipeline_type=STANDARD_PIPELINE)
        new_agent = create_agent_fn(
            new_config, vad, getattr(agent, "_memory", None), skip_greeting=True
        )
        # A model that died at connect never greeted: the rescuer does.
        deps = getattr(session, "userdata", None)
        if deps is not None and hasattr(deps, "greeting_owed") and not _has_spoken(agent):
            deps.greeting_owed = True
        state.update_agent(session, new_agent)
        if publisher is not None:
            await publisher.publish(
                {
                    "type": "voice_fallback",
                    "from": REALTIME_PIPELINE,
                    "to": STANDARD_PIPELINE,
                    "reason": reason,
                }
            )
    except Exception:
        logger.exception("Could not move the call off the realtime model")
    finally:
        rescuing["active"] = False
