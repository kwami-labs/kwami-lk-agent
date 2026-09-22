"""Steps of the job lifecycle, pulled out of the entrypoint so they can be tested.

`entrypoint` is one long coroutine wired to a live worker, a room and a
session, so nothing inside it was reachable from a test. These two steps both
guard revenue: one decides whether a session has a billable identity at all,
and the other decides whether a telephony caller hears their own persona or the
default placeholder.
"""

from __future__ import annotations

from typing import Any

from ..handlers import handle_full_config
from ..utils.logging import get_logger
from ..utils.room import is_agent_participant

logger = get_logger("lifecycle")


def resolve_identity_on_join(state: Any, participant: Any) -> bool:
    """Adopt a joining human as the session's billable identity.

    A human can join after the agent does. Without this the session ends with
    no `user_identity`, and its usage is silently dropped rather than billed.

    Returns True when this call set the identity.
    """
    if state.user_identity:
        return False
    if is_agent_participant(participant):
        return False
    identity = getattr(participant, "identity", None)
    if not identity:
        return False

    state.user_identity = identity
    logger.info("Resolved user identity on join: %s", identity)
    return True


async def apply_runtime_config(
    session: Any,
    state: Any,
    vad: Any,
    create_agent_fn: Any,
    runtime_config_task: Any,
    kwami_id: str | None = None,
) -> bool:
    """Await the telephony runtime-config fetch and apply it, if there was one.

    The fetch is started before `session.start()` so the HTTP round trip
    overlaps with bringing the room up; this is where it is collected. A
    failure must never take the session down -- the placeholder agent is a
    worse experience than the configured one, but it is still a working call.

    Returns True when a config was applied.
    """
    if runtime_config_task is None:
        return False

    try:
        runtime_config = await runtime_config_task
    except Exception:
        logger.exception("Failed to fetch runtime config for %s", kwami_id)
        return False

    if not runtime_config:
        logger.warning(
            "No runtime config for kwami_id=%s; staying on the placeholder agent", kwami_id
        )
        return False

    await state.run_serialized(
        handle_full_config(session, state, runtime_config, vad, create_agent_fn)
    )
    return True
