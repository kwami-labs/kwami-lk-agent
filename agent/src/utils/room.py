"""Room and participant utilities for Kwami agent."""

import asyncio
from typing import TYPE_CHECKING, Optional

from .logging import get_logger

if TYPE_CHECKING:
    from livekit.rtc import Participant, Room

logger = get_logger("room")


def is_agent_participant(participant: "Participant") -> bool:
    """True when a participant is an agent rather than a human.

    Uses the participant kind the server reports. The previous check --
    `identity.startswith("agent")` -- was a string heuristic that also crashed
    on a None identity.
    """
    from livekit.rtc import ParticipantKind

    # The protobuf enum wrapper exposes the fully-qualified name only:
    # `ParticipantKind.AGENT` raises AttributeError, which would have crashed
    # identity resolution on the first participant in the room.
    return getattr(participant, "kind", None) == ParticipantKind.PARTICIPANT_KIND_AGENT


def resolve_user_identity(room: "Room") -> str | None:
    """Identity of the first human participant in the room, if any.

    Must be called AFTER the room is connected. `JobContext.room` is empty
    until then, so the original pre-connect scan in the entrypoint always saw
    zero participants and left `user_identity` as None -- which silently
    dropped the session's usage report, and with it the billing.
    """
    for participant in room.remote_participants.values():
        if not participant.identity:
            continue
        if is_agent_participant(participant):
            continue
        return participant.identity
    return None


async def get_other_agents(room: "Room") -> list["Participant"]:
    """Get list of other agent participants in the room.

    Args:
        room: The LiveKit room instance.

    Returns:
        List of participants that are agents.
    """
    from livekit.rtc import ParticipantKind

    return [
        p
        for p in room.remote_participants.values()
        if p.kind == ParticipantKind.PARTICIPANT_KIND_AGENT
    ]


async def should_disconnect_as_duplicate(
    room: "Room",
    my_identity: str,
    check_delays: list[float] | None = None,
) -> bool:
    """Check if this agent should disconnect due to another agent having priority.

    Note: This check is intentionally lenient. LiveKit Cloud manages agent dispatch,
    so duplicate agents are rare. We only disconnect if we clearly see another
    active agent that has priority.

    Args:
        room: The LiveKit room instance.
        my_identity: This agent's identity string.
        check_delays: List of delays (seconds) between checks. Defaults to [0.1].

    Returns:
        True if this agent should disconnect, False if it should stay.
    """
    # Single quick check - don't be too aggressive as it can prevent agents from starting
    if check_delays is None:
        check_delays = [0.1]

    for delay in check_delays:
        await asyncio.sleep(delay)

        other_agents = await get_other_agents(room)

        if other_agents:
            # rtc.RemoteParticipant has no `is_connected` attribute -- reading it
            # raised AttributeError here, which meant this duplicate-agent guard
            # blew up precisely when another agent was present. Membership of
            # room.remote_participants already means connected, so the only thing
            # worth filtering is a participant that is on its way out.
            active_agents = [a for a in other_agents if not getattr(a, "disconnect_reason", None)]

            if not active_agents:
                logger.debug("Found agents but none are actively connected, proceeding")
                return False

            # The agent with the "smaller" identity stays
            oldest_agent = min(active_agents, key=lambda p: p.identity)

            if my_identity > oldest_agent.identity:
                logger.warning(
                    f"Another active agent ({oldest_agent.identity}) has priority. "
                    f"This agent ({my_identity}) should disconnect."
                )
                return True
            else:
                logger.info(f"This agent ({my_identity}) has priority over {oldest_agent.identity}")
                return False

    return False


async def check_duplicate_before_action(
    room: Optional["Room"],
    my_identity: str | None,
) -> bool:
    """Quick check for duplicate agents before performing an action.

    Args:
        room: The LiveKit room instance.
        my_identity: This agent's identity string.

    Returns:
        True if this agent should abort the action, False if it's safe to proceed.
    """
    if not room:
        return False

    other_agents = await get_other_agents(room)

    if not other_agents:
        return False

    if not my_identity:
        my_identity = room.local_participant.identity if room.local_participant else ""

    oldest = min(other_agents, key=lambda p: p.identity)

    if my_identity > oldest.identity:
        logger.warning(f"Aborting action - another agent ({oldest.identity}) has priority")
        return True

    return False
