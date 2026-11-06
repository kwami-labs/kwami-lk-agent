"""A caller who goes quiet on the phone is asked once, and then the line is freed.

A phone call has no hang-up button the agent can see. A caller who puts the
phone down without ending the call, walks away, or loses signal into a
half-open line used to leave the agent waiting in silence for as long as the
carrier kept the call up -- billed by the minute on the SIP trunk, on LiveKit
and on every model that stayed connected.

livekit-agents already notices: when neither side has spoken for
`user_away_timeout`, the user's state turns `away`. This acts on it, on phone
calls only. The agent asks once whether the caller is still there; if nobody
answers within `HANG_UP_AFTER_SECONDS`, it says goodbye and ends the call.

Silence is not always absence. While the brain is working on something the
caller asked for, waiting quietly is the expected thing to do, so a call with
work in progress is never ended this way. The app is not covered: a desktop
session left open is the user's choice, not a dropped line.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from livekit import rtc

from ..utils.logging import get_logger

logger = get_logger("away")

#: Silence on both sides before the framework calls the user away. Pinned here
#: rather than left to the library default, which can change under a minor bump.
USER_AWAY_SECONDS = 15.0

#: After asking, how long the caller has to answer before the call ends.
HANG_UP_AFTER_SECONDS = 20.0

#: How long the goodbye may take to play before the room is closed anyway.
GOODBYE_PLAYOUT_SECONDS = 8.0

ASK_IF_STILL_THERE = (
    "The caller has been silent for a while. Ask them, in one short sentence and in "
    "the language of the conversation, whether they are still there."
)
SAY_GOODBYE = (
    "The caller has not answered. Say a brief, polite goodbye in the language of the "
    "conversation, mention that they can call back any time, and stop."
)


def caller_on_phone(room: Any) -> bool:
    """Whether someone in the room joined through a SIP trunk."""
    participants = getattr(room, "remote_participants", None) or {}
    return any(
        getattr(p, "kind", None) == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
        for p in participants.values()
    )


class PhoneAwayWatcher:
    """Listens to `user_state_changed`; asks once, then ends a silent phone call."""

    def __init__(
        self,
        session: Any,
        room: Any,
        *,
        working: Callable[[], bool],
        end_call: Callable[[], Awaitable[Any]],
        hang_up_after: float = HANG_UP_AFTER_SECONDS,
        goodbye_playout: float = GOODBYE_PLAYOUT_SECONDS,
    ) -> None:
        self._session = session
        self._room = room
        self._working = working
        self._end_call = end_call
        self._hang_up_after = hang_up_after
        self._goodbye_playout = goodbye_playout
        self._timer: asyncio.Task[None] | None = None
        self.ended = False

    @property
    def waiting_for_answer(self) -> bool:
        return self._timer is not None and not self._timer.done()

    def on_user_state_changed(self, event: Any) -> None:
        if getattr(event, "new_state", None) != "away":
            # They spoke (or are speaking): whatever we were counting down is moot.
            self._cancel()
            return
        if self.ended or self.waiting_for_answer or not caller_on_phone(self._room):
            return
        if self._working():
            logger.info("Caller silent while the brain works for them; not asking")
            return
        logger.info(
            "Caller silent for %.0fs; asking whether they are still there", USER_AWAY_SECONDS
        )
        try:
            self._session.generate_reply(instructions=ASK_IF_STILL_THERE)
        except RuntimeError:
            # The session is already closing; there is no call left to keep.
            return
        self._timer = asyncio.get_running_loop().create_task(
            self._hang_up_if_still_silent(), name="phone_away_hang_up"
        )

    def close(self) -> None:
        self._cancel()

    def _cancel(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    async def _hang_up_if_still_silent(self) -> None:
        await asyncio.sleep(self._hang_up_after)
        if self._working():
            # Something started in the meantime; the caller is waiting on it.
            logger.info("Not hanging up: the brain started working for the caller")
            return
        self.ended = True
        logger.info("No answer after %.0fs; ending the call", self._hang_up_after)
        try:
            handle = self._session.generate_reply(instructions=SAY_GOODBYE)
            await asyncio.wait_for(handle.wait_for_playout(), timeout=self._goodbye_playout)
        except Exception:
            # A goodbye that cannot be said must not keep the line open.
            logger.warning("Could not say goodbye before hanging up", exc_info=True)
        try:
            await self._end_call()
        except Exception:
            logger.exception("Could not end the silent call")
