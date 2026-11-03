"""Hear the end of what the user is saying before answering it.

The end-of-turn model judges a pause, and a pause mid-thought often passes. Real
turns from 2026-11-02 show the result: "puedes entrar en el local, digamos, de la
cuenta de Cloud," ended on its comma, the agent answered that half, and the other
half arrived as a second question. Each half also went to memory, and with the
brain relaying everything, each half was a task of its own.

What the transcript says is a second opinion the model does not have: a turn that
stops on a comma or on "que" has not finished. Such a turn is held instead of
answered. If the user goes on, the pieces are answered as one; if they stay quiet,
what they said is answered after `HOLD_SECONDS`, so a wrong guess costs a short
wait and never an answer.

Only the standard pipeline: a realtime model takes its turns on its own server.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable

from ..utils.logging import get_logger

logger = get_logger("turn_gate")

#: Silence after a held turn before it is answered as it stands.
HOLD_SECONDS = 1.5
#: After this long without a turn committing, a transcript no longer means one is open.
OPEN_TURN_SECONDS = 6.0

#: Nobody ends a thought on these. Words that can close a sentence stay out ("más",
#: "bueno", "es"): waiting on them would slow the answers that are complete.
_OPEN_WORDS = frozenset(
    # Spanish: joining words, prepositions, articles, pronouns before a verb, fillers.
    "y e o u ni que pero porque aunque si como cuando donde mientras "  # noqa: SIM905
    "de del a al en con sin por para hacia hasta desde entre sobre según "
    "el la los las un una unos unas mi mis tu tus su sus nuestro nuestra "
    "lo le les me te se nos muy pues entonces digamos eh ehm em mm mmm "
    # English: the same.
    "and or but because although if when where while that which "
    "of to at in on with without for from by about the an my your his her our their "
    "um uh er so like".split()
)
_OPEN_PHRASES = ("o sea", "es decir", "a ver", "por ejemplo", "you know", "i mean")
_OPEN_ENDINGS = (",", ";", ":", "-", "…", "...")
_CLOSED_ENDINGS = ("?", "!")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def sounds_unfinished(text: str) -> bool:
    """Whether a transcript stops where a sentence cannot."""
    said = text.strip()
    if not said or said.endswith(_CLOSED_ENDINGS):
        return False
    if said.endswith(_OPEN_ENDINGS):
        return True
    lowered = said.lower().rstrip(".")
    if lowered.endswith(_OPEN_PHRASES):
        return True
    words = _WORD.findall(lowered)
    return bool(words) and words[-1] in _OPEN_WORDS


class TurnGate:
    """The pieces of one thing the user is saying, until it can be answered."""

    def __init__(
        self,
        answer: Callable[[str], None],
        *,
        hold_seconds: float = HOLD_SECONDS,
        now: Callable[[], float] | None = None,
    ) -> None:
        self._answer = answer
        self._hold_seconds = hold_seconds
        self._now = now
        self._held: list[str] = []
        self._timer: asyncio.Task[None] | None = None
        self._speaking = False
        self._heard_at: float | None = None

    @property
    def holding(self) -> bool:
        return bool(self._held)

    @property
    def turn_open(self) -> bool:
        """The user is mid-turn: holding, speaking, or just transcribed and not committed."""
        if self._held or self._speaking:
            return True
        return self._heard_at is not None and self._clock() - self._heard_at < OPEN_TURN_SECONDS

    def _clock(self) -> float:
        return self._now() if self._now else asyncio.get_running_loop().time()

    def take(self, text: str) -> str | None:
        """What to answer now with this finished turn, or None when it is being held."""
        self._heard_at = None
        self._cancel_timer()
        said = " ".join([*self._held, text.strip()]).strip()
        if sounds_unfinished(said):
            self._held = [said]
            logger.info("Holding a turn that sounds unfinished (%d words)", len(said.split()))
            self._start_timer()
            return None
        self._held = []
        return said

    def heard(self) -> None:
        """A final transcript arrived: a turn is open until it commits."""
        self._heard_at = self._clock()

    def speaking(self, speaking: bool) -> None:
        """The user started or stopped: the wait only counts while they are quiet."""
        self._speaking = speaking
        if speaking:
            self._cancel_timer()
        elif self._held:
            self._start_timer()

    def drain(self) -> str:
        """What was held, for a session that is ending. Nothing will answer it."""
        self._cancel_timer()
        said = " ".join(self._held)
        self._held = []
        return said

    def _start_timer(self) -> None:
        self._cancel_timer()
        self._timer = asyncio.create_task(self._wait(), name="turn_gate_hold")

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    async def _wait(self) -> None:
        await asyncio.sleep(self._hold_seconds)
        said = " ".join(self._held)
        self._held = []
        self._timer = None
        logger.info("Answering a held turn: the user stayed quiet")
        self._answer(said)
