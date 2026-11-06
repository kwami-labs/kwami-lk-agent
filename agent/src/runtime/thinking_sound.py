"""A soft keyboard while the agent thinks, for users who would rather hear that than silence.

Between the end of the user's turn and the first word of the answer there is
silence: the LLM's first token, a tool call, a delegated search. On a phone, or
with the app in the background, a few seconds of it read as a dropped call.
livekit-agents' `BackgroundAudioPlayer` plays a clip while the agent's state is
`thinking`, on a track of its own, mixed under the voice.

Off by default, and switched from the app's Enhancements panel
(`enhancements.thinkingSound`). The setting is read live on every state change,
so turning it on or off applies from the next turn without a rebuild. The extra
track is only published the first time it is needed: a session that never turns
it on never has a second audio track (which older SDK builds mistook for the
agent's voice).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from livekit.agents.voice.background_audio import (
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
)

from ..utils.logging import get_logger

logger = get_logger("thinking_sound")

#: Two takes of typing, picked at random, quiet under the voice and faded at
#: both ends so it neither starts nor stops with a click.
THINKING_CLIPS = [
    AudioConfig(
        BuiltinAudioClip.KEYBOARD_TYPING, volume=0.35, probability=0.5, fade_in=0.3, fade_out=0.3
    ),
    AudioConfig(
        BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.3, probability=0.5, fade_in=0.3, fade_out=0.3
    ),
]


class ThinkingSound:
    """Plays `THINKING_CLIPS` while the agent is `thinking`, when `enabled()` says so."""

    def __init__(
        self,
        enabled: Callable[[], bool],
        *,
        player_factory: Callable[[], Any] = BackgroundAudioPlayer,
    ) -> None:
        self._enabled = enabled
        self._player_factory = player_factory
        self._room: Any = None
        self._session: Any = None
        self._player: Any = None
        self._handle: Any = None
        self._thinking = False
        self._tasks: set[asyncio.Task[None]] = set()

    def attach(self, room: Any, session: Any) -> None:
        """The room and session the sound plays into; nothing is published yet."""
        self._room = room
        self._session = session

    def on_agent_state_changed(self, event: Any) -> None:
        thinking = getattr(event, "new_state", None) == "thinking"
        self._thinking = thinking
        if not thinking:
            self._stop()
            return
        if self._room is None or not self._safe_enabled():
            return
        task = asyncio.get_running_loop().create_task(self._play(), name="thinking_sound")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def aclose(self) -> None:
        self._stop()
        for task in list(self._tasks):
            task.cancel()
        if self._player is not None:
            player, self._player = self._player, None
            try:
                await player.aclose()
            except Exception:
                logger.warning("Could not close the thinking sound's player", exc_info=True)

    def _safe_enabled(self) -> bool:
        try:
            return bool(self._enabled())
        except Exception:
            return False

    async def _play(self) -> None:
        try:
            if self._player is None:
                player = self._player_factory()
                await player.start(room=self._room, agent_session=self._session)
                self._player = player
            # The agent may have started speaking while the track was published.
            if not self._thinking or self._handle is not None:
                return
            self._handle = self._player.play(THINKING_CLIPS, loop=True)
        except Exception:
            # A sound is a nicety; failing to play it must never touch the turn.
            logger.warning("Could not play the thinking sound", exc_info=True)

    def _stop(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.stop()
            except Exception:
                logger.debug("Thinking sound already stopped", exc_info=True)
