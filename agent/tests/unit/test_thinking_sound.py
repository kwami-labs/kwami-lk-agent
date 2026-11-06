"""`runtime/thinking_sound.py`: a soft keyboard while the agent thinks, only when asked for."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from src.domain import KwamiVoiceConfig
from src.handlers.listening import LISTENING_FIELDS, is_listening_only, parse_listening
from src.runtime.thinking_sound import THINKING_CLIPS, ThinkingSound


class Handle:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class Player:
    """BackgroundAudioPlayer's surface: start once, play clips, close."""

    instances: list[Player] = []

    def __init__(self, *, fail_start: bool = False, fail_close: bool = False) -> None:
        self.started_with: dict[str, Any] | None = None
        self.played: list[tuple[Any, bool]] = []
        self.handles: list[Handle] = []
        self.closed = False
        self._fail_start = fail_start
        self._fail_close = fail_close
        Player.instances.append(self)

    async def start(self, *, room: Any, agent_session: Any) -> None:
        if self._fail_start:
            raise RuntimeError("could not publish the track")
        self.started_with = {"room": room, "agent_session": agent_session}

    def play(self, clips: Any, *, loop: bool) -> Handle:
        self.played.append((clips, loop))
        handle = Handle()
        self.handles.append(handle)
        return handle

    async def aclose(self) -> None:
        self.closed = True
        if self._fail_close:
            raise RuntimeError("already closed")


def _state(new: str, old: str = "listening") -> SimpleNamespace:
    return SimpleNamespace(new_state=new, old_state=old)


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


def _sound(enabled: list[bool], **player_kwargs: Any) -> tuple[ThinkingSound, list[Player]]:
    made: list[Player] = []

    def factory() -> Player:
        player = Player(**player_kwargs)
        made.append(player)
        return player

    sound = ThinkingSound(lambda: enabled[0], player_factory=factory)
    sound.attach(room="room", session="session")
    return sound, made


class TestWhenSwitchedOn:
    async def test_it_plays_while_thinking_and_stops_when_speaking(self) -> None:
        sound, made = _sound([True])

        sound.on_agent_state_changed(_state("thinking"))
        await _settle()
        [player] = made
        assert player.started_with == {"room": "room", "agent_session": "session"}
        assert player.played == [(THINKING_CLIPS, True)]

        sound.on_agent_state_changed(_state("speaking", "thinking"))
        assert player.handles[0].stopped is True

    async def test_the_track_is_published_once_for_the_whole_session(self) -> None:
        sound, made = _sound([True])

        for _ in range(3):
            sound.on_agent_state_changed(_state("thinking"))
            await _settle()
            sound.on_agent_state_changed(_state("speaking", "thinking"))

        assert len(made) == 1
        assert len(made[0].played) == 3

    async def test_speech_that_starts_while_the_track_publishes_plays_nothing(self) -> None:
        sound, made = _sound([True])

        sound.on_agent_state_changed(_state("thinking"))
        sound.on_agent_state_changed(_state("speaking", "thinking"))
        await _settle()

        assert made[0].played == []

    async def test_a_second_thinking_event_does_not_stack_a_second_clip(self) -> None:
        sound, made = _sound([True])

        sound.on_agent_state_changed(_state("thinking"))
        await _settle()
        sound.on_agent_state_changed(_state("thinking"))
        await _settle()

        assert len(made[0].played) == 1


class TestWhenSwitchedOff:
    async def test_nothing_is_published_or_played(self) -> None:
        sound, made = _sound([False])

        sound.on_agent_state_changed(_state("thinking"))
        await _settle()

        assert made == []

    async def test_the_setting_is_read_on_every_turn(self) -> None:
        enabled = [False]
        sound, made = _sound(enabled)

        sound.on_agent_state_changed(_state("thinking"))
        await _settle()
        enabled[0] = True
        sound.on_agent_state_changed(_state("thinking"))
        await _settle()

        assert len(made) == 1

    async def test_a_setting_that_cannot_be_read_counts_as_off(self) -> None:
        def broken() -> bool:
            raise AttributeError("no agent yet")

        sound = ThinkingSound(broken, player_factory=Player)
        sound.attach(room="room", session="session")
        Player.instances.clear()

        sound.on_agent_state_changed(_state("thinking"))
        await _settle()

        assert Player.instances == []

    async def test_before_the_room_is_up_nothing_happens(self) -> None:
        made: list[Player] = []
        sound = ThinkingSound(
            lambda: True, player_factory=lambda: made.append(Player()) or made[-1]
        )

        sound.on_agent_state_changed(_state("thinking"))
        await _settle()

        assert made == []


class TestFailures:
    async def test_a_track_that_cannot_be_published_is_logged_not_raised(self, caplog) -> None:
        sound, made = _sound([True], fail_start=True)

        with caplog.at_level("WARNING"):
            sound.on_agent_state_changed(_state("thinking"))
            await _settle()

        assert made[0].played == []
        assert "Could not play the thinking sound" in caplog.text

    async def test_closing_stops_the_clip_and_the_player(self) -> None:
        sound, made = _sound([True])
        sound.on_agent_state_changed(_state("thinking"))
        await _settle()

        await sound.aclose()

        assert made[0].handles[0].stopped is True
        assert made[0].closed is True

    async def test_a_player_that_fails_to_close_is_logged(self, caplog) -> None:
        sound, _ = _sound([True], fail_close=True)
        sound.on_agent_state_changed(_state("thinking"))
        await _settle()

        with caplog.at_level("WARNING"):
            await sound.aclose()

        assert "Could not close the thinking sound" in caplog.text

    async def test_closing_while_the_track_is_still_publishing_cancels_it(self) -> None:
        sound, made = _sound([True])

        sound.on_agent_state_changed(_state("thinking"))
        # The publish has not run yet: closing now cancels it rather than leaking it.
        await sound.aclose()
        await _settle()

        assert all(not player.played for player in made)

    async def test_closing_a_session_that_never_played_is_a_no_op(self) -> None:
        sound, made = _sound([True])

        await sound.aclose()

        assert made == []

    async def test_a_handle_that_fails_to_stop_is_ignored(self) -> None:
        sound, made = _sound([True])
        sound.on_agent_state_changed(_state("thinking"))
        await _settle()

        def boom() -> None:
            raise RuntimeError("stream closed")

        made[0].handles[0].stop = boom  # type: ignore[method-assign]
        sound.on_agent_state_changed(_state("speaking", "thinking"))  # must not raise


class TestTheSetting:
    def test_it_is_off_by_default(self) -> None:
        assert KwamiVoiceConfig().thinking_sound is False

    @pytest.mark.parametrize("value", [True, False])
    def test_the_enhancements_panel_switches_it(self, value: bool) -> None:
        values = parse_listening({"enhancements": {"thinkingSound": value}})
        assert values["thinking_sound"] is value

    def test_it_is_a_listening_setting_carried_across_rebuilds(self) -> None:
        assert "thinking_sound" in LISTENING_FIELDS
        assert is_listening_only({"thinkingSound": True})
