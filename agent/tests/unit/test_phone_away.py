"""`runtime/away.py`: a silent phone caller is asked once, then the line is freed."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from livekit import rtc

from src.brain.bridge import BrainTask, Bridge
from src.runtime.away import (
    ASK_IF_STILL_THERE,
    SAY_GOODBYE,
    PhoneAwayWatcher,
    caller_on_phone,
)

SIP = rtc.ParticipantKind.PARTICIPANT_KIND_SIP
STANDARD = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD


class Handle:
    def __init__(self, playout: Any = None) -> None:
        self._playout = playout

    async def wait_for_playout(self) -> None:
        if isinstance(self._playout, BaseException):
            raise self._playout
        if self._playout == "forever":
            await asyncio.sleep(3600)


class Session:
    def __init__(self, *, closing: bool = False, playout: Any = None) -> None:
        self.replies: list[str] = []
        self._closing = closing
        self._playout = playout

    def generate_reply(self, *, instructions: str) -> Handle:
        if self._closing:
            raise RuntimeError("AgentSession is closing")
        self.replies.append(instructions)
        return Handle(self._playout)


def _room(*kinds: Any) -> SimpleNamespace:
    return SimpleNamespace(
        remote_participants={f"p{i}": SimpleNamespace(kind=k) for i, k in enumerate(kinds)}
    )


def _state(new_state: str) -> SimpleNamespace:
    return SimpleNamespace(new_state=new_state, old_state="listening")


class Calls:
    def __init__(self, fail: bool = False) -> None:
        self.ended = 0
        self._fail = fail

    async def end(self) -> None:
        self.ended += 1
        if self._fail:
            raise RuntimeError("twirp: room not found")


def _watcher(
    session: Session,
    room: Any,
    calls: Calls,
    *,
    working: list[bool] | None = None,
    goodbye_playout: float = 1.0,
) -> PhoneAwayWatcher:
    flag = working if working is not None else [False]
    return PhoneAwayWatcher(
        session,
        room,
        working=lambda: flag[0],
        end_call=calls.end,
        hang_up_after=0.01,
        goodbye_playout=goodbye_playout,
    )


async def _wait_for(watcher: PhoneAwayWatcher) -> None:
    for _ in range(200):
        if not watcher.waiting_for_answer:
            return
        await asyncio.sleep(0.005)
    pytest.fail("the hang-up countdown never finished")


class TestWhoIsOnThePhone:
    def test_a_sip_participant_is_a_phone_caller(self) -> None:
        assert caller_on_phone(_room(STANDARD, SIP)) is True

    def test_the_app_alone_is_not(self) -> None:
        assert caller_on_phone(_room(STANDARD)) is False
        assert caller_on_phone(SimpleNamespace()) is False


class TestASilentCaller:
    async def test_is_asked_once_then_told_goodbye_and_hung_up_on(self) -> None:
        session, calls = Session(), Calls()
        watcher = _watcher(session, _room(SIP), calls)

        watcher.on_user_state_changed(_state("away"))
        # A second "away" while the question is pending asks nothing new.
        watcher.on_user_state_changed(_state("away"))
        await _wait_for(watcher)

        assert session.replies == [ASK_IF_STILL_THERE, SAY_GOODBYE]
        assert calls.ended == 1
        assert watcher.ended is True

        # Once ended, nothing more is said or hung up.
        watcher.on_user_state_changed(_state("away"))
        assert session.replies == [ASK_IF_STILL_THERE, SAY_GOODBYE]

    async def test_who_answers_keeps_the_call(self) -> None:
        session, calls = Session(), Calls()
        watcher = PhoneAwayWatcher(
            session, _room(SIP), working=lambda: False, end_call=calls.end, hang_up_after=60
        )

        watcher.on_user_state_changed(_state("away"))
        assert watcher.waiting_for_answer
        watcher.on_user_state_changed(_state("speaking"))
        await asyncio.sleep(0)

        assert not watcher.waiting_for_answer
        assert calls.ended == 0
        assert session.replies == [ASK_IF_STILL_THERE]

    async def test_a_goodbye_that_cannot_play_does_not_keep_the_line_open(self) -> None:
        session, calls = Session(playout=RuntimeError("speech scheduling is draining")), Calls()
        watcher = _watcher(session, _room(SIP), calls)

        watcher.on_user_state_changed(_state("away"))
        await _wait_for(watcher)

        assert calls.ended == 1

    async def test_a_goodbye_that_never_finishes_is_cut_short(self) -> None:
        session, calls = Session(playout="forever"), Calls()
        watcher = _watcher(session, _room(SIP), calls, goodbye_playout=0.01)

        watcher.on_user_state_changed(_state("away"))
        await _wait_for(watcher)

        assert calls.ended == 1

    async def test_a_failed_hang_up_is_logged_not_raised(self, caplog) -> None:
        session, calls = Session(), Calls(fail=True)
        watcher = _watcher(session, _room(SIP), calls)

        watcher.on_user_state_changed(_state("away"))
        await _wait_for(watcher)

        assert calls.ended == 1
        assert "Could not end the silent call" in caplog.text


class TestSilenceThatIsNotAbsence:
    async def test_the_app_is_never_hung_up_on(self) -> None:
        session, calls = Session(), Calls()
        watcher = _watcher(session, _room(STANDARD), calls)

        watcher.on_user_state_changed(_state("away"))
        await asyncio.sleep(0.02)

        assert session.replies == []
        assert calls.ended == 0

    async def test_a_caller_waiting_on_the_brain_is_not_asked(self) -> None:
        session, calls = Session(), Calls()
        watcher = _watcher(session, _room(SIP), calls, working=[True])

        watcher.on_user_state_changed(_state("away"))

        assert session.replies == []
        assert not watcher.waiting_for_answer

    async def test_work_started_after_asking_keeps_the_call(self) -> None:
        session, calls = Session(), Calls()
        working = [False]
        watcher = _watcher(session, _room(SIP), calls, working=working)

        watcher.on_user_state_changed(_state("away"))
        working[0] = True
        await _wait_for(watcher)

        assert calls.ended == 0
        assert watcher.ended is False

    async def test_a_closing_session_is_left_alone(self) -> None:
        session, calls = Session(closing=True), Calls()
        watcher = _watcher(session, _room(SIP), calls)

        watcher.on_user_state_changed(_state("away"))

        assert not watcher.waiting_for_answer
        assert calls.ended == 0

    async def test_close_stops_a_pending_countdown(self) -> None:
        session, calls = Session(), Calls()
        watcher = PhoneAwayWatcher(
            session, _room(SIP), working=lambda: False, end_call=calls.end, hang_up_after=60
        )

        watcher.on_user_state_changed(_state("away"))
        watcher.close()
        await asyncio.sleep(0)

        assert not watcher.waiting_for_answer
        assert calls.ended == 0


def _task(task_id: str, **kwargs: Any) -> BrainTask:
    return BrainTask(
        id=task_id, title="t", task="busca vuelos", kind="research", difficulty="hard", **kwargs
    )


class TestTheBridgeSaysWhenItIsWorking:
    def test_only_a_running_task_counts(self) -> None:
        bridge = Bridge(keys={})
        assert bridge.working is False

        bridge.tasks["a"] = _task("a", status="done")
        assert bridge.working is False

        bridge.tasks["b"] = _task("b")
        assert bridge.working is True
