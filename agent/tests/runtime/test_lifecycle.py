"""Lifecycle steps that were previously buried in the entrypoint coroutine.

Both guard revenue: identity resolution decides whether a session is billable
at all, and the runtime-config step decides whether a telephony caller hears
their own persona or the default placeholder.
"""

from __future__ import annotations

from typing import Any

import pytest
from livekit import rtc

from src.runtime.lifecycle import apply_runtime_config, resolve_identity_on_join

AGENT_KIND = rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
STANDARD_KIND = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
SIP_KIND = rtc.ParticipantKind.PARTICIPANT_KIND_SIP


class FakeParticipant:
    def __init__(self, identity: str, kind: int = STANDARD_KIND) -> None:
        self.identity = identity
        self.kind = kind


class FakeState:
    def __init__(self, user_identity: str | None = None) -> None:
        self.user_identity = user_identity
        self.serialized: list[Any] = []

    async def run_serialized(self, coro: Any) -> None:
        self.serialized.append(coro)
        await coro


# -- identity on join -------------------------------------------------------


def test_a_joining_human_becomes_the_billable_identity() -> None:
    state = FakeState()
    assert resolve_identity_on_join(state, FakeParticipant("user-42")) is True
    assert state.user_identity == "user-42"


def test_a_telephony_caller_counts() -> None:
    state = FakeState()
    assert resolve_identity_on_join(state, FakeParticipant("sip_+34600", SIP_KIND)) is True


def test_another_agent_is_not_the_user() -> None:
    state = FakeState()
    assert resolve_identity_on_join(state, FakeParticipant("kwami-agent", AGENT_KIND)) is False
    assert state.user_identity is None


def test_an_existing_identity_is_not_overwritten() -> None:
    """The first human in the room owns the session; a later joiner does not steal it."""
    state = FakeState(user_identity="user-first")
    assert resolve_identity_on_join(state, FakeParticipant("user-second")) is False
    assert state.user_identity == "user-first"


def test_a_blank_identity_is_ignored() -> None:
    state = FakeState()
    assert resolve_identity_on_join(state, FakeParticipant("")) is False
    assert state.user_identity is None


# -- runtime config ---------------------------------------------------------


async def _task(value: Any) -> Any:
    return value


async def _failing_task() -> Any:
    raise RuntimeError("kwami API unreachable")


async def test_no_task_means_nothing_to_apply() -> None:
    """Non-telephony sessions never start the fetch."""
    state = FakeState()
    assert await apply_runtime_config(None, state, None, None, None) is False


async def test_a_fetched_config_is_applied_serialized(monkeypatch: pytest.MonkeyPatch) -> None:
    applied: dict[str, Any] = {}

    async def fake_handle_full_config(session, state, message, vad, create_agent_fn):
        applied["message"] = message

    monkeypatch.setattr("src.runtime.lifecycle.handle_full_config", fake_handle_full_config)
    state = FakeState()

    result = await apply_runtime_config(
        object(), state, None, None, _task({"soul": {"name": "Ada"}}), "kwami-1"
    )

    assert result is True
    assert applied["message"] == {"soul": {"name": "Ada"}}
    assert state.serialized, "config was applied without the serialization lock"


async def test_a_failed_fetch_leaves_the_session_running() -> None:
    """A dead Kwami API must not take down a live call."""
    state = FakeState()
    assert await apply_runtime_config(None, state, None, None, _failing_task(), "kwami-1") is False


async def test_an_empty_config_keeps_the_placeholder() -> None:
    state = FakeState()
    assert await apply_runtime_config(None, state, None, None, _task(None), "kwami-1") is False
    assert await apply_runtime_config(None, state, None, None, _task({}), "kwami-1") is False
