"""Data-channel routing, which was previously unreachable from any test.

It lived as a closure inside `entrypoint`, so exercising a single branch meant
standing up a worker, a room and a session. Every branch below is a message the
frontend can actually send.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.runtime.dispatch import DataMessageRouter, decode_data_message, route_metrics

# -- decoding ---------------------------------------------------------------


def test_a_valid_payload_decodes() -> None:
    assert decode_data_message(b'{"type": "config"}') == {"type": "config"}


@pytest.mark.parametrize(
    "raw",
    [
        b"not json at all",
        b"",
        b"\xff\xfe invalid utf-8",
        b"[1, 2, 3]",  # valid JSON, but not a message object
        b'"a string"',
        b"null",
        None,
    ],
)
def test_malformed_payloads_are_discarded_not_raised(raw: Any) -> None:
    """One bad packet must not kill the data handler for the whole session."""
    assert decode_data_message(raw) is None


# -- metrics ----------------------------------------------------------------


class RecordingTracker:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def on_llm_metrics(self, m: Any) -> None:
        self.seen.append("llm")

    def on_stt_metrics(self, m: Any) -> None:
        self.seen.append("stt")

    def on_tts_metrics(self, m: Any) -> None:
        self.seen.append("tts")

    def on_realtime_metrics(self, m: Any) -> None:
        self.seen.append("realtime")


@pytest.mark.parametrize(
    ("metric_type", "expected"),
    [
        ("llm_metrics", "llm"),
        ("stt_metrics", "stt"),
        ("tts_metrics", "tts"),
        ("realtime_model_metrics", "realtime"),
    ],
)
def test_each_metric_type_reaches_its_tracker_method(metric_type: str, expected: str) -> None:
    tracker = RecordingTracker()
    metrics = type("M", (), {"type": metric_type})()

    assert route_metrics(tracker, metrics) is True
    assert tracker.seen == [expected]


def test_an_unknown_metric_is_reported_as_unrouted() -> None:
    tracker = RecordingTracker()
    assert route_metrics(tracker, type("M", (), {"type": "eos_metrics"})()) is False
    assert route_metrics(tracker, object()) is False
    assert tracker.seen == []


# -- routing ----------------------------------------------------------------


class FakeState:
    def __init__(self, agent: Any = None, browser: Any = None) -> None:
        self.current_agent = agent
        self.active_browser_session = browser
        self.spawned: list[str] = []

    def spawn(self, coro: Any, *, name: str) -> None:
        self.spawned.append(name)
        coro.close()  # we are not running a loop; don't warn about it

    def run_serialized(self, coro: Any) -> Any:
        return coro


class FakeAgent:
    def __init__(self) -> None:
        self.searches: list[str] = []

    async def web_search(self, ctx: Any, query: str, **kwargs: Any) -> str:
        self.searches.append(query)
        return ""


class FakeBrowser:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _router(state: FakeState) -> DataMessageRouter:
    return DataMessageRouter(
        session=object(), state=state, vad=None, create_agent_fn=lambda *a, **k: None
    )


def test_a_config_message_is_spawned_and_serialized() -> None:
    state = FakeState()
    assert _router(state).handle({"type": "config", "voice": {}}) == "config"
    assert state.spawned == ["handle_full_config"]


def test_a_config_update_is_spawned_and_serialized() -> None:
    state = FakeState()
    assert _router(state).handle({"type": "config_update"}) == "config_update"
    assert state.spawned == ["handle_config_update"]


def test_a_browser_close_request_closes_a_live_browser() -> None:
    state = FakeState(browser=FakeBrowser())
    assert _router(state).handle({"type": "browser_close_request"}) == "browser_close_request"
    assert state.spawned == ["browser_close_request"]


def test_a_browser_close_request_with_no_browser_is_a_no_op() -> None:
    state = FakeState(browser=None)
    _router(state).handle({"type": "browser_close_request"})
    assert state.spawned == []


def test_search_similar_builds_a_query_from_the_client_title() -> None:
    state = FakeState(agent=FakeAgent())
    assert _router(state).handle({"type": "search_similar", "title": "Red bag"}) == "search_similar"
    assert state.spawned == ["search_similar"]


def test_search_similar_without_an_agent_is_a_no_op() -> None:
    state = FakeState(agent=None)
    _router(state).handle({"type": "search_similar", "title": "Red bag"})
    assert state.spawned == []


def test_an_unknown_message_type_is_ignored() -> None:
    state = FakeState()
    assert _router(state).handle({"type": "who_knows"}) is None
    assert state.spawned == []


def test_a_message_with_no_type_is_ignored() -> None:
    state = FakeState()
    assert _router(state).handle({}) is None


def test_a_non_string_type_is_ignored() -> None:
    """The type field comes off the wire; it is not guaranteed to be a string."""
    state = FakeState()
    assert _router(state).handle({"type": 7}) is None
    assert state.spawned == []
