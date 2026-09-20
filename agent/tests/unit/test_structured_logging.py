"""Correlating a log line back to the session that produced it.

Room name and `kwami_id` were interpolated into individual messages by hand
wherever someone remembered, so reconstructing one session out of a worker
serving many meant grepping for a string that appeared in three lines out of
several hundred. At a million sessions a day that is not a workflow.

The two properties that matter:

1. A field bound once reaches every record, from any logger -- including the
   SDK's, because a livekit-agents warning about *this* session should be
   findable with the same query as ours.
2. Sessions do not bleed into each other. One worker process runs many
   concurrently, so a global would be worse than nothing: it would attribute
   lines to the wrong user.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from src.utils.logging import (
    JsonFormatter,
    SessionContextFilter,
    bind_session_fields,
    configure_logging,
    get_logger,
    session_context,
    session_fields,
)


@pytest.fixture
def records(caplog):
    caplog.set_level(logging.INFO)
    return caplog


# -- Binding -------------------------------------------------------------------


def test_nothing_is_bound_by_default() -> None:
    assert session_fields() == {}


def test_a_bound_field_is_visible_inside_the_scope() -> None:
    with session_context(room="room-1"):
        assert session_fields() == {"room": "room-1"}


def test_the_binding_is_released_on_exit() -> None:
    with session_context(room="room-1"):
        pass

    assert session_fields() == {}


def test_the_binding_is_released_even_on_an_exception() -> None:
    """A job that raises must not leave its room bound for the next one."""
    with pytest.raises(RuntimeError), session_context(room="room-1"):
        raise RuntimeError("job failed")

    assert session_fields() == {}


def test_empty_values_are_not_bound() -> None:
    """`kwami_id=""` reads as "we looked and there is none", which is a
    different and usually wrong claim."""
    with session_context(room="room-1", kwami_id=None, user=""):
        assert session_fields() == {"room": "room-1"}


def test_a_field_can_be_added_partway_through() -> None:
    """A telephony `kwami_id` arrives after the room is up, so lines before it
    are correlated by room and lines after carry both."""
    with session_context(room="room-1"):
        bind_session_fields(kwami_id="kwami_1")

        assert session_fields() == {"room": "room-1", "kwami_id": "kwami_1"}


def test_nested_scopes_merge() -> None:
    with session_context(room="room-1"), session_context(kwami_id="kwami_1"):
        assert session_fields() == {"room": "room-1", "kwami_id": "kwami_1"}


def test_values_are_stringified() -> None:
    with session_context(attempt=3):  # type: ignore[arg-type]
        assert session_fields() == {"attempt": "3"}


async def test_concurrent_sessions_do_not_bleed_into_each_other() -> None:
    """The reason this is a ContextVar and not a global. Getting this wrong
    attributes log lines to the wrong user."""
    seen: dict[str, dict[str, str]] = {}

    async def job(name: str) -> None:
        with session_context(room=name):
            await asyncio.sleep(0)  # let the other task interleave
            seen[name] = session_fields()

    await asyncio.gather(job("room-a"), job("room-b"))

    assert seen == {"room-a": {"room": "room-a"}, "room-b": {"room": "room-b"}}


# -- Reaching the records -------------------------------------------------------


def test_the_filter_copies_the_fields_onto_a_record() -> None:
    record = logging.LogRecord("x", logging.INFO, "f", 1, "hello", None, None)

    with session_context(room="room-1"):
        assert SessionContextFilter().filter(record) is True

    assert record.room == "room-1"  # type: ignore[attr-defined]


def test_the_filter_never_overwrites_a_logging_attribute() -> None:
    """A field called `module` or `message` must not clobber the record's own."""
    record = logging.LogRecord("x", logging.INFO, "f", 1, "hello", None, None)
    original = record.module

    with session_context(module="mine", name="mine"):
        SessionContextFilter().filter(record)

    assert record.module == original
    assert record.name == "x"


def test_the_filter_passes_every_record_through() -> None:
    """It annotates; it must never drop a line."""
    record = logging.LogRecord("x", logging.INFO, "f", 1, "hello", None, None)

    assert SessionContextFilter().filter(record) is True


def test_a_logged_line_carries_the_binding(records) -> None:
    configure_logging(force_json=False)

    with session_context(room="room-1", kwami_id="kwami_1"):
        get_logger("test").info("something happened")

    record = records.records[-1]
    assert record.room == "room-1"
    assert record.kwami_id == "kwami_1"


def test_records_from_another_library_are_correlated_too(records) -> None:
    """The SDK's own warnings about this session should be findable with the
    same query as ours."""
    configure_logging(force_json=False)

    with session_context(room="room-1"):
        logging.getLogger("livekit.agents").warning("something from the SDK")

    assert records.records[-1].room == "room-1"


# -- JSON output ----------------------------------------------------------------


def test_the_json_formatter_emits_one_object_per_line() -> None:
    record = logging.LogRecord("kwami-agent", logging.INFO, "f", 1, "hello %s", ("world",), None)

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "kwami-agent"
    assert payload["msg"] == "hello world"
    assert "time" in payload


def test_correlation_fields_are_top_level_in_json() -> None:
    """Aggregators index top-level keys; nesting would mean every query needs a
    path expression."""
    record = logging.LogRecord("kwami-agent", logging.INFO, "f", 1, "hi", None, None)
    with session_context(room="room-1"):
        SessionContextFilter().filter(record)

    payload = json.loads(JsonFormatter().format(record))

    assert payload["room"] == "room-1"


def test_an_exception_is_included_in_json() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            "kwami-agent", logging.ERROR, "f", 1, "failed", None, sys.exc_info()
        )

    payload = json.loads(JsonFormatter().format(record))

    assert "ValueError: boom" in payload["exception"]


def test_unserialisable_values_do_not_break_the_line() -> None:
    """A log line that raises while being formatted loses the information it
    was written to record."""
    record = logging.LogRecord("kwami-agent", logging.INFO, "f", 1, "hi", None, None)
    record.thing = object()  # type: ignore[attr-defined]

    payload = json.loads(JsonFormatter().format(record))

    assert "object at" in payload["thing"]


# -- Configuration ---------------------------------------------------------------


def test_configuring_twice_does_not_stack_record_factories() -> None:
    """It is called from the process entry point; a second call must be a
    no-op rather than wrapping the factory again on every import."""
    configure_logging(force_json=False)
    once = logging.getLogRecordFactory()

    configure_logging(force_json=False)

    assert logging.getLogRecordFactory() is once


def test_json_output_is_opt_in(env_setting) -> None:
    """A human running `make dev` does not want JSON."""
    env_setting("KWAMI_LOG_FORMAT", None)
    root = logging.getLogger()
    handler = logging.StreamHandler()
    root.addHandler(handler)
    try:
        configure_logging()

        assert not isinstance(handler.formatter, JsonFormatter)
    finally:
        root.removeHandler(handler)


def test_the_env_var_turns_json_on(env_setting) -> None:
    env_setting("KWAMI_LOG_FORMAT", "json")
    root = logging.getLogger()
    handler = logging.StreamHandler()
    root.addHandler(handler)
    try:
        configure_logging()

        assert isinstance(handler.formatter, JsonFormatter)
    finally:
        root.removeHandler(handler)
        handler.setFormatter(None)


def test_json_mode_adds_a_handler_when_there_is_none() -> None:
    """Otherwise turning JSON on in a bare process silently produces no output."""
    root = logging.getLogger()
    saved = root.handlers[:]
    root.handlers = []
    try:
        configure_logging(force_json=True)

        assert root.handlers
        assert isinstance(root.handlers[0].formatter, JsonFormatter)
    finally:
        root.handlers = saved
