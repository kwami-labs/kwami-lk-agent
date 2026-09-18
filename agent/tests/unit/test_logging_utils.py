"""`utils/logging.py` is the one place errors get formatted, so both of its
shapes have to be exercised -- a truncated error message is how a production
failure becomes unreadable."""

from __future__ import annotations

import logging

from src.utils.logging import LOGGER_NAME, get_logger, log_error


def test_a_bare_get_logger_returns_the_root_agent_logger() -> None:
    assert get_logger().name == LOGGER_NAME


def test_a_named_logger_is_a_child_of_the_agent_logger() -> None:
    logger = get_logger("memory")

    assert logger.name == f"{LOGGER_NAME}.memory"
    # A child, so a level set on the parent still applies.
    assert logger.name.startswith(f"{LOGGER_NAME}.")


def test_an_empty_name_falls_back_to_the_root_logger() -> None:
    """`if name:` is falsy-checked, not None-checked; "" must not build
    "kwami-agent." with a trailing dot."""
    assert get_logger("").name == LOGGER_NAME


def test_log_error_includes_the_type_the_message_and_the_traceback(
    caplog,
) -> None:
    logger = get_logger("test-with-tb")

    with caplog.at_level(logging.ERROR):
        try:
            raise ValueError("boom")
        except ValueError as exc:
            log_error(logger, "while doing the thing", exc)

    record = caplog.records[-1]
    assert "while doing the thing" in record.message
    assert "ValueError" in record.message
    assert "boom" in record.message
    assert "Traceback" in record.message


def test_log_error_can_omit_the_traceback(caplog) -> None:
    logger = get_logger("test-no-tb")

    with caplog.at_level(logging.ERROR):
        log_error(logger, "quietly", RuntimeError("nope"), include_traceback=False)

    record = caplog.records[-1]
    assert record.message == "quietly: RuntimeError: nope"
    assert "Traceback" not in record.message
