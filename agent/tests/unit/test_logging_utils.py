"""`utils/logging.py` decides two things: what a logger is called, and what
of the user's own words is allowed to reach it. The second is the reason
`redacted` exists, so its edges are exercised rather than assumed."""

from __future__ import annotations

from src.utils.logging import EMPTY, LOGGER_NAME, NONE, get_logger, redacted


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


def test_redacted_reports_length_and_not_content() -> None:
    """The default shape: enough to spot a truncation, nothing to read."""
    fact = "the user's partner is called Marie and they live in Lisbon"

    result = redacted(fact)

    assert result == f"<{len(fact)} chars>"
    assert "Marie" not in result
    assert "Lisbon" not in result


def test_redacted_can_keep_a_leading_character_for_correlation() -> None:
    """One character is enough to follow a name across two log lines."""
    assert redacted("Alexandra", keep=1) == "A<9 chars>"
    assert redacted("Alexandra", keep=3) == "Ale<9 chars>"


def test_redacted_distinguishes_absent_from_empty() -> None:
    """ "No name stored" and "a name we are not printing" are different bugs."""
    assert redacted(None) == NONE
    assert redacted("") == EMPTY
    assert redacted(None, keep=1) == NONE


def test_redacted_never_leaks_a_value_shorter_than_keep() -> None:
    """`keep` is an upper bound, not a promise -- slicing past the end is silent."""
    assert redacted("Jo", keep=5) == "Jo<2 chars>"


def test_redacted_accepts_things_that_are_not_strings() -> None:
    """Callers pass whatever the SDK handed them; it must not raise here."""
    assert redacted(12345) == "<5 chars>"
    assert redacted(["a", "b"]) == f"<{len(str(['a', 'b']))} chars>"


def test_a_work_thread_logs_its_info_lines_to_stderr(monkeypatch) -> None:
    """Started by plain Python, a worker printed nothing below WARNING (2026-11-06)."""
    import logging

    from src.utils import logging as logging_module

    calls: list[dict] = []
    configured: list[bool] = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kw: calls.append(kw))
    monkeypatch.setattr(logging_module, "configure_logging", lambda: configured.append(True))
    logging_module.log_to_console()
    assert calls == [{"level": logging.INFO, "format": logging_module.CONSOLE_FORMAT}]
    assert configured == [True]


def test_a_work_thread_logs_failed_requests_but_not_its_polling(monkeypatch) -> None:
    """A job poll every two seconds was 40,000 "200 OK" lines a day."""
    import logging

    from src.utils import logging as logging_module

    monkeypatch.setattr(logging, "basicConfig", lambda **kw: None)
    monkeypatch.setattr(logging_module, "configure_logging", lambda: None)
    http = logging.getLogger("httpx")
    monkeypatch.setattr(http, "filters", [])
    logging_module.log_to_console()
    logging_module.log_to_console()
    assert len(http.filters) == 1

    def request(status: int) -> logging.LogRecord:
        return http.makeRecord(
            "httpx",
            logging.INFO,
            __file__,
            1,
            'HTTP Request: %s %s "%s %d %s"',
            ("POST", "https://x.supabase.co/rest/v1/rpc/brain_claim_job", "HTTP/1.1", status, "OK"),
            None,
        )

    assert not http.filter(request(200))
    assert not http.filter(request(204))
    assert http.filter(request(400))
    assert http.filter(request(503))
    assert http.filter(http.makeRecord("httpx", logging.INFO, __file__, 1, "other", None, None))
