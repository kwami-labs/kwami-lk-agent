"""Logging: what the agent writes, and what it must not.

Two jobs. `redacted` decides what of the user's own words is allowed to reach a
log at all. The rest of this module decides whether a line can be tied back to
the session that produced it.

Before this, it could not. Room name and `kwami_id` were interpolated into
individual messages by hand wherever someone remembered -- `f"Kwami session
starting in room: {room}"` -- so reconstructing one session out of a worker
serving many meant grepping for a string that appeared in three log lines out of
several hundred. At a million sessions a day that is not a workflow.

`session_context` binds them once for the whole job. Every record emitted inside
that scope carries them, whatever module wrote it, without a single call site
having to remember. A ContextVar rather than a global because a worker runs many
jobs concurrently in one process, and `copy_context` semantics mean a task
spawned inside a session inherits its binding.

JSON output is opt-in via `KWAMI_LOG_FORMAT=json`. The Cloudflare Worker beside
this already logs JSON; a human running `make dev` does not want to.
"""

from __future__ import annotations

import contextvars
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

# Single logger name for the entire agent
LOGGER_NAME = "kwami-agent"

#: Stand-in for a value that is absent rather than redacted, so a log line can
#: distinguish "the user has no name stored" from "the name is not printed here".
EMPTY = "<empty>"
NONE = "<none>"


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Optional sub-logger name. If provided, creates "kwami-agent.{name}".
              If None, returns the main "kwami-agent" logger.

    Returns:
        Logger instance.
    """
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)


def redacted(value: object, *, keep: int = 0) -> str:
    """A log-safe stand-in for something the user said or asked us to remember.

    Memory turns conversations into stored names and facts, and those were
    going to INFO verbatim -- `Remembered fact: %s`, `Found user name: %s`. That
    put the user's name and whatever they asked to be remembered into whatever
    aggregator the deployment ships logs to, for a retention period nobody chose
    with that content in mind.

    What is kept is deliberately just enough to debug with: the length, so a
    truncation or an empty extraction is still visible, and optionally the first
    `keep` characters, so two lines about the same value can be correlated. It
    is not reversible and is not meant to be.

    Args:
        value: The user content to describe rather than print.
        keep: How many leading characters to show. Zero for free text; one is
            enough to follow a name through a session.
    """
    if value is None:
        return NONE
    text = str(value)
    if not text:
        return EMPTY
    if keep <= 0:
        return f"<{len(text)} chars>"
    return f"{text[:keep]}<{len(text)} chars>"


# -- Session correlation ------------------------------------------------------

#: Bound once per job. A ContextVar, not a global: one worker process runs many
#: sessions concurrently, and a task spawned inside a session inherits the
#: binding that was current when it was created.
_session_fields: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "kwami_session_fields", default=None
)

#: Keys that never appear on a log record for other reasons, so injecting them
#: cannot collide with `logging`'s own attributes.
RESERVED_RECORD_FIELDS = frozenset(
    {"name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module", "message"}
)


def session_fields() -> dict[str, str]:
    """The correlation fields currently bound, if any."""
    return dict(_session_fields.get() or {})


@contextmanager
def session_context(**fields: str | None) -> Iterator[None]:
    """Bind correlation fields for everything logged inside this scope.

    Falsy values are dropped rather than recorded as empty strings: a line
    saying `kwami_id=""` reads as "we looked and there is none", which is a
    different and usually wrong claim.
    """
    merged = {**(_session_fields.get() or {}), **{k: str(v) for k, v in fields.items() if v}}
    token = _session_fields.set(merged)
    try:
        yield
    finally:
        _session_fields.reset(token)


def bind_session_fields(**fields: str | None) -> None:
    """Add correlation fields to the binding already in scope.

    For values that are only known partway through a job -- a telephony
    `kwami_id` arrives after the room is up -- so lines written before it are
    still correlated by room, and lines after carry both.
    """
    merged = {**(_session_fields.get() or {}), **{k: str(v) for k, v in fields.items() if v}}
    _session_fields.set(merged)


def _annotate(record: logging.LogRecord) -> logging.LogRecord:
    """Copy the bound correlation fields onto a record."""
    for key, value in (_session_fields.get() or {}).items():
        if key not in RESERVED_RECORD_FIELDS:
            setattr(record, key, value)
    return record


class SessionContextFilter(logging.Filter):
    """Annotates a record with the bound correlation fields.

    Kept as a filter for handlers that want it explicitly, but note the trap
    that made this necessary to get right: a `logging.Filter` attached to a
    *logger* only runs for records logged directly to that logger. Records
    propagated up from `kwami-agent.memory`, or from `livekit.agents`, never
    touch the root logger's filters at all -- only its handlers. Attaching here
    alone silently correlated almost nothing.

    `install_record_factory` is what actually guarantees coverage.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        _annotate(record)
        return True


def install_record_factory() -> None:
    """Annotate every record at creation, wherever it comes from.

    The record factory runs for every `LogRecord` the process makes, so this
    reaches the SDK's loggers, third-party loggers and any handler anyone
    attaches later -- including a test's `caplog`. Idempotent: wrapping twice
    would annotate twice, harmlessly but pointlessly, so the wrapper is tagged.
    """
    current = logging.getLogRecordFactory()
    if getattr(current, "_kwami_session_factory", False):
        return

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        return _annotate(current(*args, **kwargs))

    factory._kwami_session_factory = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(factory)


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with the correlation fields at the top level.

    Aggregators index top-level keys; nesting the session under a `context`
    object would mean every query needs a path expression.
    """

    #: Everything `logging` puts on a record for its own purposes. Anything else
    #: was put there by the filter above and is ours to emit.
    _STANDARD = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
        "message",
        "asctime",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        # `logging.Formatter.format` sets this, and other code reads it --
        # pytest's caplog among them. Leaving it unset makes any handler this
        # formatter touches quietly break `record.message` for everyone else.
        record.message = record.getMessage()
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.message,
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
        }
        payload.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key not in self._STANDARD and not key.startswith("_")
            }
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(*, force_json: bool | None = None) -> None:
    """Install the correlation filter, and JSON output when asked for.

    Idempotent: called from the process entry point, and safe if something calls
    it again. Attaches to the root logger so the SDK's records are correlated
    too, not only ours.
    """
    # The factory, not a filter on the root logger: a logger's filters do not
    # run for records propagated from its children, so a root filter would have
    # correlated our own direct calls and essentially nothing else.
    install_record_factory()

    root = logging.getLogger()

    if force_json is None:
        # Read through Settings like every other variable, so it is covered by
        # the inventory guard and the Worker's forwarded environment rather than
        # being a second way to configure the process.
        from ..settings import get_settings

        as_json = get_settings().log_format == "json"
    else:
        as_json = force_json
    if not as_json:
        return

    if not root.handlers:
        root.addHandler(logging.StreamHandler())
    for handler in root.handlers:
        handler.setFormatter(JsonFormatter())
