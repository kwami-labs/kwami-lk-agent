"""The credit-reporting boundary."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class UsageReporterPort(Protocol):
    """Reports a session's accumulated usage so credits can be deducted.

    Returning False means "not billed" and must be logged by the caller. This
    is the one path in the system where a silent failure costs money, so the
    contract is a bool rather than None.
    """

    async def report(self, user_id: str, session_id: str, tracker: Any) -> bool: ...
