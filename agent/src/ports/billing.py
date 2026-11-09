"""The credit-reporting boundary."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class UsageReporterPort(Protocol):
    """Reports a session's accumulated usage so credits can be deducted.

    Returning False means "not billed" and must be logged by the caller. This
    is the one path in the system where a silent failure costs money, so the
    contract is a bool rather than None.

    `kwami_id` lets a reporter find the owning user itself when the identity
    it is handed is not a user id.
    """

    async def report(
        self, user_id: str, session_id: str, tracker: Any, *, kwami_id: str | None = None
    ) -> bool: ...
