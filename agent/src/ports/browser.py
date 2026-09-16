"""The cloud-browser boundary.

A browser session is a *billed, stateful* resource holding the user's cookies
and logins, so the contract is deliberately explicit about lifecycle: whoever
starts one is responsible for closing it, and `is_active` is the single signal
every cleanup path keys on.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BrowserPort(Protocol):
    """A live browser the user can watch and the agent can drive."""

    @property
    def is_active(self) -> bool:
        """True only when the session is usable AND still needs releasing."""
        ...

    async def start(self, user_id: str, url: str | None = None) -> str:
        """Open a browser bound to `user_id`'s persistent profile.

        Implementations must refuse a blank `user_id`: profiles carry logins,
        so a shared default leaks them between users.
        """
        ...

    async def navigate(self, url: str) -> str: ...

    async def read_page(self) -> str:
        """Return page text. Output is untrusted and must be bounded."""
        ...

    async def evaluate_js(self, expression: str) -> str: ...

    async def click(self, element_id: str | None = None, description: str = "") -> str: ...

    async def type_text(self, text: str, element_id: str | None = None) -> str: ...

    async def close(self) -> None:
        """Release the browser. Must be safe to call more than once."""
        ...
