"""The memory boundary: persistent recall for a single Kwami."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MemoryContextPort(Protocol):
    """Retrieved context, ready to be folded into a system prompt."""

    context_block: str
    summary: str | None
    facts: list[str]
    recent_messages: list[dict[str, str]]

    def to_system_prompt_addition(self) -> str:
        """Render the context as prompt text, bounded in length."""
        ...


@runtime_checkable
class MemoryPort(Protocol):
    """Persistent memory for one user.

    Every method must degrade rather than raise: memory is an enhancement, and
    a failure here should cost context, never the call.
    """

    @property
    def is_initialized(self) -> bool: ...

    @property
    def user_id(self) -> str: ...

    async def initialize(self) -> bool: ...

    async def get_context(self) -> MemoryContextPort: ...

    async def get_user_name(self) -> str | None: ...

    async def set_user_name(self, name: str) -> None: ...

    async def add_fact(self, fact: str) -> None: ...

    async def buffer_user_message(self, content: str, name: str | None = None) -> None: ...

    async def add_exchange(
        self, assistant_content: str, assistant_name: str | None = None
    ) -> None: ...

    async def search(self, query: str, limit: int = 5) -> list[dict]: ...

    async def close(self) -> None: ...
