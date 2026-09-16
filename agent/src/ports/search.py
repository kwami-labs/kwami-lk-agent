"""The web-search boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class SearchResult:
    """One result, in the shape the room and the LLM both consume."""

    title: str = ""
    url: str = ""
    content: str = ""
    image: str | None = None
    price: str | None = None
    features: list[str] = field(default_factory=list)


@runtime_checkable
class SearchPort(Protocol):
    """A provider that answers a query with results.

    Implementations must not raise: a search failure becomes an empty result
    set and a logged warning, because the alternative is a dropped voice turn.
    """

    async def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]: ...

    async def search_products(self, query: str, *, max_results: int = 5) -> list[SearchResult]: ...
