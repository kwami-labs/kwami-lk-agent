"""The tool shelf: the client tools that did not fit, still within reach.

A request carries at most 128 tools (`tools/limits.py`). Past that the client
tools at the tail were dropped for the whole session: the model could not call
them, and could not know they existed, so a user who asked for one heard "I
can't do that" from an assistant whose app had the tool registered.

Now they wait on a shelf. When anything is shelved, the model gets one more
tool, `find_more_tools`: it searches the shelf by what the user wants and swaps
the matches in, in place of the lowest-priority client tools on the list (the
tail, which is what the trim would have cut next). Nothing is lost and the list
never grows past the ceiling. With nothing shelved the tool is withheld, so it
costs no one its place.
"""

from __future__ import annotations

import re
from typing import Any

from livekit.agents import RunContext, function_tool

from ..utils.logging import get_logger
from .limits import KEEP_LONGEST

logger = get_logger("tools.shelf")

SHELF_TOOL = "find_more_tools"
#: How many tools one search brings in. Each one swaps a tool out.
MAX_BROUGHT_IN = 5
#: How many shelved names a search with no match offers instead.
MAX_SUGGESTED = 20

_WORD = re.compile(r"[a-z0-9]+")


def tool_name(tool: Any) -> str:
    name = getattr(getattr(tool, "info", None), "name", None)
    return name if isinstance(name, str) else ""


def _description(tool: Any) -> str:
    info = getattr(tool, "info", None)
    raw = getattr(info, "raw_schema", None)
    if isinstance(raw, dict) and isinstance(raw.get("description"), str):
        return raw["description"]
    return getattr(info, "description", "") or ""


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower().replace("_", " ")))


def rank(shelf: list[Any], query: str) -> list[Any]:
    """The shelved tools that match `query`, best first. A name match counts double."""
    wanted = _words(query)
    if not wanted:
        return []
    scored: list[tuple[int, int, Any]] = []
    for index, tool in enumerate(shelf):
        in_name = len(wanted & _words(tool_name(tool)))
        in_text = len(wanted & _words(_description(tool)))
        score = 2 * in_name + in_text
        if score:
            scored.append((-score, index, tool))
    return [tool for _, _, tool in sorted(scored, key=lambda item: item[:2])]


def swap(
    offered: list[Any], client: list[Any], shelf: list[Any], wanted: list[Any]
) -> tuple[list[Any], list[Any], list[Any]]:
    """Bring `wanted` in from the shelf, sending as many client tools out.

    Returns (the new offered list, what came in, the new shelf). The tools sent
    out are the last client tools offered -- the tail the trim cuts first --
    never one of `KEEP_LONGEST`, and never the agent's own.
    """
    client_ids = {id(tool) for tool in client}
    movable = [
        tool
        for tool in reversed(offered)
        if id(tool) in client_ids and tool_name(tool) not in KEEP_LONGEST
    ]
    count = min(len(wanted), len(movable))
    brought, out = wanted[:count], movable[:count]
    out_ids = {id(tool) for tool in out}
    brought_ids = {id(tool) for tool in brought}
    kept = [tool for tool in offered if id(tool) not in out_ids]
    new_shelf = [tool for tool in shelf if id(tool) not in brought_ids] + out
    return [*kept, *brought], brought, new_shelf


class ToolShelfMixin:
    """`find_more_tools`, offered only while some client tools are shelved."""

    _shelf: list[Any]

    @function_tool()
    async def find_more_tools(self, context: RunContext, query: str) -> dict[str, Any]:
        """Bring in tools set aside to fit the limit on how many you can hold at once.

        Call it when the user asks for something none of your tools does, before
        saying you cannot: describe what they want in a few words. The tools found
        become available at once; then call them.

        Args:
            query: What the user wants done, in a few keywords ("calendar event",
                "export theme", "wallet").
        """
        shelf = list(getattr(self, "_shelf", None) or [])
        if not shelf:
            return {"found": [], "note": "Every tool is already available to you."}
        found = rank(shelf, query)[:MAX_BROUGHT_IN]
        if not found:
            return {
                "found": [],
                "note": "Nothing set aside matches. These are set aside; search again by one.",
                "shelved": [tool_name(tool) for tool in shelf[:MAX_SUGGESTED]],
            }
        client = list(self.client_tools.create_client_tools())  # type: ignore[attr-defined]
        offered, brought, new_shelf = swap(list(self.tools), client, shelf, found)  # type: ignore[attr-defined]
        if not brought:
            return {"found": [], "note": "No room can be made for more tools right now."}
        self._shelf = new_shelf
        await self.update_tools(offered)  # type: ignore[attr-defined]
        publish = getattr(self, "publish_tool_inventory", None)
        if publish is not None:
            await publish()
        logger.info(
            "Shelf: brought in %s for %r", ", ".join(tool_name(t) for t in brought), query[:80]
        )
        return {
            "found": [
                {"name": tool_name(tool), "description": _description(tool)[:300]}
                for tool in brought
            ],
            "note": "These are available now. Call the one that fits.",
        }
