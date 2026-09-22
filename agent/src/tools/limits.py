"""The provider's ceiling on how many tools one request may carry.

The agent hands the model its own built-ins plus every tool the frontend
registered. Neither number is bounded by anything in this repo, and the two
grow independently: the app added sixteen client tools in a single afternoon.

OpenAI's limit is 128 tools per request, and it is a hard 400:

    Invalid 'tools': array too long. Expected an array with maximum length 128,
    but got an array with length 129 instead.

Verified against the live API rather than taken from memory -- 128 succeeds,
129 does not. That failure mode is the worst shape available: it is not one
degraded feature, it is *every turn of every session* failing, so the agent
joins the room and can never speak. And because it is driven by what the
frontend registers, it would arrive on a frontend deploy with nothing in this
repo having changed.

So the list is trimmed rather than sent over the limit. A session with 128
tools that works beats a session with 129 that cannot answer at all. Built-ins
are kept in preference to client tools on purpose: they are what the agent
needs to function at all -- search, navigation, memory, its own reconfiguration
-- and losing `web_search` silently is worse than losing the 129th UI control.
"""

from __future__ import annotations

from typing import Any

from ..utils.logging import get_logger

logger = get_logger("tool_limits")

#: OpenAI's hard limit, confirmed empirically. Anthropic and Google accept more,
#: but the pipeline can be switched to OpenAI mid-session, so the floor across
#: providers is the only safe budget.
MAX_TOOLS_PER_REQUEST = 128

#: Warn from here, so the ceiling is visible in logs before it is hit rather
#: than for the first time as a dead session.
TOOL_COUNT_WARN_THRESHOLD = int(MAX_TOOLS_PER_REQUEST * 0.85)


def _name_of(tool: Any) -> str:
    info = getattr(tool, "info", None)
    name = getattr(info, "name", None) if info is not None else None
    return name if isinstance(name, str) else "<unnamed>"


def trim_client_tools(
    builtin_count: int,
    client_tools: list[Any],
    *,
    limit: int = MAX_TOOLS_PER_REQUEST,
) -> list[Any]:
    """The client tools that fit alongside `builtin_count` built-ins.

    Split out from `enforce_tool_limit` because the two call sites need
    different halves of the same decision. At construction the framework appends
    the built-ins itself (`tools + find_function_tools(self)`), so the agent
    must hand it the *client* list already trimmed; `update_tools` assembles
    both halves by hand and wants the combined list.

    Args:
        builtin_count: How many of the agent's own tools will be present.
        client_tools: Tools the frontend registered, trimmed from the tail.
        limit: Overridable so a test does not need 129 real tools to reach the
            edge, and so a provider with a different ceiling can be given its own.
    """
    total = builtin_count + len(client_tools)

    if total <= limit:
        if total >= TOOL_COUNT_WARN_THRESHOLD:
            logger.warning(
                "Tool count is %d of a %d limit (%d built-in, %d from the client). "
                "Past the limit the provider rejects every request.",
                total,
                limit,
                builtin_count,
                len(client_tools),
            )
        return list(client_tools)

    room_for_client = max(0, limit - builtin_count)
    kept = client_tools[:room_for_client]
    dropped = client_tools[room_for_client:]

    logger.error(
        "Tool limit exceeded: %d built-in + %d client tools is %d, over the %d the "
        "provider accepts. Dropping %d client tool(s): %s. Those capabilities are "
        "NOT available this session.",
        builtin_count,
        len(client_tools),
        total,
        limit,
        len(dropped),
        ", ".join(_name_of(tool) for tool in dropped) or "<none>",
    )
    return kept


def enforce_tool_limit(
    builtin_tools: list[Any],
    client_tools: list[Any],
    *,
    limit: int = MAX_TOOLS_PER_REQUEST,
) -> list[Any]:
    """The combined tool list to hand the model, never longer than `limit`."""
    kept = trim_client_tools(len(builtin_tools), client_tools, limit=limit)

    if not kept and len(builtin_tools) > limit:
        # The built-ins alone overflow the budget. Trim them too rather than
        # send an over-length request: a crippled agent still answers, a
        # rejected one cannot answer at all.
        logger.critical(
            "The built-in tools alone (%d) exceed the %d limit; no client tool can be "
            "registered and built-ins are being trimmed.",
            len(builtin_tools),
            limit,
        )
        return list(builtin_tools[:limit])

    return [*builtin_tools, *kept]
