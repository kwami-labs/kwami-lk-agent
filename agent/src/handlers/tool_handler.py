"""Tool result handling for Kwami agent."""

from typing import Any

from ..utils.logging import get_logger

logger = get_logger("tool_handler")


def handle_tool_result(
    agent: Any,
    tool_call_id: str | None,
    result: str | None,
    error: str | None = None,
) -> None:
    """Handle incoming tool result from client.

    Args:
        agent: The current KwamiAgent instance.
        tool_call_id: The ID of the tool call.
        result: The result from the client.
        error: Optional error message.
    """
    if not tool_call_id:
        logger.warning("Received tool result with no tool_call_id")
        return

    if not agent:
        logger.warning(f"No agent available to handle tool result: {tool_call_id}")
        return

    # Check if agent has client_tools manager
    if hasattr(agent, "client_tools") and agent.client_tools:
        agent.client_tools.handle_tool_result(tool_call_id, result, error)
    elif hasattr(agent, "handle_tool_result"):
        # Fallback to direct method on agent
        agent.handle_tool_result(tool_call_id, result, error)
    else:
        logger.warning(f"Agent cannot handle tool results: {tool_call_id}")
