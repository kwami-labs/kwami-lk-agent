"""Client-side tool management for Kwami agent."""

import asyncio
import json
import uuid
from typing import TYPE_CHECKING, Any

from livekit.agents import RunContext, function_tool

from ..room_context import get_current_room
from ..utils.logging import get_logger
from ..utils.validation import validate_tool_definition

if TYPE_CHECKING:
    from ..agent import KwamiAgent

logger = get_logger("client_tools")


# A client tool result goes verbatim into the LLM context.
MAX_CLIENT_TOOL_RESULT_CHARS = 4000


def _cap(text: str) -> str:
    """Bound a client-supplied string before it reaches the LLM."""
    if len(text) <= MAX_CLIENT_TOOL_RESULT_CHARS:
        return text
    dropped = len(text) - MAX_CLIENT_TOOL_RESULT_CHARS
    return f"{text[:MAX_CLIENT_TOOL_RESULT_CHARS]}\n... [truncated, {dropped} more characters]"


class ClientToolManager:
    """Manages dynamic tools that are executed on the client side.

    This class handles:
    - Registration of client-defined tools
    - Forwarding tool calls to the client via data channel
    - Handling tool results from the client
    """

    def __init__(self, kwami_agent: "KwamiAgent"):
        """Initialize the client tool manager.

        Args:
            kwami_agent: The KwamiAgent instance (needed to access the room for sending data)
        """
        self.agent = kwami_agent
        self.pending_calls: dict[str, asyncio.Future] = {}
        self.registered_tools: list[dict[str, Any]] = []
        self._tools: list[Any] = []

    def register_client_tools(self, tool_definitions: list[dict[str, Any]]) -> None:
        """Register tools defined in configuration for the LLM.

        Args:
            tool_definitions: List of tool definition dictionaries.
        """
        if not tool_definitions:
            return

        for tool_def in tool_definitions:
            # Validate tool definition
            if not validate_tool_definition(tool_def):
                continue

            # Handle different formats
            func_def = tool_def.get("function", tool_def)
            tool_name = func_def.get("name")
            description = func_def.get("description", "")
            parameters = func_def.get("parameters", {})

            logger.info(f"Registering client tool: {tool_name}")

            # Create the tool using function_tool with raw_schema
            tool = self._create_client_tool(tool_name, description, parameters)
            self._tools.append(tool)
            self.registered_tools.append(tool_def)

    def _create_client_tool(
        self,
        tool_name: str,
        description: str,
        parameters: dict,
    ) -> Any:
        """Create a function tool that forwards calls to the client.

        Args:
            tool_name: The name of the tool.
            description: Tool description for the LLM.
            parameters: JSON schema for tool parameters.

        Returns:
            A function_tool decorated handler.
        """
        # Build the raw schema for the tool
        raw_schema = {
            "type": "function",
            "name": tool_name,
            "description": description,
            "parameters": parameters
            if parameters
            else {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }

        # Create the handler function that will be called when the tool is invoked
        async def tool_handler(raw_arguments: dict, context: RunContext) -> str:
            tool_call_id = str(uuid.uuid4())
            logger.info(
                f"Calling client tool '{tool_name}' (id: {tool_call_id}) args: {raw_arguments}"
            )

            room = (
                get_current_room()
                or (getattr(context, "room", None) if context else None)
                or getattr(self.agent, "room", None)
            )

            # Check room connection
            if not room:
                logger.error(
                    "Cannot call client tool: No room connection "
                    "(current_room=%s, context_room=%s, agent_room=%s)",
                    get_current_room() is not None,
                    getattr(context, "room", None) is not None if context else False,
                    getattr(self.agent, "room", None) is not None,
                )
                return "Error: Agent not connected to room"

            result_future: asyncio.Future = asyncio.Future()
            self.pending_calls[tool_call_id] = result_future

            payload = {
                "type": "tool_call",
                "toolCallId": tool_call_id,
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(raw_arguments),
                },
            }

            try:
                data = json.dumps(payload).encode("utf-8")
                await room.local_participant.publish_data(data, reliable=True)

                try:
                    result = await asyncio.wait_for(result_future, timeout=30.0)
                    return result
                except TimeoutError:
                    logger.warning(f"Tool call timed out: {tool_name} ({tool_call_id})")
                    return "Error: Tool execution timed out"

            except Exception as e:
                logger.error(f"Error executing client tool: {e}")
                return f"Error executing tool: {str(e)}"
            finally:
                self.pending_calls.pop(tool_call_id, None)

        # Create the function tool using the raw_schema approach
        return function_tool(tool_handler, raw_schema=raw_schema)

    def handle_tool_result(
        self,
        tool_call_id: str,
        result: str | None,
        error: str | None = None,
    ) -> None:
        """Handle incoming tool result from client.

        Args:
            tool_call_id: The ID of the tool call.
            result: The result string from the client.
            error: Optional error message.
        """
        if tool_call_id not in self.pending_calls:
            logger.warning(f"Received result for unknown tool call: {tool_call_id}")
            return

        future = self.pending_calls[tool_call_id]
        if future.done():
            logger.warning(f"Tool call already completed: {tool_call_id}")
            return

        # Everything here lands straight in the LLM context. The frontend is
        # not a trusted size bound, so cap both branches rather than letting a
        # buggy or hostile client blow the prompt budget -- or the bill.
        if error:
            future.set_result(_cap(f"Error from client: {error}"))
        else:
            future.set_result(_cap(result or ""))

    def create_client_tools(self) -> list[Any]:
        """Return the list of registered client tools.

        Returns:
            List of function_tool instances.
        """
        return self._tools

    @property
    def tool_count(self) -> int:
        """Get the number of registered client tools."""
        return len(self._tools)
