"""A stand-in for `claude --claude-in-chrome-mcp`, run as a real stdio MCP server.

Only `StdioTools`' tests start it: it answers like Claude in Chrome does, without a
browser, an account or Anthropic's bridge.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp_types import ImageContent, TextContent

server = MCPServer("Claude in Chrome")


@server.tool()
def list_connected_browsers() -> str:
    return '[{"deviceId":"d1","name":"Browser 1","osPlatform":"Windows","isLocal":true}]'


@server.tool()
def computer(action: str, tabId: int) -> list[TextContent | ImageContent]:  # noqa: N803
    return [
        TextContent(type="text", text=f"did {action} on {tabId}"),
        ImageContent(type="image", data="anBlZw==", mime_type="image/jpeg"),
    ]


if __name__ == "__main__":
    server.run()
