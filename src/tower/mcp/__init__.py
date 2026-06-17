"""MCP client — unified tool access layer for all agents.

Agents call MCP tools through this client. Never directly.
"""
from tower.mcp.client import MCPClient, get_client

__all__ = ["MCPClient", "get_client"]
