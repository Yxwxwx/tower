"""MCP client — unified tool access for all agents.

Agents access MCP tools through this client. The client manages
connections to infra-mcp and hpc-mcp servers.

MVP: in-process function calls (no stdio process spawn).
Post-MVP: MCP stdio JSON-RPC protocol.
"""
from typing import Any


class MCPClient:
    """Unified MCP tool access layer.

    Agents call `client.call_tool(server, tool_name, params)`.
    The client routes to the correct MCP server.

    MVP implementation calls tool functions directly in-process.
    Post-MVP spawns MCP server processes via stdio and uses JSON-RPC.
    """

    def __init__(self):
        self._tools: dict[str, dict[str, callable]] = {}
        self._schemas: dict[str, dict[str, dict]] = {}

    # ── Registration ──

    def register_server(self, server_name: str, tools: dict[str, callable],
                        schemas: dict[str, dict] | None = None):
        """Register an MCP server's tools.

        Called at startup for infra-mcp and hpc-mcp.
        """
        self._tools[server_name] = tools
        if schemas:
            self._schemas[server_name] = schemas

    # ── Tool execution ──

    async def call_tool(self, server: str, tool_name: str,
                        params: dict[str, Any]) -> dict[str, Any]:
        """Call a tool on an MCP server.

        Args:
            server: "infra-mcp" | "hpc-mcp"
            tool_name: "filesystem.read" | "queue-status.query" | ...
            params: Tool-specific parameters.

        Returns:
            Tool result dict. Check for "error" key before using data.

        Raises:
            KeyError: server or tool not found.
        """
        server_tools = self._tools.get(server)
        if server_tools is None:
            return {"error": f"MCP server not found: {server}",
                    "error_code": "SERVER_NOT_FOUND"}

        tool_fn = server_tools.get(tool_name)
        if tool_fn is None:
            return {"error": f"Tool not found: {server}/{tool_name}",
                    "error_code": "TOOL_NOT_FOUND"}

        try:
            return await tool_fn(**params) if _is_async(tool_fn) else tool_fn(**params)
        except Exception as e:
            return {"error": str(e), "error_code": "TOOL_ERROR"}

    def call_tool_sync(self, server: str, tool_name: str,
                       params: dict[str, Any]) -> dict[str, Any]:
        """Synchronous wrapper for call_tool. For use in sync graph nodes."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.call_tool(server, tool_name, params))
        # Running in event loop — create new loop in thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run, self.call_tool(server, tool_name, params)
            )
            return future.result(timeout=30)

    # ── Discovery ──

    def list_servers(self) -> list[str]:
        return list(self._tools.keys())

    def list_tools(self, server: str) -> list[str]:
        server_tools = self._tools.get(server, {})
        return list(server_tools.keys())

    def get_schema(self, server: str, tool_name: str) -> dict | None:
        return self._schemas.get(server, {}).get(tool_name)


def _is_async(fn: callable) -> bool:
    import inspect
    return inspect.iscoroutinefunction(fn)


# ═══════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════

_client: MCPClient | None = None


def get_client() -> MCPClient:
    """Get or create the global MCP client.

    Registers infra-mcp and hpc-mcp tools on first call.
    """
    global _client
    if _client is None:
        import sys
        from pathlib import Path

        # Ensure mcp_servers/ is importable
        _mcp_root = Path(__file__).resolve().parent.parent.parent.parent / "mcp_servers"
        for _mcp_dir in ["infra-mcp/src", "hpc-mcp/src"]:
            _p = str(_mcp_root / _mcp_dir)
            if _p not in sys.path:
                sys.path.insert(0, _p)

        _client = MCPClient()

        # Register infra-mcp tools
        from infra_mcp.server import TOOLS as infra_tools
        from infra_mcp.server import TOOL_SCHEMAS as infra_schemas
        _client.register_server("infra-mcp", infra_tools, infra_schemas)

        # Register hpc-mcp tools
        from hpc_mcp.server import TOOLS as hpc_tools
        from hpc_mcp.server import TOOL_SCHEMAS as hpc_schemas
        _client.register_server("hpc-mcp", hpc_tools, hpc_schemas)

    return _client
