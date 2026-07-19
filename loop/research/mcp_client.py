"""
Loads tools from external MCP servers into the Phase 9 ReAct research agent
(Phase 12b). Loop was an MCP *server* in 12a (loop/mcp_server.py, exposing
our own tools to other clients); this module makes Loop an MCP *client* too
-- the same host can hold both roles for different connections.

Config-driven allow-list (loop.config.settings.mcp_server_configs), empty by
default -- the same "opt-in, feature-off" pattern as Phase 10's
fallback_model_id and Phase 9b's company: with no servers configured, the
research agent's tool list is byte-for-byte the Phase 9 flow, and no
subprocess is ever spawned.

Sync/async boundary: MultiServerMCPClient.get_tools() is a coroutine -- MCP
communication is inherently async (request/response over a subprocess pipe,
potentially to several servers at once). loop/nodes/research.py's research()
node is a plain sync function (the rest of the graph, and graph.stream() in
api.py, are sync), so load_mcp_tools() wraps the call in asyncio.run(). This
is safe here because nothing in the sync graph-node call path already has an
event loop running -- verified empirically: research() is invoked from
graph.stream(), which is itself sync, all the way up to the FastAPI request
handler that calls it (which does its own async dispatch in a separate
context, not nested inside this call).

Verified against installed mcp==1.28.1 / langchain-mcp-adapters==0.3.0:
    MultiServerMCPClient(connections: dict[str, StdioConnection | ...])
    StdioConnection = {"command": str, "args": list[str], "transport": "stdio", ...}
    client.get_tools(*, server_name: str | None = None) -> list[BaseTool]  (async)
"""

from __future__ import annotations

import asyncio

from langchain_core.tools import BaseTool

from loop.config import settings


def load_mcp_tools() -> list[BaseTool]:
    """Return LangChain tools loaded from every server in settings.mcp_server_configs.

    Returns [] immediately -- no client constructed, no subprocess spawned --
    when the config is empty (the default). This keeps the feature fully
    opt-in: an unconfigured Loop install never touches MCP at all.
    """
    if not settings.mcp_server_configs:
        return []

    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(settings.mcp_server_configs)
    return asyncio.run(client.get_tools())
