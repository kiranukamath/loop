"""
A minimal MCP server, for tests only.

Phase 12b needs to prove Loop can *consume* an external MCP server, not just
expose its own (12a). This tiny FastMCP server exposes exactly one canned
tool over stdio; tests/test_mcp.py spawns it as a real subprocess via
MultiServerMCPClient and round-trips a call through actual MCP-over-stdio.
No fixtures/questions.json, no loop.tools — this file stands in for
"someone else's MCP server" that Loop's research agent might load tools from.

Run standalone (used by MultiServerMCPClient, not by a human):
    uv run python -m tests.fixtures.mock_mcp_server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mock-external-server")


@mcp.tool()
def echo_fact(topic: str) -> str:
    """Return a canned fact about the given topic. Test-only stand-in for a
    real external tool (e.g. a company-facts lookup)."""
    return f"Canned fact about {topic}: this MCP server is a test fixture."


if __name__ == "__main__":
    mcp.run(transport="stdio")
