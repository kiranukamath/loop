"""
Loop's tools exposed as an MCP server (Phase 12a).

MCP (Model Context Protocol) standardizes how any client — Claude Desktop, an
IDE, another agent — discovers and calls a server's tools. This module is a
thin FastMCP wrapper around loop.tools / loop.retrieval: every @mcp.tool()
below is one line that delegates to an already-tested Phase 3/8 function.
NO logic is duplicated here — this file only translates our existing
functions into the MCP wire format (name, description, JSON input schema).

Run it as a standalone stdio server:
    uv run python -m loop.mcp_server

Register it in an MCP client (e.g. Claude Desktop's claude_desktop_config.json)
with a stdio command pointing at that invocation — see doc/phase-12-mcp.md.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from loop import tools

mcp = FastMCP("loop")


@mcp.tool()
def list_questions(modality: str) -> list[dict]:
    """List all interview questions for a modality.

    Args:
        modality: one of "coding", "system_design", "behavioral".
    """
    return tools.get_questions_by_modality(modality)


@mcp.tool()
def search_questions(query: str, modality: str | None = None, k: int = 3) -> list[dict]:
    """Semantically search interview questions by topic.

    Args:
        query: free-text description of the topic/skill to search for.
        modality: optional hard filter — "coding", "system_design", or "behavioral".
        k: maximum number of results to return.
    """
    return tools.search_questions(query, modality=modality, k=k)


@mcp.tool()
def get_rubric(question_id: str) -> dict | None:
    """Get the grading rubric for a question, by question id."""
    return tools.get_rubric(question_id)


@mcp.tool()
def get_reference_answer(question_id: str) -> str | None:
    """Get the reference/model answer for a question, by question id."""
    return tools.get_reference_answer(question_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")
