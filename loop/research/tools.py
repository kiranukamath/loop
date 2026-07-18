"""
@tool-decorated wrappers around loop.research.search — what the ReAct agent
actually sees and calls.

The docstring on search_web IS the tool description the model reads to decide
when to call it (verified: langchain_core.tools.tool infers name/description
from the function name + docstring by default). Write it for the model's
audience, not a human reader's.
"""

from __future__ import annotations

from langchain_core.tools import tool

from loop.research.search import web_search


@tool
def search_web(query: str) -> str:
    """Search the web for current information about a company: its interview
    process, tech stack, recent news, culture, or engineering blog posts.

    Use this to ground interview prep in real, up-to-date company facts
    instead of guessing from training data.
    """
    results = web_search(query, k=5)
    if not results:
        return f"No results found for query: {query!r}"

    return "\n\n".join(
        f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['snippet']}" for r in results
    )
