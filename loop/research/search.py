"""
Swappable web search provider — the same factory-seam pattern as
loop/models.py (chat model) and loop/embeddings.py (embeddings).

Every caller calls web_search().  No caller imports DDGS or a Tavily client
directly, so swapping providers is a one-file change.

v1 provider: `ddgs` (the actively-maintained DuckDuckGo search client —
`duckduckgo-search` was renamed to `ddgs`).  Keyless — no signup required.
This is the ONE place in the whole project where v1 reaches real external
data over the network (everything else in v1 is fixtures + Bedrock).

v2 seam: Tavily, selected automatically when TAVILY_API_KEY is set.

Verified against ddgs==9.14.4:
    DDGS().text(query, max_results=k) -> list[{"title", "href", "body"}]

Offline test gate: tests stub web_search() (or the underlying DDGS call)
entirely — this module is a *server* activity, like live Bedrock calls or
live embeddings, never exercised over the network in tests/.
"""

from __future__ import annotations

from loop.config import settings


def web_search(query: str, k: int = 5) -> list[dict]:
    """Search the web and return up to k results.

    Args:
        query: The search query text.
        k:     Maximum number of results to return.

    Returns:
        List of dicts: {"title": str, "url": str, "snippet": str}.
    """
    if settings.tavily_api_key:
        return _tavily_search(query, k)
    return _duckduckgo_search(query, k)


def _duckduckgo_search(query: str, k: int) -> list[dict]:
    """Keyless search via the `ddgs` client (v1 default)."""
    from ddgs import DDGS

    raw_results = DDGS().text(query, max_results=k)
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", ""),
        }
        for r in raw_results
    ]


def _tavily_search(query: str, k: int) -> list[dict]:
    """Tavily-backed search — v2 seam, not implemented yet.

    Deliberately raises, mirroring the Ollama seam in loop/models.py: the
    branch exists so switching providers later is a one-file change, but
    building it out is out of scope for v1.
    """
    raise NotImplementedError(
        "Tavily search is a v2 feature. Unset TAVILY_API_KEY to use the "
        "keyless DuckDuckGo provider for now."
    )
