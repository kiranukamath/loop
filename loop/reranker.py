"""
Swappable reranker factory (Phase 14a).

Mirrors loop/embeddings.py exactly: every caller calls get_reranker() and never
imports BedrockRerank directly. A reranker takes a query and a shortlist of
already-retrieved documents and re-scores them with a more expensive, more
accurate model than the bi-encoder embeddings used for the initial retrieval
pass -- the classic "cheap recall, expensive precision" two-stage pattern.

Verified against langchain-aws==1.5.1 (source read directly -- BedrockRerank
is not re-exported from langchain_aws.document_compressors.__init__, so it is
imported from its submodule):
    from langchain_aws.document_compressors.rerank import BedrockRerank
    BedrockRerank(model_arn=..., region_name=...)
    reranker.rerank(documents: list[str | Document | dict], query: str, top_n=...)
        -> list[{"index": int, "relevance_score": float}], sorted highest first.

Live Bedrock rerank is unreachable from this sandbox -- this factory + the
FakeReranker below are the verified seam; a real Bedrock rerank call should be
sanity-checked on the server per CLAUDE.md rule #7.
"""

from __future__ import annotations

from typing import Protocol

from loop.config import settings


class Reranker(Protocol):
    """Anything with this shape can sit behind get_reranker()."""

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[dict]:
        """Return [{"index": int, "relevance_score": float}, ...], highest first."""
        ...


def get_reranker() -> Reranker:
    """Return the configured reranker.

    Currently supports provider="bedrock" only. The v2 Ollama path raises
    NotImplementedError as a deliberate seam, same as models.py/embeddings.py.
    """
    if settings.model_provider == "bedrock":
        return _make_bedrock_reranker()

    if settings.model_provider == "ollama":
        raise NotImplementedError(
            "Ollama reranking is a v2 feature. Set MODEL_PROVIDER=bedrock for now."
        )

    raise ValueError(f"Unknown model provider: {settings.model_provider!r}")


def _make_bedrock_reranker() -> Reranker:
    """Construct a BedrockRerank for the configured rerank model id.

    AWS credentials are read from the boto3 credential chain, same as
    ChatBedrockConverse and BedrockEmbeddings -- nothing to pass explicitly.
    """
    from langchain_aws.document_compressors.rerank import BedrockRerank

    model_arn = (
        f"arn:aws:bedrock:{settings.aws_region}::foundation-model/{settings.rerank_model_id}"
    )
    return BedrockRerank(model_arn=model_arn, region_name=settings.aws_region)


class FakeReranker:
    """Deterministic, offline reranker -- no network, no AWS credentials.

    Scores each document by lexical (word-overlap) similarity to the query.
    This is a crude stand-in for a cross-encoder, but it is deterministic and
    pure-Python, so tests get a stable, inspectable ranking instead of a live
    Bedrock call. Used in tests/ via conftest.py's autouse stub_reranker fixture.
    """

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[dict]:
        query_tokens = set(query.lower().split())
        scored = [
            (i, float(len(query_tokens & set(doc.lower().split()))))
            for i, doc in enumerate(documents)
        ]
        # Stable sort on original index breaks ties deterministically.
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        if top_n is not None:
            scored = scored[:top_n]
        return [{"index": i, "relevance_score": score} for i, score in scored]
