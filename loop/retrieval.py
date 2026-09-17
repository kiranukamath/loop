"""
Semantic question retrieval, upgraded in Phase 14 into a hybrid + reranked +
corrective pipeline, still behind the unchanged retrieve_questions() signature.

Pipeline (retrieve_questions):
  1. _build_index() reads fixtures/questions.json, wraps each question as a
     LangChain Document (rich text = title + prompt + topic; metadata = the
     filterable fields + the full original question dict), embeds every
     document with get_embeddings() into InMemoryVectorStore, and builds a
     parallel BM25Okapi index over the same documents' page_content.
  2. Dense ranking: cosine similarity search (optionally modality-filtered).
  3. Lexical ranking (Phase 14a, settings.hybrid_enabled): BM25 keyword
     scoring over the same modality-filtered pool -- catches exact-term
     matches (e.g. "LRU cache") that a bi-encoder embedding can blur into
     nearby-but-wrong neighbours.
  4. Reciprocal Rank Fusion merges the two rankings into one, without needing
     the two scales (cosine similarity vs. BM25 score) to be comparable.
  5. Reranking (Phase 14a, settings.rerank_enabled): a cross-encoder-style
     reranker (get_reranker()) re-scores the fused shortlist against the raw
     query text and produces the final top-k order -- more accurate than
     either ranking alone, at the cost of an extra pass over a small pool.

Phase 14b adds rewrite_query() (query expansion before retrieval) and Phase
14c adds grade_retrieval() + crag_search() (a bounded corrective loop around
retrieve_questions()) -- both are used by loop/nodes/interviewers.py, not by
retrieve_questions() itself, so the core pipeline above stays simple.

The module-level indexes are built lazily on first access so that tests can
monkeypatch get_embeddings/get_reranker BEFORE the index is built, then call
_build_index() to rebuild with fakes.

v2 seam: swap InMemoryVectorStore -> PGVector in _build_index() only; the
         retrieve_questions() signature and callers stay unchanged.

Verified against langchain-core==1.4.7:
    InMemoryVectorStore(embedding=<Embeddings>)
    store.add_documents(docs)
    store.similarity_search(query, k, filter=<Callable[[Document], bool]>)
Verified against rank-bm25==0.2.2:
    BM25Okapi(list[list[str]]).get_scores(query_tokens) -> list[float],
    aligned index-for-index with the corpus passed to the constructor.
"""

from __future__ import annotations

import json
import pathlib

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from rank_bm25 import BM25Okapi

from loop.config import settings
from loop.embeddings import get_embeddings
from loop.models import get_chat_model
from loop.reranker import get_reranker
from loop.schemas import QueryRewrite, RetrievalGrade

_FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"
_RRF_K = 60  # standard Reciprocal Rank Fusion smoothing constant

# Module-level singletons. None until _build_index() runs for the first time.
_vector_store: InMemoryVectorStore | None = None
_documents: list[Document] | None = None
_bm25: BM25Okapi | None = None


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _build_index() -> None:
    """Build (or rebuild) the vector store + BM25 index from fixtures/questions.json.

    Called automatically on first retrieve_questions() call, and explicitly by
    tests after monkeypatching get_embeddings/get_reranker with fakes.
    """
    global _vector_store, _documents, _bm25

    raw = json.loads((_FIXTURES / "questions.json").read_text())
    questions: list[dict] = raw["questions"]

    # Each question becomes one Document.  The page_content is the rich text
    # that gets embedded (and BM25-indexed) -- concatenating title + prompt +
    # topic maximises the signal for both dense and lexical search.  Metadata
    # carries the filterable fields plus "_source", the full original dict,
    # so retrieve_questions() never needs a second fixture lookup.
    docs = [
        Document(
            page_content=f"{q['title']}. {q['prompt']} Topic: {q['topic']}",
            metadata={
                "id": q["id"],
                "modality": q["modality"],
                "topic": q["topic"],
                "difficulty": q["difficulty"],
                "_source": q,
            },
        )
        for q in questions
    ]

    store = InMemoryVectorStore(embedding=get_embeddings())
    store.add_documents(docs)

    _vector_store = store
    _documents = docs
    _bm25 = BM25Okapi([_tokenize(d.page_content) for d in docs])


def _get_store() -> InMemoryVectorStore:
    """Return the singleton store, building it on first access."""
    if _vector_store is None:
        _build_index()
    assert _vector_store is not None
    return _vector_store


def _get_documents() -> list[Document]:
    if _documents is None:
        _build_index()
    assert _documents is not None
    return _documents


def _modality_filter(modality: str | None):
    if modality is None:
        return None
    return lambda doc: doc.metadata.get("modality") == modality  # noqa: E731


def _dense_rank_ids(query: str, modality: str | None, pool_size: int) -> list[str]:
    """Cosine-similarity ranking, highest first, as a list of question ids."""
    store = _get_store()
    docs = store.similarity_search(query, k=pool_size, filter=_modality_filter(modality))
    return [doc.metadata["id"] for doc in docs]


def _bm25_rank_ids(query: str, modality: str | None) -> list[str]:
    """BM25 keyword ranking, highest first, as a list of question ids.

    Scores are computed against the FULL corpus (correct BM25 idf statistics),
    then filtered to the requested modality before ranking.
    """
    bm25 = _bm25
    docs = _get_documents()
    if bm25 is None:
        _build_index()
        bm25 = _bm25
    assert bm25 is not None

    scores = bm25.get_scores(_tokenize(query))
    scored = [
        (doc.metadata["id"], score)
        for doc, score in zip(docs, scores, strict=True)
        if modality is None or doc.metadata["modality"] == modality
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [doc_id for doc_id, _ in scored]


def _reciprocal_rank_fusion(rankings: list[list[str]], k: int = _RRF_K) -> list[str]:
    """Merge several ranked id lists into one, via RRF: score(d) = sum 1/(k + rank).

    RRF needs no score normalisation between rankers -- it only uses rank
    position -- which is exactly why it fuses cosine similarity (0..1) and
    BM25 (unbounded) scores without one dominating the other.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)


def retrieve_questions(
    query: str,
    modality: str | None = None,
    k: int = 3,
) -> list[dict]:
    """Return up to k questions closest to the query.

    Args:
        query:    Free-text description of the topic or skill to find questions
                  for.  E.g. "distributed systems consistency and consensus".
        modality: Optional hard filter -- "coding", "system_design", or
                  "behavioral".  When None, all modalities are searched.
        k:        Maximum number of results to return.

    Returns:
        List of original question dicts (id, modality, topic, difficulty,
        title, prompt), ranked highest-relevance first.  May return fewer
        than k items if the filtered pool is smaller than k.

    Pipeline (Phase 14a): dense + (if hybrid_enabled) BM25, fused via RRF,
    then (if rerank_enabled) reranked against the raw query text.  With both
    flags off this is exactly the Phase 8 cosine-only top-k behaviour.
    """
    docs = _get_documents()
    filtered = [d for d in docs if modality is None or d.metadata["modality"] == modality]
    if not filtered:
        return []
    by_id = {d.metadata["id"]: d for d in filtered}

    dense_ids = _dense_rank_ids(query, modality, pool_size=len(filtered))

    if settings.hybrid_enabled:
        bm25_ids = _bm25_rank_ids(query, modality)
        fused_ids = _reciprocal_rank_fusion([dense_ids, bm25_ids])
    else:
        fused_ids = dense_ids

    if settings.rerank_enabled and fused_ids:
        pool_ids = fused_ids[: max(k * 4, k)]
        pool_docs = [by_id[doc_id] for doc_id in pool_ids]
        reranked = get_reranker().rerank(query, [doc.page_content for doc in pool_docs], top_n=k)
        final_ids = [pool_docs[r["index"]].metadata["id"] for r in reranked]
    else:
        final_ids = fused_ids[:k]

    return [by_id[doc_id].metadata["_source"] for doc_id in final_ids]


# ── Phase 14b: query rewriting / HyDE ─────────────────────────────────────────

_MULTIQUERY_SYSTEM = """You expand one interview-prep search need into several
distinct search queries that together cover it better than the original alone.
Vary phrasing and vocabulary; do not just restate the input."""

_MULTIQUERY_HUMAN = """Focus: {focus}
Topics: {topics}

Produce 2-3 alternate search queries for finding a matching interview question."""

_HYDE_SYSTEM = """You write a short, hypothetical interview-question description
that would perfectly match the given focus and topics. This hypothetical text is
used purely as a search query -- it embeds like a real question so retrieval can
find real questions that are semantically close to it."""

_HYDE_HUMAN = """Focus: {focus}
Topics: {topics}

Write one hypothetical interview-question description (2-3 sentences)."""

_multiquery_prompt = ChatPromptTemplate.from_messages(
    [("system", _MULTIQUERY_SYSTEM), ("human", _MULTIQUERY_HUMAN)]
)
_hyde_prompt = ChatPromptTemplate.from_messages([("system", _HYDE_SYSTEM), ("human", _HYDE_HUMAN)])


def rewrite_query(focus: str, topics: list[str]) -> list[str]:
    """Expand a session's focus/topics into one or more retrieval queries.

    settings.query_rewrite_mode controls the strategy:
      - "off" (default): returns exactly [base_query] -- byte-identical to the
        pre-Phase-14 single-query behaviour.
      - "multiquery": asks the model for a few alternate phrasings, in
        addition to the base query.
      - "hyde": asks the model to write a hypothetical matching question and
        uses that as the (single) query instead of the raw focus/topics.

    The base_query format matches loop/nodes/interviewers.py's pre-Phase-14
    query construction exactly, so "off" mode changes nothing.
    """
    base_query = f"{focus}. Topics: {', '.join(topics)}" if topics else focus
    mode = settings.query_rewrite_mode

    if mode == "off":
        return [base_query]

    topics_str = ", ".join(topics) if topics else "(none given)"
    model = get_chat_model()

    if mode == "multiquery":
        chain = _multiquery_prompt | model.with_structured_output(QueryRewrite)
        result: QueryRewrite = chain.invoke({"focus": focus, "topics": topics_str})
        return [base_query, *result.queries]

    if mode == "hyde":
        chain = _hyde_prompt | model.with_structured_output(QueryRewrite)
        # HyDE conventionally asks for one hypothetical document; QueryRewrite's
        # `queries` list is reused here so both modes share one schema.
        result = chain.invoke({"focus": focus, "topics": topics_str})
        return result.queries[:1] or [base_query]

    raise ValueError(f"Unknown query_rewrite_mode: {mode!r}")


# ── Phase 14c: Corrective RAG (CRAG) ──────────────────────────────────────────

_GRADE_RETRIEVAL_SYSTEM = """You judge how well a candidate interview question
matches an interview session's intended focus. Score 1.0 if it's a strong match,
0.0 if it's unrelated, and anything in between for a partial match."""

_GRADE_RETRIEVAL_HUMAN = """Session focus: {focus}

Retrieved question:
Title: {title}
Topic: {topic}
Prompt: {prompt}

How relevant is this question to the session focus?"""

_grade_retrieval_prompt = ChatPromptTemplate.from_messages(
    [("system", _GRADE_RETRIEVAL_SYSTEM), ("human", _GRADE_RETRIEVAL_HUMAN)]
)


def grade_retrieval(question: dict, focus: str) -> float:
    """Score (0.0-1.0) how well a retrieved question matches the session focus."""
    model = get_chat_model()
    chain = _grade_retrieval_prompt | model.with_structured_output(RetrievalGrade)
    result: RetrievalGrade = chain.invoke(
        {
            "focus": focus,
            "title": question["title"],
            "topic": question["topic"],
            "prompt": question["prompt"],
        }
    )
    return result.relevance


def crag_search(query: str, modality: str | None, k: int, focus: str) -> list[dict]:
    """Corrective-RAG wrapper around retrieve_questions().

    When settings.crag_enabled is False, this is exactly retrieve_questions(query,
    modality, k) -- a no-op wrapper, so CRAG is off by default with zero behaviour
    change. When enabled:
      1. Retrieve normally.
      2. Grade the top result against `focus`. If it scores at or above
         crag_min_relevance, return the results as-is.
      3. Otherwise, re-retrieve with a broadened query, up to
         settings.crag_max_retries times total.
      4. If relevance is still low after every retry, fall back to search_web
         for grounding context and fold it into one final retrieval query.

    Bounded: at most crag_max_retries re-retrievals plus one web-search fallback
    pass, regardless of what the grader returns -- this can never loop forever.
    """
    candidates = retrieve_questions(query, modality=modality, k=k)
    if not settings.crag_enabled or not candidates:
        return candidates

    attempt_query = query
    for _ in range(settings.crag_max_retries):
        if grade_retrieval(candidates[0], focus) >= settings.crag_min_relevance:
            return candidates
        attempt_query = f"{attempt_query} {focus}"
        candidates = retrieve_questions(attempt_query, modality=modality, k=k)
        if not candidates:
            return candidates

    if grade_retrieval(candidates[0], focus) >= settings.crag_min_relevance:
        return candidates

    from loop.research.tools import search_web

    web_context = search_web.invoke(focus)
    fallback_query = f"{query} {web_context}"[:500]
    return retrieve_questions(fallback_query, modality=modality, k=k) or candidates
