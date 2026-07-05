"""
Semantic question retrieval using an InMemoryVectorStore.

How it works:
  1. _build_index() reads fixtures/questions.json, wraps each question as a
     LangChain Document (rich text = title + prompt + topic; metadata = the
     filterable fields + the full original question dict).
  2. It embeds every document using get_embeddings() and stores the resulting
     (vector, text, metadata) triples in InMemoryVectorStore.
  3. retrieve_questions() embeds the caller's query, runs cosine similarity
     against all stored vectors, applies an optional modality filter, and
     returns the top-k original question dicts.

The module-level _vector_store is built lazily on first access so that tests
can monkeypatch get_embeddings BEFORE the index is built, then call
_build_index() to rebuild with the fake embeddings.

v2 seam: swap InMemoryVectorStore → PGVector in _build_index() only; the
         retrieve_questions() signature and callers stay unchanged.

Verified against langchain-core==1.4.7:
    InMemoryVectorStore(embedding=<Embeddings>)
    store.add_documents(docs)
    store.similarity_search(query, k, filter=<Callable[[Document], bool]>)
"""

from __future__ import annotations

import json
import pathlib

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from loop.embeddings import get_embeddings

_FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"

# Module-level singleton.  None until _build_index() runs for the first time.
_vector_store: InMemoryVectorStore | None = None


def _build_index() -> None:
    """Build (or rebuild) the in-memory vector store from fixtures/questions.json.

    Called automatically on first retrieve_questions() call, and explicitly by
    tests after monkeypatching get_embeddings with a fake implementation.
    """
    global _vector_store

    raw = json.loads((_FIXTURES / "questions.json").read_text())
    questions: list[dict] = raw["questions"]

    # Each question becomes one Document.  The page_content is the rich text
    # that gets embedded — concatenating title + prompt + topic maximises the
    # semantic signal.  The metadata carries the filterable fields so that
    # similarity_search() can apply a modality filter without touching
    # page_content, plus "_source" to return the full original dict cheaply
    # without a second fixture lookup.
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


def _get_store() -> InMemoryVectorStore:
    """Return the singleton store, building it on first access."""
    if _vector_store is None:
        _build_index()
    # _build_index guarantees _vector_store is not None after it runs
    assert _vector_store is not None
    return _vector_store


def retrieve_questions(
    query: str,
    modality: str | None = None,
    k: int = 3,
) -> list[dict]:
    """Return up to k questions semantically closest to the query.

    Args:
        query:    Free-text description of the topic or skill to find questions
                  for.  E.g. "distributed systems consistency and consensus".
        modality: Optional hard filter — "coding", "system_design", or
                  "behavioral".  When None, all modalities are searched.
        k:        Maximum number of results to return.

    Returns:
        List of original question dicts (id, modality, topic, difficulty,
        title, prompt), ranked by cosine similarity, highest first.
        May return fewer than k items if the filtered pool is smaller than k.
    """
    store = _get_store()

    # InMemoryVectorStore.similarity_search accepts filter as a callable
    # Callable[[Document], bool] — NOT a dict.  Verified against
    # langchain-core==1.4.7 source (similarity_search_with_score_by_vector).
    doc_filter = None
    if modality is not None:
        doc_filter = lambda doc: doc.metadata.get("modality") == modality  # noqa: E731

    docs = store.similarity_search(query, k=k, filter=doc_filter)

    # _source is the full original question dict — no second fixture read needed.
    return [doc.metadata["_source"] for doc in docs]
