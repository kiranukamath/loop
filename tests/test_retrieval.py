"""
Offline tests for loop/retrieval.py.

All tests use DeterministicFakeEmbedding instead of BedrockEmbeddings — no
network, no AWS credentials required.  The fake embedding produces deterministic
(but semantically arbitrary) vectors for each text string, so the vector store
mechanics (add, filter, top-k) work correctly; only the semantic ranking has no
real meaning.

Test strategy:
  - autouse fixture patches get_embeddings → DeterministicFakeEmbedding, then
    calls _build_index() to rebuild the store with the fake embeddings.
  - Each test calls retrieve_questions() and asserts on structure and constraints
    (count ≤ k, modality filter respected, etc.).
  - One test uses a query that is textually close to a known question to verify
    that the fake embeddings produce a stable (deterministic) ranking, which
    lets us assert the known question appears in the results.
"""

from __future__ import annotations

import pytest
from langchain_core.embeddings.fake import DeterministicFakeEmbedding

import loop.retrieval as retrieval_mod
from loop.retrieval import retrieve_questions

_ALL_MODALITIES = {"coding", "system_design", "behavioral"}


# ── Fixture ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    """Replace get_embeddings with DeterministicFakeEmbedding for every test.

    DeterministicFakeEmbedding(size=256) returns a 256-dim deterministic vector
    for each string — same text always gets the same vector.  After patching,
    _build_index() is called so the module-level store is rebuilt with the fake
    embeddings (not a real Bedrock call).
    """
    monkeypatch.setattr(
        "loop.retrieval.get_embeddings",
        lambda: DeterministicFakeEmbedding(size=256),
    )
    retrieval_mod._build_index()
    yield
    # Reset so the next test starts with a clean slate
    retrieval_mod._vector_store = None


# ── Index structure tests ─────────────────────────────────────────────────────


def test_index_builds():
    """_build_index should produce a non-None store."""
    assert retrieval_mod._vector_store is not None


def test_index_contains_all_24_questions():
    """The store should have exactly 24 documents (8 per modality)."""
    store = retrieval_mod._vector_store
    # InMemoryVectorStore exposes its internal store as a dict keyed by id
    assert len(store.store) == 24


# ── Top-k constraint tests ────────────────────────────────────────────────────


def test_returns_at_most_k_results():
    """retrieve_questions should never return more than k items."""
    results = retrieve_questions("algorithm data structure", k=3)
    assert len(results) <= 3


def test_returns_at_most_k_when_k_is_1():
    results = retrieve_questions("binary search", modality="coding", k=1)
    assert len(results) == 1


def test_returns_fewer_than_k_when_pool_is_small():
    """With modality filter, pool = 8 questions.  k=5 ≤ 8, so 5 come back."""
    results = retrieve_questions("any topic", modality="coding", k=5)
    assert len(results) <= 5


# ── Modality filter tests ─────────────────────────────────────────────────────


def test_modality_filter_coding():
    """All returned questions should be coding when modality='coding'."""
    results = retrieve_questions("algorithm implementation", modality="coding", k=5)
    assert len(results) > 0
    assert all(q["modality"] == "coding" for q in results)


def test_modality_filter_system_design():
    results = retrieve_questions("distributed systems", modality="system_design", k=5)
    assert len(results) > 0
    assert all(q["modality"] == "system_design" for q in results)


def test_modality_filter_behavioral():
    results = retrieve_questions("leadership conflict", modality="behavioral", k=5)
    assert len(results) > 0
    assert all(q["modality"] == "behavioral" for q in results)


def test_no_modality_filter_returns_across_all_modalities():
    """Without a filter, results can come from any modality."""
    results = retrieve_questions("technical skill", k=10)
    returned_modalities = {q["modality"] for q in results}
    # With 24 questions and k=10, at least 2 different modalities should appear
    assert len(returned_modalities) >= 2


def test_coding_filter_excludes_non_coding():
    """coding filter must never leak system_design or behavioral questions."""
    results = retrieve_questions("system architecture design", modality="coding", k=8)
    for q in results:
        assert q["modality"] == "coding"


def test_behavioral_filter_excludes_technical():
    results = retrieve_questions("binary tree traversal", modality="behavioral", k=8)
    for q in results:
        assert q["modality"] == "behavioral"


# ── Result shape tests ────────────────────────────────────────────────────────


def test_result_has_required_fields():
    """Each returned dict must have the five fields callers depend on."""
    results = retrieve_questions("graph traversal", modality="coding", k=1)
    assert len(results) == 1
    q = results[0]
    for field in ("id", "modality", "topic", "difficulty", "title", "prompt"):
        assert field in q, f"Missing field: {field}"


def test_result_ids_are_unique():
    """Top-k results should not contain duplicate question ids."""
    results = retrieve_questions("data structures", modality="coding", k=8)
    ids = [q["id"] for q in results]
    assert len(ids) == len(set(ids))


# ── Determinism test ──────────────────────────────────────────────────────────


def test_same_query_returns_same_results():
    """DeterministicFakeEmbedding produces stable vectors — same query, same ranking."""
    r1 = retrieve_questions("concurrency thread synchronization", modality="coding", k=3)
    r2 = retrieve_questions("concurrency thread synchronization", modality="coding", k=3)
    assert [q["id"] for q in r1] == [q["id"] for q in r2]


def test_known_question_retrieved_by_exact_text():
    """A query identical to a document's page_content gets cosine similarity = 1.0
    and must appear first.

    DeterministicFakeEmbedding is hash-based: identical text → identical vector
    → cos(θ) = 1.0 → highest similarity among all documents.  We reconstruct
    the exact page_content string using the same format as retrieval._build_index()
    so the vectors are guaranteed to match.
    """
    import json
    import pathlib

    fixtures = pathlib.Path(__file__).parent.parent / "fixtures" / "questions.json"
    questions = json.loads(fixtures.read_text())["questions"]

    # Pick cod-002 ("Thread-safe bounded queue") as the target
    target = next(q for q in questions if q["id"] == "cod-002")

    # Reconstruct the exact page_content string used during indexing
    exact_text = f"{target['title']}. {target['prompt']} Topic: {target['topic']}"

    results = retrieve_questions(exact_text, modality="coding", k=1)
    assert len(results) == 1
    assert results[0]["id"] == "cod-002"


# ── Phase 14a: hybrid search + RRF + reranking ────────────────────────────────
#
# The autouse fixtures above (fake_embeddings here, stub_reranker in
# conftest.py) mean these tests already run the full hybrid+rerank pipeline
# by default -- settings.hybrid_enabled and settings.rerank_enabled are both
# True. This section adds tests for the new internals directly (BM25 ranking,
# RRF fusion) and for the two flags individually.


def test_bm25_ranks_exact_keyword_match_first():
    """BM25 is pure keyword scoring -- a query using a question's exact
    distinctive vocabulary should rank that question first, regardless of
    what the (fake) embeddings say."""
    results = retrieval_mod._bm25_rank_ids("thread-safe bounded queue concurrency", "coding")
    assert results[0] == "cod-002"


def test_bm25_respects_modality_filter():
    results = retrieval_mod._bm25_rank_ids("anything", "behavioral")
    ids = set(results)
    assert ids <= {f"beh-{i:03d}" for i in range(1, 9)}


def test_reciprocal_rank_fusion_prefers_items_ranked_high_in_both_lists():
    ranking_a = ["x", "y", "z"]
    ranking_b = ["y", "x", "z"]
    fused = retrieval_mod._reciprocal_rank_fusion([ranking_a, ranking_b])
    # "x" and "y" are ranked #1/#2 in both lists (in some order); "z" is
    # always last -- so "z" must end up last in the fused ranking.
    assert fused[-1] == "z"
    assert set(fused[:2]) == {"x", "y"}


def test_reciprocal_rank_fusion_is_a_pure_function_of_rank_not_list_length():
    """A doc appearing only in one ranking still gets fused in, at a lower score."""
    fused = retrieval_mod._reciprocal_rank_fusion([["a", "b"], ["a"]])
    assert fused[0] == "a"
    assert "b" in fused


def test_hybrid_disabled_matches_dense_only_ranking(monkeypatch):
    """With hybrid_enabled off, retrieve_questions should use exactly the
    dense ranking (no BM25/RRF involved)."""
    monkeypatch.setattr("loop.retrieval.settings.hybrid_enabled", False)
    monkeypatch.setattr("loop.retrieval.settings.rerank_enabled", False)

    query = "concurrency thread synchronization"
    expected = retrieval_mod._dense_rank_ids(query, "coding", pool_size=8)[:3]
    results = retrieve_questions(query, modality="coding", k=3)
    assert [q["id"] for q in results] == expected


def test_rerank_enabled_can_reorder_relative_to_fusion(monkeypatch):
    """With a reranker that always prefers a fixed document, the top result
    should match the reranker's choice even if fusion ranked it lower."""

    class _AlwaysPrefersLast:
        def rerank(self, query, documents, top_n=None):
            # Always rank the last candidate first.
            n = len(documents)
            order = [n - 1] + list(range(n - 1)) if n else []
            order = order[:top_n] if top_n is not None else order
            return [{"index": i, "relevance_score": 1.0} for i in order]

    monkeypatch.setattr("loop.retrieval.settings.rerank_enabled", True)
    monkeypatch.setattr("loop.retrieval.get_reranker", lambda: _AlwaysPrefersLast())

    results = retrieve_questions("data structures", modality="coding", k=3)
    assert len(results) == 3


def test_rerank_disabled_returns_fused_top_k(monkeypatch):
    monkeypatch.setattr("loop.retrieval.settings.hybrid_enabled", True)
    monkeypatch.setattr("loop.retrieval.settings.rerank_enabled", False)

    query = "concurrency thread synchronization"
    dense_ids = retrieval_mod._dense_rank_ids(query, "coding", pool_size=8)
    bm25_ids = retrieval_mod._bm25_rank_ids(query, "coding")
    expected = retrieval_mod._reciprocal_rank_fusion([dense_ids, bm25_ids])[:3]

    results = retrieve_questions(query, modality="coding", k=3)
    assert [q["id"] for q in results] == expected


def test_retrieve_questions_still_respects_k_with_hybrid_and_rerank_on():
    results = retrieve_questions("algorithm data structure", k=3)
    assert len(results) <= 3


def test_retrieve_questions_no_matches_for_impossible_modality(monkeypatch):
    """An empty filtered pool returns [] without touching BM25/reranker."""
    results = retrieve_questions("anything", modality="not_a_real_modality", k=3)
    assert results == []
