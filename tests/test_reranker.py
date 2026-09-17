"""
Offline tests for loop/reranker.py (Phase 14a).

FakeReranker is pure-Python word-overlap scoring -- fully deterministic, no
network. The Bedrock construction path is exercised too, but only up to
building the BedrockRerank object (no .rerank() call, which would hit the
network) -- construction itself reads no live AWS state.
"""

from __future__ import annotations

import pytest

from loop.reranker import FakeReranker, get_reranker


class TestFakeReranker:
    def test_orders_by_word_overlap_descending(self):
        reranker = FakeReranker()
        documents = [
            "a completely unrelated document about gardening",
            "binary search tree traversal in order",
            "binary search algorithm implementation",
        ]
        results = reranker.rerank("binary search algorithm", documents)
        assert results[0]["index"] == 2  # most word overlap with the query
        assert results[-1]["index"] == 0  # least overlap

    def test_relevance_scores_are_non_increasing(self):
        reranker = FakeReranker()
        documents = ["binary search", "search only", "nothing in common here"]
        results = reranker.rerank("binary search query", documents)
        scores = [r["relevance_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_respects_top_n(self):
        reranker = FakeReranker()
        documents = ["a", "b", "c", "d"]
        results = reranker.rerank("a b c d", documents, top_n=2)
        assert len(results) == 2

    def test_empty_documents_returns_empty(self):
        reranker = FakeReranker()
        assert reranker.rerank("anything", []) == []

    def test_deterministic_tie_break_by_original_index(self):
        """Two documents with identical overlap keep their original relative order."""
        reranker = FakeReranker()
        documents = ["binary search", "binary search"]
        results = reranker.rerank("binary search", documents)
        assert [r["index"] for r in results] == [0, 1]


class TestGetReranker:
    def test_bedrock_provider_returns_bedrock_reranker(self, monkeypatch):
        monkeypatch.setattr("loop.reranker.settings.model_provider", "bedrock")
        monkeypatch.setattr("loop.reranker.settings.rerank_model_id", "amazon.rerank-v1:0")
        monkeypatch.setattr("loop.reranker.settings.aws_region", "us-east-1")

        reranker = get_reranker()

        from langchain_aws.document_compressors.rerank import BedrockRerank

        assert isinstance(reranker, BedrockRerank)
        assert reranker.model_arn == (
            "arn:aws:bedrock:us-east-1::foundation-model/amazon.rerank-v1:0"
        )

    def test_ollama_provider_raises_not_implemented(self, monkeypatch):
        monkeypatch.setattr("loop.reranker.settings.model_provider", "ollama")
        with pytest.raises(NotImplementedError, match="Ollama"):
            get_reranker()

    def test_unknown_provider_raises_value_error(self, monkeypatch):
        monkeypatch.setattr("loop.reranker.settings.model_provider", "made-up-provider")
        with pytest.raises(ValueError, match="Unknown model provider"):
            get_reranker()
