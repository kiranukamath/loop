"""
Offline tests for loop/retrieval.py::grade_retrieval + crag_search (Phase 14c).

grade_retrieval's model call is stubbed directly (monkeypatching the function
itself, since crag_search is what's under test); search_web is stubbed too so
no network call happens on the fallback path.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import loop.retrieval as retrieval_mod
from loop.retrieval import crag_search


def test_crag_disabled_is_a_pure_passthrough(monkeypatch):
    monkeypatch.setattr("loop.retrieval.settings.crag_enabled", False)
    grade_spy = MagicMock()
    monkeypatch.setattr("loop.retrieval.grade_retrieval", grade_spy)

    result = crag_search("binary search", "coding", 3, "algorithms")

    assert result == retrieval_mod.retrieve_questions("binary search", modality="coding", k=3)
    grade_spy.assert_not_called()


def test_crag_returns_immediately_on_high_relevance(monkeypatch):
    monkeypatch.setattr("loop.retrieval.settings.crag_enabled", True)
    monkeypatch.setattr("loop.retrieval.settings.crag_min_relevance", 0.5)
    monkeypatch.setattr("loop.retrieval.grade_retrieval", lambda question, focus: 0.9)

    retrieve_spy = MagicMock(wraps=retrieval_mod.retrieve_questions)
    monkeypatch.setattr("loop.retrieval.retrieve_questions", retrieve_spy)

    result = crag_search("binary search", "coding", 3, "algorithms")

    assert len(result) <= 3
    retrieve_spy.assert_called_once()


def test_crag_retries_bounded_times_then_falls_back_to_web_search(monkeypatch):
    monkeypatch.setattr("loop.retrieval.settings.crag_enabled", True)
    monkeypatch.setattr("loop.retrieval.settings.crag_min_relevance", 0.5)
    monkeypatch.setattr("loop.retrieval.settings.crag_max_retries", 2)

    # Always "irrelevant" -- forces every retry plus the final web-search fallback.
    grade_calls = []

    def _always_low(question, focus):
        grade_calls.append(question["id"])
        return 0.1

    monkeypatch.setattr("loop.retrieval.grade_retrieval", _always_low)

    retrieve_calls = []
    original_retrieve = retrieval_mod.retrieve_questions

    def _counting_retrieve(query, modality=None, k=3):
        retrieve_calls.append(query)
        return original_retrieve(query, modality=modality, k=k)

    monkeypatch.setattr("loop.retrieval.retrieve_questions", _counting_retrieve)

    fake_search_web = MagicMock()
    fake_search_web.invoke.return_value = "some grounding context from the web"
    monkeypatch.setattr("loop.research.tools.search_web", fake_search_web)

    result = crag_search("binary search", "coding", 3, "algorithms")

    # 1 initial retrieve + crag_max_retries (2) re-retrieves + 1 final fallback
    # retrieve = 4 total. Bounded regardless of grade_retrieval always failing.
    assert len(retrieve_calls) == 4
    fake_search_web.invoke.assert_called_once_with("algorithms")
    assert isinstance(result, list)


def test_crag_stops_retrying_early_if_retrieval_goes_empty(monkeypatch):
    monkeypatch.setattr("loop.retrieval.settings.crag_enabled", True)
    monkeypatch.setattr("loop.retrieval.settings.crag_min_relevance", 0.5)
    monkeypatch.setattr("loop.retrieval.settings.crag_max_retries", 3)
    monkeypatch.setattr("loop.retrieval.grade_retrieval", lambda question, focus: 0.1)

    calls = {"n": 0}
    original_retrieve = retrieval_mod.retrieve_questions

    def _empty_after_first(query, modality=None, k=3):
        calls["n"] += 1
        if calls["n"] == 1:
            return original_retrieve(query, modality=modality, k=k)
        return []

    monkeypatch.setattr("loop.retrieval.retrieve_questions", _empty_after_first)

    result = crag_search("binary search", "coding", 3, "algorithms")

    assert result == []
    assert calls["n"] == 2  # initial retrieve + one re-retrieve that came back empty
