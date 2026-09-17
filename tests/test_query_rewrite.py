"""
Offline tests for loop/retrieval.py::rewrite_query (Phase 14b).

All model calls are stubbed with RunnableLambda -- pure functions of the
prompt input -- so nothing here touches Bedrock.
"""

from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableLambda

from loop.retrieval import rewrite_query
from loop.schemas import QueryRewrite


def test_off_mode_returns_single_base_query(monkeypatch):
    monkeypatch.setattr("loop.retrieval.settings.query_rewrite_mode", "off")
    result = rewrite_query("distributed systems", ["consensus", "replication"])
    assert result == ["distributed systems. Topics: consensus, replication"]


def test_off_mode_with_no_topics_returns_bare_focus(monkeypatch):
    monkeypatch.setattr("loop.retrieval.settings.query_rewrite_mode", "off")
    assert rewrite_query("behavioral", []) == ["behavioral"]


def test_multiquery_mode_expands_one_to_many(monkeypatch):
    monkeypatch.setattr("loop.retrieval.settings.query_rewrite_mode", "multiquery")

    fake_model = RunnableLambda(
        lambda _: QueryRewrite(queries=["alt phrasing one", "alt phrasing two"])
    )
    monkeypatch.setattr(
        "loop.retrieval.get_chat_model",
        lambda: type("M", (), {"with_structured_output": lambda self, schema: fake_model})(),
    )

    result = rewrite_query("distributed systems", ["consensus"])
    assert result[0] == "distributed systems. Topics: consensus"
    assert "alt phrasing one" in result
    assert "alt phrasing two" in result
    assert len(result) == 3


def test_hyde_mode_returns_single_hypothetical_query(monkeypatch):
    monkeypatch.setattr("loop.retrieval.settings.query_rewrite_mode", "hyde")

    fake_model = RunnableLambda(
        lambda _: QueryRewrite(queries=["a hypothetical matching question about consensus"])
    )
    monkeypatch.setattr(
        "loop.retrieval.get_chat_model",
        lambda: type("M", (), {"with_structured_output": lambda self, schema: fake_model})(),
    )

    result = rewrite_query("distributed systems", ["consensus"])
    assert result == ["a hypothetical matching question about consensus"]


def test_hyde_mode_falls_back_to_base_query_on_empty_response(monkeypatch):
    monkeypatch.setattr("loop.retrieval.settings.query_rewrite_mode", "hyde")

    fake_model = RunnableLambda(lambda _: QueryRewrite(queries=[]))
    monkeypatch.setattr(
        "loop.retrieval.get_chat_model",
        lambda: type("M", (), {"with_structured_output": lambda self, schema: fake_model})(),
    )

    result = rewrite_query("behavioral", [])
    assert result == ["behavioral"]


def test_unknown_mode_raises_value_error(monkeypatch):
    monkeypatch.setattr("loop.retrieval.settings.query_rewrite_mode", "not-a-real-mode")
    with pytest.raises(ValueError, match="Unknown query_rewrite_mode"):
        rewrite_query("behavioral", [])
