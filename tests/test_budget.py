"""
Phase 10c tests — token/cost budget accountant.

All offline: no real model call. LLMResult/AIMessage objects are built by
hand (the same shapes langchain_core uses internally) to drive
BudgetCallbackHandler.on_llm_end() directly.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from loop.budget import BudgetCallbackHandler, BudgetExceeded, SessionBudget, price_usage
from loop.config import MODEL_PRICES_PER_1K, settings

_MODEL_ID = next(iter(MODEL_PRICES_PER_1K))  # any real, priced model id


def _llm_result(input_tokens: int, output_tokens: int) -> LLMResult:
    msg = AIMessage(
        content="hi",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


# ── price_usage ────────────────────────────────────────────────────────────────


class TestPriceUsage:
    def test_matches_price_table_math(self):
        in_price, out_price = MODEL_PRICES_PER_1K[_MODEL_ID]
        cost = price_usage(_MODEL_ID, 1000, 1000)
        assert cost == pytest.approx(in_price + out_price)

    def test_unknown_model_id_prices_at_zero(self):
        assert price_usage("no-such-model", 1000, 1000) == 0.0

    def test_zero_tokens_costs_zero(self):
        assert price_usage(_MODEL_ID, 0, 0) == 0.0


# ── SessionBudget ────────────────────────────────────────────────────────────


class TestSessionBudget:
    def test_accumulates_tokens_across_calls(self, monkeypatch):
        monkeypatch.setattr(settings, "max_session_tokens", 0)  # unlimited
        budget = SessionBudget()
        budget.add(_MODEL_ID, 100, 50)
        budget.add(_MODEL_ID, 200, 25)
        assert budget.tokens_used == 375

    def test_accumulates_cost_across_calls(self, monkeypatch):
        monkeypatch.setattr(settings, "max_session_tokens", 0)
        in_price, out_price = MODEL_PRICES_PER_1K[_MODEL_ID]
        budget = SessionBudget()
        budget.add(_MODEL_ID, 1000, 1000)
        budget.add(_MODEL_ID, 1000, 1000)
        assert budget.cost_usd == pytest.approx(2 * (in_price + out_price))

    def test_unlimited_when_max_session_tokens_is_zero(self, monkeypatch):
        monkeypatch.setattr(settings, "max_session_tokens", 0)
        budget = SessionBudget()
        budget.add(_MODEL_ID, 10_000_000, 0)  # should not raise
        assert budget.tokens_used == 10_000_000

    def test_raises_budget_exceeded_when_ceiling_breached(self, monkeypatch):
        monkeypatch.setattr(settings, "max_session_tokens", 100)
        budget = SessionBudget()
        with pytest.raises(BudgetExceeded):
            budget.add(_MODEL_ID, 80, 30)

    def test_does_not_raise_when_under_ceiling(self, monkeypatch):
        monkeypatch.setattr(settings, "max_session_tokens", 100)
        budget = SessionBudget()
        budget.add(_MODEL_ID, 40, 40)  # 80, under ceiling
        assert budget.tokens_used == 80


# ── BudgetCallbackHandler ──────────────────────────────────────────────────────


class TestBudgetCallbackHandler:
    def test_on_llm_end_feeds_usage_into_budget(self, monkeypatch):
        monkeypatch.setattr(settings, "max_session_tokens", 0)
        budget = SessionBudget()
        cb = BudgetCallbackHandler(budget, model_id=_MODEL_ID)

        cb.on_llm_end(_llm_result(100, 50), run_id=uuid4())

        assert budget.tokens_used == 150

    def test_multiple_generations_all_counted(self, monkeypatch):
        monkeypatch.setattr(settings, "max_session_tokens", 0)
        budget = SessionBudget()
        cb = BudgetCallbackHandler(budget, model_id=_MODEL_ID)

        result = LLMResult(
            generations=[
                [
                    ChatGeneration(
                        message=AIMessage(
                            content="a",
                            usage_metadata={
                                "input_tokens": 10,
                                "output_tokens": 5,
                                "total_tokens": 15,
                            },
                        )
                    )
                ],
                [
                    ChatGeneration(
                        message=AIMessage(
                            content="b",
                            usage_metadata={
                                "input_tokens": 20,
                                "output_tokens": 10,
                                "total_tokens": 30,
                            },
                        )
                    )
                ],
            ]
        )
        cb.on_llm_end(result, run_id=uuid4())
        assert budget.tokens_used == 45

    def test_missing_usage_metadata_is_ignored_not_errored(self, monkeypatch):
        monkeypatch.setattr(settings, "max_session_tokens", 0)
        budget = SessionBudget()
        cb = BudgetCallbackHandler(budget, model_id=_MODEL_ID)

        result = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="no usage"))]])
        cb.on_llm_end(result, run_id=uuid4())  # must not raise
        assert budget.tokens_used == 0

    def test_defaults_model_id_to_settings_bedrock_model_id(self):
        budget = SessionBudget()
        cb = BudgetCallbackHandler(budget)
        assert cb.model_id == settings.bedrock_model_id

    def test_raises_budget_exceeded_through_callback(self, monkeypatch):
        monkeypatch.setattr(settings, "max_session_tokens", 10)
        budget = SessionBudget()
        cb = BudgetCallbackHandler(budget, model_id=_MODEL_ID)

        with pytest.raises(BudgetExceeded):
            cb.on_llm_end(_llm_result(100, 50), run_id=uuid4())
