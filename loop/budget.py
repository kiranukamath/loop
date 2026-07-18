"""
Token / cost budget accountant — Phase 10c.

Every Bedrock Converse response carries a `usage_metadata` block on the
returned AIMessage: {"input_tokens": int, "output_tokens": int,
"total_tokens": int}. This module accumulates that usage across a session
and enforces a hard ceiling — the same shape as a rate limiter guarding an
expensive downstream call.

Design note: this hooks in as a LangChain callback (`on_llm_end`), NOT by
changing what any node returns. Every node already builds its `config` dict
and passes it to `chain.invoke(..., config=config)` — LangGraph propagates a
top-level `config["callbacks"]` list down into every nested model call
automatically, so attaching a BudgetCallbackHandler at the graph-invoke level
(see loop/api.py) tracks usage across ALL nodes without touching grader.py,
coach.py, planner.py, or readiness.py at all.

Known simplification: the callback prices every call against a single
`model_id` (the primary model), since a callback has no reliable way to know
which model actually served a given call (relevant only on the rare occasion
a Phase 10a fallback fires). Good enough for a v1 cost *estimate*, not
billing-accurate.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from loop.config import MODEL_PRICES_PER_1K, settings


class BudgetExceeded(Exception):
    """Raised when a session's accumulated tokens exceed max_session_tokens."""


def price_usage(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Return the USD cost of one model call, using MODEL_PRICES_PER_1K.

    Unknown model ids price at $0 rather than raising — a missing price
    entry should not crash a graph run; it just means cost_usd under-reports.
    """
    input_price, output_price = MODEL_PRICES_PER_1K.get(model_id, (0.0, 0.0))
    return (input_tokens / 1000) * input_price + (output_tokens / 1000) * output_price


class SessionBudget:
    """Running token/cost total for one session (one thread_id).

    Plain, callback-independent accumulator — easy to unit test on its own,
    separate from the LangChain glue in BudgetCallbackHandler below.
    """

    def __init__(self) -> None:
        self.tokens_used = 0
        self.cost_usd = 0.0

    def add(self, model_id: str, input_tokens: int, output_tokens: int) -> None:
        """Fold one model call's usage into the running total.

        Raises BudgetExceeded if the new total breaches max_session_tokens
        (0 = unlimited, never raises).
        """
        self.tokens_used += input_tokens + output_tokens
        self.cost_usd += price_usage(model_id, input_tokens, output_tokens)

        if settings.max_session_tokens and self.tokens_used > settings.max_session_tokens:
            raise BudgetExceeded(
                f"Session token budget exceeded: {self.tokens_used} > {settings.max_session_tokens}"
            )


class BudgetCallbackHandler(BaseCallbackHandler):
    """LangChain callback that feeds every model response into a SessionBudget.

    Attach via config={"callbacks": [BudgetCallbackHandler(budget, model_id)]}
    at the top-level graph.stream()/invoke() call — see loop/api.py.
    """

    def __init__(self, budget: SessionBudget, model_id: str | None = None) -> None:
        self.budget = budget
        self.model_id = model_id or settings.bedrock_model_id

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        for generation_list in response.generations:
            for generation in generation_list:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None) if message else None
                if not usage:
                    continue
                self.budget.add(
                    self.model_id,
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                )
