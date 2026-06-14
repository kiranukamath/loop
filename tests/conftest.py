"""
Shared pytest fixtures.

stub_graph_nodes: autouse fixture for test_graph.py only.

Why we patch loop.graph.grader / loop.graph.coach (not the module functions):
  graph.py does `from loop.nodes.grader import grader` at import time.
  After that, `loop.graph.grader` is a local name pointing to the real function.
  compile_graph() → build_graph() uses these local names when it registers nodes.
  Patching the name in loop.graph's namespace means each _run_graph() call
  (which calls compile_graph() fresh) registers the stub instead of the real node.
  Patching only the model inside grader wouldn't help — the grader function itself
  would still try to look up answers and fail before ever calling the model.
"""

import pytest
from langchain_core.runnables import RunnableLambda

from loop.schemas import Feedback, Grade, PrepPlan, Session

# ── Stub plan ─────────────────────────────────────────────────────────────────

_STUB_PLAN = PrepPlan(
    role_summary="stub",
    total_sessions=1,
    sessions=[Session(session_number=1, modality="coding", topics=["x"], focus="y")],
    key_gaps=[],
    rationale="stub",
)

# ── Stub grade ────────────────────────────────────────────────────────────────

_STUB_GRADE = Grade(
    question_id="cod-001",
    score=7,
    criterion_scores={
        "correctness": 3,
        "optimal_complexity": 2,
        "code_quality": 1,
        "communication": 1,
    },
    strengths=["good sliding-window intuition"],
    improvements=["handle edge cases explicitly"],
    overall_feedback="Solid approach. Practice edge cases.",
)

# ── Stub feedback ─────────────────────────────────────────────────────────────

_STUB_FEEDBACK = Feedback(
    summary="Good session overall.",
    action_items=["practice sliding-window edge cases"],
    weak_areas_update=["sliding-window", "communication"],
)


@pytest.fixture(autouse=True)
def stub_graph_nodes(request, monkeypatch):
    """Auto-stub all model-calling nodes for tests in test_graph.py.

    Excluded for test_planner.py and test_phase3.py — those files manage their
    own stubs.
    """
    if "test_graph" not in request.fspath.basename:
        return

    from unittest.mock import MagicMock

    # ── Stub planner model ────────────────────────────────────────────────────
    # Planner calls get_chat_model() and uses with_structured_output.
    # We patch the model factory in the planner module's namespace.
    fake_planner = MagicMock()
    fake_planner.with_structured_output.return_value = RunnableLambda(lambda _: _STUB_PLAN)
    monkeypatch.setattr("loop.nodes.planner.get_chat_model", lambda: fake_planner)

    # ── Stub grader node function ─────────────────────────────────────────────
    # Grader needs answers in state before it can call the model.
    # For graph-mechanics tests we stub the whole node (not just the model) so
    # tests don't need to pre-inject canned answers into state.
    monkeypatch.setattr("loop.graph.grader", lambda state: {"grades": [_STUB_GRADE.model_dump()]})

    # ── Stub coach node function ──────────────────────────────────────────────
    monkeypatch.setattr(
        "loop.graph.coach",
        lambda state: {"weak_areas": list(_STUB_FEEDBACK.weak_areas_update)},
    )
