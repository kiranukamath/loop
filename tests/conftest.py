"""
Shared pytest fixtures.

stub_planner: an autouse fixture for test_graph.py that replaces the planner
node's model call with a fast RunnableLambda.  This keeps graph-mechanics
tests (state shape, intake, reducer) offline and under 1 second.

Tests that explicitly want to test the planner (test_planner.py) patch
get_chat_model themselves and are not affected by this fixture.
"""

import pytest
from langchain_core.runnables import RunnableLambda

from loop.schemas import PrepPlan, Session

_STUB_PLAN = PrepPlan(
    role_summary="stub",
    total_sessions=1,
    sessions=[Session(session_number=1, modality="coding", topics=["x"], focus="y")],
    key_gaps=[],
    rationale="stub",
)


@pytest.fixture(autouse=True)
def stub_planner(request, monkeypatch):
    """Auto-stub the planner model for all tests in test_graph.py.

    Skipped for test_planner.py — those tests set up their own stubs.
    """
    if "test_graph" not in request.fspath.basename:
        return  # let test_planner.py manage its own patching

    from unittest.mock import MagicMock

    fake_model = MagicMock()
    fake_model.with_structured_output.return_value = RunnableLambda(lambda _: _STUB_PLAN)
    monkeypatch.setattr("loop.nodes.planner.get_chat_model", lambda: fake_model)
