"""
Phase 7a tests — multi-session loop.

Verifies:
- session_router picks the right session based on session_index
- advance_session increments session_index
- _route_after_session returns 'continue' when more sessions remain, 'done' otherwise
- single-session plan: advance_session → readiness (no loop)
- two-session plan: graph loops through both sessions, then reaches readiness
- session_index out of range raises a clear error
"""

import pytest

from loop.graph import _route_after_session, advance_session, session_router
from loop.schemas import PrepPlan, Session

# ── Helpers ───────────────────────────────────────────────────────────────────


def _plan_dict(*modalities: str) -> dict:
    """Build a minimal PrepPlan dict with one session per modality."""
    sessions = [
        Session(session_number=i + 1, modality=m, topics=["x"], focus="y").model_dump()
        for i, m in enumerate(modalities)
    ]
    return PrepPlan(
        role_summary="test",
        total_sessions=len(sessions),
        sessions=sessions,
        key_gaps=[],
        rationale="test",
    ).model_dump()


# ── session_router ────────────────────────────────────────────────────────────


class TestSessionRouter:
    def test_picks_first_session_when_index_is_zero(self):
        state = {"plan": _plan_dict("coding", "system_design"), "session_index": 0}
        result = session_router(state)
        assert result["current_modality"] == "coding"
        assert result["session_number"] == 1

    def test_picks_second_session_when_index_is_one(self):
        state = {"plan": _plan_dict("coding", "system_design"), "session_index": 1}
        result = session_router(state)
        assert result["current_modality"] == "system_design"
        assert result["session_number"] == 2

    def test_defaults_to_zero_when_index_is_none(self):
        state = {"plan": _plan_dict("behavioral"), "session_index": None}
        result = session_router(state)
        assert result["current_modality"] == "behavioral"

    def test_raises_on_index_out_of_range(self):
        state = {"plan": _plan_dict("coding"), "session_index": 5}
        with pytest.raises(ValueError, match="out of range"):
            session_router(state)

    def test_raises_when_no_sessions(self):
        state = {"plan": {"sessions": []}, "session_index": 0}
        with pytest.raises(ValueError, match="no sessions"):
            session_router(state)


# ── advance_session ───────────────────────────────────────────────────────────


class TestAdvanceSession:
    def test_increments_from_zero(self):
        result = advance_session({"session_index": 0})
        assert result["session_index"] == 1

    def test_increments_from_one(self):
        result = advance_session({"session_index": 1})
        assert result["session_index"] == 2

    def test_treats_none_as_zero(self):
        result = advance_session({"session_index": None})
        assert result["session_index"] == 1

    def test_only_returns_session_index(self):
        # Node should only update what it owns — don't touch other state fields.
        result = advance_session({"session_index": 0, "plan": {"sessions": []}})
        assert list(result.keys()) == ["session_index"]


# ── _route_after_session ──────────────────────────────────────────────────────


class TestRouteAfterSession:
    def test_continue_when_more_sessions_remain(self):
        # Two sessions, index=1 → one remaining → continue
        state = {"plan": _plan_dict("coding", "system_design"), "session_index": 1}
        assert _route_after_session(state) == "continue"

    def test_done_when_all_sessions_complete(self):
        # Two sessions, index=2 → none remaining → done
        state = {"plan": _plan_dict("coding", "system_design"), "session_index": 2}
        assert _route_after_session(state) == "done"

    def test_done_for_single_session_after_advance(self):
        # One session, index=1 after advance → done
        state = {"plan": _plan_dict("coding"), "session_index": 1}
        assert _route_after_session(state) == "done"

    def test_continue_at_index_zero_of_two(self):
        state = {"plan": _plan_dict("coding", "behavioral"), "session_index": 0}
        assert _route_after_session(state) == "continue"


# ── Full graph multi-session loop ─────────────────────────────────────────────


class TestMultiSessionGraph:
    def _build_app(self, monkeypatch, num_sessions: int):
        """Compile graph with stubbed nodes and a plan of num_sessions sessions."""
        from unittest.mock import MagicMock

        from langchain_core.runnables import RunnableLambda
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.types import Command

        from loop.graph import build_graph
        from loop.schemas import Feedback, Grade

        stub_grade = Grade(
            question_id="cod-001",
            score=7,
            criterion_scores={
                "correctness": 3,
                "optimal_complexity": 2,
                "code_quality": 1,
                "communication": 1,
            },
            strengths=["ok"],
            improvements=[],
            overall_feedback="ok",
        )
        stub_feedback = Feedback(summary="ok", action_items=[], weak_areas_update=["loops"])

        # Build a plan with the requested number of sessions (all coding for simplicity)
        modalities = ["coding"] * num_sessions
        stub_plan_dict = _plan_dict(*modalities)

        # Stub planner to return our N-session plan
        fake_model = MagicMock()
        fake_model.with_structured_output.return_value = RunnableLambda(
            lambda _: PrepPlan(**stub_plan_dict)
        )
        monkeypatch.setattr("loop.nodes.planner.get_chat_model", lambda: fake_model)

        # plan_approval must return Command (no static edge)
        monkeypatch.setattr(
            "loop.graph.plan_approval",
            lambda state: Command(goto="session_router", update={"plan_approved": True}),
        )
        # Stub interviewers — bypass the answer-gate interrupt and set a canned question_id
        monkeypatch.setattr(
            "loop.graph.coding_interviewer",
            lambda state: {"current_question_id": "cod-001", "messages": []},
        )
        monkeypatch.setattr(
            "loop.graph.sd_interviewer",
            lambda state: {"current_question_id": "sys-001", "messages": []},
        )
        monkeypatch.setattr(
            "loop.graph.beh_interviewer",
            lambda state: {"current_question_id": "beh-001", "messages": []},
        )
        # Stub grader (bypasses answer-lookup)
        monkeypatch.setattr(
            "loop.graph.grader",
            lambda state: {"grades": [stub_grade.model_dump()]},
        )
        # Stub coach
        monkeypatch.setattr(
            "loop.graph.coach",
            lambda state: {"weak_areas": list(stub_feedback.weak_areas_update)},
        )
        # Stub readiness (bypasses interrupt)
        monkeypatch.setattr(
            "loop.graph.readiness",
            lambda state: {
                "readiness_verdict": {
                    "verdict": "ready",
                    "confidence": 0.9,
                    "strengths": ["ok"],
                    "gaps": [],
                    "recommendation": "go",
                },
                "verdict_approved": True,
            },
        )

        app = build_graph().compile(checkpointer=MemorySaver())
        return app

    def test_single_session_plan_reaches_readiness(self, monkeypatch):
        app = self._build_app(monkeypatch, num_sessions=1)
        from loop.state import initial_state

        result = app.invoke(
            initial_state(),
            config={"configurable": {"thread_id": "ms-single"}},
        )
        assert result.get("readiness_verdict") is not None
        assert result.get("verdict_approved") is True

    def test_single_session_final_index_is_one(self, monkeypatch):
        app = self._build_app(monkeypatch, num_sessions=1)
        from loop.state import initial_state

        result = app.invoke(
            initial_state(),
            config={"configurable": {"thread_id": "ms-single-idx"}},
        )
        # After the single session, advance_session sets index to 1
        assert result.get("session_index") == 1

    def test_two_session_plan_loops_back(self, monkeypatch):
        """Graph must visit session_router twice for a two-session plan."""
        from loop.state import initial_state

        visited = []

        # Wrap session_router to record visits without changing behaviour
        original_router = session_router

        def spy_router(state):
            visited.append(state.get("session_index", 0))
            return original_router(state)

        monkeypatch.setattr("loop.graph.session_router", spy_router)
        app = self._build_app(monkeypatch, num_sessions=2)

        app.invoke(
            initial_state(),
            config={"configurable": {"thread_id": "ms-two"}},
        )
        assert len(visited) == 2, f"Expected 2 session_router visits, got {len(visited)}"
        assert visited == [0, 1]

    def test_two_session_plan_reaches_readiness(self, monkeypatch):
        app = self._build_app(monkeypatch, num_sessions=2)
        from loop.state import initial_state

        result = app.invoke(
            initial_state(),
            config={"configurable": {"thread_id": "ms-two-done"}},
        )
        assert result.get("readiness_verdict") is not None
        assert result.get("session_index") == 2

    def test_two_session_plan_accumulates_grades(self, monkeypatch):
        """Each session's grader output should append to grades list."""
        app = self._build_app(monkeypatch, num_sessions=2)
        from loop.state import initial_state

        result = app.invoke(
            initial_state(),
            config={"configurable": {"thread_id": "ms-grades"}},
        )
        # Both sessions ran the stubbed grader → two grade entries
        assert len(result.get("grades") or []) == 2
