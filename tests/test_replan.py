"""
Phase 15b tests — bounded replanning.

Unit-level tests for _grade_divergence, replan(), and _route_after_advance
(all offline: planner's model is stubbed exactly like tests/test_planner.py).
Also one full-graph test proving the bound holds: sustained low grades never
replan more than settings.replan_max_times.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.runnables import RunnableLambda

from loop.schemas import PrepPlan, Session

# ── Helpers ───────────────────────────────────────────────────────────────────


def _plan_dict(*modalities: str) -> dict:
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


def _grade(score: int) -> dict:
    return {
        "question_id": "cod-001",
        "score": score,
        "criterion_scores": {"correctness": score},
        "strengths": [],
        "improvements": [],
        "overall_feedback": "x",
    }


@pytest.fixture(autouse=True)
def _replan_defaults(monkeypatch):
    """Every test controls replan settings explicitly."""
    from loop.config import settings

    monkeypatch.setattr(settings, "replan_enabled", False)
    monkeypatch.setattr(settings, "replan_score_threshold", 5)
    monkeypatch.setattr(settings, "replan_max_times", 1)
    yield


def _stub_planner_model(monkeypatch, new_plan: dict):
    """Stub loop.nodes.planner.get_chat_model so planner() returns new_plan."""
    plan_obj = PrepPlan(**new_plan)
    fake = MagicMock()
    fake.with_structured_output.return_value = RunnableLambda(lambda _: plan_obj)
    monkeypatch.setattr("loop.nodes.planner.get_chat_model", lambda: fake)


# ── _grade_divergence ─────────────────────────────────────────────────────────


class TestGradeDivergence:
    def test_no_divergence_when_no_grades(self):
        from loop.graph import _grade_divergence

        assert _grade_divergence({"grades": []}) is False
        assert _grade_divergence({}) is False

    def test_no_divergence_when_last_grade_at_or_above_threshold(self):
        from loop.config import settings
        from loop.graph import _grade_divergence

        settings.replan_score_threshold = 5
        assert _grade_divergence({"grades": [_grade(7), _grade(5)]}) is False

    def test_divergence_when_last_grade_below_threshold(self):
        from loop.config import settings
        from loop.graph import _grade_divergence

        settings.replan_score_threshold = 5
        assert _grade_divergence({"grades": [_grade(9), _grade(2)]}) is True


# ── replan() node ─────────────────────────────────────────────────────────────


class TestReplanNode:
    def test_keeps_completed_sessions_and_replaces_the_rest(self, monkeypatch):
        from loop.graph import replan

        original = _plan_dict("coding", "system_design", "behavioral")
        new_plan = _plan_dict("system_design", "behavioral")  # planner's fresh suggestion
        _stub_planner_model(monkeypatch, new_plan)

        state = {"plan": original, "session_index": 1, "jd": "jd", "profile": "profile"}
        result = replan(state)

        sessions = result["plan"]["sessions"]
        # session 1 (already completed) is untouched
        assert sessions[0]["modality"] == "coding"
        assert sessions[0]["session_number"] == 1
        # remaining slots (2 of them) are replaced from the fresh plan
        assert len(sessions) == 3
        assert sessions[1]["modality"] == "system_design"
        assert sessions[2]["modality"] == "behavioral"
        assert sessions[1]["session_number"] == 2
        assert sessions[2]["session_number"] == 3

    def test_caps_replacement_to_remaining_slots(self, monkeypatch):
        from loop.graph import replan

        original = _plan_dict("coding", "system_design")
        # planner suggests MORE sessions than the one slot left -- must be capped.
        new_plan = _plan_dict("system_design", "behavioral", "coding")
        _stub_planner_model(monkeypatch, new_plan)

        state = {"plan": original, "session_index": 1, "jd": "jd", "profile": "profile"}
        result = replan(state)

        sessions = result["plan"]["sessions"]
        assert len(sessions) == 2  # total_sessions never grows past the original
        assert result["plan"]["total_sessions"] == 2

    def test_increments_replan_count(self, monkeypatch):
        from loop.graph import replan

        _stub_planner_model(monkeypatch, _plan_dict("coding"))
        state = {
            "plan": _plan_dict("coding", "coding"),
            "session_index": 1,
            "jd": "jd",
            "profile": "profile",
            "replan_count": 1,
        }
        result = replan(state)
        assert result["replan_count"] == 2

    def test_replan_count_defaults_to_zero(self, monkeypatch):
        from loop.graph import replan

        _stub_planner_model(monkeypatch, _plan_dict("coding"))
        state = {
            "plan": _plan_dict("coding", "coding"),
            "session_index": 1,
            "jd": "jd",
            "profile": "profile",
        }
        result = replan(state)
        assert result["replan_count"] == 1


# ── _route_after_advance ──────────────────────────────────────────────────────


class TestRouteAfterAdvance:
    def test_disabled_by_default_defers_to_route_after_session(self):
        from loop.graph import _route_after_advance

        state = {
            "plan": _plan_dict("coding", "system_design"),
            "session_index": 1,
            "grades": [_grade(1)],  # would diverge if enabled
        }
        assert _route_after_advance(state) == "continue"

    def test_replans_when_enabled_and_diverging_and_more_sessions(self, monkeypatch):
        from loop.config import settings
        from loop.graph import _route_after_advance

        monkeypatch.setattr(settings, "replan_enabled", True)
        state = {
            "plan": _plan_dict("coding", "system_design"),
            "session_index": 1,
            "grades": [_grade(2)],
            "replan_count": 0,
        }
        assert _route_after_advance(state) == "replan"

    def test_no_replan_when_no_divergence(self, monkeypatch):
        from loop.config import settings
        from loop.graph import _route_after_advance

        monkeypatch.setattr(settings, "replan_enabled", True)
        state = {
            "plan": _plan_dict("coding", "system_design"),
            "session_index": 1,
            "grades": [_grade(9)],
            "replan_count": 0,
        }
        assert _route_after_advance(state) == "continue"

    def test_no_replan_when_bound_already_hit(self, monkeypatch):
        from loop.config import settings
        from loop.graph import _route_after_advance

        monkeypatch.setattr(settings, "replan_enabled", True)
        monkeypatch.setattr(settings, "replan_max_times", 1)
        state = {
            "plan": _plan_dict("coding", "system_design"),
            "session_index": 1,
            "grades": [_grade(2)],
            "replan_count": 1,  # bound already used up
        }
        assert _route_after_advance(state) == "continue"

    def test_no_replan_on_last_session_even_if_diverging(self, monkeypatch):
        from loop.config import settings
        from loop.graph import _route_after_advance

        monkeypatch.setattr(settings, "replan_enabled", True)
        state = {
            "plan": _plan_dict("coding", "system_design"),
            "session_index": 2,  # all sessions done -- nothing left to replan
            "grades": [_grade(1)],
            "replan_count": 0,
        }
        assert _route_after_advance(state) == "done"


# ── Full-graph bound test ─────────────────────────────────────────────────────


class TestReplanBoundHolds:
    def test_sustained_low_grades_replan_at_most_replan_max_times(self, monkeypatch):
        """Even with grades that always diverge, the graph must not replan
        more than settings.replan_max_times -- this is the "cannot loop
        forever" guarantee the spec requires.
        """
        from langchain_core.runnables import RunnableLambda
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.types import Command

        from loop.config import settings
        from loop.graph import build_graph
        from loop.schemas import Feedback, Grade, ReadinessVerdict
        from loop.state import initial_state

        monkeypatch.setattr(settings, "replan_enabled", True)
        monkeypatch.setattr(settings, "replan_score_threshold", 10)  # every grade "diverges"
        monkeypatch.setattr(settings, "replan_max_times", 2)
        # planner() hard-caps sessions to settings.max_sessions -- bump it so
        # the 3-session stub plan below isn't truncated before replan() runs.
        monkeypatch.setattr(settings, "max_sessions", 5)

        stub_plan = PrepPlan(
            role_summary="t",
            total_sessions=3,
            sessions=[
                Session(session_number=i + 1, modality="coding", topics=["x"], focus="y")
                for i in range(3)
            ],
            key_gaps=[],
            rationale="t",
        )
        stub_grade = Grade(
            question_id="cod-001",
            score=1,
            criterion_scores={"correctness": 1},
            strengths=[],
            improvements=[],
            overall_feedback="weak",
        )
        stub_feedback = Feedback(summary="ok", action_items=[], weak_areas_update=[])
        stub_verdict = ReadinessVerdict(
            verdict="not_ready",
            confidence=0.5,
            strengths=[],
            gaps=[],
            recommendation="keep practicing",
        )

        def fake_model(rv):
            m = MagicMock()
            m.with_structured_output.return_value = RunnableLambda(lambda _: rv)
            return m

        monkeypatch.setattr("loop.nodes.planner.get_chat_model", lambda: fake_model(stub_plan))
        monkeypatch.setattr(
            "loop.graph.coding_interviewer",
            lambda state: {"current_question_id": "cod-001", "messages": []},
        )
        monkeypatch.setattr(
            "loop.graph.grader", lambda state: {"grades": [stub_grade.model_dump()]}
        )
        monkeypatch.setattr("loop.nodes.coach.get_chat_model", lambda: fake_model(stub_feedback))
        monkeypatch.setattr("loop.nodes.readiness.get_chat_model", lambda: fake_model(stub_verdict))
        monkeypatch.setattr(
            "loop.graph.readiness",
            lambda state: {
                "readiness_verdict": stub_verdict.model_dump(),
                "verdict_approved": True,
            },
        )

        app = build_graph().compile(checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "replan-bound-test"}, "recursion_limit": 100}

        state = initial_state()
        state["answers"] = [{"question_id": "cod-001", "text": "brute force"}]

        result = app.invoke(state, config=cfg)
        # Approve the plan to enter the session loop.
        result = app.invoke(Command(resume={"decision": "approve"}), config=cfg)

        assert result.get("replan_count", 0) <= settings.replan_max_times
        # It DID replan at least once, proving the path is exercised, not just vacuously bounded.
        assert result.get("replan_count", 0) == settings.replan_max_times
