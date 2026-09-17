"""
Phase 15a tests — Reflexion self-critique in loop/nodes/grader.py.

All offline: get_chat_model is stubbed everywhere a model would be called, so
no live Bedrock request happens. Verifies:
  - reflexion_enabled=False (the default) never calls the critique pass, and
    grader() returns exactly the first-pass grade.
  - reflexion_enabled=True runs a second pass; a stubbed critic that flips a
    wrong score changes the committed grade.
  - reflexion_enabled=True with a critic that returns the same grade leaves
    the grade unchanged.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.runnables import RunnableLambda

from loop.schemas import Grade

_QID = "cod-001"

_FIRST_PASS_GRADE = Grade(
    question_id=_QID,
    score=3,
    criterion_scores={
        "correctness": 1,
        "optimal_complexity": 1,
        "code_quality": 1,
        "communication": 0,
    },
    strengths=["identified the right data structure"],
    improvements=["missed the sliding window optimization"],
    overall_feedback="Weak: brute force approach.",
)

_REVISED_GRADE = Grade(
    question_id=_QID,
    score=8,
    criterion_scores={
        "correctness": 3,
        "optimal_complexity": 2,
        "code_quality": 2,
        "communication": 1,
    },
    strengths=["correct sliding window", "clear complexity analysis"],
    improvements=["could mention edge cases"],
    overall_feedback="Strong answer, mis-scored on the first pass.",
)


def _state():
    return {
        "current_question_id": _QID,
        "answers": [
            {
                "question_id": _QID,
                "text": "sliding window with a hash set, O(n) time",
            }
        ],
        "grades": [],
    }


@pytest.fixture(autouse=True)
def _reset_reflexion_flag(monkeypatch):
    """Every test controls reflexion_enabled explicitly -- don't inherit .env."""
    from loop.config import settings

    monkeypatch.setattr(settings, "reflexion_enabled", False)
    monkeypatch.setattr(settings, "fallback_model_id", "")
    yield


def _stub_two_call_model(monkeypatch, first_output, second_output):
    """Stub get_chat_model() so the grader's PRIMARY call returns first_output
    and any subsequent call (the critique pass) returns second_output.
    """
    calls = {"n": 0}

    def fake_get_chat_model(model_id=None):
        m = MagicMock()

        def structured(_schema):
            calls["n"] += 1
            output = first_output if calls["n"] == 1 else second_output
            return RunnableLambda(lambda _: output)

        m.with_structured_output.side_effect = structured
        return m

    monkeypatch.setattr("loop.nodes.grader.get_chat_model", fake_get_chat_model)
    return calls


class TestReflexionDisabledByDefault:
    def test_disabled_returns_first_pass_grade_unchanged(self, monkeypatch):
        from loop.config import settings
        from loop.nodes.grader import grader

        monkeypatch.setattr(settings, "reflexion_enabled", False)
        calls = _stub_two_call_model(monkeypatch, _FIRST_PASS_GRADE, _REVISED_GRADE)

        result = grader(_state())

        assert result["grades"][0]["score"] == 3
        assert calls["n"] == 1  # only the primary grading call ran, no critique pass


class TestReflexionEnabled:
    def test_critique_can_revise_a_wrong_score(self, monkeypatch):
        from loop.config import settings
        from loop.nodes.grader import grader

        monkeypatch.setattr(settings, "reflexion_enabled", True)
        calls = _stub_two_call_model(monkeypatch, _FIRST_PASS_GRADE, _REVISED_GRADE)

        result = grader(_state())

        assert calls["n"] == 2  # primary pass + critique pass
        assert result["grades"][0]["score"] == 8
        assert (
            result["grades"][0]["overall_feedback"]
            == "Strong answer, mis-scored on the first pass."
        )

    def test_critique_leaves_a_correct_grade_unchanged(self, monkeypatch):
        from loop.config import settings
        from loop.nodes.grader import grader

        monkeypatch.setattr(settings, "reflexion_enabled", True)
        # Critic agrees with the first pass -- returns the same grade back.
        calls = _stub_two_call_model(monkeypatch, _FIRST_PASS_GRADE, _FIRST_PASS_GRADE)

        result = grader(_state())

        assert calls["n"] == 2
        assert result["grades"][0]["score"] == 3


class TestSelfCritiqueDirectly:
    def test_self_critique_returns_model_output(self, monkeypatch):
        from loop.nodes.grader import _self_critique
        from loop.tools import get_rubric

        rubric = get_rubric(_QID)
        calls = _stub_two_call_model(monkeypatch, _REVISED_GRADE, _REVISED_GRADE)

        result = _self_critique(
            _FIRST_PASS_GRADE,
            rubric,
            "reference text",
            "candidate answer text",
            "question prompt text",
        )

        assert calls["n"] == 1
        assert result.score == 8
