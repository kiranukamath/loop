"""
Phase 3 tests — tools, interviewers, grader, coach, routing.
All offline: model calls are stubbed via monkeypatch.

These tests exercise Phase 3 nodes in isolation, not the full graph.
Full-graph smoke tests live in test_graph.py (conftest stubs all models there).
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.runnables import RunnableLambda

from loop.schemas import Feedback, Grade

# ── Helpers ───────────────────────────────────────────────────────────────────


def _stub_model(monkeypatch, target: str, return_value):
    """Patch get_chat_model in `target` module to return a fake structured model."""
    fake = MagicMock()
    fake.with_structured_output.return_value = RunnableLambda(lambda _: return_value)
    monkeypatch.setattr(target, lambda: fake)


_STUB_GRADE = Grade(
    question_id="cod-001",
    score=8,
    criterion_scores={
        "correctness": 4,
        "optimal_complexity": 2,
        "code_quality": 1,
        "communication": 1,
    },
    strengths=["correct sliding window"],
    improvements=["handle empty string"],
    overall_feedback="Good attempt.",
)

_STUB_FEEDBACK = Feedback(
    summary="Solid session.",
    action_items=["practice edge cases"],
    weak_areas_update=["edge-case handling"],
)


# ── Tools ─────────────────────────────────────────────────────────────────────


class TestTools:
    def test_get_questions_by_modality_coding(self):
        from loop.tools import get_questions_by_modality

        qs = get_questions_by_modality("coding")
        assert len(qs) >= 1
        assert all(q["modality"] == "coding" for q in qs)

    def test_get_questions_by_modality_behavioral(self):
        from loop.tools import get_questions_by_modality

        qs = get_questions_by_modality("behavioral")
        assert len(qs) >= 1
        assert all(q["modality"] == "behavioral" for q in qs)

    def test_get_questions_unknown_modality_returns_empty(self):
        from loop.tools import get_questions_by_modality

        assert get_questions_by_modality("unknown") == []

    def test_get_question_by_id_found(self):
        from loop.tools import get_question_by_id

        q = get_question_by_id("cod-001")
        assert q is not None
        assert q["id"] == "cod-001"
        assert "modality" in q

    def test_get_question_by_id_not_found(self):
        from loop.tools import get_question_by_id

        assert get_question_by_id("nonexistent") is None

    def test_get_rubric_found(self):
        from loop.tools import get_rubric

        r = get_rubric("cod-001")
        assert r is not None
        assert r["question_id"] == "cod-001"
        assert "criteria" in r
        assert r["max_score"] == 10

    def test_get_rubric_not_found(self):
        from loop.tools import get_rubric

        assert get_rubric("nonexistent-id") is None

    def test_all_questions_have_rubrics(self):
        """Every fixture question has a matching rubric."""
        from loop.tools import _load_questions, get_rubric

        for q in _load_questions():
            rubric = get_rubric(q["id"])
            assert rubric is not None, f"Missing rubric for question {q['id']}"


# ── Interviewers ──────────────────────────────────────────────────────────────


class TestInterviewers:
    """Test the question-picking and answer-recording logic of the interviewer nodes.

    The real interviewer calls interrupt() to pause for a human answer.
    These tests stub interrupt() in the interviewers module so we can call
    the functions directly and assert on question selection + state structure.
    The stub returns a canned answer string, as Command(resume="...") would.
    """

    def _stub_interrupt(self, monkeypatch, answer: str = "stub answer"):
        monkeypatch.setattr("loop.nodes.interviewers.interrupt", lambda payload: answer)

    def test_coding_interviewer_sets_question_id(self, monkeypatch):
        self._stub_interrupt(monkeypatch)
        from loop.nodes.interviewers import coding_interviewer

        result = coding_interviewer({"answers": []})
        assert result["current_question_id"] is not None
        from loop.tools import get_question_by_id

        q = get_question_by_id(result["current_question_id"])
        assert q["modality"] == "coding"

    def test_coding_interviewer_adds_ai_and_human_messages(self, monkeypatch):
        self._stub_interrupt(monkeypatch, answer="my answer")
        from langchain_core.messages import AIMessage, HumanMessage

        from loop.nodes.interviewers import coding_interviewer

        result = coding_interviewer({"answers": []})
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][0], AIMessage)
        assert isinstance(result["messages"][1], HumanMessage)
        assert result["messages"][1].content == "my answer"

    def test_coding_interviewer_stores_answer_in_state(self, monkeypatch):
        self._stub_interrupt(monkeypatch, answer="sliding window approach")
        from loop.nodes.interviewers import coding_interviewer

        result = coding_interviewer({"answers": []})
        assert len(result["answers"]) == 1
        assert result["answers"][0]["text"] == "sliding window approach"
        assert result["answers"][0]["question_id"] == result["current_question_id"]

    def test_sd_interviewer_picks_system_design(self, monkeypatch):
        self._stub_interrupt(monkeypatch)
        from loop.nodes.interviewers import sd_interviewer
        from loop.tools import get_question_by_id

        result = sd_interviewer({"answers": []})
        q = get_question_by_id(result["current_question_id"])
        assert q["modality"] == "system_design"

    def test_beh_interviewer_picks_behavioral(self, monkeypatch):
        self._stub_interrupt(monkeypatch)
        from loop.nodes.interviewers import beh_interviewer
        from loop.tools import get_question_by_id

        result = beh_interviewer({"answers": []})
        q = get_question_by_id(result["current_question_id"])
        assert q["modality"] == "behavioral"

    def test_interviewer_skips_already_answered_question(self, monkeypatch):
        """If cod-001 is in answers, interviewer picks the next coding question."""
        self._stub_interrupt(monkeypatch)
        from loop.nodes.interviewers import coding_interviewer

        state = {"answers": [{"question_id": "cod-001", "text": "..."}]}
        result = coding_interviewer(state)
        assert result["current_question_id"] != "cod-001"

    def test_interviewer_falls_back_when_all_answered(self, monkeypatch):
        """If all questions answered, falls back to first question (no crash)."""
        self._stub_interrupt(monkeypatch)
        from loop.nodes.interviewers import coding_interviewer
        from loop.tools import get_questions_by_modality

        all_ids = [q["id"] for q in get_questions_by_modality("coding")]
        state = {"answers": [{"question_id": qid, "text": "..."} for qid in all_ids]}
        result = coding_interviewer(state)
        assert result["current_question_id"] is not None

    def test_interviewer_selects_by_focus_not_fixture_order(self, monkeypatch):
        """Phase 8b: question selection follows current_focus, not fixture position.

        cod-003 ("Binary search and its variants") is NOT first in fixture order
        (cod-001 is). Setting current_focus to text drawn directly from cod-003's
        own indexed content should surface cod-003 first, proving the interviewer
        is ranking by semantic similarity rather than always picking questions[0].
        """
        self._stub_interrupt(monkeypatch)
        from loop.nodes.interviewers import coding_interviewer
        from loop.tools import get_question_by_id

        target = get_question_by_id("cod-003")
        # Reconstruct the exact page_content string retrieval.py embeds for cod-003
        # so DeterministicFakeEmbedding gives it cosine similarity = 1.0.
        exact_focus = f"{target['title']}. {target['prompt']} Topic: {target['topic']}"

        state = {"answers": [], "current_focus": exact_focus, "current_topics": []}
        result = coding_interviewer(state)
        assert result["current_question_id"] == "cod-003"

    def test_interviewer_no_focus_falls_back_to_modality_query(self, monkeypatch):
        """With no current_focus set, the interviewer still returns a valid
        question for the requested modality (falls back to modality as the query).
        """
        self._stub_interrupt(monkeypatch)
        from loop.nodes.interviewers import beh_interviewer
        from loop.tools import get_question_by_id

        result = beh_interviewer({"answers": []})
        q = get_question_by_id(result["current_question_id"])
        assert q["modality"] == "behavioral"

    def test_interviewer_skips_answered_even_with_focus_match(self, monkeypatch):
        """If the top semantic match is already answered, the next-best unanswered
        question is picked instead — the answered-skip logic still applies under
        semantic ranking.
        """
        self._stub_interrupt(monkeypatch)
        from loop.nodes.interviewers import coding_interviewer
        from loop.tools import get_question_by_id

        target = get_question_by_id("cod-003")
        exact_focus = f"{target['title']}. {target['prompt']} Topic: {target['topic']}"

        state = {
            "answers": [{"question_id": "cod-003", "text": "already answered"}],
            "current_focus": exact_focus,
            "current_topics": [],
        }
        result = coding_interviewer(state)
        assert result["current_question_id"] != "cod-003"


# ── Grader ────────────────────────────────────────────────────────────────────


class TestGrader:
    def test_grader_appends_grade(self, monkeypatch):
        monkeypatch.setattr("loop.nodes.grader.get_chat_model", lambda: _make_fake_grader())

        from loop.nodes.grader import grader

        state = {
            "current_question_id": "cod-001",
            "answers": [{"question_id": "cod-001", "text": "sliding window..."}],
            "grades": None,
        }
        result = grader(state)
        assert "grades" in result
        assert len(result["grades"]) == 1
        assert result["grades"][0]["question_id"] == "cod-001"

    def test_grader_returns_only_new_grade(self, monkeypatch):
        """Grader returns only the new grade dict, not the full accumulated list.

        Accumulation across sessions is handled by the _append_list reducer on
        state["grades"] at the LangGraph level. Returning the full list from the
        node would cause duplicates when the reducer appends it to the existing list.
        """
        monkeypatch.setattr("loop.nodes.grader.get_chat_model", lambda: _make_fake_grader())

        from loop.nodes.grader import grader

        existing = [
            {
                "question_id": "beh-001",
                "score": 9,
                "criterion_scores": {},
                "strengths": [],
                "improvements": [],
                "overall_feedback": "",
            }
        ]
        state = {
            "current_question_id": "cod-001",
            "answers": [{"question_id": "cod-001", "text": "..."}],
            "grades": existing,
        }
        result = grader(state)
        # Grader returns only the new grade — the reducer in state.py handles merging.
        assert len(result["grades"]) == 1
        assert result["grades"][0]["question_id"] == "cod-001"

    def test_grader_raises_when_no_answer(self, monkeypatch):
        monkeypatch.setattr("loop.nodes.grader.get_chat_model", lambda: _make_fake_grader())

        from loop.nodes.grader import grader

        state = {
            "current_question_id": "cod-001",
            "answers": [],  # no answer for cod-001
            "grades": None,
        }
        with pytest.raises(ValueError, match="No answer in state"):
            grader(state)

    def test_grader_raises_for_missing_rubric(self, monkeypatch):
        """Grader raises if question_id has no matching rubric."""
        monkeypatch.setattr("loop.nodes.grader.get_chat_model", lambda: _make_fake_grader())
        monkeypatch.setattr("loop.nodes.grader.get_rubric", lambda _: None)

        from loop.nodes.grader import grader

        state = {
            "current_question_id": "cod-001",
            "answers": [{"question_id": "cod-001", "text": "..."}],
            "grades": None,
        }
        with pytest.raises(ValueError, match="No rubric"):
            grader(state)

    def test_grader_prompt_includes_reference_answer(self, monkeypatch):
        """Phase 8c: the retrieved reference answer must reach the model prompt.

        Captures the formatted ChatPromptValue passed to the (stubbed) structured
        model and asserts the known cod-001 reference text appears in it.
        """
        from loop.tools import get_reference_answer

        captured = {}

        def _capturing_grader():
            fake = MagicMock()

            def _capture(prompt_value):
                captured["prompt_text"] = prompt_value.to_string()
                return _STUB_GRADE

            fake.with_structured_output.return_value = RunnableLambda(_capture)
            return fake

        monkeypatch.setattr("loop.nodes.grader.get_chat_model", _capturing_grader)

        from loop.nodes.grader import grader

        state = {
            "current_question_id": "cod-001",
            "answers": [{"question_id": "cod-001", "text": "sliding window with a hash set"}],
            "grades": None,
        }
        result = grader(state)

        assert "grades" in result  # grading still completes normally
        reference_text = get_reference_answer("cod-001")
        assert reference_text is not None
        assert reference_text in captured["prompt_text"]

    def test_grader_falls_back_when_no_reference_answer(self, monkeypatch):
        """A question with no reference_answers.json entry still grades successfully."""
        monkeypatch.setattr("loop.nodes.grader.get_chat_model", lambda: _make_fake_grader())
        monkeypatch.setattr("loop.nodes.grader.get_reference_answer", lambda _: None)

        from loop.nodes.grader import grader

        state = {
            "current_question_id": "cod-001",
            "answers": [{"question_id": "cod-001", "text": "..."}],
            "grades": None,
        }
        result = grader(state)
        assert "grades" in result
        assert len(result["grades"]) == 1


# ── Coach ─────────────────────────────────────────────────────────────────────


class TestCoach:
    def test_coach_updates_weak_areas(self, monkeypatch):
        monkeypatch.setattr("loop.nodes.coach.get_chat_model", lambda: _make_fake_coach())

        from loop.nodes.coach import coach

        state = {
            "grades": [
                {
                    "question_id": "cod-001",
                    "score": 7,
                    "criterion_scores": {},
                    "strengths": ["good approach"],
                    "improvements": ["edge cases"],
                    "overall_feedback": "Decent.",
                }
            ]
        }
        result = coach(state)
        assert "weak_areas" in result
        assert isinstance(result["weak_areas"], list)
        assert len(result["weak_areas"]) > 0

    def test_coach_returns_empty_weak_areas_on_no_grades(self):
        from loop.nodes.coach import coach

        result = coach({"grades": []})
        assert result["weak_areas"] == []

    def test_coach_handles_none_grades(self):
        from loop.nodes.coach import coach

        result = coach({"grades": None})
        assert result["weak_areas"] == []


# ── Graph routing ─────────────────────────────────────────────────────────────


class TestGraphRouting:
    def test_session_router_sets_coding_modality(self):
        from loop.graph import session_router

        state = {
            "plan": {
                "sessions": [{"session_number": 1, "modality": "coding", "topics": [], "focus": ""}]
            }
        }
        result = session_router(state)
        assert result["current_modality"] == "coding"
        assert result["session_number"] == 1

    def test_session_router_sets_behavioral_modality(self):
        from loop.graph import session_router

        state = {
            "plan": {
                "sessions": [
                    {"session_number": 1, "modality": "behavioral", "topics": [], "focus": ""}
                ]
            }
        }
        result = session_router(state)
        assert result["current_modality"] == "behavioral"

    def test_session_router_raises_on_empty_plan(self):
        from loop.graph import session_router

        with pytest.raises(ValueError, match="no sessions"):
            session_router({"plan": {"sessions": []}})

    def test_route_by_modality_coding(self):
        from loop.graph import _route_by_modality

        assert _route_by_modality({"current_modality": "coding"}) == "coding"

    def test_route_by_modality_system_design(self):
        from loop.graph import _route_by_modality

        assert _route_by_modality({"current_modality": "system_design"}) == "system_design"

    def test_route_by_modality_raises_on_unknown(self):
        from loop.graph import _route_by_modality

        with pytest.raises(ValueError, match="Unknown modality"):
            _route_by_modality({"current_modality": "magic"})


# ── Internal helpers ──────────────────────────────────────────────────────────


def _make_fake_grader():
    fake = MagicMock()
    fake.with_structured_output.return_value = RunnableLambda(lambda _: _STUB_GRADE)
    return fake


def _make_fake_coach():
    fake = MagicMock()
    fake.with_structured_output.return_value = RunnableLambda(lambda _: _STUB_FEEDBACK)
    return fake
