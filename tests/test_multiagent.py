"""
Phase 13 tests — multi-agent orchestration (supervisor · parallel Send · handoffs).

All offline: model calls are stubbed with RunnableLambdas, exactly like every
prior phase's tests. Both Phase 13 features are flag-gated and default OFF —
tests that don't pass orchestration_mode/panel_grading explicitly exercise
the SAME Phase 12 graph the rest of the suite already covers; this file only
targets the new, explicitly-opted-into code paths.

Sections:
  A. grade_dispatch / panel_grader / grade_aggregator (13a) — unit-level,
     hand-built state so the map/reduce math is asserted directly.
  B. Panel grading through the full compiled graph (13a) — proves the
     fan-out/fan-in mechanics (aggregator runs once, not once per persona)
     and the reset-sentinel reducer, end to end.
  C. panel_debate (13c, stretch) — the extra round, aggregated from the
     LATEST round only.
  D. interview_supervisor (13b) — unit-level bound check, then the full
     compiled graph reproducing today's plan-based sequence, then the
     GraphRecursionError backstop for a supervisor that never terminates.
"""

import re
from unittest.mock import MagicMock

import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command, Send

from loop.nodes.panel import grade_aggregator, grade_dispatch, panel_grader
from loop.nodes.supervisor import interview_supervisor
from loop.schemas import PersonaGrade, PrepPlan, Session, SupervisorDecision
from loop.state import initial_state

# ── Shared helpers ─────────────────────────────────────────────────────────────


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


def _fake_model(return_value):
    """Ignore-input stub: same pattern as conftest.py's planner stub."""
    fake = MagicMock()
    fake.with_structured_output.return_value = RunnableLambda(lambda _: return_value)
    return fake


def _partial(persona: str, round_: int, scores: dict, notes: str = "note") -> dict:
    return {
        "persona": persona,
        "question_id": "cod-001",
        "round": round_,
        "criterion_scores": scores,
        "strengths": [f"{persona}-strength"],
        "improvements": [f"{persona}-improvement"],
        "notes": notes,
    }


# ── A. grade_dispatch ──────────────────────────────────────────────────────────


class TestGradeDispatch:
    def test_fans_out_one_send_per_persona(self):
        state = {
            "current_question_id": "cod-001",
            "answers": [{"question_id": "cod-001", "text": "sliding window answer"}],
        }
        sends = grade_dispatch(state)
        assert len(sends) == 3  # default settings.grader_personas
        assert all(isinstance(s, Send) for s in sends)
        assert all(s.node == "panel_grader" for s in sends)
        personas = {s.arg["persona"] for s in sends}
        assert personas == {"correctness", "communication", "depth"}

    def test_each_payload_carries_full_context(self):
        state = {
            "current_question_id": "cod-001",
            "answers": [{"question_id": "cod-001", "text": "sliding window answer"}],
        }
        sends = grade_dispatch(state)
        for s in sends:
            assert s.arg["question_id"] == "cod-001"
            assert s.arg["round"] == 0
            assert s.arg["peer_scores"] is None
            assert s.arg["answer_text"] == "sliding window answer"
            assert s.arg["max_score"] == 10
            assert "correctness" in s.arg["rubric_criteria"]

    def test_respects_configured_personas(self, monkeypatch):
        monkeypatch.setattr("loop.nodes.panel.settings.grader_personas", ["correctness"])
        state = {
            "current_question_id": "cod-001",
            "answers": [{"question_id": "cod-001", "text": "x"}],
        }
        sends = grade_dispatch(state)
        assert len(sends) == 1
        assert sends[0].arg["persona"] == "correctness"

    def test_raises_when_no_matching_answer(self):
        state = {"current_question_id": "cod-001", "answers": []}
        with pytest.raises(ValueError, match="No answer in state"):
            grade_dispatch(state)


# ── A. panel_grader ────────────────────────────────────────────────────────────


class TestPanelGrader:
    _STUB_PERSONA_GRADE = PersonaGrade(
        criterion_scores={
            "correctness": 3,
            "optimal_complexity": 2,
            "code_quality": 1,
            "communication": 1,
        },
        strengths=["good intuition"],
        improvements=["edge cases"],
        notes="solid",
    )

    def test_payload_fields_win_over_model_output(self, monkeypatch):
        """persona/question_id/round come from the payload, not the model."""
        monkeypatch.setattr(
            "loop.nodes.panel.get_chat_model", lambda: _fake_model(self._STUB_PERSONA_GRADE)
        )
        payload = {
            "persona": "depth",
            "question_id": "cod-001",
            "round": 0,
            "peer_scores": None,
            "question_prompt": "q",
            "reference_answer": "ref",
            "answer_text": "ans",
            "max_score": 10,
            "rubric_criteria": "- correctness (weight 4): ...",
        }
        result = panel_grader(payload)
        [partial] = result["panel_grades"]
        assert partial["persona"] == "depth"
        assert partial["question_id"] == "cod-001"
        assert partial["round"] == 0
        assert partial["criterion_scores"] == self._STUB_PERSONA_GRADE.criterion_scores

    def test_handles_peer_scores_without_crashing(self, monkeypatch):
        """Round-1 payloads carry peer_scores — panel_grader must accept them."""
        monkeypatch.setattr(
            "loop.nodes.panel.get_chat_model", lambda: _fake_model(self._STUB_PERSONA_GRADE)
        )
        payload = {
            "persona": "correctness",
            "question_id": "cod-001",
            "round": 1,
            "peer_scores": [_partial("communication", 0, {"correctness": 2})],
            "question_prompt": "q",
            "reference_answer": "ref",
            "answer_text": "ans",
            "max_score": 10,
            "rubric_criteria": "- correctness (weight 4): ...",
        }
        result = panel_grader(payload)
        assert result["panel_grades"][0]["round"] == 1


# ── A. grade_aggregator ────────────────────────────────────────────────────────


class TestGradeAggregator:
    def test_reduces_round_zero_and_resets_scratch(self):
        partials = [
            _partial(
                "correctness",
                0,
                {"correctness": 4, "optimal_complexity": 2, "code_quality": 1, "communication": 1},
            ),
            _partial(
                "communication",
                0,
                {"correctness": 2, "optimal_complexity": 3, "code_quality": 2, "communication": 1},
            ),
            _partial(
                "depth",
                0,
                {"correctness": 3, "optimal_complexity": 1, "code_quality": 2, "communication": 1},
            ),
        ]
        state = {"current_question_id": "cod-001", "panel_grades": partials}

        result = grade_aggregator(state)

        assert isinstance(result, Command)
        assert result.goto == "coach"
        [grade] = result.update["grades"]
        assert grade["criterion_scores"] == {
            "correctness": 3,  # round((4+2+3)/3) = round(3.0)
            "optimal_complexity": 2,  # round((2+3+1)/3) = round(2.0)
            "code_quality": 2,  # round((1+2+2)/3) = round(1.667)
            "communication": 1,
        }
        assert grade["score"] == 3 + 2 + 2 + 1
        assert len(grade["strengths"]) == 3  # one per persona, no dupes here
        assert "[correctness]" in grade["overall_feedback"]
        assert "[depth]" in grade["overall_feedback"]
        # Reset sentinel: None, not [] — see loop/state.py's _reset_or_append.
        assert result.update["panel_grades"] is None

    def test_raises_when_no_partials_for_question(self):
        state = {
            "current_question_id": "cod-001",
            "panel_grades": [_partial("correctness", 0, {"correctness": 1}, "n")],
        }
        state["panel_grades"][0]["question_id"] = "other-question"
        with pytest.raises(ValueError, match="No panel grades found"):
            grade_aggregator(state)


# ── C. panel_debate (13c stretch) ──────────────────────────────────────────────


class TestPanelDebate:
    _ROUND0 = [
        _partial("correctness", 0, {"correctness": 4}, "r0-correctness"),
        _partial("communication", 0, {"correctness": 2}, "r0-communication"),
        _partial("depth", 0, {"correctness": 3}, "r0-depth"),
    ]

    def test_round_zero_triggers_a_debate_fanout(self, monkeypatch):
        monkeypatch.setattr("loop.nodes.panel.settings.panel_debate", True)
        state = {
            "current_question_id": "cod-001",
            "panel_grades": list(self._ROUND0),
            "answers": [{"question_id": "cod-001", "text": "some answer"}],
        }

        result = grade_aggregator(state)

        assert isinstance(result, Command)
        assert isinstance(result.goto, list)
        assert len(result.goto) == 3
        assert all(isinstance(s, Send) and s.node == "panel_grader" for s in result.goto)
        for send in result.goto:
            assert send.arg["round"] == 1
            peers = send.arg["peer_scores"]
            assert all(p["persona"] != send.arg["persona"] for p in peers)
            assert len(peers) == 2
        # No state update on the fan-out leg — round-0 partials stay in place.
        assert not result.update

    def test_latest_round_only_is_reduced(self, monkeypatch):
        monkeypatch.setattr("loop.nodes.panel.settings.panel_debate", True)
        round1 = [
            _partial("correctness", 1, {"correctness": 10}, "r1-correctness"),
            _partial("communication", 1, {"correctness": 10}, "r1-communication"),
            _partial("depth", 1, {"correctness": 10}, "r1-depth"),
        ]
        state = {
            "current_question_id": "cod-001",
            # Both rounds present — the round-0 partials must be IGNORED.
            "panel_grades": list(self._ROUND0) + round1,
        }

        result = grade_aggregator(state)

        assert result.goto == "coach"
        [grade] = result.update["grades"]
        assert grade["criterion_scores"]["correctness"] == 10  # only round 1 used
        assert result.update["panel_grades"] is None


# ── B. Panel grading through the full compiled graph ──────────────────────────


class TestPanelGradingGraph:
    def _build_app(self, monkeypatch):
        from loop.graph import build_graph

        stub_persona_grade = PersonaGrade(
            criterion_scores={
                "correctness": 4,
                "optimal_complexity": 3,
                "code_quality": 2,
                "communication": 1,
            },
            strengths=["clear approach"],
            improvements=["state complexity upfront"],
            notes="strong",
        )
        monkeypatch.setattr(
            "loop.nodes.panel.get_chat_model", lambda: _fake_model(stub_persona_grade)
        )

        monkeypatch.setattr(
            "loop.nodes.planner.get_chat_model",
            lambda: _fake_model(PrepPlan(**_plan_dict("coding"))),
        )
        monkeypatch.setattr(
            "loop.graph.plan_approval",
            lambda state: Command(goto="session_router", update={"plan_approved": True}),
        )
        monkeypatch.setattr(
            "loop.graph.coding_interviewer",
            lambda state: {
                "current_question_id": "cod-001",
                "answers": [{"question_id": "cod-001", "text": "sliding window, O(n)"}],
                "messages": [],
            },
        )
        monkeypatch.setattr("loop.graph.coach", lambda state: {"weak_areas": ["edge cases"]})
        monkeypatch.setattr(
            "loop.graph.readiness",
            lambda state: {
                "readiness_verdict": {
                    "verdict": "ready",
                    "confidence": 0.9,
                    "strengths": [],
                    "gaps": [],
                    "recommendation": "go",
                },
                "verdict_approved": True,
            },
        )

        return build_graph(panel_grading=True).compile(checkpointer=MemorySaver())

    def test_aggregator_runs_once_not_once_per_persona(self, monkeypatch):
        from loop.nodes.panel import grade_aggregator as real_aggregator

        calls = []

        def spy(state):
            calls.append(1)
            return real_aggregator(state)

        monkeypatch.setattr("loop.graph.grade_aggregator", spy)
        app = self._build_app(monkeypatch)

        app.invoke(initial_state(), config={"configurable": {"thread_id": "panel-once"}})

        assert len(calls) == 1, f"Expected grade_aggregator to run exactly once, ran {len(calls)}x"

    def test_final_grade_matches_averaged_panel_and_scratch_is_reset(self, monkeypatch):
        app = self._build_app(monkeypatch)

        result = app.invoke(initial_state(), config={"configurable": {"thread_id": "panel-reset"}})

        assert len(result.get("grades") or []) == 1
        grade = result["grades"][0]
        # All 3 personas got the SAME stub PersonaGrade -> averaging is a no-op.
        assert grade["criterion_scores"] == {
            "correctness": 4,
            "optimal_complexity": 3,
            "code_quality": 2,
            "communication": 1,
        }
        assert grade["score"] == 10
        # The scratch channel must be reset, not left with 3 stale partials.
        assert result.get("panel_grades") == []


# ── D. interview_supervisor (13b) ──────────────────────────────────────────────


class TestInterviewSupervisorUnit:
    def test_stops_at_bound_without_calling_the_model(self, monkeypatch):
        fake_model = MagicMock()
        monkeypatch.setattr("loop.nodes.supervisor.get_chat_model", lambda: fake_model)
        monkeypatch.setattr("loop.nodes.supervisor.settings.max_sessions", 2)

        state = {
            "plan": _plan_dict("coding", "system_design"),
            "session_index": 2,
            "weak_areas": [],
        }
        result = interview_supervisor(state)

        assert isinstance(result, Command)
        assert result.goto == "readiness"
        fake_model.with_structured_output.assert_not_called()

    def test_stops_when_plan_has_no_sessions(self, monkeypatch):
        fake_model = MagicMock()
        monkeypatch.setattr("loop.nodes.supervisor.get_chat_model", lambda: fake_model)

        result = interview_supervisor({"plan": {"sessions": []}, "session_index": 0})
        assert result.goto == "readiness"
        fake_model.with_structured_output.assert_not_called()

    def test_maps_decision_to_the_right_interviewer_node(self, monkeypatch):
        decision = SupervisorDecision(next_modality="system_design", focus="f", topics=["t"])
        monkeypatch.setattr("loop.nodes.supervisor.get_chat_model", lambda: _fake_model(decision))
        state = {"plan": _plan_dict("coding"), "session_index": 0, "weak_areas": []}

        result = interview_supervisor(state)

        assert result.goto == "sd_interviewer"
        assert result.update["current_modality"] == "system_design"
        assert result.update["supervisor_decisions"][0]["chosen_modality"] == "system_design"


def _echo_plan_modality(prompt_value) -> SupervisorDecision:
    """A model stub that's an actual (non-constant) function of its input —
    reads the rendered prompt's "suggested modality" line back out, so the
    supervisor reproduces whatever modality the plan suggested for this
    session. This is what lets a stubbed supervisor "reproduce today's
    modality sequence" instead of always answering the same thing."""
    human_text = prompt_value.messages[-1].content
    modality = re.search(r"suggested modality for this session: (\w+)", human_text).group(1)
    return SupervisorDecision(next_modality=modality, focus="stub focus", topics=["stub-topic"])


class TestSupervisorGraph:
    def _build_app(self, monkeypatch):
        from loop.graph import build_graph

        monkeypatch.setattr(
            "loop.nodes.planner.get_chat_model",
            lambda: _fake_model(PrepPlan(**_plan_dict("coding", "system_design"))),
        )
        monkeypatch.setattr(
            "loop.graph.plan_approval",
            lambda state, next_node="session_router": Command(
                goto=next_node, update={"plan_approved": True}
            ),
        )
        fake_supervisor_model = MagicMock()
        fake_supervisor_model.with_structured_output.return_value = RunnableLambda(
            _echo_plan_modality
        )
        monkeypatch.setattr("loop.nodes.supervisor.get_chat_model", lambda: fake_supervisor_model)

        monkeypatch.setattr(
            "loop.graph.coding_interviewer",
            lambda state: {
                "current_question_id": "cod-001",
                "answers": [{"question_id": "cod-001", "text": "ans"}],
                "messages": [],
            },
        )
        monkeypatch.setattr(
            "loop.graph.sd_interviewer",
            lambda state: {
                "current_question_id": "sys-001",
                "answers": [{"question_id": "sys-001", "text": "ans"}],
                "messages": [],
            },
        )
        monkeypatch.setattr("loop.graph.grader", lambda state: {"grades": []})
        monkeypatch.setattr("loop.graph.coach", lambda state: {"weak_areas": []})
        monkeypatch.setattr(
            "loop.graph.readiness",
            lambda state: {
                "readiness_verdict": {
                    "verdict": "ready",
                    "confidence": 0.9,
                    "strengths": [],
                    "gaps": [],
                    "recommendation": "go",
                },
                "verdict_approved": True,
            },
        )

        return build_graph(orchestration_mode="supervisor").compile(checkpointer=MemorySaver())

    def test_reproduces_todays_modality_sequence_then_lands_on_readiness(self, monkeypatch):
        visited = []
        from loop.nodes.supervisor import interview_supervisor as real_supervisor

        def spy(state):
            result = real_supervisor(state)
            if result.update:
                visited.append(result.update["current_modality"])
            return result

        monkeypatch.setattr("loop.graph.interview_supervisor", spy)
        app = self._build_app(monkeypatch)

        result = app.invoke(
            initial_state(), config={"configurable": {"thread_id": "supervisor-seq"}}
        )

        # Plan was ["coding", "system_design"] — the stubbed model just echoes
        # each session's planned modality back, so this reproduces the exact
        # sequence _route_by_modality would have produced in fixed mode.
        assert visited == ["coding", "system_design"]
        assert result.get("readiness_verdict") is not None
        assert result.get("session_index") == 2

    def test_model_never_called_once_bound_is_reached(self, monkeypatch):
        """Even with a plan that would keep suggesting sessions, the
        supervisor's own bound (checked before the model call) stops it —
        proving termination doesn't depend on the model 'agreeing' to stop."""
        app = self._build_app(monkeypatch)

        call_count = {"n": 0}
        original = _echo_plan_modality

        def counting_echo(prompt_value):
            call_count["n"] += 1
            return original(prompt_value)

        monkeypatch.setattr(
            "loop.nodes.supervisor.get_chat_model", lambda: _mock_with(counting_echo)
        )

        app.invoke(initial_state(), config={"configurable": {"thread_id": "supervisor-bound"}})

        # 2 sessions in the plan -> exactly 2 model calls, never a 3rd.
        assert call_count["n"] == 2


def _mock_with(fn):
    fake = MagicMock()
    fake.with_structured_output.return_value = RunnableLambda(fn)
    return fake


class TestSupervisorRecursionBackstop:
    def test_graph_recursion_error_is_the_real_backstop(self, monkeypatch):
        """A supervisor that (due to a bug) NEVER routes to readiness must be
        stopped by LangGraph's recursion_limit, not loop forever. This proves
        the graph-level backstop actually works — it's not just documentation."""
        from loop.graph import build_graph

        monkeypatch.setattr(
            "loop.nodes.planner.get_chat_model",
            lambda: _fake_model(PrepPlan(**_plan_dict("coding"))),
        )
        monkeypatch.setattr(
            "loop.graph.plan_approval",
            lambda state, next_node="session_router": Command(
                goto=next_node, update={"plan_approved": True}
            ),
        )
        # Deliberately broken: ignores session_index, always sends the
        # candidate back into another coding session. Bypasses the real
        # interview_supervisor's own bound entirely.
        monkeypatch.setattr(
            "loop.graph.interview_supervisor",
            lambda state: Command(goto="coding_interviewer", update={"current_modality": "coding"}),
        )
        monkeypatch.setattr(
            "loop.graph.coding_interviewer",
            lambda state: {
                "current_question_id": "cod-001",
                "answers": [{"question_id": "cod-001", "text": "ans"}],
                "messages": [],
            },
        )
        monkeypatch.setattr("loop.graph.grader", lambda state: {"grades": []})
        monkeypatch.setattr("loop.graph.coach", lambda state: {"weak_areas": []})

        app = build_graph(orchestration_mode="supervisor").compile(checkpointer=MemorySaver())

        with pytest.raises(GraphRecursionError):
            app.invoke(
                initial_state(),
                config={"configurable": {"thread_id": "supervisor-runaway"}, "recursion_limit": 8},
            )
