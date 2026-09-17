"""
Phase 6 eval tests — all offline (no Langfuse, no Bedrock required).

Tests verify:
- grader_labels.json is valid JSON with the expected schema
- score_agreement evaluator computes correct in-range and absolute-error scores
- aggregate_mae computes mean correctly and handles empty input
- extract_trajectory derives the correct node visit sequence from state history
- assert_trajectory passes on a correct trajectory and fails on wrong/short ones
- Full trajectory matches the expected Loop node sequence (using stubbed graph)
"""

import json
import pathlib

import pytest

_FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"


# ── Fixture data validation ───────────────────────────────────────────────────


class TestGraderLabelsFixture:
    def test_labels_json_is_valid(self):
        data = json.loads((_FIXTURES / "grader_labels.json").read_text())
        assert "items" in data
        assert len(data["items"]) > 0

    def test_each_item_has_required_keys(self):
        data = json.loads((_FIXTURES / "grader_labels.json").read_text())
        required = {"id", "question_id", "answer", "expected_score", "score_min", "score_max"}
        for item in data["items"]:
            missing = required - item.keys()
            assert not missing, f"Item {item.get('id')!r} missing keys: {missing}"

    def test_score_ranges_are_valid(self):
        data = json.loads((_FIXTURES / "grader_labels.json").read_text())
        for item in data["items"]:
            assert 0 <= item["score_min"] <= item["expected_score"] <= item["score_max"] <= 10, (
                f"Item {item['id']!r}: score range invalid: "
                f"{item['score_min']}–{item['expected_score']}–{item['score_max']}"
            )

    def test_all_question_ids_are_known(self):
        data = json.loads((_FIXTURES / "grader_labels.json").read_text())
        questions = json.loads((_FIXTURES / "questions.json").read_text())
        known_ids = {q["id"] for q in questions["questions"]}
        for item in data["items"]:
            assert item["question_id"] in known_ids, (
                f"Item {item['id']!r}: unknown question_id={item['question_id']!r}"
            )

    def test_covers_all_modalities(self):
        """Dataset must include at least one coding, one system_design, one behavioral example."""
        questions = json.loads((_FIXTURES / "questions.json").read_text())
        qid_to_modality = {q["id"]: q["modality"] for q in questions["questions"]}
        data = json.loads((_FIXTURES / "grader_labels.json").read_text())
        modalities = {qid_to_modality[item["question_id"]] for item in data["items"]}
        assert "coding" in modalities
        assert "system_design" in modalities
        assert "behavioral" in modalities


# ── score_agreement evaluator ─────────────────────────────────────────────────


class TestScoreAgreementEvaluator:
    def _run(self, actual_score, expected_score, score_min, score_max):
        from evals.run_grader_eval import score_agreement

        return score_agreement(
            input={"question_id": "cod-001", "answer": "..."},
            output={"score": actual_score},
            expected_output={
                "score": expected_score,
                "score_min": score_min,
                "score_max": score_max,
            },
            metadata=None,
        )

    def test_score_in_range_when_exact_match(self):
        results = self._run(7, 7, 6, 8)
        by_name = {r["name"]: r for r in results}
        assert by_name["score_in_range"]["value"] == 1.0

    def test_score_in_range_when_within_tolerance(self):
        results = self._run(6, 7, 6, 8)  # 6 is in [6, 8]
        by_name = {r["name"]: r for r in results}
        assert by_name["score_in_range"]["value"] == 1.0

    def test_score_out_of_range(self):
        results = self._run(4, 7, 6, 8)  # 4 < 6
        by_name = {r["name"]: r for r in results}
        assert by_name["score_in_range"]["value"] == 0.0

    def test_absolute_error_exact_match(self):
        results = self._run(7, 7, 6, 8)
        by_name = {r["name"]: r for r in results}
        assert by_name["absolute_error"]["value"] == 0.0

    def test_absolute_error_two_off(self):
        results = self._run(5, 7, 6, 8)
        by_name = {r["name"]: r for r in results}
        assert by_name["absolute_error"]["value"] == 2.0

    def test_returns_two_metrics(self):
        results = self._run(7, 7, 6, 8)
        names = {r["name"] for r in results}
        assert names == {"score_in_range", "absolute_error"}

    def test_comment_contains_both_scores(self):
        results = self._run(5, 8, 7, 9)
        by_name = {r["name"]: r for r in results}
        comment = by_name["score_in_range"]["comment"]
        assert "5" in comment  # actual
        assert "8" in comment  # expected


# ── aggregate_mae run evaluator ───────────────────────────────────────────────


class TestAggregateMae:
    def _make_item_result(self, actual, expected):
        """Minimal mock of the ItemResult structure aggregate_mae expects."""

        class _IR:
            output = {"score": actual}
            expected_output = {"score": expected}

        return _IR()

    def test_mae_exact(self):
        from evals.run_grader_eval import aggregate_mae

        results = [self._make_item_result(7, 7), self._make_item_result(5, 5)]
        out = aggregate_mae(item_results=results)
        by_name = {r["name"]: r for r in out}
        assert by_name["mean_absolute_error"]["value"] == 0.0

    def test_mae_mixed(self):
        from evals.run_grader_eval import aggregate_mae

        # errors: 1, 2  → MAE = 1.5
        results = [self._make_item_result(6, 7), self._make_item_result(5, 7)]
        out = aggregate_mae(item_results=results)
        by_name = {r["name"]: r for r in out}
        assert by_name["mean_absolute_error"]["value"] == 1.5

    def test_agreement_rate_all_within_1(self):
        from evals.run_grader_eval import aggregate_mae

        results = [self._make_item_result(7, 8), self._make_item_result(6, 7)]
        out = aggregate_mae(item_results=results)
        by_name = {r["name"]: r for r in out}
        assert by_name["agreement_rate"]["value"] == 1.0

    def test_agreement_rate_none_within_1(self):
        from evals.run_grader_eval import aggregate_mae

        results = [self._make_item_result(3, 7), self._make_item_result(2, 7)]
        out = aggregate_mae(item_results=results)
        by_name = {r["name"]: r for r in out}
        assert by_name["agreement_rate"]["value"] == 0.0

    def test_empty_returns_empty(self):
        from evals.run_grader_eval import aggregate_mae

        assert aggregate_mae(item_results=[]) == []


# ── extract_trajectory ────────────────────────────────────────────────────────


class TestExtractTrajectory:
    def _run_stubbed_graph(self, monkeypatch):
        """Run the full graph with stubbed nodes through both HITL gates.

        Stubs the grader NODE FUNCTION directly (not just its model) because
        the grader checks state["answers"] before calling the model — without a
        stub, it would raise if the interviewer picks a question we didn't pre-answer.
        This mirrors the pattern used across all other graph-level tests.
        """
        from unittest.mock import MagicMock

        from langchain_core.runnables import RunnableLambda
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.types import Command

        from loop.graph import build_graph
        from loop.schemas import Feedback, Grade, PrepPlan, ReadinessVerdict, Session
        from loop.state import initial_state

        stub_plan = PrepPlan(
            role_summary="t",
            total_sessions=1,
            sessions=[Session(session_number=1, modality="coding", topics=["x"], focus="y")],
            key_gaps=[],
            rationale="t",
        )
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
        stub_verdict = ReadinessVerdict(
            verdict="ready",
            confidence=0.8,
            strengths=["ok"],
            gaps=[],
            recommendation="go",
        )
        stub_feedback = Feedback(
            summary="ok",
            action_items=[],
            weak_areas_update=[],
        )

        def fake_model(rv):
            m = MagicMock()
            m.with_structured_output.return_value = RunnableLambda(lambda _: rv)
            return m

        monkeypatch.setattr("loop.nodes.planner.get_chat_model", lambda: fake_model(stub_plan))
        # Stub the grader function itself — bypasses the answer-lookup check
        monkeypatch.setattr("loop.graph.grader", lambda s: {"grades": [stub_grade.model_dump()]})
        monkeypatch.setattr("loop.nodes.coach.get_chat_model", lambda: fake_model(stub_feedback))
        monkeypatch.setattr("loop.nodes.readiness.get_chat_model", lambda: fake_model(stub_verdict))

        app = build_graph().compile(checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "traj-test"}}
        state = initial_state()
        state["answers"] = [{"question_id": "cod-001", "text": "sliding window..."}]

        app.invoke(state, config=cfg)
        # Gate 1: approve the plan
        app.invoke(Command(resume={"decision": "approve"}), config=cfg)
        # Gate 2: answer the interviewer's question (grader is stubbed, so any string works)
        app.invoke(Command(resume="sliding window approach"), config=cfg)
        # Gate 3: approve the readiness verdict
        app.invoke(Command(resume={"decision": "approve"}), config=cfg)

        return app, cfg

    def test_trajectory_has_correct_length(self, monkeypatch):
        app, cfg = self._run_stubbed_graph(monkeypatch)
        from evals.trajectory_check import extract_trajectory

        traj = extract_trajectory(app, cfg)
        # intake, planner, plan_approval, session_router, interviewer,
        # grader, coach, advance_session, readiness, reflect (Phase 16b)
        assert len(traj) == 10

    def test_trajectory_starts_with_intake(self, monkeypatch):
        app, cfg = self._run_stubbed_graph(monkeypatch)
        from evals.trajectory_check import extract_trajectory

        traj = extract_trajectory(app, cfg)
        assert traj[0] == "intake"

    def test_trajectory_ends_with_reflect(self, monkeypatch):
        """Phase 16b: reflect() always runs last in the topology (as a no-op
        when reflection_enabled is off) -- readiness is now second-to-last."""
        app, cfg = self._run_stubbed_graph(monkeypatch)
        from evals.trajectory_check import extract_trajectory

        traj = extract_trajectory(app, cfg)
        assert traj[-1] == "reflect"
        assert traj[-2] == "readiness"

    def test_trajectory_contains_all_expected_nodes(self, monkeypatch):
        app, cfg = self._run_stubbed_graph(monkeypatch)
        from evals.trajectory_check import INTERVIEWER_NODES, extract_trajectory

        traj = extract_trajectory(app, cfg)
        traj_set = set(traj)

        assert "intake" in traj_set
        assert "planner" in traj_set
        assert "plan_approval" in traj_set
        assert "session_router" in traj_set
        assert traj_set & INTERVIEWER_NODES, "Expected at least one interviewer node"
        assert "grader" in traj_set
        assert "coach" in traj_set
        assert "advance_session" in traj_set
        assert "readiness" in traj_set

    def test_assert_trajectory_passes_on_valid(self, monkeypatch):
        app, cfg = self._run_stubbed_graph(monkeypatch)
        from evals.trajectory_check import assert_trajectory, extract_trajectory

        traj = extract_trajectory(app, cfg)
        assert_trajectory(traj)  # must not raise

    def test_assert_trajectory_fails_on_missing_grader(self):
        from evals.trajectory_check import assert_trajectory

        bad_traj = [
            "intake",
            "planner",
            "plan_approval",
            "session_router",
            "coding_interviewer",
            # grader missing
            "coach",
            "advance_session",
            "readiness",
        ]
        with pytest.raises(AssertionError):
            assert_trajectory(bad_traj)

    def test_assert_trajectory_fails_on_wrong_prefix(self):
        from evals.trajectory_check import assert_trajectory

        bad_traj = [
            "intake",
            "session_router",  # planner skipped
            "plan_approval",
            "coding_interviewer",
            "grader",
            "coach",
            "advance_session",
            "readiness",
        ]
        with pytest.raises(AssertionError):
            assert_trajectory(bad_traj)

    def test_assert_trajectory_fails_on_too_short(self):
        from evals.trajectory_check import assert_trajectory

        with pytest.raises(AssertionError):
            assert_trajectory(["intake", "planner"])
