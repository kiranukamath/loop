"""
Phase 17a — eval-in-CI gate tests. All offline (no live model calls) — this
IS the per-commit gate: pytest running this file is Tier 1 in production.

Verifies:
- check_offline_gate() passes on the real fixtures (good path)
- check_trajectory_gate() / check_grader_determinism_gate() individually pass
- check_grader_determinism_gate() raises on an intentionally regressed
  fixture (expected_score outside its own labeled band)
- check_grader_determinism_gate() raises when grader() is non-deterministic
  given a fixed model
- check_live_gate() raises GateFailure when Langfuse isn't configured
  (run_eval() returns None) and when agreement regresses below the threshold
  — both without any network call, by stubbing evals.run_grader_eval.run_eval
"""

from __future__ import annotations

import pytest

from evals.ci_gate import (
    GateFailure,
    check_grader_determinism_gate,
    check_live_gate,
    check_offline_gate,
    check_trajectory_gate,
)

# ── Tier 1: good-path ───────────────────────────────────────────────────────────


class TestOfflineGateGoodPath:
    def test_trajectory_gate_passes(self):
        check_trajectory_gate()  # must not raise

    def test_grader_determinism_gate_passes_on_real_fixtures(self):
        check_grader_determinism_gate()  # must not raise

    def test_full_offline_gate_passes(self):
        check_offline_gate()  # must not raise


# ── Tier 1: seeded regressions ──────────────────────────────────────────────────


class TestOfflineGateRegressions:
    def test_fails_on_inconsistent_score_band(self):
        """expected_score outside its own [score_min, score_max] band."""
        bad_items = [
            {
                "id": "regressed-1",
                "question_id": "cod-001",
                "answer": "sliding window with a hash set",
                "expected_score": 9,
                "score_min": 2,
                "score_max": 4,  # 9 is nowhere near [2, 4] — a seeded regression
            }
        ]
        with pytest.raises(GateFailure, match="outside its own labeled band"):
            check_grader_determinism_gate(bad_items)

    def test_fails_when_grader_is_nondeterministic(self, monkeypatch):
        """If grader() returns different output for identical input given a
        fixed model, that's a real regression (hidden state/randomness)."""
        import itertools

        from loop.schemas import Grade

        counter = itertools.count()

        def flaky_grader(state):
            # Score drifts across calls even though the "model" is stubbed —
            # simulates a node that leaked non-deterministic state.
            n = next(counter)
            return {
                "grades": [
                    Grade(
                        question_id=state["current_question_id"],
                        score=7 + n,
                        criterion_scores={"overall": 7 + n},
                        strengths=[],
                        improvements=[],
                        overall_feedback="x",
                    ).model_dump()
                ]
            }

        monkeypatch.setattr("evals.ci_gate.grader", flaky_grader)

        items = [
            {
                "id": "flaky-1",
                "question_id": "cod-001",
                "answer": "sliding window",
                "expected_score": 7,
                "score_min": 6,
                "score_max": 8,
            }
        ]
        with pytest.raises(GateFailure, match="non-deterministic"):
            check_grader_determinism_gate(items)


# ── Tier 2: live gate, stubbed (no network) ─────────────────────────────────────


class TestLiveGateStubbed:
    def test_fails_when_langfuse_not_configured(self, monkeypatch):
        monkeypatch.setattr("evals.ci_gate.run_eval", lambda run_name=None: None)
        with pytest.raises(GateFailure, match="not configured"):
            check_live_gate(min_agreement=0.7)

    def test_fails_below_threshold(self, monkeypatch):
        class _FakeResult:
            run_evaluations = [{"name": "agreement_rate", "value": 0.4}]

        monkeypatch.setattr("evals.ci_gate.run_eval", lambda run_name=None: _FakeResult())
        with pytest.raises(GateFailure, match="agreement_rate=0.40"):
            check_live_gate(min_agreement=0.7)

    def test_passes_above_threshold(self, monkeypatch):
        class _FakeResult:
            run_evaluations = [{"name": "agreement_rate", "value": 0.9}]

        monkeypatch.setattr("evals.ci_gate.run_eval", lambda run_name=None: _FakeResult())
        agreement = check_live_gate(min_agreement=0.7)
        assert agreement == 0.9

    def test_fails_when_no_agreement_score_produced(self, monkeypatch):
        class _FakeResult:
            run_evaluations = [{"name": "mean_absolute_error", "value": 1.0}]

        monkeypatch.setattr("evals.ci_gate.run_eval", lambda run_name=None: _FakeResult())
        with pytest.raises(GateFailure, match="no 'agreement_rate'"):
            check_live_gate(min_agreement=0.7)
