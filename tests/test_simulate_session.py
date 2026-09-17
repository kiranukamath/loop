"""
Phase 17b — agent simulation tests. All offline: relies on the same
autouse stub_embeddings/stub_reranker fixtures (tests/conftest.py) every
other graph test uses, plus evals.simulate_session's own LLM-node stubs.

Verifies:
- SimulatedCandidate.respond() resolves all three interrupt actions and
  raises on an unrecognized one
- A full 2-session simulated curriculum run completes end-to-end offline:
  both sessions graded, readiness verdict produced and approved
- run_to_completion raises RuntimeError if max_turns is exceeded (a stuck
  candidate that can't resume an interrupt shouldn't hang forever)
"""

from __future__ import annotations

import pytest

from evals.simulate_session import SimulatedCandidate, run_simulated_session

# ── SimulatedCandidate.respond ──────────────────────────────────────────────────


class TestSimulatedCandidateRespond:
    def test_approve_plan_returns_plan_decision(self):
        candidate = SimulatedCandidate()
        assert candidate.respond({"action": "approve_plan"}) == {"decision": "approve"}

    def test_answer_question_uses_scripted_answer(self):
        candidate = SimulatedCandidate(answers={"cod-001": "scripted answer"})
        response = candidate.respond({"action": "answer_question", "question_id": "cod-001"})
        assert response == "scripted answer"

    def test_answer_question_falls_back_to_default(self):
        candidate = SimulatedCandidate(answers={}, default_answer="fallback")
        response = candidate.respond({"action": "answer_question", "question_id": "unknown-id"})
        assert response == "fallback"

    def test_approve_verdict_returns_verdict_decision(self):
        candidate = SimulatedCandidate(
            verdict_decision={"decision": "override", "verdict": "ready"}
        )
        assert candidate.respond({"action": "approve_verdict"}) == {
            "decision": "override",
            "verdict": "ready",
        }

    def test_unknown_action_raises(self):
        candidate = SimulatedCandidate()
        with pytest.raises(ValueError, match="doesn't know how to resume"):
            candidate.respond({"action": "something_else"})


# ── Full end-to-end simulated run ────────────────────────────────────────────────


class TestFullSimulatedRun:
    def test_two_session_run_completes_offline(self):
        result = run_simulated_session(user_id="sim-user", thread_id="sim-thread-test")

        # Both sessions of the stub 2-session plan ran and advanced the index.
        assert result.get("session_index") == 2
        assert len(result.get("grades") or []) == 2

        # Readiness gate reached and approved.
        assert result.get("readiness_verdict") is not None
        assert result.get("verdict_approved") is True

        # No leftover interrupt — the run truly completed.
        assert "__interrupt__" not in result

    def test_plan_was_approved(self):
        result = run_simulated_session(user_id="sim-user", thread_id="sim-thread-test-2")
        assert result.get("plan_approved") is True


# ── Safety bound ──────────────────────────────────────────────────────────────


class TestMaxTurnsBound:
    def test_raises_when_candidate_cannot_resolve_interrupts(self, monkeypatch):
        """A candidate that always raises on respond() should hit max_turns
        (via the RuntimeError path) rather than loop forever. Simulated with
        a fake app whose invoke() always returns an interrupt payload the
        candidate happens to resolve trivially but which never terminates —
        exercises the max_turns guard directly rather than needing a real
        stuck graph."""

        class _FakeApp:
            def invoke(self, _state_or_command, config):
                # Always interrupts again — simulates a runaway/looping graph.
                return {"__interrupt__": [_FakeInterrupt()]}

        class _FakeInterrupt:
            value = {"action": "approve_plan"}

        candidate = SimulatedCandidate()
        with pytest.raises(RuntimeError, match="max_turns"):
            candidate.run_to_completion(_FakeApp(), config={}, initial_state={}, max_turns=3)
