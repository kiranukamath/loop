"""
Phase 5 HITL tests — interrupt/resume flow for both approval gates.
All offline: model calls are stubbed.

Tests verify:
- First invoke pauses at plan_approval with the PrepPlan in the interrupt payload
- Resuming with 'approve' passes plan_approval and pauses at readiness
- Resuming readiness 'approve' completes the graph with verdict_approved=True
- Resuming plan_approval with 'edit' updates state['plan'] before continuing
- Resuming plan_approval with 'reject' ends the graph without running the interview
- Resuming readiness with 'override' stores the human's verdict (not the model's)
- plan_approved and verdict_approved fields are set correctly in all paths
"""

from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from loop.schemas import Feedback, Grade, PrepPlan, ReadinessVerdict, Session

# ── Shared stubs ──────────────────────────────────────────────────────────────

_STUB_PLAN = PrepPlan(
    role_summary="stub",
    total_sessions=1,
    sessions=[Session(session_number=1, modality="coding", topics=["x"], focus="y")],
    key_gaps=["Kafka", "STAR"],
    rationale="stub",
)

_STUB_GRADE = Grade(
    question_id="cod-001",
    score=6,
    criterion_scores={
        "correctness": 2,
        "optimal_complexity": 1,
        "code_quality": 2,
        "communication": 1,
    },
    strengths=["good approach"],
    improvements=["handle edge cases"],
    overall_feedback="Decent attempt.",
)

_STUB_VERDICT = ReadinessVerdict(
    verdict="not_ready",
    confidence=0.7,
    strengths=["strong systems knowledge"],
    gaps=["Kafka", "STAR storytelling"],
    recommendation="Two more sessions recommended before interviewing.",
)

_STUB_FEEDBACK = Feedback(
    summary="session summary",
    action_items=["practice Kafka"],
    weak_areas_update=["Kafka"],
)


def _fake_model(return_value):
    from unittest.mock import MagicMock

    fake = MagicMock()
    fake.with_structured_output.return_value = RunnableLambda(lambda _: return_value)
    return fake


def _build_app(monkeypatch):
    """Build and compile the graph with MemorySaver; stub all model-calling nodes.

    plan_approval and readiness use the real implementations (they call interrupt),
    so the HITL tests can exercise the actual pause/resume mechanics.
    """
    monkeypatch.setattr("loop.nodes.planner.get_chat_model", lambda: _fake_model(_STUB_PLAN))
    monkeypatch.setattr("loop.graph.grader", lambda s: {"grades": [_STUB_GRADE.model_dump()]})
    monkeypatch.setattr("loop.nodes.coach.get_chat_model", lambda: _fake_model(_STUB_FEEDBACK))
    monkeypatch.setattr("loop.nodes.readiness.get_chat_model", lambda: _fake_model(_STUB_VERDICT))

    from loop.graph import build_graph

    return build_graph().compile(checkpointer=MemorySaver())


def _initial(answers=None):
    from loop.state import initial_state

    s = initial_state()
    s["answers"] = answers or [{"question_id": "cod-001", "text": "sliding window..."}]
    return s


# ── Gate 1: plan_approval ─────────────────────────────────────────────────────


class TestPlanApprovalGate:
    def test_first_invoke_pauses_at_plan_approval(self, monkeypatch):
        """Graph pauses on first invoke — __interrupt__ present with plan payload."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-pa-1"}}

        result = app.invoke(_initial(), config=cfg)

        assert "__interrupt__" in result, "Expected graph to pause at plan_approval"
        ipt = result["__interrupt__"][0]
        assert ipt.value["action"] == "approve_plan"
        assert ipt.value["plan"] is not None

    def test_plan_payload_contains_prep_plan_keys(self, monkeypatch):
        """Interrupt payload includes the full PrepPlan dict."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-pa-payload"}}

        result = app.invoke(_initial(), config=cfg)
        plan = result["__interrupt__"][0].value["plan"]

        for key in ("role_summary", "total_sessions", "sessions", "key_gaps", "rationale"):
            assert key in plan, f"PrepPlan key '{key}' missing from interrupt payload"

    def test_next_nodes_after_plan_approval_interrupt(self, monkeypatch):
        """get_state().next shows plan_approval is the pending node."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-pa-next"}}
        app.invoke(_initial(), config=cfg)

        snap = app.get_state(cfg)
        assert "plan_approval" in snap.next

    def test_resume_approve_continues_to_readiness(self, monkeypatch):
        """Resuming with 'approve' passes gate 1 and pauses at readiness (gate 2)."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-pa-approve"}}

        # Gate 1 interrupt
        app.invoke(_initial(), config=cfg)

        # Resume gate 1
        result2 = app.invoke(Command(resume={"decision": "approve"}), config=cfg)

        # Should now be at gate 2
        assert "__interrupt__" in result2
        ipt2 = result2["__interrupt__"][0]
        assert ipt2.value["action"] == "approve_verdict"

    def test_resume_approve_sets_plan_approved_true(self, monkeypatch):
        """plan_approved=True after human approves the plan."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-pa-approved-flag"}}

        app.invoke(_initial(), config=cfg)
        app.invoke(Command(resume={"decision": "approve"}), config=cfg)
        # Auto-approve readiness too to get final state
        result3 = app.invoke(Command(resume={"decision": "approve"}), config=cfg)

        assert result3["plan_approved"] is True

    def test_resume_edit_updates_plan_in_state(self, monkeypatch):
        """Resuming with 'edit' writes the human's updated_plan into state['plan']."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-pa-edit"}}

        app.invoke(_initial(), config=cfg)

        edited_plan = _STUB_PLAN.model_dump()
        edited_plan["rationale"] = "human-edited rationale"

        app.invoke(
            Command(resume={"decision": "edit", "updated_plan": edited_plan}),
            config=cfg,
        )
        # Gate 2 interrupt — but state should already carry the edited plan
        snap = app.get_state(cfg)
        assert snap.values.get("plan", {}).get("rationale") == "human-edited rationale"
        assert snap.values.get("plan_approved") is True

    def test_resume_reject_ends_graph_immediately(self, monkeypatch):
        """Resuming with 'reject' stops the graph — no interview, no grades."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-pa-reject"}}

        app.invoke(_initial(), config=cfg)
        result2 = app.invoke(Command(resume={"decision": "reject"}), config=cfg)

        # Graph ended — no more interrupts
        assert "__interrupt__" not in result2
        assert result2.get("plan_approved") is False
        # Interview never ran
        assert not result2.get("grades")
        assert result2.get("readiness_verdict") is None


# ── Gate 2: readiness ─────────────────────────────────────────────────────────


class TestReadinessGate:
    def _reach_gate2(self, monkeypatch, thread_id: str):
        """Helper: run through gate 1, return (app, cfg, gate2_result)."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": thread_id}}
        app.invoke(_initial(), config=cfg)
        result2 = app.invoke(Command(resume={"decision": "approve"}), config=cfg)
        return app, cfg, result2

    def test_gate2_pauses_at_readiness(self, monkeypatch):
        """After gate 1 approved, graph pauses at readiness with verdict payload."""
        _, _, result2 = self._reach_gate2(monkeypatch, "t-r-pause")

        assert "__interrupt__" in result2
        ipt = result2["__interrupt__"][0]
        assert ipt.value["action"] == "approve_verdict"
        assert "verdict" in ipt.value
        assert ipt.value["verdict"]["verdict"] in ("ready", "not_ready")

    def test_gate2_verdict_payload_has_all_fields(self, monkeypatch):
        """ReadinessVerdict payload has all required fields."""
        _, _, result2 = self._reach_gate2(monkeypatch, "t-r-fields")
        verdict = result2["__interrupt__"][0].value["verdict"]

        for key in ("verdict", "confidence", "strengths", "gaps", "recommendation"):
            assert key in verdict, f"Verdict key '{key}' missing from interrupt payload"

    def test_resume_approve_completes_graph(self, monkeypatch):
        """Resuming gate 2 with 'approve' completes the graph — no more interrupts."""
        app, cfg, _ = self._reach_gate2(monkeypatch, "t-r-complete")
        result3 = app.invoke(Command(resume={"decision": "approve"}), config=cfg)

        assert "__interrupt__" not in result3
        assert result3.get("readiness_verdict") is not None
        assert result3.get("verdict_approved") is True

    def test_resume_approve_stores_model_verdict(self, monkeypatch):
        """On approve, stored verdict matches the model's output (not overridden)."""
        app, cfg, result2 = self._reach_gate2(monkeypatch, "t-r-store-model")
        model_verdict = result2["__interrupt__"][0].value["verdict"]["verdict"]

        result3 = app.invoke(Command(resume={"decision": "approve"}), config=cfg)

        assert result3["readiness_verdict"]["verdict"] == model_verdict
        # No override_reason on a plain approve
        assert "override_reason" not in result3["readiness_verdict"]

    def test_resume_override_stores_human_verdict(self, monkeypatch):
        """On override, stored verdict reflects the human's choice and reason."""
        app, cfg, _ = self._reach_gate2(monkeypatch, "t-r-override")

        result3 = app.invoke(
            Command(
                resume={
                    "decision": "override",
                    "verdict": "ready",
                    "reason": "Strong system-design fundamentals outweigh coding gaps.",
                }
            ),
            config=cfg,
        )

        assert result3["readiness_verdict"]["verdict"] == "ready"
        assert result3["readiness_verdict"]["override_reason"] == (
            "Strong system-design fundamentals outweigh coding gaps."
        )
        assert result3.get("verdict_approved") is True

    def test_override_different_from_model(self, monkeypatch):
        """Override flips the verdict — human's choice wins over the model's."""
        # The stub model returns 'not_ready'; the human overrides to 'ready'
        app, cfg, result2 = self._reach_gate2(monkeypatch, "t-r-flip")
        model_verdict = result2["__interrupt__"][0].value["verdict"]["verdict"]

        opposite = "ready" if model_verdict == "not_ready" else "not_ready"
        result3 = app.invoke(
            Command(
                resume={"decision": "override", "verdict": opposite, "reason": "human knows best"}
            ),
            config=cfg,
        )

        assert result3["readiness_verdict"]["verdict"] == opposite
        assert result3["readiness_verdict"]["verdict"] != model_verdict


# ── Full two-gate flow ────────────────────────────────────────────────────────


class TestFullHITLFlow:
    def test_full_happy_path_approve_both_gates(self, monkeypatch):
        """Approve both gates — graph runs to completion with all fields set."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-full-happy"}}

        r1 = app.invoke(_initial(), config=cfg)
        assert "__interrupt__" in r1  # gate 1

        r2 = app.invoke(Command(resume={"decision": "approve"}), config=cfg)
        assert "__interrupt__" in r2  # gate 2

        r3 = app.invoke(Command(resume={"decision": "approve"}), config=cfg)
        assert "__interrupt__" not in r3  # done

        assert r3["plan_approved"] is True
        assert r3["verdict_approved"] is True
        assert r3["readiness_verdict"] is not None
        assert r3["grades"] is not None
        assert r3["weak_areas"] is not None

    def test_full_flow_state_persisted_across_resumes(self, monkeypatch):
        """get_state() after each resume shows cumulative state."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-full-persist"}}

        app.invoke(_initial(), config=cfg)
        app.invoke(Command(resume={"decision": "approve"}), config=cfg)
        app.invoke(Command(resume={"decision": "approve"}), config=cfg)

        final_snap = app.get_state(cfg)
        assert final_snap.values["plan"] is not None
        assert final_snap.values["plan_approved"] is True
        assert final_snap.values["readiness_verdict"] is not None
        assert final_snap.values["verdict_approved"] is True

    def test_two_threads_are_independent(self, monkeypatch):
        """Different thread_ids produce independent interrupt/resume cycles."""
        app = _build_app(monkeypatch)
        cfg_a = {"configurable": {"thread_id": "t-iso-A"}}
        cfg_b = {"configurable": {"thread_id": "t-iso-B"}}

        # Both hit gate 1
        app.invoke(_initial(), config=cfg_a)
        app.invoke(_initial(), config=cfg_b)

        # Approve A only
        app.invoke(Command(resume={"decision": "approve"}), config=cfg_a)

        # B is still at gate 1
        snap_b = app.get_state(cfg_b)
        assert "plan_approval" in snap_b.next

        # A is now at gate 2
        snap_a = app.get_state(cfg_a)
        assert "readiness" in snap_a.next
