"""
Phase 5 + 7b HITL tests — interrupt/resume flow for all three approval gates.
All offline: model calls are stubbed.

Three interrupt gates in order:
  Gate 1 — plan_approval  : human approves / edits / rejects the PrepPlan
  Gate 2 — answer_question: human types their answer to the interviewer's question
  Gate 3 — readiness      : human approves / overrides the readiness verdict

Tests verify:
- Gate 1 pauses with the PrepPlan in the interrupt payload
- Resuming gate 1 with 'approve' passes through to gate 2 (answer gate)
- Resuming gate 1 with 'edit' updates state['plan']
- Resuming gate 1 with 'reject' ends the graph without running the interview
- Gate 2 pauses with the question payload (action, question_id, question_prompt)
- Resuming gate 2 stores the answer in state['answers'] and reaches gate 3
- Gate 3 pauses with the readiness verdict payload
- Resuming gate 3 with 'approve' stores the model's verdict
- Resuming gate 3 with 'override' stores the human's choice
- Full 3-gate happy path completes with all fields set
- Thread isolation: two threads' interrupt/resume cycles are independent
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

_CANNED_ANSWER = "Use a sliding window with a hash set. O(n) time, O(k) space."


def _fake_model(return_value):
    from unittest.mock import MagicMock

    fake = MagicMock()
    fake.with_structured_output.return_value = RunnableLambda(lambda _: return_value)
    return fake


def _build_app(monkeypatch):
    """Build and compile the graph with MemorySaver; stub all model-calling nodes.

    plan_approval, the interviewers (answer gate), and readiness all use their
    real implementations so HITL tests exercise the actual pause/resume mechanics.
    Grader, coach model, and readiness model are stubbed.
    """
    monkeypatch.setattr("loop.nodes.planner.get_chat_model", lambda: _fake_model(_STUB_PLAN))
    monkeypatch.setattr("loop.graph.grader", lambda s: {"grades": [_STUB_GRADE.model_dump()]})
    monkeypatch.setattr("loop.nodes.coach.get_chat_model", lambda: _fake_model(_STUB_FEEDBACK))
    monkeypatch.setattr("loop.nodes.readiness.get_chat_model", lambda: _fake_model(_STUB_VERDICT))

    from loop.graph import build_graph

    return build_graph().compile(checkpointer=MemorySaver())


def _initial():
    """Return a blank initial state — no pre-injected answers (interviewer provides them)."""
    from loop.state import initial_state

    return initial_state()


# ── Helpers to advance through gates ─────────────────────────────────────────


def _reach_answer_gate(app, cfg):
    """Run graph to gate 1, approve plan — returns result paused at gate 2."""
    app.invoke(_initial(), config=cfg)
    return app.invoke(Command(resume={"decision": "approve"}), config=cfg)


def _reach_readiness_gate(app, cfg):
    """Run graph through gates 1 and 2, arrive at gate 3 (readiness)."""
    _reach_answer_gate(app, cfg)
    return app.invoke(Command(resume=_CANNED_ANSWER), config=cfg)


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

    def test_resume_approve_passes_gate1_and_reaches_answer_gate(self, monkeypatch):
        """Resuming gate 1 with 'approve' passes the plan and pauses at answer gate."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-pa-approve"}}

        app.invoke(_initial(), config=cfg)
        result2 = app.invoke(Command(resume={"decision": "approve"}), config=cfg)

        assert "__interrupt__" in result2
        assert result2["__interrupt__"][0].value["action"] == "answer_question"

    def test_resume_approve_sets_plan_approved_true(self, monkeypatch):
        """plan_approved=True is set after the human approves the plan."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-pa-approved-flag"}}

        app.invoke(_initial(), config=cfg)
        app.invoke(Command(resume={"decision": "approve"}), config=cfg)
        app.invoke(Command(resume=_CANNED_ANSWER), config=cfg)
        result4 = app.invoke(Command(resume={"decision": "approve"}), config=cfg)

        assert result4["plan_approved"] is True

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
        # Paused at answer gate — but state should already carry the edited plan
        snap = app.get_state(cfg)
        assert snap.values.get("plan", {}).get("rationale") == "human-edited rationale"
        assert snap.values.get("plan_approved") is True

    def test_resume_reject_ends_graph_immediately(self, monkeypatch):
        """Resuming with 'reject' stops the graph — no interview, no grades."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-pa-reject"}}

        app.invoke(_initial(), config=cfg)
        result2 = app.invoke(Command(resume={"decision": "reject"}), config=cfg)

        assert "__interrupt__" not in result2
        assert result2.get("plan_approved") is False
        assert not result2.get("grades")
        assert result2.get("readiness_verdict") is None


# ── Gate 2: answer_question ───────────────────────────────────────────────────


class TestAnswerGate:
    def test_answer_gate_pauses_with_question_payload(self, monkeypatch):
        """After plan approved, graph pauses at interviewer with question payload."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-aq-pause"}}

        result = _reach_answer_gate(app, cfg)

        assert "__interrupt__" in result
        ipt = result["__interrupt__"][0]
        assert ipt.value["action"] == "answer_question"

    def test_answer_gate_payload_has_question_fields(self, monkeypatch):
        """Question payload contains question_id, title, and prompt."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-aq-fields"}}

        result = _reach_answer_gate(app, cfg)
        payload = result["__interrupt__"][0].value

        for key in ("question_id", "question_title", "question_prompt"):
            assert key in payload, f"Answer gate payload missing key: {key!r}"

    def test_answer_gate_question_is_coding_modality(self, monkeypatch):
        """Plan has one coding session — answer gate should ask a coding question."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-aq-modality"}}

        result = _reach_answer_gate(app, cfg)
        qid = result["__interrupt__"][0].value["question_id"]

        from loop.tools import get_question_by_id

        q = get_question_by_id(qid)
        assert q["modality"] == "coding"

    def test_next_node_at_answer_gate(self, monkeypatch):
        """get_state().next shows the interviewer node as pending."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-aq-next"}}
        _reach_answer_gate(app, cfg)

        snap = app.get_state(cfg)
        # The pending node is whichever interviewer ran (coding_interviewer for coding plan)
        interviewer_nodes = {"coding_interviewer", "sd_interviewer", "beh_interviewer"}
        assert snap.next[0] in interviewer_nodes

    def test_resume_answer_stores_in_state(self, monkeypatch):
        """Resuming with answer text stores it in state['answers']."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-aq-store"}}

        _reach_answer_gate(app, cfg)
        result = app.invoke(Command(resume=_CANNED_ANSWER), config=cfg)

        # Graph should now be at readiness gate (gate 3)
        assert "__interrupt__" in result
        assert result["__interrupt__"][0].value["action"] == "approve_verdict"

        # Answer is in state
        answers = result.get("answers") or []
        assert len(answers) == 1
        assert answers[0]["text"] == _CANNED_ANSWER

    def test_resume_answer_then_grade_appears(self, monkeypatch):
        """After answering, grader runs and grade lands in state."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-aq-grade"}}

        _reach_answer_gate(app, cfg)
        result = app.invoke(Command(resume=_CANNED_ANSWER), config=cfg)

        grades = result.get("grades") or []
        assert len(grades) == 1
        assert grades[0]["score"] == _STUB_GRADE.score


# ── Gate 3: readiness ─────────────────────────────────────────────────────────


class TestReadinessGate:
    def test_gate3_pauses_at_readiness(self, monkeypatch):
        """After answer given, graph pauses at readiness with verdict payload."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-r-pause"}}

        result = _reach_readiness_gate(app, cfg)

        assert "__interrupt__" in result
        ipt = result["__interrupt__"][0]
        assert ipt.value["action"] == "approve_verdict"
        assert "verdict" in ipt.value
        assert ipt.value["verdict"]["verdict"] in ("ready", "not_ready")

    def test_gate3_verdict_payload_has_all_fields(self, monkeypatch):
        """ReadinessVerdict payload has all required fields."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-r-fields"}}

        result = _reach_readiness_gate(app, cfg)
        verdict = result["__interrupt__"][0].value["verdict"]

        for key in ("verdict", "confidence", "strengths", "gaps", "recommendation"):
            assert key in verdict, f"Verdict key '{key}' missing from interrupt payload"

    def test_resume_approve_completes_graph(self, monkeypatch):
        """Resuming gate 3 with 'approve' completes the graph — no more interrupts."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-r-complete"}}

        _reach_readiness_gate(app, cfg)
        result4 = app.invoke(Command(resume={"decision": "approve"}), config=cfg)

        assert "__interrupt__" not in result4
        assert result4.get("readiness_verdict") is not None
        assert result4.get("verdict_approved") is True

    def test_resume_approve_stores_model_verdict(self, monkeypatch):
        """On approve, stored verdict matches the model's output (not overridden)."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-r-store-model"}}

        result3 = _reach_readiness_gate(app, cfg)
        model_verdict = result3["__interrupt__"][0].value["verdict"]["verdict"]

        result4 = app.invoke(Command(resume={"decision": "approve"}), config=cfg)

        assert result4["readiness_verdict"]["verdict"] == model_verdict
        assert "override_reason" not in result4["readiness_verdict"]

    def test_resume_override_stores_human_verdict(self, monkeypatch):
        """On override, stored verdict reflects the human's choice and reason."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-r-override"}}

        _reach_readiness_gate(app, cfg)
        result4 = app.invoke(
            Command(
                resume={
                    "decision": "override",
                    "verdict": "ready",
                    "reason": "Strong system-design fundamentals outweigh coding gaps.",
                }
            ),
            config=cfg,
        )

        assert result4["readiness_verdict"]["verdict"] == "ready"
        assert result4["readiness_verdict"]["override_reason"] == (
            "Strong system-design fundamentals outweigh coding gaps."
        )
        assert result4.get("verdict_approved") is True

    def test_override_different_from_model(self, monkeypatch):
        """Override flips the verdict — human's choice wins over the model's."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-r-flip"}}

        result3 = _reach_readiness_gate(app, cfg)
        model_verdict = result3["__interrupt__"][0].value["verdict"]["verdict"]
        opposite = "ready" if model_verdict == "not_ready" else "not_ready"

        result4 = app.invoke(
            Command(
                resume={
                    "decision": "override",
                    "verdict": opposite,
                    "reason": "human knows best",
                }
            ),
            config=cfg,
        )

        assert result4["readiness_verdict"]["verdict"] == opposite
        assert result4["readiness_verdict"]["verdict"] != model_verdict


# ── Full 3-gate flow ──────────────────────────────────────────────────────────


class TestFullHITLFlow:
    def test_full_happy_path_approve_all_gates(self, monkeypatch):
        """Approve all three gates — graph runs to completion with all fields set."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-full-happy"}}

        r1 = app.invoke(_initial(), config=cfg)
        assert "__interrupt__" in r1  # gate 1: plan
        assert r1["__interrupt__"][0].value["action"] == "approve_plan"

        r2 = app.invoke(Command(resume={"decision": "approve"}), config=cfg)
        assert "__interrupt__" in r2  # gate 2: answer
        assert r2["__interrupt__"][0].value["action"] == "answer_question"

        r3 = app.invoke(Command(resume=_CANNED_ANSWER), config=cfg)
        assert "__interrupt__" in r3  # gate 3: readiness
        assert r3["__interrupt__"][0].value["action"] == "approve_verdict"

        r4 = app.invoke(Command(resume={"decision": "approve"}), config=cfg)
        assert "__interrupt__" not in r4  # done

        assert r4["plan_approved"] is True
        assert r4["verdict_approved"] is True
        assert r4["readiness_verdict"] is not None
        assert r4["grades"] is not None
        assert r4["weak_areas"] is not None
        assert len(r4["answers"]) == 1
        assert r4["answers"][0]["text"] == _CANNED_ANSWER

    def test_full_flow_state_persisted_across_resumes(self, monkeypatch):
        """get_state() after final resume shows all cumulative state."""
        app = _build_app(monkeypatch)
        cfg = {"configurable": {"thread_id": "t-full-persist"}}

        app.invoke(_initial(), config=cfg)
        app.invoke(Command(resume={"decision": "approve"}), config=cfg)
        app.invoke(Command(resume=_CANNED_ANSWER), config=cfg)
        app.invoke(Command(resume={"decision": "approve"}), config=cfg)

        final_snap = app.get_state(cfg)
        assert final_snap.values["plan"] is not None
        assert final_snap.values["plan_approved"] is True
        assert final_snap.values["readiness_verdict"] is not None
        assert final_snap.values["verdict_approved"] is True
        assert final_snap.values["answers"] is not None

    def test_two_threads_are_independent(self, monkeypatch):
        """Different thread_ids produce independent interrupt/resume cycles."""
        app = _build_app(monkeypatch)
        cfg_a = {"configurable": {"thread_id": "t-iso-A"}}
        cfg_b = {"configurable": {"thread_id": "t-iso-B"}}

        # Both hit gate 1
        app.invoke(_initial(), config=cfg_a)
        app.invoke(_initial(), config=cfg_b)

        # Advance A through gate 1 only
        app.invoke(Command(resume={"decision": "approve"}), config=cfg_a)

        # B is still at gate 1
        snap_b = app.get_state(cfg_b)
        assert "plan_approval" in snap_b.next

        # A is now at gate 2 (answer gate — inside the interviewer node)
        snap_a = app.get_state(cfg_a)
        interviewer_nodes = {"coding_interviewer", "sd_interviewer", "beh_interviewer"}
        assert snap_a.next[0] in interviewer_nodes
