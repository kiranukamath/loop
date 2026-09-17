"""
Agent simulation — Phase 17b.

A SimulatedCandidate auto-resumes every interrupt() the graph raises with a
scripted, deterministic response, driving a full multi-session curriculum
run unattended — through the exact same interrupt/resume machinery a real
UI uses (loop/graph.py's plan_approval, interviewer nodes, and readiness;
see also _run_session_with_hitl() there for the human-driven equivalent).

Why simulate, when trajectory_check.py already asserts the node path?
  A trajectory check proves the graph CAN take the right path when every
  interrupt is bypassed by hand, one gate at a time (the stub returns a
  Command directly instead of pausing). Agent simulation proves the graph's
  OWN interrupt/resume contract works end-to-end, unattended, across a real
  multi-session loop — exactly where flow bugs hide (a gate that never
  resumes cleanly, an interrupt payload shape the client can't parse, a
  session_index off-by-one across a loop iteration).

Offline vs. live (see PLAN.md Phase 17b):
  This phase builds the OFFLINE scripted candidate only — canned answers
  keyed by question_id, no model in the loop. A "live LLM candidate" (a
  model plays the candidate and free-forms each answer) is a natural server
  extension, like eval-gate Tier 2 — it would need Bedrock and is
  deliberately not built here; `uv run pytest` must stay network-free.

Run the demo:
    uv run python -m evals.simulate_session
"""

from __future__ import annotations

from typing import Any

from langgraph.types import Command

# ── The simulated candidate ────────────────────────────────────────────────────


class SimulatedCandidate:
    """Drives a compiled Loop graph to completion by auto-resuming every
    interrupt() it raises with a scripted, deterministic response.

    Loop's three interrupt actions (see loop/graph.py's plan_approval,
    loop/nodes/interviewers.py's _ask_question, loop/nodes/readiness.py) are:
      "approve_plan"    -> resume with a decision dict (default: approve)
      "answer_question" -> resume with an answer string, looked up by
                            question_id (falling back to a generic default
                            so an unscripted question doesn't stall the run)
      "approve_verdict" -> resume with a decision dict (default: approve)
    """

    def __init__(
        self,
        answers: dict[str, str] | None = None,
        default_answer: str = (
            "I'd approach this methodically: clarify requirements, consider "
            "the tradeoffs of the obvious approaches, then implement and "
            "verify with a couple of edge cases."
        ),
        plan_decision: dict | None = None,
        verdict_decision: dict | None = None,
    ) -> None:
        self.answers = answers or {}
        self.default_answer = default_answer
        self.plan_decision = plan_decision or {"decision": "approve"}
        self.verdict_decision = verdict_decision or {"decision": "approve"}

    def respond(self, interrupt_payload: dict) -> Any:
        """Return the scripted resume value for one interrupt() payload."""
        action = interrupt_payload.get("action")
        if action == "approve_plan":
            return self.plan_decision
        if action == "answer_question":
            qid = interrupt_payload.get("question_id")
            return self.answers.get(qid, self.default_answer)
        if action == "approve_verdict":
            return self.verdict_decision
        raise ValueError(
            f"SimulatedCandidate doesn't know how to resume interrupt action={action!r}"
        )

    def run_to_completion(
        self,
        app: Any,
        config: dict,
        initial_state: dict,
        max_turns: int = 25,
    ) -> dict:
        """Invoke `app`, then keep resuming interrupts with scripted
        responses until the graph stops interrupting (a curriculum run
        completed) or `max_turns` is hit.

        max_turns is a safety bound against an interrupt loop (e.g. a bug
        that re-raises the same interrupt forever) — same purpose as
        settings.research_max_iterations bounding the ReAct research agent.
        Returns the final state dict.
        """
        result = app.invoke(initial_state, config=config)
        turns = 1
        while "__interrupt__" in result:
            if turns >= max_turns:
                raise RuntimeError(
                    f"SimulatedCandidate exceeded max_turns={max_turns} without the "
                    "graph completing — likely an interrupt loop or an unscripted "
                    "action this candidate can't resume"
                )
            payload = result["__interrupt__"][0].value
            response = self.respond(payload)
            result = app.invoke(Command(resume=response), config=config)
            turns += 1
        return result


# ── Deterministic stubs for the model-calling nodes ────────────────────────────
# The candidate's job is to drive interrupts, not to exercise live Bedrock —
# that's what evals/ci_gate.py's live tier and run_grader_eval.py are for.
# Every node that pauses for a human (plan_approval, interviewers, readiness)
# stays REAL; only the LLM call inside planner/coach/readiness is stubbed,
# and grader is stubbed as a whole node (same pattern as
# tests/test_multisession.py) since its job here is "produce *a* grade", not
# exercise rubric grounding.


def _fake_structured_model(value: Any) -> Any:
    from unittest.mock import MagicMock

    from langchain_core.runnables import RunnableLambda

    model = MagicMock()
    model.with_structured_output.return_value = RunnableLambda(lambda _: value)
    return model


def _make_stub_plan():
    from loop.schemas import PrepPlan, Session

    return PrepPlan(
        role_summary="Simulated two-session curriculum for Phase 17b agent simulation.",
        total_sessions=2,
        sessions=[
            Session(
                session_number=1,
                modality="coding",
                topics=["arrays", "hashing"],
                focus="core data structure fluency",
            ),
            Session(
                session_number=2,
                modality="system_design",
                topics=["scalability"],
                focus="distributed systems tradeoffs",
            ),
        ],
        key_gaps=["dynamic programming"],
        rationale="Deterministic plan fixture — no model call.",
    )


def _stub_grader(state: dict) -> dict:
    """Whole-node stub (not just its model): produces a Grade keyed to
    whichever question_id the (real) interviewer node actually picked."""
    from loop.schemas import Grade

    qid = state.get("current_question_id", "unknown")
    grade = Grade(
        question_id=qid,
        score=7,
        criterion_scores={"overall": 7},
        strengths=["clear, structured reasoning"],
        improvements=["cover more edge cases"],
        overall_feedback="Solid answer — deterministic simulated grade.",
    )
    return {"grades": [grade.model_dump()]}


def _make_stub_feedback():
    from loop.schemas import Feedback

    return Feedback(
        summary="Good session — steady progress on the plan's focus areas.",
        action_items=["Practice a few more edge-case-heavy problems."],
        weak_areas_update=["edge-case handling"],
    )


def _make_stub_verdict():
    from loop.schemas import ReadinessVerdict

    return ReadinessVerdict(
        verdict="ready",
        confidence=0.8,
        strengths=["consistent, structured answers across both sessions"],
        gaps=[],
        recommendation="Ready to schedule real interviews.",
    )


def stub_llm_nodes():
    """Return an ExitStack of mock.patch(...) context managers stubbing
    every LLM-calling node this simulation exercises. Use as:

        with stub_llm_nodes():
            ...drive the graph...

    Kept as one helper so both the pytest test and the standalone demo below
    patch the identical set of seams.
    """
    from contextlib import ExitStack
    from unittest import mock

    stack = ExitStack()
    stack.enter_context(
        mock.patch(
            "loop.nodes.planner.get_chat_model",
            lambda: _fake_structured_model(_make_stub_plan()),
        )
    )
    stack.enter_context(mock.patch("loop.graph.grader", _stub_grader))
    stack.enter_context(
        mock.patch(
            "loop.nodes.coach.get_chat_model", lambda: _fake_structured_model(_make_stub_feedback())
        )
    )
    stack.enter_context(
        mock.patch(
            "loop.nodes.readiness.get_chat_model",
            lambda: _fake_structured_model(_make_stub_verdict()),
        )
    )
    return stack


# ── Runner ────────────────────────────────────────────────────────────────────

# Canned answers keyed by question_id — covers the fixture bank's coding and
# system_design questions (the two modalities the stub plan above uses).
# Any question_id not listed here falls back to SimulatedCandidate's generic
# default_answer, so this stays correct even if the fixture bank changes.
_SCRIPTED_ANSWERS = {
    "cod-001": (
        "Sliding window with a hash set tracking characters currently in the "
        "window. Advance the right pointer; on a duplicate, advance the left "
        "pointer until the duplicate is removed. Track the max window size "
        "seen. O(n) time, O(k) space where k is the charset size."
    ),
    "cod-002": (
        "Use a ReentrantLock with two Conditions, notFull and notEmpty. put() "
        "checks capacity and awaits notFull; take() checks emptiness and "
        "awaits notEmpty. Always loop on the condition (while, not if) to "
        "handle spurious wakeups."
    ),
    "sys-001": (
        "API gateway takes the payment request with a client-supplied "
        "idempotency key, deduped via Redis. Async queue (Kafka) hands off "
        "to the payment processor; an outbox table gives at-least-once "
        "delivery. Postgres holds an event-sourced audit log, partitioned by "
        "merchant_id for scale."
    ),
    "sys-002": (
        "Token bucket implemented in Redis via a Lua script for atomic "
        "check-and-decrement. Fail-open if Redis is unavailable so an outage "
        "doesn't take down the whole API. A sliding-window log for "
        "accuracy-critical endpoints where token-bucket bursts are unacceptable."
    ),
}


def run_simulated_session(
    user_id: str,
    thread_id: str,
    candidate: SimulatedCandidate | None = None,
    checkpointer: Any = None,
    store: Any = None,
) -> dict:
    """Compile Loop's real graph and drive one full curriculum run (a
    two-session plan) to completion with a SimulatedCandidate.

    Only the LLM calls are stubbed (see stub_llm_nodes()) — plan_approval,
    both interviewer interrupts, the multi-session loop, and readiness's
    interrupt are all real graph mechanics, exercised exactly as a live UI
    would exercise them.

    checkpointer/store default to a FRESH MemorySaver()/InMemoryStore() per
    call (not loop.memory's process-wide singletons) — same isolation every
    other graph test in this repo uses (e.g. tests/test_hitl.py compiles
    with its own `MemorySaver()` per test). Deliberately NOT
    compile_with_memory(): that reads settings.db_path, so on a machine
    whose .env points db_path at a real SqliteSaver file, two calls with the
    same thread_id would resume EACH OTHER's persisted state across
    separate runs — exactly the kind of environment-dependent flakiness the
    laptop test gate must not have. Pass an explicit checkpointer/store (see
    run_multi_session_simulation) to opt into real cross-session memory.
    """
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.store.memory import InMemoryStore

    from loop.graph import build_graph
    from loop.state import initial_state

    candidate = candidate or SimulatedCandidate(answers=_SCRIPTED_ANSWERS)
    checkpointer = checkpointer if checkpointer is not None else MemorySaver()
    store = store if store is not None else InMemoryStore()

    with stub_llm_nodes():
        app = build_graph().compile(checkpointer=checkpointer, store=store)
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        return candidate.run_to_completion(app, config, initial_state())


def run_multi_session_simulation(user_id: str, thread_ids: list[str]) -> list[dict]:
    """Drive several separate curriculum runs (one per thread_id) for the
    same user_id, sharing ONE checkpointer + store across them — the same
    cross-session pattern loop/graph.py's main() demo exercises (session 1's
    coach writes weak_areas; session 2's planner reads them), just driven by
    a scripted candidate instead of auto-approving a human runner.
    """
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.store.memory import InMemoryStore

    checkpointer = MemorySaver()
    store = InMemoryStore()
    return [
        run_simulated_session(user_id, thread_id, checkpointer=checkpointer, store=store)
        for thread_id in thread_ids
    ]


def main() -> None:
    """Offline demo: one 2-session simulated curriculum run, printed."""
    from unittest import mock

    from langchain_core.embeddings.fake import DeterministicFakeEmbedding

    import loop.retrieval as retrieval_mod
    from loop.reranker import FakeReranker

    print("Loop — Phase 17b agent simulation (offline)")
    print("=" * 60)

    # run_simulated_session() builds its own fresh, unindexed InMemoryStore
    # (see its docstring), so only the interviewer's retrieval pipeline needs
    # fake embeddings/reranker here.
    with (
        mock.patch("loop.retrieval.get_embeddings", lambda: DeterministicFakeEmbedding(size=256)),
        mock.patch("loop.retrieval.get_reranker", lambda: FakeReranker()),
    ):
        retrieval_mod._build_index()
        result = run_simulated_session(user_id="sim-candidate", thread_id="sim-thread-1")

    print(f"\nSessions completed: {result.get('session_index')}")
    print(f"Grades: {[g['score'] for g in result.get('grades', [])]}")
    verdict = result.get("readiness_verdict") or {}
    print(f"Readiness verdict: {verdict.get('verdict')} (confidence {verdict.get('confidence')})")
    print(f"Verdict approved: {result.get('verdict_approved')}")
    print("\nSimulation complete.")


if __name__ == "__main__":
    main()
