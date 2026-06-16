"""
Graph assembly — Phase 5: interview loop + orchestration + memory + HITL.

Full topology:
  START → intake → planner → plan_approval [INTERRUPT] → session_router
    ─[conditional on current_modality]─►
      coding_interviewer  ─┐
      sd_interviewer       ├─► grader → coach → readiness [INTERRUPT] → END
      beh_interviewer     ─┘

Phase 5 additions:
  - plan_approval node: interrupt after planner — human approves/edits/rejects PrepPlan
  - readiness node: model verdict + interrupt — human approves or overrides readiness call
  - main() demonstrates the two-gate flow with auto-approve responses

Run with:  uv run python -m loop.graph
"""

import pathlib

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from loop.nodes.coach import coach
from loop.nodes.grader import grader
from loop.nodes.interviewers import beh_interviewer, coding_interviewer, sd_interviewer
from loop.nodes.planner import planner
from loop.nodes.readiness import readiness
from loop.observability import get_langfuse_callback
from loop.state import LoopState, initial_state

_FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"


# ── Intake node ───────────────────────────────────────────────────────────────


def intake(state: dict) -> dict:
    """Load JD and profile text from fixtures into state."""
    jd = (_FIXTURES / "sample_jd.md").read_text()
    profile = (_FIXTURES / "sample_profile.md").read_text()
    return {"jd": jd, "profile": profile}


# ── Session router node ───────────────────────────────────────────────────────


def session_router(state: dict) -> dict:
    """Pick the first session from the PrepPlan and set current_modality.

    Phase 3 runs one session end-to-end.  Multi-session looping (iterate
    over all sessions) is added in Phase 4+.
    """
    plan = state.get("plan") or {}
    sessions = plan.get("sessions") or []
    if not sessions:
        raise ValueError("PrepPlan has no sessions — planner node must run first")
    session = sessions[0]
    return {
        "current_modality": session["modality"],
        "session_number": session["session_number"],
    }


# ── Plan approval node (HITL gate 1) ─────────────────────────────────────────


def plan_approval(state: dict) -> Command:
    """Show the PrepPlan to the human and wait for approval.

    Calls interrupt() to pause the graph.  The human's response dict drives
    routing via Command(goto=...):

      {'decision': 'approve'}
          → Command(goto='session_router', update={'plan_approved': True})
      {'decision': 'edit', 'updated_plan': {...}}
          → Command(goto='session_router', update={'plan': ..., 'plan_approved': True})
      {'decision': 'reject'}
          → Command(goto=END, update={'plan_approved': False})

    IMPORTANT: this node has NO static edge defined — it always returns Command.
    A static edge would compete with Command(goto=END) and cause both paths to run.

    Analogy: a pull-request that can be approved, edited, or closed without merge.
    """
    human_response: dict = interrupt(
        {
            "action": "approve_plan",
            "plan": state.get("plan"),
        }
    )

    decision = human_response.get("decision", "approve")

    if decision == "reject":
        return Command(goto=END, update={"plan_approved": False})

    if decision == "edit":
        return Command(
            goto="session_router",
            update={"plan": human_response["updated_plan"], "plan_approved": True},
        )

    # 'approve'
    return Command(goto="session_router", update={"plan_approved": True})


# ── Routing function (used by conditional edge) ───────────────────────────────


def _route_by_modality(state: dict) -> str:
    """Return the node name to route to based on state["current_modality"].

    This function is the 'switch statement' of the conditional edge.
    LangGraph calls it after session_router runs, then follows the returned key
    through the path_map to find the next node.
    """
    modality = state.get("current_modality", "")
    valid = {"coding", "system_design", "behavioral"}
    if modality not in valid:
        raise ValueError(f"Unknown modality: {modality!r}. Must be one of {valid}")
    return modality


# ── Graph definition ──────────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """Define the Phase 5 graph topology (blueprint — not yet runnable)."""
    graph = StateGraph(LoopState)

    # Register all nodes
    graph.add_node("intake", intake)
    graph.add_node("planner", planner)
    graph.add_node("plan_approval", plan_approval)  # HITL gate 1
    graph.add_node("session_router", session_router)
    graph.add_node("coding_interviewer", coding_interviewer)
    graph.add_node("sd_interviewer", sd_interviewer)
    graph.add_node("beh_interviewer", beh_interviewer)
    graph.add_node("grader", grader)
    graph.add_node("coach", coach)
    graph.add_node("readiness", readiness)  # HITL gate 2

    # Fixed edges: START → intake → planner → plan_approval
    # plan_approval has NO static outgoing edge — it always returns Command(goto=...)
    # so routing is determined entirely by the human's decision at runtime.
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "planner")
    graph.add_edge("planner", "plan_approval")

    # Conditional edge: session_router → one of the three interviewers
    graph.add_conditional_edges(
        "session_router",
        _route_by_modality,
        path_map={
            "coding": "coding_interviewer",
            "system_design": "sd_interviewer",
            "behavioral": "beh_interviewer",
        },
    )

    # All interviewers converge on grader → coach → readiness → END
    graph.add_edge("coding_interviewer", "grader")
    graph.add_edge("sd_interviewer", "grader")
    graph.add_edge("beh_interviewer", "grader")
    graph.add_edge("grader", "coach")
    graph.add_edge("coach", "readiness")
    graph.add_edge("readiness", END)

    return graph


def compile_graph():
    """Compile the graph without checkpointer/store (used by tests).

    Tests stub nodes in loop.graph's namespace and call compile_graph() fresh
    each time — no thread_id required in the invoke config.
    """
    return build_graph().compile()


def compile_graph_with_memory():
    """Compile the graph with MemorySaver checkpointer + InMemoryStore.

    Required for production use:
    - invoke must pass config={'configurable': {'thread_id': '...', 'user_id': '...'}}
    - thread_id isolates graph state per session
    - user_id keys the long-term store (weak_areas, session_count)
    """
    from loop.memory import compile_with_memory

    return compile_with_memory(build_graph())


# ── Module-level compiled graph (with memory for the runner) ──────────────────
compiled = compile_graph_with_memory()


# ── Manual runner ─────────────────────────────────────────────────────────────


_CANNED_ANSWERS = [
    {
        "question_id": "cod-001",
        "text": (
            "Use a sliding window with a hash set to track characters. "
            "Right pointer advances; on duplicate, advance left pointer until "
            "duplicate removed. Track max window size. O(n) time, O(k) space."
        ),
    },
    {
        "question_id": "cod-002",
        "text": (
            "Use ReentrantLock with two Conditions: notFull and notEmpty. "
            "put() checks capacity, awaits notFull; take() checks empty, awaits "
            "notEmpty. Always use while-loops not if for spurious wakeups."
        ),
    },
    {
        "question_id": "sys-001",
        "text": (
            "API gateway accepts payment with idempotency key. Key stored in Redis "
            "for dedup. Async queue (Kafka) to payment processor. Outbox pattern for "
            "at-least-once delivery. PostgreSQL event-sourced audit log. Partition by "
            "merchant_id for scale."
        ),
    },
    {
        "question_id": "sys-002",
        "text": (
            "Token bucket in Redis with Lua script for atomic check-and-decrement. "
            "Fail-open on Redis outage. Sliding window log for accuracy-critical endpoints."
        ),
    },
    {
        "question_id": "beh-001",
        "text": (
            "Team chose MongoDB for a financial ledger. I disagreed on ACID grounds, "
            "wrote an RFC comparing Mongo vs Postgres, presented it. We switched to "
            "Postgres — no consistency issues since."
        ),
    },
    {
        "question_id": "beh-002",
        "text": (
            "A critical reconciliation job was failing silently with no owner. "
            "I traced the root cause (timezone bug), fixed it, added alerting. "
            "Zero missed reconciliations since; team adopted the pattern."
        ),
    },
]


def _run_session_with_hitl(user_id: str, thread_id: str) -> dict:
    """Run one full graph session through both HITL gates and return final result.

    The graph pauses twice:
      1. plan_approval gate — after planner produces a PrepPlan
      2. readiness gate    — after coach synthesizes feedback

    This runner auto-approves both gates (demo mode).
    In a real UI the human would inspect the payloads and respond interactively.
    """
    cb = get_langfuse_callback()
    callbacks = [cb] if cb else []

    state = initial_state()
    state["answers"] = list(_CANNED_ANSWERS)

    config = {
        "configurable": {"thread_id": thread_id, "user_id": user_id},
        "callbacks": callbacks,
    }

    # ── First invoke — runs until the first interrupt ──────────────
    result = compiled.invoke(state, config=config)

    # ── Gate 1: plan_approval ──────────────────────────────────────
    if "__interrupt__" in result:
        ipt = result["__interrupt__"][0]
        payload = ipt.value
        print(f"\n  [GATE 1 — {payload['action']}]")
        plan = payload.get("plan") or {}
        print(
            f"  PrepPlan: {plan.get('total_sessions', '?')} sessions, "
            f"gaps: {plan.get('key_gaps', [])[:2]}"
        )
        print("  → Auto-approving plan (demo mode)")

        # Resume: approve the plan as-is
        result = compiled.invoke(
            Command(resume={"decision": "approve"}),
            config=config,
        )

    # ── Gate 2: readiness ──────────────────────────────────────────
    if "__interrupt__" in result:
        ipt = result["__interrupt__"][0]
        payload = ipt.value
        verdict = payload.get("verdict") or {}
        print(f"\n  [GATE 2 — {payload['action']}]")
        print(
            f"  Readiness verdict: {verdict.get('verdict', '?')} "
            f"(confidence {verdict.get('confidence', 0):.0%})"
        )
        print(f"  Gaps: {verdict.get('gaps', [])[:2]}")
        print("  → Auto-approving verdict (demo mode)")

        # Resume: approve the verdict as-is
        result = compiled.invoke(
            Command(resume={"decision": "approve"}),
            config=config,
        )

    return result


def main() -> None:
    """Phase 5 demo: two-gate HITL flow + cross-session memory.

    Session 1: cold start — two approval gates, auto-approved.
    Session 2: planner reads weak_areas stored by session 1's coach.
    """
    print("Loop — Phase 5 graph run (HITL demo)")
    print("=" * 60)

    # ── Session 1 ─────────────────────────────────────────────────
    print("\n[ SESSION 1 — cold start ]\n")
    r1 = _run_session_with_hitl(user_id="kiran", thread_id="loop-s1")

    print(f"\n  Modality:        {r1.get('current_modality')}")
    print(f"  Question ID:     {r1.get('current_question_id')}")
    print(f"  Grade score:     {r1['grades'][0]['score'] if r1.get('grades') else 'n/a'}/10")
    print(f"  Plan approved:   {r1.get('plan_approved')}")
    rv = r1.get("readiness_verdict") or {}
    print(f"  Readiness:       {rv.get('verdict', 'n/a')}")
    print(f"  Verdict approved:{r1.get('verdict_approved')}")
    print(f"  Weak areas:      {r1.get('weak_areas')}")

    from loop.memory import get_store_instance

    item = get_store_instance().get(("loop", "users"), "kiran")
    if item:
        print(f"\n  Store after session 1: {item.value}")

    # ── Session 2 ─────────────────────────────────────────────────
    print("\n[ SESSION 2 — planner reads stored weak areas ]\n")
    r2 = _run_session_with_hitl(user_id="kiran", thread_id="loop-s2")

    print(f"\n  Modality:        {r2.get('current_modality')}")
    print(f"  Grade score:     {r2['grades'][0]['score'] if r2.get('grades') else 'n/a'}/10")
    rv2 = r2.get("readiness_verdict") or {}
    print(f"  Readiness:       {rv2.get('verdict', 'n/a')}")
    print(f"  Weak areas:      {r2.get('weak_areas')}")

    print("\nPhase 5 run complete.")


if __name__ == "__main__":
    main()
