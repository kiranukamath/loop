"""
Graph assembly — Phase 3: interview loop + orchestration.

Full topology:
  START → intake → planner → session_router
    ─[conditional on current_modality]─►
      coding_interviewer  ─┐
      sd_interviewer       ├─► grader → coach → END
      beh_interviewer     ─┘

session_router reads the first session from the PrepPlan and sets
current_modality.  The conditional edge reads current_modality and routes to
the matching interviewer.

Run with:  uv run python -m loop.graph
"""

import json
import pathlib

from langgraph.graph import END, START, StateGraph

from loop.nodes.coach import coach
from loop.nodes.grader import grader
from loop.nodes.interviewers import beh_interviewer, coding_interviewer, sd_interviewer
from loop.nodes.planner import planner
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
    """Define the Phase 3 graph topology (blueprint — not yet runnable)."""
    graph = StateGraph(LoopState)

    # Register all nodes
    graph.add_node("intake", intake)
    graph.add_node("planner", planner)
    graph.add_node("session_router", session_router)
    graph.add_node("coding_interviewer", coding_interviewer)
    graph.add_node("sd_interviewer", sd_interviewer)
    graph.add_node("beh_interviewer", beh_interviewer)
    graph.add_node("grader", grader)
    graph.add_node("coach", coach)

    # Fixed edges: linear chain up to session_router
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "planner")
    graph.add_edge("planner", "session_router")

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

    # All interviewers converge on grader → coach → END
    graph.add_edge("coding_interviewer", "grader")
    graph.add_edge("sd_interviewer", "grader")
    graph.add_edge("beh_interviewer", "grader")
    graph.add_edge("grader", "coach")
    graph.add_edge("coach", END)

    return graph


def compile_graph():
    """Compile the graph into a runnable CompiledStateGraph."""
    return build_graph().compile()


# ── Module-level compiled graph ───────────────────────────────────────────────
compiled = compile_graph()


# ── Manual runner ─────────────────────────────────────────────────────────────


def main() -> None:
    """Invoke the graph with a canned answer and pretty-print the result."""
    print("Loop — Phase 3 graph run")
    print("Invoking: START → intake → planner → session_router → interviewer → grader → coach\n")

    cb = get_langfuse_callback()
    config = {"callbacks": [cb]} if cb else {}

    # Pre-inject canned answers for every fixture question so the grader always
    # has a matching answer regardless of which modality the planner chooses.
    # Phase 5 will replace this with a real human interrupt.
    state = initial_state()
    state["answers"] = [
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
                "for dedup before processing. Async queue (Kafka) to payment processor. "
                "Outbox pattern for at-least-once delivery. PostgreSQL with event-sourced "
                "audit log. Horizontal scaling via partition by merchant_id."
            ),
        },
        {
            "question_id": "sys-002",
            "text": (
                "Token bucket in Redis with Lua script for atomic check-and-decrement. "
                "Fail-open on Redis outage (accept traffic, log for post-hoc audit). "
                "Sliding window log for accuracy-critical endpoints. Per-user key with "
                "TTL equal to the window size."
            ),
        },
        {
            "question_id": "beh-001",
            "text": (
                "Situation: team decided to use MongoDB for a financial ledger. "
                "Task: I disagreed due to ACID concerns. Actions: wrote a short RFC "
                "comparing Mongo vs Postgres for our access patterns, presented to "
                "the team. Result: we switched to Postgres; no data consistency issues "
                "in production."
            ),
        },
        {
            "question_id": "beh-002",
            "text": (
                "Situation: a critical payment reconciliation job was failing silently "
                "with no owner. Task: I volunteered to fix it. Actions: traced root "
                "cause to a timezone bug, fixed it, added monitoring and alerting. "
                "Result: zero missed reconciliations since; team adopted the alerting "
                "pattern for other jobs."
            ),
        },
    ]

    result = compiled.invoke(state, config=config)

    print(f"jd loaded:           {len(result['jd'])} chars")
    print(f"current_modality:    {result['current_modality']}")
    print(f"current_question_id: {result['current_question_id']}")
    print("\n── PrepPlan (first session) ──")
    if result["plan"] and result["plan"].get("sessions"):
        print(json.dumps(result["plan"]["sessions"][0], indent=2))

    print("\n── Grade ──")
    if result["grades"]:
        print(json.dumps(result["grades"][0], indent=2))

    print(f"\n── Weak areas ──\n{result.get('weak_areas')}")
    print("\nGraph run complete.")


if __name__ == "__main__":
    main()
