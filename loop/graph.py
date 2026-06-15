"""
Graph assembly — Phase 4: interview loop + orchestration + memory.

Full topology:
  START → intake → planner → session_router
    ─[conditional on current_modality]─►
      coding_interviewer  ─┐
      sd_interviewer       ├─► grader → coach → END
      beh_interviewer     ─┘

Phase 4 additions:
  - compile_graph_with_memory(): attaches checkpointer + store via loop.memory
  - planner reads cross-session weak_areas from the store
  - coach writes weak_areas to the store after each session
  - main() demonstrates two sessions showing the feedback loop

Run with:  uv run python -m loop.graph
"""

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


def _run_session(session_label: str, user_id: str, thread_id: str) -> dict:
    """Run one full graph session and return the result."""
    cb = get_langfuse_callback()
    callbacks = [cb] if cb else []

    state = initial_state()
    state["answers"] = list(_CANNED_ANSWERS)

    # thread_id = unique per session (state isolation)
    # user_id   = stable per user (store lookup / write)
    config = {
        "configurable": {"thread_id": thread_id, "user_id": user_id},
        "callbacks": callbacks,
    }
    return compiled.invoke(state, config=config)


def main() -> None:
    """Two-session demo showing the memory feedback loop.

    Session 1: cold start — planner sees no stored weak_areas.
    Session 2: planner reads weak_areas stored by session 1's coach.
    The second PrepPlan should focus on the areas identified as weak in session 1.
    """
    print("Loop — Phase 4 graph run (memory demo)")
    print("=" * 60)

    # ── Session 1 ─────────────────────────────────────────────────
    print("\n[ SESSION 1 — cold start, no stored weak areas ]\n")
    r1 = _run_session("session-1", user_id="kiran", thread_id="loop-session-1")

    print(f"Modality:     {r1['current_modality']}")
    print(f"Question ID:  {r1['current_question_id']}")
    print(f"Grade score:  {r1['grades'][0]['score'] if r1.get('grades') else 'n/a'}/10")
    print(f"Weak areas written to store: {r1.get('weak_areas')}")

    # Check what's now in the store
    from loop.memory import get_store_instance

    item = get_store_instance().get(("loop", "users"), "kiran")
    if item:
        print(f"\nStore after session 1: {item.value}")

    # ── Session 2 ─────────────────────────────────────────────────
    print("\n[ SESSION 2 — planner reads stored weak areas ]\n")
    r2 = _run_session("session-2", user_id="kiran", thread_id="loop-session-2")

    print(f"Modality:     {r2['current_modality']}")
    print(f"Question ID:  {r2['current_question_id']}")
    print(f"Grade score:  {r2['grades'][0]['score'] if r2.get('grades') else 'n/a'}/10")
    print(f"Weak areas written to store: {r2.get('weak_areas')}")

    item2 = get_store_instance().get(("loop", "users"), "kiran")
    if item2:
        print(f"\nStore after session 2: {item2.value}")

    # ── Thread isolation check ─────────────────────────────────────
    print("\n[ Thread isolation check ]")
    s1 = compiled.get_state({"configurable": {"thread_id": "loop-session-1"}})
    s2 = compiled.get_state({"configurable": {"thread_id": "loop-session-2"}})
    plan1_sessions = s1.values.get("plan", {}).get("total_sessions", "?") if s1 else "?"
    plan2_sessions = s2.values.get("plan", {}).get("total_sessions", "?") if s2 else "?"
    print(f"Session-1 saved plan total_sessions: {plan1_sessions}")
    print(f"Session-2 saved plan total_sessions: {plan2_sessions}")
    print("\nPhase 4 run complete.")


if __name__ == "__main__":
    main()
