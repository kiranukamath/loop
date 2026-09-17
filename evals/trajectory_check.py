"""
Trajectory check — verify a recorded graph run visited the expected nodes.

What is a trajectory check?
  An output-quality eval (grader eval) checks "was the answer good?"
  A trajectory check asks a different question: "did the agent take the right path?"
  These are independent — an agent can produce good output via a wrong path,
  or take the right path and still produce poor output.

  Analogy: in distributed tracing you might assert that an API call touched the
  auth service before the business logic service, regardless of the response body.

How it works:
  LangGraph checkpoints every node's state to the checkpointer.
  app.get_state_history(cfg) returns all saved snapshots, newest-first.
  Each snapshot has a `.next` field: the node(s) about to run after this snapshot.
  Reading snapshots in chronological order, collecting `.next[0]`, gives the
  sequence of nodes that ran — the trajectory.

Run the demo:
    uv run python -m evals.trajectory_check

This demo runs a fully stubbed graph (no Bedrock, no Langfuse) and asserts
the trajectory. It's also used by tests/test_evals.py.
"""

from __future__ import annotations

from typing import Any

# Nodes that appear in `snap.next` but are graph-internal, not real nodes.
_SYSTEM_NODES = {"__start__", "__end__", "__resume__"}

# The expected trajectory for a normal Loop run (both gates approved).
# Modality in position 4 varies — the check allows any of the three interviewers.
EXPECTED_TRAJECTORY_PREFIX = [
    "intake",
    "planner",
    "plan_approval",
    "session_router",
]
INTERVIEWER_NODES = {"coding_interviewer", "sd_interviewer", "beh_interviewer"}
EXPECTED_TRAJECTORY_SUFFIX = [
    "grader",
    "coach",
    "advance_session",
    "readiness",
    "reflect",  # Phase 16b: curriculum-boundary consolidation, always in the
    # topology (no-op when settings.reflection_enabled is off)
]


def extract_trajectory(app: Any, cfg: dict) -> list[str]:
    """Return the ordered list of graph nodes that ran for a given thread.

    Reads checkpointed state history and derives the visit sequence from each
    snapshot's `.next` field (what node was about to run after that snapshot).

    Newest-first snapshots are reversed to get chronological order.
    System-internal node names are filtered out.
    """
    history = list(app.get_state_history(cfg))
    history.reverse()  # oldest first = chronological

    visited: list[str] = []
    for snap in history:
        for node in snap.next:
            if node not in _SYSTEM_NODES:
                visited.append(node)
    return visited


def assert_trajectory(trajectory: list[str]) -> None:
    """Assert the trajectory matches the expected Loop node sequence.

    Raises AssertionError with a clear message if it doesn't match.
    """
    n = len(trajectory)

    # 1. Check prefix (intake → planner → plan_approval → session_router)
    prefix_len = len(EXPECTED_TRAJECTORY_PREFIX)
    actual_prefix = trajectory[:prefix_len]
    assert actual_prefix == EXPECTED_TRAJECTORY_PREFIX, (
        f"Trajectory prefix mismatch.\n"
        f"  Expected : {EXPECTED_TRAJECTORY_PREFIX}\n"
        f"  Got      : {actual_prefix}\n"
        f"  Full     : {trajectory}"
    )

    # 2. Position 4 must be one of the three interviewer nodes
    if n <= prefix_len:
        raise AssertionError(
            f"Trajectory too short — missing interviewer node.\n  Got: {trajectory}"
        )
    interviewer = trajectory[prefix_len]
    assert interviewer in INTERVIEWER_NODES, (
        f"Expected an interviewer node at position {prefix_len}, got {interviewer!r}.\n"
        f"Full trajectory: {trajectory}"
    )

    # 3. Check suffix (grader → coach → readiness)
    suffix_len = len(EXPECTED_TRAJECTORY_SUFFIX)
    actual_suffix = trajectory[prefix_len + 1 : prefix_len + 1 + suffix_len]
    assert actual_suffix == EXPECTED_TRAJECTORY_SUFFIX, (
        f"Trajectory suffix mismatch.\n"
        f"  Expected : {EXPECTED_TRAJECTORY_SUFFIX}\n"
        f"  Got      : {actual_suffix}\n"
        f"  Full     : {trajectory}"
    )

    # 4. Total length must be exactly prefix + interviewer + suffix
    expected_len = prefix_len + 1 + suffix_len
    assert n == expected_len, (
        f"Trajectory length mismatch: expected {expected_len} nodes, got {n}.\n"
        f"Full trajectory: {trajectory}"
    )


# ── Demo runner ───────────────────────────────────────────────────────────────


def _run_demo() -> None:
    """Run the full graph with stubbed nodes and print + assert the trajectory."""
    from unittest.mock import MagicMock

    from langchain_core.runnables import RunnableLambda
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from loop.graph import build_graph
    from loop.schemas import Feedback, Grade, PrepPlan, ReadinessVerdict, Session
    from loop.state import initial_state

    stub_plan = PrepPlan(
        role_summary="demo",
        total_sessions=1,
        sessions=[Session(session_number=1, modality="coding", topics=["x"], focus="y")],
        key_gaps=[],
        rationale="demo",
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
        strengths=["good"],
        improvements=["edge cases"],
        overall_feedback="Solid.",
    )
    stub_verdict = ReadinessVerdict(
        verdict="ready",
        confidence=0.8,
        strengths=["strong fundamentals"],
        gaps=[],
        recommendation="Go for it.",
    )
    stub_feedback = Feedback(
        summary="Good session.",
        action_items=["review edge cases"],
        weak_areas_update=["edge-case handling"],
    )

    def fake_model(rv):
        m = MagicMock()
        m.with_structured_output.return_value = RunnableLambda(lambda _: rv)
        return m

    # Build the graph with real plan_approval and readiness (the HITL nodes)
    from unittest.mock import patch

    with (
        patch("loop.nodes.planner.get_chat_model", lambda: fake_model(stub_plan)),
        patch("loop.nodes.grader.get_chat_model", lambda: fake_model(stub_grade)),
        patch("loop.nodes.coach.get_chat_model", lambda: fake_model(stub_feedback)),
        patch("loop.nodes.readiness.get_chat_model", lambda: fake_model(stub_verdict)),
    ):
        app = build_graph().compile(checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "traj-demo"}}
        state = initial_state()
        state["answers"] = [{"question_id": "cod-001", "text": "sliding window..."}]

        # First invoke — pauses at plan_approval
        app.invoke(state, config=cfg)
        # Resume gate 1
        app.invoke(Command(resume={"decision": "approve"}), config=cfg)
        # Resume gate 2
        app.invoke(Command(resume={"decision": "approve"}), config=cfg)

        traj = extract_trajectory(app, cfg)

    print("Trajectory:")
    for i, node in enumerate(traj):
        print(f"  {i + 1:2d}. {node}")

    assert_trajectory(traj)
    print("\nTrajectory check PASSED ✓")


if __name__ == "__main__":
    _run_demo()
