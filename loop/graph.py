"""
Graph assembly — Phase 2: intake → planner → END.

Phase 1 proved the graph mechanics with no LLM.
Phase 2 adds the planner node: the first real model call, returning a
structured PrepPlan via with_structured_output().

Run with:  uv run python -m loop.graph
"""

import json
import pathlib

from langgraph.graph import END, START, StateGraph

from loop.nodes.planner import planner
from loop.observability import get_langfuse_callback
from loop.state import LoopState, initial_state

# ── Fixtures directory ────────────────────────────────────────────────────────
_FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"


# ── Node functions ────────────────────────────────────────────────────────────


def intake(state: dict) -> dict:
    """Load JD and profile text from fixtures into state.

    Node contract: receives full state, returns ONLY the keys it changes.
    LangGraph merges the returned dict back into state before the next node.
    """
    jd = (_FIXTURES / "sample_jd.md").read_text()
    profile = (_FIXTURES / "sample_profile.md").read_text()
    return {"jd": jd, "profile": profile}


# ── Graph definition ──────────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """Define the graph topology (blueprint — not yet runnable).

    Phase 2 topology:  START → intake → planner → END
    """
    graph = StateGraph(LoopState)

    graph.add_node("intake", intake)
    graph.add_node("planner", planner)  # new in Phase 2

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "planner")  # intake feeds planner
    graph.add_edge("planner", END)

    return graph


def compile_graph():
    """Compile the graph into a runnable CompiledStateGraph."""
    return build_graph().compile()


# ── Module-level compiled graph ───────────────────────────────────────────────
compiled = compile_graph()


# ── Manual runner ─────────────────────────────────────────────────────────────


def main() -> None:
    """Invoke the graph and pretty-print the resulting PrepPlan."""
    print("Loop — Phase 2 graph run")
    print("Invoking graph: START → intake → planner → END\n")

    cb = get_langfuse_callback()
    config = {"callbacks": [cb]} if cb else {}

    result = compiled.invoke(initial_state(), config=config)

    print(f"jd loaded:      {len(result['jd'])} chars")
    print(f"profile loaded: {len(result['profile'])} chars")
    print("\n── PrepPlan ──")
    print(json.dumps(result["plan"], indent=2))
    print("\nGraph run complete.")


if __name__ == "__main__":
    main()
