"""
Graph assembly — Phase 1: intake → END.

This is the minimal graph: one node that loads JD + profile from fixtures
into state, then terminates.  The goal is to prove the graph mechanics
(StateGraph, node, edge, compile, invoke) before adding any LLM calls.

Run with:  uv run python -m loop.graph
"""

import pathlib

from langgraph.graph import END, START, StateGraph

from loop.observability import get_langfuse_callback
from loop.state import LoopState, initial_state

# ── Fixtures directory ────────────────────────────────────────────────────────
_FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"


# ── Node functions ────────────────────────────────────────────────────────────


def intake(state: dict) -> dict:
    """Load JD and profile text from fixtures into state.

    Node contract: receives full state, returns ONLY the keys it changes.
    LangGraph merges the returned dict back into state before the next node.

    This is Phase 1's only node — no LLM call, just data loading.
    """
    jd = (_FIXTURES / "sample_jd.md").read_text()
    profile = (_FIXTURES / "sample_profile.md").read_text()

    return {
        "jd": jd,
        "profile": profile,
    }


# ── Graph definition ──────────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """Define the graph topology (blueprint only — not yet runnable).

    StateGraph(LoopState) tells LangGraph what keys and reducers to expect.
    Nodes and edges are registered here; compile() makes it runnable.
    """
    # StateGraph takes the state schema so it knows the shape + reducers.
    # Analogy: declaring the bean definitions before the ApplicationContext starts.
    graph = StateGraph(LoopState)

    # Register nodes — name → function.
    # The function signature is (state: dict) -> dict (partial update).
    graph.add_node("intake", intake)

    # Wire edges: START → intake → END.
    # START and END are LangGraph sentinels (not real nodes).
    graph.add_edge(START, "intake")
    graph.add_edge("intake", END)

    return graph


def compile_graph():
    """Compile the graph into a runnable CompiledStateGraph.

    compile() is like ApplicationContext.refresh() — it validates the topology,
    wires reducers, and returns a Runnable you can .invoke() / .stream().
    No checkpointer yet (added in Phase 4).
    """
    return build_graph().compile()


# ── Module-level compiled graph (import this in tests / other modules) ────────
compiled = compile_graph()


# ── Manual runner ─────────────────────────────────────────────────────────────


def main() -> None:
    """Invoke the graph and print the resulting state."""
    print("Loop — Phase 1 graph run")
    print("Invoking graph: START → intake → END\n")

    cb = get_langfuse_callback()
    config = {"callbacks": [cb]} if cb else {}

    # invoke() runs the graph to completion and returns the final state dict.
    # We pass initial_state() so every key exists with a safe default.
    result = compiled.invoke(initial_state(), config=config)

    print(f"jd loaded:      {len(result['jd'])} chars")
    print(f"profile loaded: {len(result['profile'])} chars")
    print(f"messages:       {result['messages']}")
    print(f"plan:           {result['plan']}")
    print("\nGraph run complete.")


if __name__ == "__main__":
    main()
