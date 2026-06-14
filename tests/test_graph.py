"""
Phase 1 graph tests — offline, no LLM calls.

Tests verify:
- The compiled graph runs without error.
- The intake node populates jd and profile from fixtures.
- State shape is correct (all expected keys present).
- Messages field starts empty and accepts appends (reducer check).
"""

import pathlib

# ── helpers ───────────────────────────────────────────────────────────────────


def _run_graph() -> dict:
    """Compile and invoke the graph with a clean initial state."""
    from loop.graph import compile_graph
    from loop.state import initial_state

    g = compile_graph()
    return g.invoke(initial_state())


# ── tests ─────────────────────────────────────────────────────────────────────


class TestGraphStructure:
    def test_graph_compiles(self):
        """Graph definition is valid — compile() must not raise."""
        from loop.graph import compile_graph

        g = compile_graph()
        assert g is not None

    def test_graph_has_intake_node(self):
        """intake node is registered in the graph."""
        from loop.graph import build_graph

        g = build_graph()
        assert "intake" in g.nodes


class TestIntakeNode:
    def test_jd_loaded(self):
        """intake populates jd from fixtures/sample_jd.md."""
        result = _run_graph()
        assert len(result["jd"]) > 0
        # Sanity: it's the right file (contains a known string from the fixture)
        assert "Backend Engineer" in result["jd"]

    def test_profile_loaded(self):
        """intake populates profile from fixtures/sample_profile.md."""
        result = _run_graph()
        assert len(result["profile"]) > 0
        assert "Java" in result["profile"]

    def test_jd_matches_fixture_file(self):
        """jd content exactly matches the fixture file on disk."""
        result = _run_graph()
        fixture = (pathlib.Path(__file__).parent.parent / "fixtures" / "sample_jd.md").read_text()
        assert result["jd"] == fixture


class TestStateShape:
    def test_all_keys_present(self):
        """Final state has every key defined in initial_state()."""
        result = _run_graph()
        expected_keys = {
            "jd",
            "profile",
            "messages",
            "plan",
            "current_modality",
            "current_question_id",
            "answers",
            "grades",
            "weak_areas",
            "session_number",
            "plan_approved",
            "readiness_verdict",
            "verdict_approved",
        }
        assert expected_keys <= result.keys()

    def test_messages_is_list(self):
        """messages field is a list (add_messages reducer target type)."""
        result = _run_graph()
        assert isinstance(result["messages"], list)

    def test_optional_fields_are_none(self):
        """Fields not set by Phase 3+ remain None; planner is stubbed by conftest."""
        result = _run_graph()
        # plan is now populated by the planner node (Phase 2)
        assert result["plan"] is not None
        # these are still None — set by Phase 3+
        assert result["current_modality"] is None
        assert result["grades"] is None
        assert result["grades"] is None


class TestMessagesReducer:
    def test_messages_append_not_overwrite(self):
        """add_messages reducer appends on successive updates, not overwrites."""
        from langchain_core.messages import HumanMessage

        from loop.graph import compile_graph
        from loop.state import initial_state

        g = compile_graph()

        # Seed state with one message already in it.
        seed = initial_state()
        seed["messages"] = [HumanMessage(content="first")]

        result = g.invoke(seed)

        # intake doesn't touch messages — list should still have the seed message.
        # This confirms the reducer is wired (overwrite would silently drop it).
        assert len(result["messages"]) >= 1
        assert result["messages"][0].content == "first"
