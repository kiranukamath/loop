"""
Phase 4 memory tests — InMemoryStore, checkpointer, planner/coach store integration.
All offline: model calls are stubbed.

Tests verify:
- InMemoryStore put/get/namespace isolation
- planner reads stored weak_areas and merges them with state weak_areas
- coach writes weak_areas to the store (merging with existing entries)
- thread isolation: different thread_ids produce independent checkpointed states
- second-session planner receives stored weak_areas from first session's coach
"""

from unittest.mock import MagicMock

from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from loop.schemas import Feedback, Grade, PrepPlan, Session

# ── Shared stubs ──────────────────────────────────────────────────────────────

_STUB_PLAN = PrepPlan(
    role_summary="stub",
    total_sessions=1,
    sessions=[Session(session_number=1, modality="coding", topics=["x"], focus="y")],
    key_gaps=[],
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
    strengths=["understands the concept"],
    improvements=["needs to handle edge cases", "improve time complexity"],
    overall_feedback="Decent attempt; edge cases missed.",
)

_STUB_FEEDBACK = Feedback(
    summary="Needs work on edge cases.",
    action_items=["practice edge cases", "review sliding window"],
    weak_areas_update=["edge-case handling", "sliding-window"],
)


def _fake_model(return_value):
    fake = MagicMock()
    fake.with_structured_output.return_value = RunnableLambda(lambda _: return_value)
    return fake


def _stub_plan_approval(state):
    """Stub for plan_approval: skip interrupt, auto-approve and route to session_router."""
    return Command(goto="session_router", update={"plan_approved": True})


def _compile_with_fresh_memory(graph):
    """Compile graph with brand-new checkpointer + store (test isolation)."""
    store = InMemoryStore()
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer, store=store), store


def _run_graph_with_memory(monkeypatch, store, checkpointer, user_id="u1", thread_id="t1"):
    """Run the full graph with fresh memory, returning the result and the store."""
    monkeypatch.setattr("loop.nodes.planner.get_chat_model", lambda: _fake_model(_STUB_PLAN))
    monkeypatch.setattr("loop.graph.plan_approval", _stub_plan_approval)
    # Stub interviewers — bypass the answer-gate interrupt and return a canned question_id.
    # Answers come from real interviewers via interrupt in production; for memory tests
    # we only care that coach/store wiring works, not that the answer gate fires.
    monkeypatch.setattr(
        "loop.graph.coding_interviewer",
        lambda state: {"current_question_id": "cod-001", "messages": []},
    )
    monkeypatch.setattr(
        "loop.graph.sd_interviewer",
        lambda state: {"current_question_id": "sys-001", "messages": []},
    )
    monkeypatch.setattr(
        "loop.graph.beh_interviewer",
        lambda state: {"current_question_id": "beh-001", "messages": []},
    )
    monkeypatch.setattr("loop.graph.grader", lambda state: {"grades": [_STUB_GRADE.model_dump()]})
    monkeypatch.setattr(
        "loop.graph.coach", lambda state: _persist_and_return(state, _STUB_FEEDBACK, store, user_id)
    )
    monkeypatch.setattr(
        "loop.graph.readiness",
        lambda state: {"readiness_verdict": {"verdict": "ready"}, "verdict_approved": True},
    )

    from loop.graph import build_graph
    from loop.state import initial_state

    app = build_graph().compile(checkpointer=checkpointer, store=store)
    cfg = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    return app.invoke(initial_state(), config=cfg)


def _persist_and_return(state, feedback, store, user_id):
    """Simulate coach: write to store and return weak_areas."""
    existing = store.get(("loop", "users"), user_id)
    existing_areas = existing.value.get("weak_areas", []) if existing else []
    session_count = (existing.value.get("session_count", 0) if existing else 0) + 1
    merged = list(dict.fromkeys(existing_areas + feedback.weak_areas_update))
    store.put(("loop", "users"), user_id, {"weak_areas": merged, "session_count": session_count})
    return {"weak_areas": feedback.weak_areas_update}


# ── Store basics ──────────────────────────────────────────────────────────────


class TestInMemoryStore:
    def test_put_and_get(self):
        store = InMemoryStore()
        store.put(("loop", "users"), "kiran", {"weak_areas": ["Kafka"], "session_count": 1})
        item = store.get(("loop", "users"), "kiran")
        assert item is not None
        assert item.value["weak_areas"] == ["Kafka"]
        assert item.value["session_count"] == 1

    def test_get_missing_returns_none(self):
        store = InMemoryStore()
        assert store.get(("loop", "users"), "nobody") is None

    def test_namespace_isolation(self):
        """Two different namespaces with the same key are independent."""
        store = InMemoryStore()
        store.put(("loop", "users"), "alice", {"weak_areas": ["A"]})
        store.put(("loop", "admins"), "alice", {"weak_areas": ["B"]})
        assert store.get(("loop", "users"), "alice").value["weak_areas"] == ["A"]
        assert store.get(("loop", "admins"), "alice").value["weak_areas"] == ["B"]

    def test_overwrite_merges(self):
        """put() with same key overwrites the entire value."""
        store = InMemoryStore()
        store.put(("ns",), "k", {"x": 1})
        store.put(("ns",), "k", {"x": 2, "y": 3})
        assert store.get(("ns",), "k").value == {"x": 2, "y": 3}


# ── Planner reads from store ──────────────────────────────────────────────────


class TestPlannerReadsStore:
    def test_planner_reads_stored_weak_areas(self, monkeypatch):
        """When store has weak_areas, planner merges them into the prompt."""
        # Capture the prompt rendered by the planner
        captured = {}

        def fake_chain_factory(return_val):
            def factory(_):
                return return_val

            class _FakeModel:
                def with_structured_output(self, schema):
                    from langchain_core.runnables import RunnableLambda as RL

                    def interceptor(prompt_value):
                        captured["weak_areas_text"] = str(prompt_value)
                        return return_val

                    return RL(interceptor)

            return _FakeModel()

        monkeypatch.setattr(
            "loop.nodes.planner.get_chat_model", lambda: fake_chain_factory(_STUB_PLAN)
        )

        store = InMemoryStore()
        store.put(("loop", "users"), "kiran", {"weak_areas": ["Kafka", "STAR storytelling"]})

        from loop.graph import build_graph
        from loop.state import initial_state

        app = build_graph().compile(store=store)
        state = initial_state()
        state["answers"] = [{"question_id": "cod-001", "text": "..."}]

        # Stub interrupt nodes + model-calling nodes to avoid interrupts and model calls
        import loop.graph as gmod

        monkeypatch.setattr(gmod, "plan_approval", _stub_plan_approval)
        monkeypatch.setattr(gmod, "grader", lambda s: {"grades": [_STUB_GRADE.model_dump()]})
        monkeypatch.setattr(gmod, "coach", lambda s: {"weak_areas": []})
        monkeypatch.setattr(
            gmod,
            "readiness",
            lambda s: {"readiness_verdict": {"verdict": "ready"}, "verdict_approved": True},
        )

        app = build_graph().compile(store=store)
        app.invoke(state, config={"configurable": {"user_id": "kiran"}})

        assert "Kafka" in captured.get("weak_areas_text", "")
        assert "STAR storytelling" in captured.get("weak_areas_text", "")

    def test_planner_works_with_no_store(self, monkeypatch):
        """Planner works fine when compiled without a store (store=None path)."""
        monkeypatch.setattr("loop.nodes.planner.get_chat_model", lambda: _fake_model(_STUB_PLAN))
        monkeypatch.setattr("loop.graph.plan_approval", _stub_plan_approval)
        monkeypatch.setattr("loop.graph.grader", lambda s: {"grades": [_STUB_GRADE.model_dump()]})
        monkeypatch.setattr("loop.graph.coach", lambda s: {"weak_areas": []})
        monkeypatch.setattr(
            "loop.graph.readiness",
            lambda s: {"readiness_verdict": {"verdict": "ready"}, "verdict_approved": True},
        )

        from loop.graph import build_graph
        from loop.state import initial_state

        app = build_graph().compile()  # no store
        state = initial_state()
        state["answers"] = [{"question_id": "cod-001", "text": "..."}]
        result = app.invoke(state)
        assert result["plan"] is not None

    def test_planner_merges_store_and_state_weak_areas(self, monkeypatch):
        """Planner merges store areas + state areas (deduplicating)."""
        captured = {}

        class _CaptureModel:
            def with_structured_output(self, schema):
                from langchain_core.runnables import RunnableLambda as RL

                def interceptor(prompt_value):
                    captured["text"] = str(prompt_value)
                    return _STUB_PLAN

                return RL(interceptor)

        monkeypatch.setattr("loop.nodes.planner.get_chat_model", lambda: _CaptureModel())

        store = InMemoryStore()
        store.put(("loop", "users"), "u1", {"weak_areas": ["Kafka"]})

        import loop.graph as gmod
        from loop.graph import build_graph
        from loop.state import initial_state

        monkeypatch.setattr(gmod, "plan_approval", _stub_plan_approval)
        monkeypatch.setattr(gmod, "grader", lambda s: {"grades": [_STUB_GRADE.model_dump()]})
        monkeypatch.setattr(gmod, "coach", lambda s: {"weak_areas": []})
        monkeypatch.setattr(
            gmod,
            "readiness",
            lambda s: {"readiness_verdict": {"verdict": "ready"}, "verdict_approved": True},
        )

        app = build_graph().compile(store=store)
        state = initial_state()
        state["answers"] = [{"question_id": "cod-001", "text": "..."}]
        # Also pass state-level weak_areas (e.g. from a same-session earlier run)
        state["weak_areas"] = ["distributed-systems"]
        app.invoke(state, config={"configurable": {"user_id": "u1"}})

        text = captured.get("text", "")
        assert "Kafka" in text
        assert "distributed-systems" in text


# ── Coach writes to store ─────────────────────────────────────────────────────


class TestCoachWritesStore:
    def _run_coach_in_graph(self, monkeypatch, store, user_id="u1"):
        """Run graph with real coach node (model stubbed) and return result."""
        monkeypatch.setattr("loop.nodes.coach.get_chat_model", lambda: _fake_model(_STUB_FEEDBACK))
        monkeypatch.setattr("loop.nodes.planner.get_chat_model", lambda: _fake_model(_STUB_PLAN))
        monkeypatch.setattr("loop.graph.plan_approval", _stub_plan_approval)
        # Stub interviewers — bypass answer-gate interrupt; coach is what we're testing here.
        monkeypatch.setattr(
            "loop.graph.coding_interviewer",
            lambda s: {"current_question_id": "cod-001", "messages": []},
        )
        monkeypatch.setattr(
            "loop.graph.sd_interviewer",
            lambda s: {"current_question_id": "sys-001", "messages": []},
        )
        monkeypatch.setattr(
            "loop.graph.beh_interviewer",
            lambda s: {"current_question_id": "beh-001", "messages": []},
        )
        monkeypatch.setattr("loop.graph.grader", lambda s: {"grades": [_STUB_GRADE.model_dump()]})
        monkeypatch.setattr(
            "loop.graph.readiness",
            lambda s: {"readiness_verdict": {"verdict": "ready"}, "verdict_approved": True},
        )

        from loop.graph import build_graph
        from loop.state import initial_state

        checkpointer = MemorySaver()
        app = build_graph().compile(checkpointer=checkpointer, store=store)
        cfg = {"configurable": {"thread_id": "t1", "user_id": user_id}}
        return app.invoke(initial_state(), config=cfg)

    def test_coach_writes_weak_areas_to_store(self, monkeypatch):
        from loop.memory import semantic_namespace

        store = InMemoryStore()
        self._run_coach_in_graph(monkeypatch, store)
        item = store.get(semantic_namespace("u1"), "weak_areas")
        assert item is not None
        assert set(_STUB_FEEDBACK.weak_areas_update) <= set(item.value["weak_areas"])

    def test_coach_sets_session_count(self, monkeypatch):
        from loop.memory import semantic_namespace

        store = InMemoryStore()
        self._run_coach_in_graph(monkeypatch, store)
        item = store.get(semantic_namespace("u1"), "weak_areas")
        assert item.value["session_count"] == 1

    def test_coach_writes_episodic_record(self, monkeypatch):
        """Phase 16a: coach also writes an immutable per-session episodic record."""
        from loop.memory import episodic_namespace

        store = InMemoryStore()
        self._run_coach_in_graph(monkeypatch, store)
        item = store.get(episodic_namespace("u1"), "session-1")
        assert item is not None
        assert item.value["session_number"] == 1
        assert item.value["new_weak_areas"] == _STUB_FEEDBACK.weak_areas_update

    def test_coach_merges_with_existing_weak_areas(self, monkeypatch):
        from loop.memory import semantic_namespace

        store = InMemoryStore()
        # Pre-seed the store (typed shape this time -- see
        # TestBackwardCompatMigration below for the legacy-shape case)
        store.put(
            semantic_namespace("u1"), "weak_areas", {"weak_areas": ["Kafka"], "session_count": 1}
        )
        self._run_coach_in_graph(monkeypatch, store)
        item = store.get(semantic_namespace("u1"), "weak_areas")
        # Should have both old and new areas
        assert "Kafka" in item.value["weak_areas"]
        for area in _STUB_FEEDBACK.weak_areas_update:
            assert area in item.value["weak_areas"]
        assert item.value["session_count"] == 2

    def test_coach_migrates_legacy_flat_shape_forward(self, monkeypatch):
        """Phase 16a: pre-existing Phase 4 flat-shape data is read (not lost)
        and the next write lands in the typed semantic namespace."""
        from loop.memory import semantic_namespace

        store = InMemoryStore()
        store.put(("loop", "users"), "u1", {"weak_areas": ["Kafka"], "session_count": 1})
        self._run_coach_in_graph(monkeypatch, store)

        migrated = store.get(semantic_namespace("u1"), "weak_areas")
        assert migrated is not None
        assert "Kafka" in migrated.value["weak_areas"]
        assert migrated.value["session_count"] == 2

    def test_coach_no_store_does_not_crash(self, monkeypatch):
        """Coach works fine when compiled without a store."""
        monkeypatch.setattr("loop.nodes.coach.get_chat_model", lambda: _fake_model(_STUB_FEEDBACK))
        monkeypatch.setattr("loop.nodes.planner.get_chat_model", lambda: _fake_model(_STUB_PLAN))
        monkeypatch.setattr("loop.graph.plan_approval", _stub_plan_approval)
        monkeypatch.setattr(
            "loop.graph.coding_interviewer",
            lambda s: {"current_question_id": "cod-001", "messages": []},
        )
        monkeypatch.setattr(
            "loop.graph.sd_interviewer",
            lambda s: {"current_question_id": "sys-001", "messages": []},
        )
        monkeypatch.setattr(
            "loop.graph.beh_interviewer",
            lambda s: {"current_question_id": "beh-001", "messages": []},
        )
        monkeypatch.setattr("loop.graph.grader", lambda s: {"grades": [_STUB_GRADE.model_dump()]})
        monkeypatch.setattr(
            "loop.graph.readiness",
            lambda s: {"readiness_verdict": {"verdict": "ready"}, "verdict_approved": True},
        )

        from loop.graph import build_graph
        from loop.state import initial_state

        app = build_graph().compile()  # no store
        result = app.invoke(initial_state())
        assert result["weak_areas"] is not None


# ── Thread isolation ──────────────────────────────────────────────────────────


class TestThreadIsolation:
    def test_different_threads_have_independent_state(self, monkeypatch):
        """Two thread_ids produce independent checkpointed states."""
        monkeypatch.setattr("loop.nodes.planner.get_chat_model", lambda: _fake_model(_STUB_PLAN))
        monkeypatch.setattr("loop.graph.plan_approval", _stub_plan_approval)
        monkeypatch.setattr("loop.graph.grader", lambda s: {"grades": [_STUB_GRADE.model_dump()]})
        monkeypatch.setattr("loop.graph.coach", lambda s: {"weak_areas": ["topic-A"]})
        monkeypatch.setattr(
            "loop.graph.readiness",
            lambda s: {"readiness_verdict": {"verdict": "ready"}, "verdict_approved": True},
        )

        from loop.graph import build_graph
        from loop.state import initial_state

        checkpointer = MemorySaver()
        store = InMemoryStore()
        app = build_graph().compile(checkpointer=checkpointer, store=store)

        state = initial_state()
        state["answers"] = [{"question_id": "cod-001", "text": "..."}]

        cfg1 = {"configurable": {"thread_id": "thread-A", "user_id": "alice"}}
        cfg2 = {"configurable": {"thread_id": "thread-B", "user_id": "bob"}}

        r1 = app.invoke(state, config=cfg1)
        r2 = app.invoke(state, config=cfg2)

        # Both ran independently — states saved per thread
        s1 = app.get_state(cfg1)
        s2 = app.get_state(cfg2)
        assert s1.values["current_question_id"] == r1["current_question_id"]
        assert s2.values["current_question_id"] == r2["current_question_id"]
        # thread-A and thread-B are separate checkpoints
        assert s1.config["configurable"]["thread_id"] == "thread-A"
        assert s2.config["configurable"]["thread_id"] == "thread-B"

    def test_get_state_restores_complete_state(self, monkeypatch):
        """After a run, get_state returns the final state."""
        monkeypatch.setattr("loop.nodes.planner.get_chat_model", lambda: _fake_model(_STUB_PLAN))
        monkeypatch.setattr("loop.graph.plan_approval", _stub_plan_approval)
        monkeypatch.setattr("loop.graph.grader", lambda s: {"grades": [_STUB_GRADE.model_dump()]})
        monkeypatch.setattr("loop.graph.coach", lambda s: {"weak_areas": ["X"]})
        monkeypatch.setattr(
            "loop.graph.readiness",
            lambda s: {"readiness_verdict": {"verdict": "ready"}, "verdict_approved": True},
        )

        from loop.graph import build_graph
        from loop.state import initial_state

        checkpointer = MemorySaver()
        app = build_graph().compile(checkpointer=checkpointer)

        state = initial_state()
        state["answers"] = [{"question_id": "cod-001", "text": "..."}]
        cfg = {"configurable": {"thread_id": "t-snapshot"}}
        result = app.invoke(state, config=cfg)

        saved = app.get_state(cfg)
        assert saved.values["plan"] is not None
        assert saved.values["current_modality"] == result["current_modality"]
        assert saved.values["grades"] == result["grades"]


# ── Cross-session feedback loop ───────────────────────────────────────────────


class TestCrossSessionFeedbackLoop:
    def test_second_session_planner_receives_stored_weak_areas(self, monkeypatch):
        """Session 2 planner prompt contains weak areas stored by session 1's coach."""
        captured_prompts = []

        class _CapturePlanner:
            def with_structured_output(self, schema):
                from langchain_core.runnables import RunnableLambda as RL

                def intercept(prompt_value):
                    captured_prompts.append(str(prompt_value))
                    return _STUB_PLAN

                return RL(intercept)

        monkeypatch.setattr("loop.nodes.planner.get_chat_model", lambda: _CapturePlanner())
        monkeypatch.setattr("loop.nodes.coach.get_chat_model", lambda: _fake_model(_STUB_FEEDBACK))
        monkeypatch.setattr("loop.graph.plan_approval", _stub_plan_approval)
        monkeypatch.setattr(
            "loop.graph.coding_interviewer",
            lambda s: {"current_question_id": "cod-001", "messages": []},
        )
        monkeypatch.setattr(
            "loop.graph.sd_interviewer",
            lambda s: {"current_question_id": "sys-001", "messages": []},
        )
        monkeypatch.setattr(
            "loop.graph.beh_interviewer",
            lambda s: {"current_question_id": "beh-001", "messages": []},
        )
        monkeypatch.setattr("loop.graph.grader", lambda s: {"grades": [_STUB_GRADE.model_dump()]})
        monkeypatch.setattr(
            "loop.graph.readiness",
            lambda s: {"readiness_verdict": {"verdict": "ready"}, "verdict_approved": True},
        )

        from loop.graph import build_graph
        from loop.state import initial_state

        checkpointer = MemorySaver()
        store = InMemoryStore()
        app = build_graph().compile(checkpointer=checkpointer, store=store)

        # Session 1: coach writes weak_areas to store
        cfg1 = {"configurable": {"thread_id": "s1", "user_id": "kiran"}}
        app.invoke(initial_state(), config=cfg1)

        # Session 2: planner should read stored weak_areas
        cfg2 = {"configurable": {"thread_id": "s2", "user_id": "kiran"}}
        app.invoke(initial_state(), config=cfg2)

        # Two planner calls captured; second one should mention weak areas
        assert len(captured_prompts) == 2
        for area in _STUB_FEEDBACK.weak_areas_update:
            assert area in captured_prompts[1], (
                f"Stored weak area '{area}' not found in second session planner prompt"
            )


# ── Checkpointer factory ──────────────────────────────────────────────────────


class TestCheckpointerFactory:
    """Test _make_checkpointer() selects the right backend from config."""

    def test_no_db_path_returns_memory_saver(self, monkeypatch):
        monkeypatch.setattr("loop.memory.settings.db_path", "")
        from loop.memory import _make_checkpointer

        cp = _make_checkpointer()
        assert isinstance(cp, MemorySaver)

    def test_db_path_set_returns_sqlite_saver(self, monkeypatch, tmp_path):
        db = str(tmp_path / "test.sqlite")
        monkeypatch.setattr("loop.memory.settings.db_path", db)
        from langgraph.checkpoint.sqlite import SqliteSaver

        from loop.memory import _make_checkpointer

        cp = _make_checkpointer()
        assert isinstance(cp, SqliteSaver)

    def test_sqlite_saver_persists_state(self, monkeypatch, tmp_path):
        """Graph state written to SqliteSaver survives a second invocation."""
        db = str(tmp_path / "persist.sqlite")
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        # Build and run a graph once
        monkeypatch.setattr("loop.nodes.planner.get_chat_model", lambda: _fake_model(_STUB_PLAN))
        monkeypatch.setattr("loop.graph.plan_approval", _stub_plan_approval)
        monkeypatch.setattr(
            "loop.graph.coding_interviewer",
            lambda s: {"current_question_id": "cod-001", "messages": []},
        )  # noqa: E501
        monkeypatch.setattr(
            "loop.graph.sd_interviewer",
            lambda s: {"current_question_id": "sys-001", "messages": []},
        )  # noqa: E501
        monkeypatch.setattr(
            "loop.graph.beh_interviewer",
            lambda s: {"current_question_id": "beh-001", "messages": []},
        )  # noqa: E501
        monkeypatch.setattr("loop.graph.grader", lambda s: {"grades": [_STUB_GRADE.model_dump()]})
        monkeypatch.setattr("loop.graph.coach", lambda s: {"weak_areas": ["binary-search"]})
        monkeypatch.setattr(
            "loop.graph.readiness",
            lambda s: {"readiness_verdict": {"verdict": "ready"}, "verdict_approved": True},
        )

        from loop.graph import build_graph
        from loop.state import initial_state

        conn1 = sqlite3.connect(db, check_same_thread=False)
        saver1 = SqliteSaver(conn1)
        saver1.setup()
        app = build_graph().compile(checkpointer=saver1)
        cfg = {"configurable": {"thread_id": "persist-test"}}
        app.invoke(initial_state(), config=cfg)
        conn1.close()

        # Re-open the same file and read state — proves data survived
        conn2 = sqlite3.connect(db, check_same_thread=False)
        saver2 = SqliteSaver(conn2)
        app2 = build_graph().compile(checkpointer=saver2)
        snap = app2.get_state(cfg)
        assert snap.values.get("plan") is not None
        assert snap.values.get("weak_areas") == ["binary-search"]
        conn2.close()


# ── Phase 18a: Postgres checkpointer + store seam ─────────────────────────────
#
# A real Postgres server is a *server* activity — these tests mock psycopg's
# Connection.connect (so nothing touches a network socket) and PostgresSaver/
# PostgresStore's own .setup() (which would otherwise try to run real DDL over
# that mocked connection), then assert only that _make_checkpointer()/
# _make_store() DISPATCH to the Postgres classes when pg_conn_string is set —
# exactly the same "verify the seam picks the right backend" discipline as
# TestCheckpointerFactory above.


class TestPostgresCheckpointerFactory:
    def test_pg_conn_string_returns_postgres_saver(self, monkeypatch):
        from unittest.mock import MagicMock

        from langgraph.checkpoint.postgres import PostgresSaver

        monkeypatch.setattr("loop.memory.settings.pg_conn_string", "postgresql://fake/db")
        monkeypatch.setattr("psycopg.Connection.connect", lambda *a, **k: MagicMock())
        monkeypatch.setattr(PostgresSaver, "setup", lambda self: None)

        from loop.memory import _make_checkpointer

        cp = _make_checkpointer()
        assert isinstance(cp, PostgresSaver)

    def test_pg_conn_string_takes_precedence_over_db_path(self, monkeypatch):
        """pg_conn_string wins over db_path if both happen to be set."""
        from unittest.mock import MagicMock

        from langgraph.checkpoint.postgres import PostgresSaver

        monkeypatch.setattr("loop.memory.settings.pg_conn_string", "postgresql://fake/db")
        monkeypatch.setattr("loop.memory.settings.db_path", "somewhere.sqlite")
        monkeypatch.setattr("psycopg.Connection.connect", lambda *a, **k: MagicMock())
        monkeypatch.setattr(PostgresSaver, "setup", lambda self: None)

        from loop.memory import _make_checkpointer

        assert isinstance(_make_checkpointer(), PostgresSaver)


class TestPostgresStoreFactory:
    def test_pg_conn_string_returns_postgres_store(self, monkeypatch):
        from unittest.mock import MagicMock

        from langgraph.store.postgres import PostgresStore

        monkeypatch.setattr("loop.memory.settings.pg_conn_string", "postgresql://fake/db")
        monkeypatch.setattr("psycopg.Connection.connect", lambda *a, **k: MagicMock())
        monkeypatch.setattr(PostgresStore, "setup", lambda self: None)

        from loop.memory import _make_store

        assert isinstance(_make_store(), PostgresStore)

    def test_no_pg_conn_string_returns_in_memory_store(self, monkeypatch):
        monkeypatch.setattr("loop.memory.settings.pg_conn_string", "")

        from loop.memory import _make_store

        assert isinstance(_make_store(), InMemoryStore)


# ── Phase 16a: typed namespace helpers ────────────────────────────────────────


class TestMemoryTypingHelpers:
    """loop/memory.py's namespace + read/write helpers, unit-level (no graph)."""

    def test_namespaces_are_distinct_and_typed(self):
        from loop.memory import episodic_namespace, procedural_namespace, semantic_namespace

        assert episodic_namespace("u1") == ("loop", "users", "u1", "episodic")
        assert semantic_namespace("u1") == ("loop", "users", "u1", "semantic")
        assert procedural_namespace("u1") == ("loop", "users", "u1", "procedural")

    def test_get_weak_areas_state_reads_typed_shape(self):
        from loop.memory import get_weak_areas_state, put_weak_areas_state

        store = InMemoryStore()
        put_weak_areas_state(store, "u1", ["Kafka"], 3)
        areas, count = get_weak_areas_state(store, "u1")
        assert areas == ["Kafka"]
        assert count == 3

    def test_get_weak_areas_state_falls_back_to_legacy_shape(self):
        """Back-compat: pre-Phase-16 flat data is still readable when no typed
        entry exists yet."""
        from loop.memory import get_weak_areas_state

        store = InMemoryStore()
        store.put(("loop", "users"), "u1", {"weak_areas": ["Kafka"], "session_count": 5})
        areas, count = get_weak_areas_state(store, "u1")
        assert areas == ["Kafka"]
        assert count == 5

    def test_typed_shape_takes_precedence_over_legacy(self):
        """Once migrated, the typed entry wins even if a stale legacy entry
        still exists."""
        from loop.memory import get_weak_areas_state, put_weak_areas_state

        store = InMemoryStore()
        store.put(("loop", "users"), "u1", {"weak_areas": ["stale"], "session_count": 1})
        put_weak_areas_state(store, "u1", ["fresh"], 2)
        areas, count = get_weak_areas_state(store, "u1")
        assert areas == ["fresh"]
        assert count == 2

    def test_get_weak_areas_state_missing_user_returns_empty(self):
        from loop.memory import get_weak_areas_state

        store = InMemoryStore()
        areas, count = get_weak_areas_state(store, "nobody")
        assert areas == []
        assert count == 0

    def test_episode_round_trip(self):
        from loop.memory import get_recent_episodes, put_episode

        store = InMemoryStore()
        put_episode(store, "u1", 1, ["sliding-window"])
        put_episode(store, "u1", 2, ["concurrency"])
        episodes = get_recent_episodes(store, "u1", session_count=2, window=10)
        assert [e["session_number"] for e in episodes] == [1, 2]
        assert episodes[0]["new_weak_areas"] == ["sliding-window"]
        assert episodes[1]["new_weak_areas"] == ["concurrency"]

    def test_get_recent_episodes_is_bounded_by_window(self):
        from loop.memory import get_recent_episodes, put_episode

        store = InMemoryStore()
        for n in range(1, 6):
            put_episode(store, "u1", n, [f"topic-{n}"])
        episodes = get_recent_episodes(store, "u1", session_count=5, window=2)
        assert [e["session_number"] for e in episodes] == [4, 5]


# ── Phase 16c: decay/conflict policy ──────────────────────────────────────────


class TestConflictResolution:
    def test_no_existing_value_returns_candidate(self):
        from loop.memory import resolve_conflict

        candidate = {"text": "x", "confidence": 0.5, "created_session_count": 1}
        assert resolve_conflict(None, candidate) == candidate

    def test_newer_session_wins(self):
        from loop.memory import resolve_conflict

        old = {"text": "old fact", "confidence": 0.9, "created_session_count": 1}
        new = {"text": "new fact", "confidence": 0.2, "created_session_count": 5}
        assert resolve_conflict(old, new) == new

    def test_older_candidate_loses_even_with_higher_confidence(self):
        from loop.memory import resolve_conflict

        existing = {"text": "current", "confidence": 0.4, "created_session_count": 5}
        stale_candidate = {"text": "stale", "confidence": 0.99, "created_session_count": 1}
        assert resolve_conflict(existing, stale_candidate) == existing

    def test_tie_breaks_on_higher_confidence(self):
        from loop.memory import resolve_conflict

        existing = {"text": "a", "confidence": 0.3, "created_session_count": 3}
        candidate = {"text": "b", "confidence": 0.8, "created_session_count": 3}
        assert resolve_conflict(existing, candidate) == candidate


class TestPutSemanticInsight:
    def test_same_topic_resolves_via_conflict_policy(self):
        from loop.memory import put_semantic_insight, semantic_namespace

        store = InMemoryStore()
        put_semantic_insight(
            store,
            "u1",
            topic="pacing",
            text="rushes system design",
            confidence=0.5,
            session_count=1,
        )
        put_semantic_insight(
            store,
            "u1",
            topic="pacing",
            text="now paces system design well",
            confidence=0.5,
            session_count=3,
        )
        item = store.get(semantic_namespace("u1"), "pacing")
        assert item.value["text"] == "now paces system design well"
        assert item.value["created_session_count"] == 3

    def test_different_topics_coexist(self):
        from loop.memory import put_semantic_insight, semantic_namespace

        store = InMemoryStore()
        put_semantic_insight(store, "u1", topic="pacing", text="a", confidence=0.5, session_count=1)
        put_semantic_insight(store, "u1", topic="depth", text="b", confidence=0.5, session_count=1)
        assert store.get(semantic_namespace("u1"), "pacing").value["text"] == "a"
        assert store.get(semantic_namespace("u1"), "depth").value["text"] == "b"


# ── Phase 16c: semantic recall (embedding similarity) + decay ────────────────


def _indexed_store(dims=256):
    """An InMemoryStore wired for semantic search, using DeterministicFakeEmbedding
    (offline -- no Bedrock call). Mirrors loop.memory._make_store()'s index
    config shape, but built directly so these tests don't depend on the
    process-level singleton."""
    from langchain_core.embeddings.fake import DeterministicFakeEmbedding

    return InMemoryStore(
        index={"dims": dims, "embed": DeterministicFakeEmbedding(size=dims), "fields": ["text"]}
    )


class TestSemanticRecall:
    def test_recall_ranks_exact_text_match_first(self):
        """DeterministicFakeEmbedding has no real semantic meaning, but the
        SAME text always embeds to the SAME vector -- querying with a stored
        insight's exact text must rank it first over an unrelated insight."""
        from loop.memory import put_semantic_insight, recall_semantic_memories

        store = _indexed_store()
        put_semantic_insight(
            store,
            "u1",
            topic="a",
            text="struggles with concurrency edge cases",
            confidence=0.8,
            session_count=1,
        )
        put_semantic_insight(
            store,
            "u1",
            topic="b",
            text="completely unrelated fact about STAR format",
            confidence=0.8,
            session_count=1,
        )

        results = recall_semantic_memories(
            store, "u1", query="struggles with concurrency edge cases", session_count=1, k=1
        )
        assert len(results) == 1
        assert results[0]["text"] == "struggles with concurrency edge cases"

    def test_recall_excludes_non_text_items(self):
        """weak_areas (no 'text' field) never leaks into semantic recall."""
        from loop.memory import put_semantic_insight, put_weak_areas_state, recall_semantic_memories

        store = _indexed_store()
        put_weak_areas_state(store, "u1", ["Kafka"], 1)
        put_semantic_insight(
            store, "u1", topic="a", text="an insight", confidence=0.8, session_count=1
        )

        results = recall_semantic_memories(store, "u1", query="an insight", session_count=1)
        assert all("text" in r for r in results)
        assert any(r["text"] == "an insight" for r in results)

    def test_recall_respects_k(self):
        from loop.memory import put_semantic_insight, recall_semantic_memories

        store = _indexed_store()
        for i in range(5):
            put_semantic_insight(
                store,
                "u1",
                topic=f"t{i}",
                text=f"insight number {i}",
                confidence=0.5,
                session_count=1,
            )

        results = recall_semantic_memories(store, "u1", query="insight", session_count=1, k=2)
        assert len(results) == 2

    def test_recall_degrades_gracefully_without_index(self):
        """A plain InMemoryStore() (no index config) still returns items --
        just not similarity-ranked -- rather than raising."""
        from loop.memory import put_semantic_insight, recall_semantic_memories

        store = InMemoryStore()
        put_semantic_insight(
            store, "u1", topic="a", text="some insight", confidence=0.8, session_count=1
        )
        results = recall_semantic_memories(store, "u1", query="some insight", session_count=1)
        assert results and results[0]["text"] == "some insight"


class TestMemoryDecay:
    def test_stale_insight_excluded_from_recall(self):
        from loop.memory import put_semantic_insight, recall_semantic_memories

        store = _indexed_store()
        put_semantic_insight(
            store, "u1", topic="old", text="an old fact", confidence=0.8, session_count=1
        )

        # Session count has advanced far past the default TTL (10 sessions).
        results = recall_semantic_memories(
            store, "u1", query="an old fact", session_count=20, ttl=10
        )
        assert results == []

    def test_fresh_insight_within_ttl_is_recalled(self):
        from loop.memory import put_semantic_insight, recall_semantic_memories

        store = _indexed_store()
        put_semantic_insight(
            store, "u1", topic="recent", text="a recent fact", confidence=0.8, session_count=8
        )

        results = recall_semantic_memories(
            store, "u1", query="a recent fact", session_count=10, ttl=10
        )
        assert any(r["text"] == "a recent fact" for r in results)
