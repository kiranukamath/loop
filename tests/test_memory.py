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


def _compile_with_fresh_memory(graph):
    """Compile graph with brand-new checkpointer + store (test isolation)."""
    store = InMemoryStore()
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer, store=store), store


def _run_graph_with_memory(monkeypatch, store, checkpointer, user_id="u1", thread_id="t1"):
    """Run the full graph with fresh memory, returning the result and the store."""
    monkeypatch.setattr("loop.nodes.planner.get_chat_model", lambda: _fake_model(_STUB_PLAN))
    monkeypatch.setattr("loop.graph.grader", lambda state: {"grades": [_STUB_GRADE.model_dump()]})
    monkeypatch.setattr(
        "loop.graph.coach", lambda state: _persist_and_return(state, _STUB_FEEDBACK, store, user_id)
    )

    from loop.graph import build_graph
    from loop.state import initial_state

    app = build_graph().compile(checkpointer=checkpointer, store=store)
    state = initial_state()
    state["answers"] = [{"question_id": "cod-001", "text": "sliding window..."}]
    cfg = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    return app.invoke(state, config=cfg)


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

        # Stub grader and coach to avoid needing answers / model
        import loop.graph as gmod

        monkeypatch.setattr(gmod, "grader", lambda s: {"grades": [_STUB_GRADE.model_dump()]})
        monkeypatch.setattr(gmod, "coach", lambda s: {"weak_areas": []})

        app = build_graph().compile(store=store)
        app.invoke(state, config={"configurable": {"user_id": "kiran"}})

        assert "Kafka" in captured.get("weak_areas_text", "")
        assert "STAR storytelling" in captured.get("weak_areas_text", "")

    def test_planner_works_with_no_store(self, monkeypatch):
        """Planner works fine when compiled without a store (store=None path)."""
        monkeypatch.setattr("loop.nodes.planner.get_chat_model", lambda: _fake_model(_STUB_PLAN))
        monkeypatch.setattr("loop.graph.grader", lambda s: {"grades": [_STUB_GRADE.model_dump()]})
        monkeypatch.setattr("loop.graph.coach", lambda s: {"weak_areas": []})

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

        monkeypatch.setattr(gmod, "grader", lambda s: {"grades": [_STUB_GRADE.model_dump()]})
        monkeypatch.setattr(gmod, "coach", lambda s: {"weak_areas": []})

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
        monkeypatch.setattr("loop.graph.grader", lambda s: {"grades": [_STUB_GRADE.model_dump()]})

        from loop.graph import build_graph
        from loop.state import initial_state

        checkpointer = MemorySaver()
        app = build_graph().compile(checkpointer=checkpointer, store=store)
        state = initial_state()
        state["answers"] = [{"question_id": "cod-001", "text": "..."}]
        cfg = {"configurable": {"thread_id": "t1", "user_id": user_id}}
        return app.invoke(state, config=cfg)

    def test_coach_writes_weak_areas_to_store(self, monkeypatch):
        store = InMemoryStore()
        self._run_coach_in_graph(monkeypatch, store)
        item = store.get(("loop", "users"), "u1")
        assert item is not None
        assert set(_STUB_FEEDBACK.weak_areas_update) <= set(item.value["weak_areas"])

    def test_coach_sets_session_count(self, monkeypatch):
        store = InMemoryStore()
        self._run_coach_in_graph(monkeypatch, store)
        item = store.get(("loop", "users"), "u1")
        assert item.value["session_count"] == 1

    def test_coach_merges_with_existing_weak_areas(self, monkeypatch):
        store = InMemoryStore()
        # Pre-seed the store
        store.put(("loop", "users"), "u1", {"weak_areas": ["Kafka"], "session_count": 1})
        self._run_coach_in_graph(monkeypatch, store)
        item = store.get(("loop", "users"), "u1")
        # Should have both old and new areas
        assert "Kafka" in item.value["weak_areas"]
        for area in _STUB_FEEDBACK.weak_areas_update:
            assert area in item.value["weak_areas"]
        assert item.value["session_count"] == 2

    def test_coach_no_store_does_not_crash(self, monkeypatch):
        """Coach works fine when compiled without a store."""
        monkeypatch.setattr("loop.nodes.coach.get_chat_model", lambda: _fake_model(_STUB_FEEDBACK))
        monkeypatch.setattr("loop.nodes.planner.get_chat_model", lambda: _fake_model(_STUB_PLAN))
        monkeypatch.setattr("loop.graph.grader", lambda s: {"grades": [_STUB_GRADE.model_dump()]})

        from loop.graph import build_graph
        from loop.state import initial_state

        app = build_graph().compile()  # no store
        state = initial_state()
        state["answers"] = [{"question_id": "cod-001", "text": "..."}]
        result = app.invoke(state)
        assert result["weak_areas"] is not None


# ── Thread isolation ──────────────────────────────────────────────────────────


class TestThreadIsolation:
    def test_different_threads_have_independent_state(self, monkeypatch):
        """Two thread_ids produce independent checkpointed states."""
        monkeypatch.setattr("loop.nodes.planner.get_chat_model", lambda: _fake_model(_STUB_PLAN))
        monkeypatch.setattr("loop.graph.grader", lambda s: {"grades": [_STUB_GRADE.model_dump()]})
        monkeypatch.setattr("loop.graph.coach", lambda s: {"weak_areas": ["topic-A"]})

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
        monkeypatch.setattr("loop.graph.grader", lambda s: {"grades": [_STUB_GRADE.model_dump()]})
        monkeypatch.setattr("loop.graph.coach", lambda s: {"weak_areas": ["X"]})

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
        monkeypatch.setattr("loop.graph.grader", lambda s: {"grades": [_STUB_GRADE.model_dump()]})

        from loop.graph import build_graph
        from loop.state import initial_state

        checkpointer = MemorySaver()
        store = InMemoryStore()
        app = build_graph().compile(checkpointer=checkpointer, store=store)

        state = initial_state()
        state["answers"] = [{"question_id": "cod-001", "text": "..."}]

        # Session 1: coach writes weak_areas to store
        cfg1 = {"configurable": {"thread_id": "s1", "user_id": "kiran"}}
        app.invoke(state, config=cfg1)

        # Session 2: planner should read stored weak_areas
        cfg2 = {"configurable": {"thread_id": "s2", "user_id": "kiran"}}
        app.invoke(state, config=cfg2)

        # Two planner calls captured; second one should mention weak areas
        assert len(captured_prompts) == 2
        for area in _STUB_FEEDBACK.weak_areas_update:
            assert area in captured_prompts[1], (
                f"Stored weak area '{area}' not found in second session planner prompt"
            )
