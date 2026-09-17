"""
Phase 7d API tests — FastAPI endpoints.
All offline: graph nodes are stubbed so no model calls are made.

Tests verify:
- POST /sessions creates a unique thread_id and registers the session
- POST /sessions/{id}/resume with each action shape returns 200 + "queued"
- POST /sessions/{id}/resume with an unknown action returns 422
- GET  /sessions/{id}/stream yields SSE events in the correct format
- GET  /sessions/{id}/stream emits "interrupt" events at HITL gates
- GET  /sessions/{id}/stream emits "done" when the graph completes
- Two concurrent threads produce independent SSE streams
"""

import json

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from loop.schemas import Feedback, Grade, PrepPlan, ReadinessVerdict, Session

# ── Shared stubs ──────────────────────────────────────────────────────────────

_STUB_PLAN = PrepPlan(
    role_summary="stub",
    total_sessions=1,
    sessions=[Session(session_number=1, modality="coding", topics=["x"], focus="y")],
    key_gaps=["Kafka"],
    rationale="stub",
)

_STUB_GRADE = Grade(
    question_id="cod-001",
    score=7,
    criterion_scores={
        "correctness": 3,
        "optimal_complexity": 2,
        "code_quality": 1,
        "communication": 1,
    },
    strengths=["ok"],
    improvements=[],
    overall_feedback="ok",
)

_STUB_VERDICT = ReadinessVerdict(
    verdict="ready",
    confidence=0.85,
    strengths=["ok"],
    gaps=[],
    recommendation="go",
)

_STUB_FEEDBACK = Feedback(
    summary="ok",
    action_items=[],
    weak_areas_update=["binary-search"],
)


def _fake_model(rv):
    from unittest.mock import MagicMock

    from langchain_core.runnables import RunnableLambda

    fake = MagicMock()
    fake.with_structured_output.return_value = RunnableLambda(lambda _: rv)
    return fake


# ── Client fixture ────────────────────────────────────────────────────────────


@pytest.fixture()
def client(monkeypatch):
    """TestClient with all model-calling nodes stubbed.

    plan_approval, interviewers, and readiness use their real implementations
    so we exercise the actual interrupt mechanics.  Models + grader are stubbed.
    """
    monkeypatch.setattr("loop.nodes.planner.get_chat_model", lambda: _fake_model(_STUB_PLAN))
    monkeypatch.setattr("loop.nodes.coach.get_chat_model", lambda: _fake_model(_STUB_FEEDBACK))
    monkeypatch.setattr("loop.nodes.readiness.get_chat_model", lambda: _fake_model(_STUB_VERDICT))

    # Stub interviewers — bypass the answer-gate interrupt
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
    monkeypatch.setattr(
        "loop.graph.grader",
        lambda s: {"grades": [_STUB_GRADE.model_dump()]},
    )

    # Replace the module-level _graph with a fresh MemorySaver instance so tests
    # are isolated from each other (no shared SQLite file).
    from loop.graph import build_graph

    fresh_graph = build_graph().compile(checkpointer=MemorySaver())
    import loop.api as api_mod

    monkeypatch.setattr(api_mod, "_graph", fresh_graph)

    # Phase 18a: pending-resume + budget state now lives in loop/memory.py's
    # long-term store (so it can be Postgres-backed and survive a restart),
    # not a process-local dict. Point api.py at a brand-new InMemoryStore per
    # test so tests stay isolated from each other exactly like the old
    # `_pending = {}` reset did.
    from langgraph.store.memory import InMemoryStore

    fresh_store = InMemoryStore()
    monkeypatch.setattr(api_mod, "get_store_instance", lambda: fresh_store)

    from loop.api import app

    return TestClient(app)


def _parse_sse(raw: str) -> list[dict]:
    """Parse raw SSE text into a list of event dicts."""
    events = []
    for line in raw.strip().splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


# ── POST /sessions ────────────────────────────────────────────────────────────


class TestCreateSession:
    def test_returns_200(self, client):
        resp = client.post("/sessions")
        assert resp.status_code == 200

    def test_returns_thread_id(self, client):
        resp = client.post("/sessions")
        data = resp.json()
        assert "thread_id" in data
        assert len(data["thread_id"]) > 0

    def test_thread_ids_are_unique(self, client):
        ids = {client.post("/sessions").json()["thread_id"] for _ in range(5)}
        assert len(ids) == 5

    def test_default_user_id(self, client):
        resp = client.post("/sessions")
        assert resp.json()["user_id"] == "default"

    def test_custom_user_id(self, client):
        resp = client.post("/sessions?user_id=kiran")
        assert resp.json()["user_id"] == "kiran"


# ── POST /sessions/{id}/resume ────────────────────────────────────────────────


class TestResumeEndpoint:
    def _new_thread(self, client) -> str:
        return client.post("/sessions").json()["thread_id"]

    def test_approve_plan_returns_200(self, client):
        tid = self._new_thread(client)
        resp = client.post(
            f"/sessions/{tid}/resume",
            json={"action": "approve_plan", "decision": "approve"},
        )
        assert resp.status_code == 200

    def test_approve_plan_status_queued(self, client):
        tid = self._new_thread(client)
        resp = client.post(
            f"/sessions/{tid}/resume",
            json={"action": "approve_plan", "decision": "approve"},
        )
        assert resp.json()["status"] == "queued"

    def test_answer_question_returns_200(self, client):
        tid = self._new_thread(client)
        resp = client.post(
            f"/sessions/{tid}/resume",
            json={"action": "answer_question", "answer": "sliding window"},
        )
        assert resp.status_code == 200

    def test_answer_question_missing_answer_returns_422(self, client):
        tid = self._new_thread(client)
        resp = client.post(f"/sessions/{tid}/resume", json={"action": "answer_question"})
        assert resp.status_code == 422

    def test_approve_verdict_returns_200(self, client):
        tid = self._new_thread(client)
        resp = client.post(
            f"/sessions/{tid}/resume",
            json={"action": "approve_verdict", "decision": "approve"},
        )
        assert resp.status_code == 200

    def test_unknown_action_returns_422(self, client):
        tid = self._new_thread(client)
        resp = client.post(f"/sessions/{tid}/resume", json={"action": "teleport"})
        assert resp.status_code == 422


# ── GET /sessions/{id}/stream ─────────────────────────────────────────────────


class TestStreamEndpoint:
    def _start(self, client, user_id="test-user") -> tuple[str, list[dict]]:
        """Create a session and stream until first interrupt/done."""
        tid = client.post(f"/sessions?user_id={user_id}").json()["thread_id"]
        resp = client.get(f"/sessions/{tid}/stream?user_id={user_id}")
        assert resp.status_code == 200
        return tid, _parse_sse(resp.text)

    def test_stream_returns_200(self, client):
        tid = client.post("/sessions").json()["thread_id"]
        resp = client.get(f"/sessions/{tid}/stream")
        assert resp.status_code == 200

    def test_stream_content_type_is_sse(self, client):
        tid = client.post("/sessions").json()["thread_id"]
        resp = client.get(f"/sessions/{tid}/stream")
        assert "text/event-stream" in resp.headers["content-type"]

    def test_first_segment_emits_node_events(self, client):
        _, events = self._start(client)
        node_events = [e for e in events if e.get("type") == "node"]
        assert len(node_events) > 0

    def test_first_segment_includes_intake_and_planner(self, client):
        _, events = self._start(client)
        nodes = {e["node"] for e in events if e.get("type") == "node"}
        assert "intake" in nodes
        assert "planner" in nodes

    def test_first_segment_pauses_at_plan_approval(self, client):
        _, events = self._start(client)
        interrupt_events = [e for e in events if e.get("type") == "interrupt"]
        assert len(interrupt_events) == 1
        assert interrupt_events[0]["action"] == "approve_plan"

    def test_interrupt_event_contains_plan(self, client):
        _, events = self._start(client)
        ipt = next(e for e in events if e.get("type") == "interrupt")
        assert "plan" in ipt

    def test_resume_approve_plan_and_stream_reaches_done(self, client):
        """Full flow: start → approve plan → answer gate stubbed → approve verdict → done."""
        tid, events1 = self._start(client)

        # Gate 1: approve plan
        client.post(
            f"/sessions/{tid}/resume",
            json={"action": "approve_plan", "decision": "approve"},
        )
        events2 = _parse_sse(client.get(f"/sessions/{tid}/stream").text)

        # Interviewers are stubbed — no answer gate interrupt; graph runs to readiness.
        # readiness calls interrupt() so gate 2 fires.
        interrupt2 = next((e for e in events2 if e.get("type") == "interrupt"), None)
        assert interrupt2 is not None, "Expected readiness interrupt after plan approval"
        assert interrupt2["action"] == "approve_verdict"

        # Gate 2: approve verdict
        client.post(
            f"/sessions/{tid}/resume",
            json={"action": "approve_verdict", "decision": "approve"},
        )
        events3 = _parse_sse(client.get(f"/sessions/{tid}/stream").text)

        done = next((e for e in events3 if e.get("type") == "done"), None)
        assert done is not None, "Expected 'done' event after final approval"

    def test_node_events_have_type_and_node_fields(self, client):
        _, events = self._start(client)
        for e in events:
            if e.get("type") == "node":
                assert "node" in e

    def test_two_threads_are_independent(self, client):
        """Each session's SSE stream is isolated by thread_id."""
        tid1 = client.post("/sessions?user_id=alice").json()["thread_id"]
        tid2 = client.post("/sessions?user_id=bob").json()["thread_id"]

        events1 = _parse_sse(client.get(f"/sessions/{tid1}/stream?user_id=alice").text)
        events2 = _parse_sse(client.get(f"/sessions/{tid2}/stream?user_id=bob").text)

        # Both hit plan_approval interrupt independently
        ipt1 = next(e for e in events1 if e.get("type") == "interrupt")
        ipt2 = next(e for e in events2 if e.get("type") == "interrupt")
        assert ipt1["action"] == "approve_plan"
        assert ipt2["action"] == "approve_plan"


# ── GET /sessions (Phase 11a) ─────────────────────────────────────────────────


class TestListSessions:
    def test_memorysaver_backend_returns_empty_with_none_persistence(self, client):
        """The `client` fixture wires a fresh MemorySaver — no durable history."""
        resp = client.get("/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert data == {"sessions": [], "persistence": "none"}

    def test_sqlite_backend_lists_completed_session(self, client, monkeypatch, tmp_path):
        """A SqliteSaver-backed graph shows up in the listing after a full run."""
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        import loop.api as api_mod
        from loop.graph import build_graph

        conn = sqlite3.connect(str(tmp_path / "sessions.sqlite"), check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        sqlite_graph = build_graph().compile(checkpointer=saver)
        monkeypatch.setattr(api_mod, "_graph", sqlite_graph)

        tid = "sqlite-thread-1"
        config = {"configurable": {"thread_id": tid}}
        from loop.state import initial_state

        result = sqlite_graph.invoke(initial_state(), config=config)
        assert "__interrupt__" in result  # plan_approval gate
        sqlite_graph.invoke({"decision": "approve"}, config=config)

        resp = client.get("/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["persistence"] == "sqlite"
        thread_ids = {s["thread_id"] for s in data["sessions"]}
        assert tid in thread_ids

    def test_unknown_action_returns_422_still_works_alongside(self, client):
        """Sanity: adding /sessions didn't disturb the existing resume validation."""
        tid = client.post("/sessions").json()["thread_id"]
        resp = client.post(f"/sessions/{tid}/resume", json={"action": "teleport"})
        assert resp.status_code == 422


# ── Phase 18a: durable session state (pending resume + budget) ───────────────
#
# Before Phase 18a, `_pending`/`_budgets` were process-local dicts — lost on
# restart even when the graph's own checkpoint (SqliteSaver/PostgresSaver)
# survived. They now live in loop/memory.py's long-term STORE (see
# loop/api.py's _SESSION_NAMESPACE helpers), so a "restart" that keeps the
# store (e.g. a real PostgresStore) keeps this too. These tests exercise the
# helper functions directly against one InMemoryStore instance — the offline
# stand-in for a durable Postgres-backed store — without needing api.py's
# module-level dicts at all (there are none left).


class TestDurableSessionState:
    def test_pending_resume_survives_across_separate_lookups(self, monkeypatch):
        """Simulates a restart: nothing process-local is read back — every
        helper call re-reads the (shared) store from scratch."""
        from langgraph.store.memory import InMemoryStore

        import loop.api as api_mod

        shared_store = InMemoryStore()
        monkeypatch.setattr(api_mod, "get_store_instance", lambda: shared_store)

        api_mod._register_session("thread-durable-1")
        api_mod._set_pending_resume("thread-durable-1", {"decision": "approve"})

        # A brand-new call to _pop_pending_command reads straight from the
        # store — nothing cached in a Python dict survives "between" these
        # two lines except what's in shared_store, exactly modelling a
        # process restart that keeps the store but not any local state.
        cmd, was_registered = api_mod._pop_pending_command("thread-durable-1")
        assert was_registered is True
        assert cmd is not None
        assert cmd.resume == {"decision": "approve"}

        # Once popped, it's gone (matches the old dict.pop() semantics).
        cmd2, _ = api_mod._pop_pending_command("thread-durable-1")
        assert cmd2 is None

    def test_unknown_thread_is_not_registered(self, monkeypatch):
        from langgraph.store.memory import InMemoryStore

        import loop.api as api_mod

        monkeypatch.setattr(api_mod, "get_store_instance", lambda: InMemoryStore())
        cmd, was_registered = api_mod._pop_pending_command("never-created")
        assert cmd is None
        assert was_registered is False

    def test_budget_round_trips_through_the_store(self, monkeypatch):
        from langgraph.store.memory import InMemoryStore

        import loop.api as api_mod
        from loop.budget import SessionBudget

        shared_store = InMemoryStore()
        monkeypatch.setattr(api_mod, "get_store_instance", lambda: shared_store)

        budget = SessionBudget()
        budget.tokens_used = 1234
        budget.cost_usd = 0.0567
        api_mod._save_budget("thread-durable-2", budget)

        # Fresh load — simulates reading it back after a restart.
        reloaded = api_mod._load_budget("thread-durable-2")
        assert reloaded.tokens_used == 1234
        assert reloaded.cost_usd == 0.0567

    def test_fresh_thread_loads_a_zeroed_budget(self, monkeypatch):
        from langgraph.store.memory import InMemoryStore

        import loop.api as api_mod

        monkeypatch.setattr(api_mod, "get_store_instance", lambda: InMemoryStore())
        budget = api_mod._load_budget("never-seen-thread")
        assert budget.tokens_used == 0
        assert budget.cost_usd == 0.0

    def test_answer_question_resume_stores_a_plain_string(self, monkeypatch):
        """answer_question's resume value is a bare string, not a dict —
        confirms the store round-trips both shapes correctly."""
        from langgraph.store.memory import InMemoryStore

        import loop.api as api_mod

        shared_store = InMemoryStore()
        monkeypatch.setattr(api_mod, "get_store_instance", lambda: shared_store)

        api_mod._register_session("thread-durable-3")
        api_mod._set_pending_resume("thread-durable-3", "sliding window answer")

        cmd, _ = api_mod._pop_pending_command("thread-durable-3")
        assert cmd.resume == "sliding window answer"


# ── GET /sessions/{thread_id}/history (Phase 11a) ─────────────────────────────


class TestSessionHistory:
    def test_unknown_thread_id_returns_404(self, client):
        resp = client.get("/sessions/nonexistent-thread/history")
        assert resp.status_code == 404

    def test_returns_plan_and_verdict_after_full_run(self, client):
        """Full flow through both gates, then fetch the history for that thread."""
        tid, events1 = TestStreamEndpoint()._start(client)

        client.post(
            f"/sessions/{tid}/resume", json={"action": "approve_plan", "decision": "approve"}
        )
        client.get(f"/sessions/{tid}/stream")  # runs interviewer→grader→coach→readiness

        client.post(
            f"/sessions/{tid}/resume", json={"action": "approve_verdict", "decision": "approve"}
        )
        client.get(f"/sessions/{tid}/stream")  # reaches END

        resp = client.get(f"/sessions/{tid}/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["thread_id"] == tid
        assert data["plan"]["total_sessions"] == 1
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["question_id"] == "cod-001"
        assert data["sessions"][0]["grade"]["score"] == 7
        assert data["readiness_verdict"]["verdict"] == "ready"
