"""
FastAPI backend for Loop — Phase 7d.

Endpoints
─────────
  POST /sessions
      Start a new interview session.  Returns thread_id.
      No graph run yet — browser opens the SSE stream to kick off the first run.

  GET  /sessions/{thread_id}/stream?user_id=kiran
      SSE stream of one graph segment (start → first interrupt, or resume → next interrupt).
      Emits:
        data: {"type": "node",      "node": "<name>", ...state fields}
        data: {"type": "interrupt", "action": "<name>", ...payload}
        data: {"type": "done"}
      Closes after the interrupt or done event — browser re-opens for each new segment.

  POST /sessions/{thread_id}/resume
      Deliver the human's response at any gate.  Stores the Command so the next
      GET /stream call picks it up and runs the next graph segment.
      Body shape (discriminated by "action"):
        approve_plan:    {"action": "approve_plan", "decision": "approve"|"edit"|"reject",
                          "updated_plan": {...}}
        answer_question: {"action": "answer_question", "answer": "sliding window..."}
        approve_verdict: {"action": "approve_verdict", "decision": "approve"|"override",
                          "verdict": "ready"|"not_ready"}

  GET  /sessions   (Phase 11a)
      List every past session, newest first. Reads the checkpointer directly —
      no separate "sessions" table is maintained.
        {"sessions": [{"thread_id", "started_at", "verdict", "sessions_completed"}, ...],
         "persistence": "sqlite" | "none"}
      "none" (with an empty list) when running on MemorySaver — it holds no
      durable history once the process exits, so there is nothing to list.

  GET  /sessions/{thread_id}/history   (Phase 11a)
      Full structured interview timeline for one session, read from the FINAL
      checkpoint snapshot (LangGraph merges each node's delta into the full
      state, so the last snapshot already has everything — no replay needed).
      404 if the thread_id has no checkpoint.

Analogy (Spring):
  compile_graph_with_memory() ≈ ApplicationContext.getBean(Graph)
  GET /stream                 ≈ SseEmitter — one per segment, closed after interrupt/done
  POST /resume                ≈ writing to a BlockingQueue that the SSE endpoint drains

Threading note:
  LangGraph graph.stream() is synchronous.  FastAPI runs sync routes in a threadpool
  automatically, so the event loop is never blocked.
"""

from __future__ import annotations

import json
import pathlib
import uuid
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel

from loop.budget import BudgetCallbackHandler, BudgetExceeded, SessionBudget
from loop.graph import compile_graph_with_memory
from loop.state import initial_state

app = FastAPI(title="Loop — Interview Coach API", version="0.1.0")

# Serve the Tailwind UI from loop/static/
_STATIC_DIR = pathlib.Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect browser root to the single-page UI."""
    return RedirectResponse(url="/static/index.html")


# Shared compiled graph — uses memory.py singletons (SqliteSaver + InMemoryStore).
_graph = compile_graph_with_memory()

# ── Per-session pending state ──────────────────────────────────────────────────
# Keyed by thread_id.
# Value is one of:
#   None         → thread exists but has not been started yet (fresh initial_state)
#   Command(...)  → a resume command ready for the next GET /stream call
#
# This is process-local memory (not persisted).  If the server restarts mid-session,
# the SQLite checkpoint survives but the pending-command dict is lost.  v2 fix:
# store pending commands in Redis or a DB table.
_pending: dict[str, Command | None] = {}

# Phase 10c — per-session token/cost accountant, keyed by thread_id.
# Same lifecycle caveat as _pending: process-local, lost on restart.
_budgets: dict[str, SessionBudget] = {}


# ── Request / response models ──────────────────────────────────────────────────


class StartResponse(BaseModel):
    thread_id: str
    user_id: str


class ResumeRequest(BaseModel):
    action: str
    # approve_plan / approve_verdict
    decision: str | None = None
    updated_plan: dict | None = None
    # approve_verdict override
    verdict: str | None = None
    reason: str | None = None
    # answer_question
    answer: str | None = None


class ResumeResponse(BaseModel):
    status: str  # "queued"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _build_command(body: ResumeRequest) -> Command:
    """Translate a ResumeRequest body into a LangGraph Command(resume=...)."""
    if body.action == "approve_plan":
        payload: dict = {"decision": body.decision or "approve"}
        if body.decision == "edit" and body.updated_plan:
            payload["updated_plan"] = body.updated_plan
        return Command(resume=payload)

    if body.action == "answer_question":
        if not body.answer:
            raise HTTPException(status_code=422, detail="answer is required for answer_question")
        return Command(resume=body.answer)

    if body.action == "approve_verdict":
        payload = {"decision": body.decision or "approve"}
        if body.decision == "override":
            payload["verdict"] = body.verdict or "not_ready"
            payload["reason"] = body.reason or ""
        return Command(resume=payload)

    raise HTTPException(status_code=422, detail=f"Unknown action: {body.action!r}")


# Fields safe to include in SSE payloads (serialisable, not too large).
_SAFE_FIELDS = frozenset(
    {
        "current_modality",
        "session_number",
        "session_index",
        "current_question_id",
        "plan_approved",
        "verdict_approved",
        "weak_areas",
    }
)


def _safe_payload(updates: dict) -> dict:
    """Extract SSE-safe state fields from a node's update dict."""
    out: dict = {}
    for key in _SAFE_FIELDS:
        if key in updates:
            out[key] = updates[key]
    # Grades: include only scores (skip heavy text)
    if "grades" in updates:
        out["grades"] = [
            {"question_id": g.get("question_id"), "score": g.get("score")}
            for g in (updates["grades"] or [])
        ]
    # Plan: include summary only
    if "plan" in updates and updates["plan"]:
        p = updates["plan"]
        out["plan_summary"] = {
            "total_sessions": p.get("total_sessions"),
            "key_gaps": p.get("key_gaps", [])[:3],
        }
    # Readiness verdict
    if "readiness_verdict" in updates and updates["readiness_verdict"]:
        v = updates["readiness_verdict"]
        out["readiness_verdict"] = {
            "verdict": v.get("verdict"),
            "confidence": v.get("confidence"),
            "gaps": (v.get("gaps") or [])[:3],
        }
    # Phase 10b: surface any guardrail flags raised by this node (e.g. a
    # suspected prompt-injection attempt in the JD or an answer).
    if updates.get("flagged_inputs"):
        out["flagged_inputs"] = updates["flagged_inputs"]
    return out


def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.post("/sessions", response_model=StartResponse)
def create_session(user_id: str = "default") -> StartResponse:
    """Create a new interview session and return a thread_id.

    The graph is NOT started here — the browser opens GET /stream to kick off
    the first run.  This mirrors the pattern where a controller creates a job
    record before the worker picks it up.
    """
    thread_id = str(uuid.uuid4())
    _pending[thread_id] = None  # None = fresh start, not yet running
    return StartResponse(thread_id=thread_id, user_id=user_id)


@app.post("/sessions/{thread_id}/resume", response_model=ResumeResponse)
def resume_session(thread_id: str, body: ResumeRequest) -> ResumeResponse:
    """Store a human response for the next GET /stream call.

    The browser calls this after seeing an 'interrupt' SSE event, then
    immediately re-opens the SSE stream to continue the graph run.
    """
    if thread_id not in _pending and thread_id not in _pending:
        # Thread_id unknown — but we allow it anyway; SQLite may have the checkpoint.
        pass
    _pending[thread_id] = _build_command(body)
    return ResumeResponse(status="queued")


@app.get("/sessions/{thread_id}/stream")
def stream_session(thread_id: str, user_id: str = "default") -> StreamingResponse:
    """SSE stream for one graph segment.

    Runs the graph from initial_state (first call) or from a Command (resume),
    emitting an SSE event after each node completes.  Closes the stream once
    the graph hits an interrupt or finishes.

    The browser re-opens this endpoint after each interrupt (once it has POSTed
    the human response to /resume).
    """

    def generate() -> Iterator[str]:
        # Phase 10c: one SessionBudget per thread_id, reused across every
        # segment of the same session so tokens accumulate for the whole
        # interview, not just one SSE stream call.
        budget = _budgets.setdefault(thread_id, SessionBudget())
        budget_cb = BudgetCallbackHandler(budget)

        config = {
            "configurable": {"thread_id": thread_id, "user_id": user_id},
            "callbacks": [budget_cb],
        }

        # Determine what to feed to graph.stream()
        cmd = _pending.pop(thread_id, None)
        if cmd is None and thread_id not in _pending:
            # cmd was None → fresh start; or thread_id was never registered → still fresh
            stream_input = initial_state()
        else:
            # cmd is a Command — resume from checkpoint
            stream_input = cmd

        try:
            # stream_mode="updates" → yields {node_name: state_delta} after every node.
            # When interrupt() is called, LangGraph emits {"__interrupt__": (...,)} instead.
            for chunk in _graph.stream(stream_input, config=config, stream_mode="updates"):
                for node, updates in chunk.items():
                    if node == "__interrupt__":
                        # updates is a tuple of Interrupt objects
                        for ipt in updates:
                            yield _sse({"type": "interrupt", **ipt.value})
                        return  # close stream after interrupt

                    if node.startswith("__"):
                        continue  # skip internal LangGraph nodes

                    payload = {"type": "node", "node": node}
                    payload.update(_safe_payload(updates))
                    payload["tokens_used"] = budget.tokens_used
                    payload["cost_usd"] = round(budget.cost_usd, 6)
                    yield _sse(payload)
        except BudgetExceeded as exc:
            # Stop the session gracefully — the browser sees a clear message
            # instead of a raw 500 / stack trace.
            yield _sse(
                {
                    "type": "error",
                    "reason": "budget_exceeded",
                    "message": str(exc),
                    "tokens_used": budget.tokens_used,
                    "cost_usd": round(budget.cost_usd, 6),
                }
            )
            return

        # Graph reached END without an interrupt
        yield _sse(
            {
                "type": "done",
                "tokens_used": budget.tokens_used,
                "cost_usd": round(budget.cost_usd, 6),
            }
        )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


# ── Phase 11a: session history ────────────────────────────────────────────────


def _sqlite_conn():
    """Return the underlying sqlite3.Connection if _graph is SqliteSaver-backed.

    None means MemorySaver (or any other non-SQLite checkpointer) — no durable
    thread listing is possible, since LangGraph itself has no "list all
    threads" API; we query the checkpointer's own storage directly.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    checkpointer = _graph.checkpointer
    if isinstance(checkpointer, SqliteSaver):
        return checkpointer.conn
    return None


@app.get("/sessions")
def list_sessions() -> dict:
    """List every past session, newest first (Phase 11a).

    MemorySaver holds no durable history once the process exits, so we return
    an empty list with persistence="none" rather than pretending to have data.
    """
    conn = _sqlite_conn()
    if conn is None:
        return {"sessions": [], "persistence": "none"}

    # checkpoint_id is a time-sortable UUID6 — MAX() per thread_id gives the
    # most recent checkpoint without needing a separate timestamp column.
    rows = conn.execute(
        "SELECT thread_id, MAX(checkpoint_id) AS latest FROM checkpoints "
        "WHERE checkpoint_ns = '' GROUP BY thread_id ORDER BY latest DESC"
    ).fetchall()

    sessions = []
    for thread_id, _latest in rows:
        snapshot = _graph.get_state({"configurable": {"thread_id": thread_id}})
        values = snapshot.values or {}
        plan = values.get("plan") or {}
        verdict = values.get("readiness_verdict") or {}
        sessions.append(
            {
                "thread_id": thread_id,
                "started_at": snapshot.created_at,
                "verdict": verdict.get("verdict"),
                "sessions_completed": values.get("session_index", 0),
                "total_sessions": plan.get("total_sessions"),
            }
        )
    return {"sessions": sessions, "persistence": "sqlite"}


@app.get("/sessions/{thread_id}/history")
def get_session_history(thread_id: str) -> dict:
    """Full structured interview timeline for one session (Phase 11a).

    Reads only the FINAL checkpoint snapshot — get_state_history() yields
    snapshots newest-first, and the newest one already holds the fully
    accumulated state (every node's delta merged in), so there is no need to
    replay the whole history to reconstruct it.
    """
    config = {"configurable": {"thread_id": thread_id}}
    history = _graph.get_state_history(config)
    snapshot = next(history, None)

    if snapshot is None or not snapshot.values:
        raise HTTPException(status_code=404, detail=f"No session found for thread_id={thread_id!r}")

    values = snapshot.values
    plan = values.get("plan") or {}
    answers = {a["question_id"]: a for a in (values.get("answers") or [])}
    grades = {g["question_id"]: g for g in (values.get("grades") or [])}

    from loop.tools import get_question_by_id

    sessions = []
    for question_id, grade in grades.items():
        question = get_question_by_id(question_id) or {}
        answer = answers.get(question_id) or {}
        sessions.append(
            {
                "question_id": question_id,
                "question_title": question.get("title"),
                "question_prompt": question.get("prompt"),
                "answer": answer.get("text"),
                "grade": grade,
            }
        )

    return {
        "thread_id": thread_id,
        "started_at": snapshot.created_at,
        "plan": {
            "role_summary": plan.get("role_summary"),
            "total_sessions": plan.get("total_sessions"),
            "key_gaps": plan.get("key_gaps", []),
        }
        if plan
        else None,
        "sessions": sessions,
        "weak_areas": values.get("weak_areas") or [],
        "readiness_verdict": values.get("readiness_verdict"),
    }
