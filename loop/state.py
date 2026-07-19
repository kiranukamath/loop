"""
LoopState — the single typed dict that flows through every node in the graph.

Think of this as the "request context" accumulator: each node reads what it
needs, returns only the fields it changed, and LangGraph merges the update
back in before the next node runs.

Fields are added phase by phase — placeholders for later phases are typed
Optional so the graph can start with partial state.
"""

from __future__ import annotations

from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def _append_list(left: list | None, right: list | None) -> list:
    """Reducer: append right to left, treating None as an empty list.

    Used for fields (grades, answers) that accumulate across loop iterations.
    Without this, a second session's grader would overwrite the first's grades.
    Same principle as add_messages, but for plain dicts rather than BaseMessage.
    """
    return (left or []) + (right or [])


def _reset_or_append(left: list | None, right: list | None) -> list:
    """Reducer for state["panel_grades"] (Phase 13a): append like _append_list,
    but a `right` of None means "reset", not "append nothing".

    Why this exists: an append-only reducer channel can't be cleared by a
    normal node return. `{"panel_grades": []}` does NOT clear it — the
    reducer runs as `(left or []) + ([] or [])`, which is just `left`
    unchanged (verified empirically: InvalidUpdateError-free, but no-op).
    `grade_aggregator` needs to wipe the scratch channel after reducing one
    question's persona partials into a final Grade, so the *reset signal*
    has to be something a normal partial-write never sends: None, which
    _append_list already treats as "nothing to append" (also a no-op).  This
    reducer instead treats None as "clear it" — safe here because the only
    two writers are panel_grader (always appends a real partial dict, never
    None) and grade_aggregator (which sends None exactly when it means reset).
    """
    if right is None:
        return []
    return (left or []) + right


class LoopState(dict):
    """
    LangGraph state for the interview coach.

    We extend dict rather than TypedDict so IDE tooling and LangGraph both
    work smoothly with Python 3.14.  The fields below document what keys
    the graph uses — each node accesses them via state["key"].

    Reducer note: `messages` uses the add_messages reducer (via Annotated)
    so node updates APPEND to the list rather than replacing it.  Every
    other field is a plain overwrite.
    """

    # ── Input (set by intake, never changed) ────────────────────────────────
    jd: str  # job description text
    profile: str  # candidate profile text
    company: Optional[str]  # target company name — set by intake, Phase 9

    # ── Phase 9 (tool-calling research agent) ────────────────────────────────
    company_research: Optional[dict]  # CompanyResearch.model_dump(), or None if skipped

    # ── Conversation messages (reducer: append, not overwrite) ───────────────
    # Annotated[list[BaseMessage], add_messages] tells LangGraph to call
    # add_messages(old, new) instead of old = new when this field is updated.
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Phase 2+ (Planning) ──────────────────────────────────────────────────
    plan: Optional[dict]  # PrepPlan — added in Phase 2

    # ── Phase 3+ (Interview loop) ────────────────────────────────────────────
    current_modality: Optional[str]  # "coding" | "system_design" | "behavioral"
    current_question_id: Optional[str]
    # ── Phase 8b (semantic question selection) ───────────────────────────────
    current_focus: Optional[str]  # session.focus — what the interviewer searches for
    current_topics: Optional[list[str]]  # session.topics — folded into the search query
    # Annotated with _append_list so successive sessions accumulate, not overwrite.
    answers: Annotated[Optional[list[dict]], _append_list]
    grades: Annotated[Optional[list[dict]], _append_list]

    # ── Phase 4+ (Memory) ────────────────────────────────────────────────────
    weak_areas: Optional[list[str]]  # topic strings that need more work
    session_number: Optional[int]

    # ── Phase 5+ (HITL) ──────────────────────────────────────────────────────
    plan_approved: Optional[bool]
    readiness_verdict: Optional[dict]  # ReadinessVerdict.model_dump() — may include override_reason
    verdict_approved: Optional[bool]

    # ── Phase 7+ (multi-session loop) ────────────────────────────────────────
    session_index: Optional[int]  # index into plan["sessions"]; starts at 0

    # ── Phase 10b (guardrails) ────────────────────────────────────────────────
    # Accumulates across the whole session (reducer: append), one entry per
    # piece of input that tripped detect_injection(). Empty/None = nothing
    # flagged. See loop/guardrails.py.
    flagged_inputs: Annotated[Optional[list[dict]], _append_list]

    # ── Phase 13a (panel grading — parallel fan-out/fan-in) ──────────────────
    # Transient scratch channel: each panel_grader Send writes one persona's
    # partial grade here; grade_aggregator reduces every partial for the
    # current question into one Grade, then resets this to [] by returning
    # None (see _reset_or_append above). Never read outside panel.py.
    panel_grades: Annotated[Optional[list[dict]], _reset_or_append]

    # ── Phase 13b (supervisor + handoffs) ────────────────────────────────────
    # Traceability only — one entry per interview_supervisor decision
    # (which modality it picked and why). Not read by any node; a window into
    # the model's reasoning for debugging/observability, the same role
    # Langfuse traces play for individual model calls.
    supervisor_decisions: Annotated[Optional[list[dict]], _append_list]


def initial_state() -> dict:
    """Return a blank starting state with safe defaults for all optional fields.

    Pass this as the initial input to compiled_graph.invoke().
    Every field that isn't set by intake is None; the graph fills them in
    as it progresses through phases.
    """
    return {
        "jd": "",
        "profile": "",
        "company": None,
        "company_research": None,
        "messages": [],
        "plan": None,
        "current_modality": None,
        "current_question_id": None,
        "current_focus": None,
        "current_topics": None,
        "answers": None,
        "grades": None,
        "weak_areas": None,
        "session_number": 1,
        "plan_approved": None,
        "readiness_verdict": None,
        "verdict_approved": None,
        "session_index": 0,
        "flagged_inputs": None,
        "panel_grades": None,
        "supervisor_decisions": None,
    }
