"""
Coach node — turns session grades into actionable feedback.

Reads:  state["grades"]
Writes: state["weak_areas"]  (list of topics to focus on next session)
        long-term store[user_id]["weak_areas"]  (Phase 4: cross-session memory)

The weak_areas written to the store are read by the planner at the START of the
next session, closing the feedback loop: grade → store → next plan adapts.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from loop.models import get_chat_model
from loop.observability import get_langfuse_callback
from loop.schemas import Feedback

_SYSTEM = """\
You are a supportive technical interview coach.

You receive a candidate's grades from a mock interview session.
Synthesize the feedback into clear, actionable guidance.
Be specific, constructive, and encouraging.
Focus on the most impactful things to improve before the next session.
"""

_HUMAN = """\
Session grades:
{grades_summary}

Provide coaching feedback with concrete action items and the top weak areas
this candidate should focus on before the next interview."""

_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])


# ── Store helper ──────────────────────────────────────────────────────────────


def _persist_weak_areas(new_areas: list[str]) -> None:
    """Merge new weak_areas into the long-term store for this user.

    No-op when called outside a graph context or when no store is wired.
    Merges (not replaces) so areas accumulate across sessions.
    """
    try:
        from langgraph.config import get_config, get_store

        store = get_store()
        if store is None:
            return
        cfg = get_config()
        user_id = cfg.get("configurable", {}).get("user_id", "default")
    except RuntimeError:
        return

    existing = store.get(("loop", "users"), user_id)
    existing_areas = existing.value.get("weak_areas", []) if existing else []
    session_count = (existing.value.get("session_count", 0) if existing else 0) + 1

    # Merge: existing first (older, higher priority for ordering) + new
    merged = list(dict.fromkeys(existing_areas + new_areas))
    store.put(("loop", "users"), user_id, {"weak_areas": merged, "session_count": session_count})


# ── Node ──────────────────────────────────────────────────────────────────────


def coach(state: dict) -> dict:
    grades = state.get("grades") or []
    if not grades:
        return {"weak_areas": []}

    grades_summary = "\n\n".join(
        f"Question {g['question_id']}: {g['score']}/10\n"
        f"Strengths: {', '.join(g.get('strengths', []))}\n"
        f"Areas to improve: {', '.join(g.get('improvements', []))}\n"
        f"Feedback: {g.get('overall_feedback', '')}"
        for g in grades
    )

    model = get_chat_model()
    structured_model = model.with_structured_output(Feedback)
    chain = _PROMPT | structured_model

    cb = get_langfuse_callback()
    config = {"callbacks": [cb]} if cb else {}

    feedback: Feedback = chain.invoke(
        {"grades_summary": grades_summary},
        config=config,
    )

    # Persist to store (no-op if store not wired).
    _persist_weak_areas(feedback.weak_areas_update)

    return {"weak_areas": feedback.weak_areas_update}
