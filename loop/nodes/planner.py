"""
Planner node — Phase 2 (updated Phase 4: reads cross-session weak areas from store).

Reads jd + profile + weak_areas from:
  1. Long-term store (cross-session, keyed by user_id) — Phase 4 addition
  2. state["weak_areas"] (current session, from a previous coach run)

Both sources are merged and passed to the model.  If no store is wired or the
node is called outside a graph context, it falls back to state-only weak areas.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from loop.models import get_chat_model
from loop.observability import get_langfuse_callback
from loop.schemas import PrepPlan

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """You are an expert technical-interview coach.
Given a job description and a candidate profile, produce a structured prep plan.
Be specific: name real topics (e.g. "sliding window", "outbox pattern", "STAR format").
Tailor the plan to the gap between the JD requirements and the candidate's current skills."""

_HUMAN = """## Job Description
{jd}

## Candidate Profile
{profile}

## Known Weak Areas (from previous sessions — empty on first session)
{weak_areas}

Produce a PrepPlan for this candidate."""

_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])


# ── Store helper ──────────────────────────────────────────────────────────────


def _get_stored_weak_areas() -> list[str]:
    """Read weak_areas from the long-term store.

    Returns [] if:
    - called outside a graph context (RuntimeError from get_config)
    - the graph was compiled without a store (get_store returns None)
    - no entry yet for this user_id
    """
    try:
        from langgraph.config import get_config, get_store

        store = get_store()
        if store is None:
            return []
        cfg = get_config()
        user_id = cfg.get("configurable", {}).get("user_id", "default")
    except RuntimeError:
        return []

    item = store.get(("loop", "users"), user_id)
    return item.value.get("weak_areas", []) if item else []


# ── Node ──────────────────────────────────────────────────────────────────────


def planner(state: dict) -> dict:
    """Plan a prep curriculum from JD + profile and write it to state["plan"].

    Node contract: receive full state, return only the keys changed.

    The chain is:  prompt | model.with_structured_output(PrepPlan)
    - prompt.invoke(vars) → ChatPromptValue (the filled-in messages)
    - model.with_structured_output(PrepPlan) wraps the model so it returns a
      PrepPlan instance instead of a raw AIMessage string.
    - The | pipe chains them: output of prompt feeds into the model call.
    """
    model = get_chat_model()
    structured_model = model.with_structured_output(PrepPlan)
    chain = _PROMPT | structured_model

    cb = get_langfuse_callback()
    config = {"callbacks": [cb]} if cb else {}

    # Merge cross-session weak areas (from store) with current-session ones (from state).
    # store_areas covers past sessions; state areas cover any coach run earlier this session.
    store_areas = _get_stored_weak_areas()
    state_areas = list(state.get("weak_areas") or [])
    # dict.fromkeys preserves insertion order while deduplicating
    all_weak_areas = list(dict.fromkeys(store_areas + state_areas))
    weak_areas_str = ", ".join(all_weak_areas) if all_weak_areas else "none"

    plan: PrepPlan = chain.invoke(
        {
            "jd": state["jd"],
            "profile": state["profile"],
            "weak_areas": weak_areas_str,
        },
        config=config,
    )

    return {"plan": plan.model_dump()}
