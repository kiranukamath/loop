"""
Planner node — Phase 2 (updated Phase 4: reads cross-session weak areas from store).

Reads jd + profile + weak_areas from:
  1. Long-term store (cross-session, keyed by user_id) — Phase 4 addition
  2. state["weak_areas"] (current session, from a previous coach run)

Both sources are merged and passed to the model.  If no store is wired or the
node is called outside a graph context, it falls back to state-only weak areas.

Phase 9b: also reads state["company_research"] (populated by the research node
when a target company was set) and folds it into the prompt, so the plan can
be grounded in real company signal — e.g. "focus more on distributed systems
since this company's interview format emphasises system design."  Absent when
no company was set; the prompt says so plainly rather than guessing.

Phase 16: _get_stored_weak_areas() reads the typed semantic namespace (with
back-compat fallback to the old Phase 4 flat shape -- see loop/memory.py).
_get_semantic_insights() additionally recalls reflect()'s (Phase 16b)
consolidated insights by embedding similarity (Phase 16c), so the plan can
be grounded in durable, higher-level facts about the candidate -- not just
the raw weak_areas list.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from loop.config import settings
from loop.models import get_chat_model, with_resilience
from loop.observability import get_langfuse_callback
from loop.schemas import PrepPlan

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """You are an expert technical-interview coach.
Given a job description and a candidate profile, produce a structured prep plan.
Be specific: name real topics (e.g. "sliding window", "outbox pattern", "STAR format").
Tailor the plan to the gap between the JD requirements and the candidate's current skills.
If company research is provided, use it to ground the plan in that company's actual
interview format and focus areas."""

_HUMAN = """## Job Description
{jd}

## Candidate Profile
{profile}

## Known Weak Areas (from previous sessions — empty on first session)
{weak_areas}

## Consolidated Insights (from reflection on past sessions — Phase 16)
{insights}

## Company Research
{company_research}

Produce a PrepPlan for this candidate."""

_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])

_NO_COMPANY_RESEARCH_TEXT = "(no company research available — no target company was set)"


def _format_company_research(company_research: dict | None) -> str:
    """Render a CompanyResearch dict into readable prompt text.

    Returns a clear placeholder when no research was performed, rather than
    an empty string that could read as "researched and found nothing."
    """
    if not company_research:
        return _NO_COMPANY_RESEARCH_TEXT

    lines = [
        f"Company: {company_research.get('company', '')}",
        f"Interview format: {company_research.get('interview_format', '')}",
        f"Focus areas: {', '.join(company_research.get('focus_areas', []))}",
        f"Tech stack: {', '.join(company_research.get('tech_stack', []))}",
        f"Recent news: {', '.join(company_research.get('recent_news', []))}",
    ]
    return "\n".join(lines)


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

    from loop.memory import get_weak_areas_state

    areas, _ = get_weak_areas_state(store, user_id)
    return areas


_NO_INSIGHTS_TEXT = "(none yet — reflection hasn't run, or this is a new candidate)"


def _get_semantic_insights(query: str) -> list[str]:
    """Recall reflect()'s (Phase 16b) consolidated semantic insights by
    embedding similarity (Phase 16c), scoped to this user.

    Returns [] under the same conditions as _get_stored_weak_areas: no graph
    context, no store wired, or (here) no insights have been written yet --
    reflection_enabled defaults to False, so this is [] for every run unless
    reflection has been turned on and has actually run at least once.
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

    from loop.memory import get_weak_areas_state, recall_semantic_memories

    _, session_count = get_weak_areas_state(store, user_id)
    memories = recall_semantic_memories(store, user_id, query=query, session_count=session_count)
    return [m["text"] for m in memories]


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

    # Phase 10a: retry the primary model; fall back to a secondary model (if
    # configured) after retries are exhausted.
    fallback_chain = None
    if settings.fallback_model_id:
        fallback_model = get_chat_model(settings.fallback_model_id)
        fallback_chain = _PROMPT | fallback_model.with_structured_output(PrepPlan)
    chain = with_resilience(chain, fallback_chain)

    cb = get_langfuse_callback()
    config = {"callbacks": [cb]} if cb else {}

    # Merge cross-session weak areas (from store) with current-session ones (from state).
    # store_areas covers past sessions; state areas cover any coach run earlier this session.
    store_areas = _get_stored_weak_areas()
    state_areas = list(state.get("weak_areas") or [])
    # dict.fromkeys preserves insertion order while deduplicating
    all_weak_areas = list(dict.fromkeys(store_areas + state_areas))
    weak_areas_str = ", ".join(all_weak_areas) if all_weak_areas else "none"

    # Phase 16: recall consolidated insights relevant to this JD/profile,
    # rather than dumping every stored insight in.
    insights = _get_semantic_insights(query=f"{state['jd']}\n{state['profile']}")
    insights_str = "\n".join(f"- {i}" for i in insights) if insights else _NO_INSIGHTS_TEXT

    plan: PrepPlan = chain.invoke(
        {
            "jd": state["jd"],
            "profile": state["profile"],
            "weak_areas": weak_areas_str,
            "insights": insights_str,
            "company_research": _format_company_research(state.get("company_research")),
        },
        config=config,
    )

    # Hard-cap sessions to settings.max_sessions regardless of what the model suggests.
    # This keeps demo runs short and costs predictable.
    cap = settings.max_sessions
    if len(plan.sessions) > cap:
        plan = plan.model_copy(update={"sessions": plan.sessions[:cap], "total_sessions": cap})

    return {"plan": plan.model_dump()}
