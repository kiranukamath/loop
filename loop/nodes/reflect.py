"""
Reflect node — Phase 16b: reflection / consolidation.

Raw episodic memories (one immutable record per session, written by
coach._persist_weak_areas) are just a log -- they don't by themselves tell
you much about the CANDIDATE. reflect() periodically reads a bounded window
of recent episodes and asks the model to distil them into two kinds of
durable, higher-level memory (the Generative-Agents "reflection" pattern --
memory that periodically *thinks* about itself):

  - semantic insights    -- durable FACTS about the candidate's skill gaps
  - procedural insights  -- durable guidance on HOW to coach this candidate

Wired in graph.py at the curriculum boundary: readiness -> reflect -> END.
That means reflect runs AT MOST ONCE per graph invocation (one full
prep-curriculum run), never per-session and never in a loop -- "bounded" in
the PLAN.md sense. It's also bounded in how much history it reads at once
(_MAX_EPISODES), so a long-lived user never blows up one reflection prompt.

Off by default (settings.reflection_enabled): with it off, this node is a
no-op passthrough and the Phase 15 graph is unaffected byte-for-byte.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from loop.config import settings
from loop.models import get_chat_model, with_resilience
from loop.observability import get_langfuse_callback
from loop.schemas import ReflectionInsights

_MAX_EPISODES = 10  # bound on how many past sessions get reflected on at once

_SYSTEM = """You are an interview coach reflecting on a candidate's recent
mock-interview sessions to distil durable insights.

Given a log of recent sessions (each session's newly-identified weak areas),
produce:
- semantic_insights: durable FACTS about the candidate's skill gaps that
  persist across sessions -- not just a restatement of one session's list.
- procedural_insights: durable guidance on HOW to coach this specific
  candidate going forward (pacing, format, what seems to land).

Keep each insight short (one sentence) and non-redundant."""

_HUMAN = """Recent session log:
{episodes}

Distil this into consolidated insights."""

_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])


def _format_episodes(episodes: list[dict]) -> str:
    lines = []
    for ep in episodes:
        areas = ", ".join(ep.get("new_weak_areas", [])) or "(none noted)"
        lines.append(f"Session {ep.get('session_number', '?')}: {areas}")
    return "\n".join(lines)


def reflect(state: dict) -> dict:
    """Consolidate recent episodic memories into semantic + procedural insights.

    Node contract: receive full state, return only the keys changed -- reflect
    changes no graph state (it's a pure store side effect, like
    coach._persist_weak_areas), so it always returns {}.

    No-op when: reflection_enabled is off, no store is wired, the node is
    called outside a graph context, or there's no episodic history yet --
    same defensive pattern as coach._persist_weak_areas / planner's store
    helpers.
    """
    if not settings.reflection_enabled:
        return {}

    try:
        from langgraph.config import get_config, get_store

        store = get_store()
        if store is None:
            return {}
        cfg = get_config()
        user_id = cfg.get("configurable", {}).get("user_id", "default")
    except RuntimeError:
        return {}

    from loop.memory import (
        get_recent_episodes,
        get_weak_areas_state,
        put_procedural_note,
        put_semantic_insight,
    )

    _, session_count = get_weak_areas_state(store, user_id)
    episodes = get_recent_episodes(store, user_id, session_count, _MAX_EPISODES)
    if not episodes:
        return {}

    model = get_chat_model()
    structured_model = model.with_structured_output(ReflectionInsights)
    chain = _PROMPT | structured_model

    # Phase 10a: retry the primary model; fall back to a secondary model (if
    # configured) after retries are exhausted -- same resilience wrapper
    # every other model-calling node uses.
    fallback_chain = None
    if settings.fallback_model_id:
        fallback_model = get_chat_model(settings.fallback_model_id)
        fallback_chain = _PROMPT | fallback_model.with_structured_output(ReflectionInsights)
    chain = with_resilience(chain, fallback_chain)

    cb = get_langfuse_callback()
    config = {"callbacks": [cb]} if cb else {}

    insights: ReflectionInsights = chain.invoke(
        {"episodes": _format_episodes(episodes)},
        config=config,
    )

    for text in insights.semantic_insights:
        put_semantic_insight(
            store,
            user_id,
            topic=text,
            text=text,
            confidence=0.8,
            session_count=session_count,
        )

    if insights.procedural_insights:
        put_procedural_note(
            store, user_id, notes=insights.procedural_insights, session_count=session_count
        )

    return {}
