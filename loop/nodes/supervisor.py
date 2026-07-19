"""
interview_supervisor — Phase 13b: a supervisor node with handoffs.

Every routing decision Loop has made through Phase 12 is the GRAPH deciding:
session_router reads the PrepPlan's pre-set session.modality and _route_by_modality
switches on it; _route_after_session compares session_index to len(sessions).
All three are pure functions of state the graph itself produced — no model call,
no judgment, just topology.

orchestration_mode="supervisor" replaces all three with ONE node that reasons
about what to do next and returns a Command(goto=...) — a "handoff". Loop's
graph already does exactly this once (plan_approval, graph.py:147, hands off
to session_router/END based on a human's decision); this generalizes the same
mechanism to a MODEL's decision instead of a human's, and to picking between
three specialists instead of two terminal states.

Why this matters pedagogically: this is the line between "a fixed workflow
with one dynamic node" (Phase 9's ReAct researcher was the only other one) and
"a real multi-agent system" — control flow itself becomes something the model
owns, not something baked into the graph's static edges.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command

from loop.config import settings
from loop.models import get_chat_model, with_resilience
from loop.observability import get_langfuse_callback
from loop.schemas import SupervisorDecision

_MODALITY_TO_NODE = {
    "coding": "coding_interviewer",
    "system_design": "sd_interviewer",
    "behavioral": "beh_interviewer",
}

_SYSTEM = """\
You are the supervisor of a mock-interview coaching session, deciding which \
specialist runs the next session.

You may follow the prep plan's suggested modality for this session number, or \
deviate from it if the candidate's weak areas make a different modality more \
valuable right now. Pick exactly one: coding, system_design, or behavioral.
"""

_HUMAN = """\
Session number: {session_number}
Plan's suggested modality for this session: {planned_modality}
Plan's suggested topics: {planned_topics}
Plan's suggested focus: {planned_focus}

Candidate's current weak areas (from prior sessions, most recent last): {weak_areas}

Decide the modality, focus, and topics for this session."""

_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])


def interview_supervisor(
    state: dict,
) -> Command[Literal["coding_interviewer", "sd_interviewer", "beh_interviewer", "readiness"]]:
    """Decide the next specialist (or readiness) and hand off via Command(goto=...).

    Replaces session_router / _route_by_modality / _route_after_session as a
    unit — advance_session (graph.py) STAYS; only its outgoing edge changes to
    point here instead of to session_router.

    Bound: session_index >= min(len(sessions), settings.max_sessions) -- same
    guard _route_after_session enforced, checked BEFORE calling the model so a
    finished plan never spends a model call deciding to stop. This is the
    supervisor's OWN bound; the graph-level recursion_limit (langgraph.errors.
    GraphRecursionError) is only the backstop if a bug in this node (or a
    misbehaving model) ignores it — see tests/test_multiagent.py for both.
    """
    plan = state.get("plan") or {}
    sessions = plan.get("sessions") or []
    idx = state.get("session_index") or 0

    if not sessions or idx >= min(len(sessions), settings.max_sessions):
        return Command(goto="readiness")

    session = sessions[idx]
    weak_areas = state.get("weak_areas") or []

    model = get_chat_model()
    structured_model = model.with_structured_output(SupervisorDecision)
    chain = _PROMPT | structured_model

    fallback_chain = None
    if settings.fallback_model_id:
        fallback_model = get_chat_model(settings.fallback_model_id)
        fallback_chain = _PROMPT | fallback_model.with_structured_output(SupervisorDecision)
    chain = with_resilience(chain, fallback_chain)

    cb = get_langfuse_callback()
    config = {"callbacks": [cb]} if cb else {}

    decision: SupervisorDecision = chain.invoke(
        {
            "session_number": session.get("session_number", idx + 1),
            "planned_modality": session.get("modality", ""),
            "planned_topics": ", ".join(session.get("topics") or []),
            "planned_focus": session.get("focus", ""),
            "weak_areas": ", ".join(weak_areas) if weak_areas else "none yet",
        },
        config=config,
    )

    return Command(
        goto=_MODALITY_TO_NODE[decision.next_modality],
        update={
            "current_modality": decision.next_modality,
            "current_focus": decision.focus,
            "current_topics": decision.topics,
            "session_number": session.get("session_number", idx + 1),
            "supervisor_decisions": [
                {
                    "session_index": idx,
                    "planned_modality": session.get("modality"),
                    "chosen_modality": decision.next_modality,
                    "focus": decision.focus,
                }
            ],
        },
    )
