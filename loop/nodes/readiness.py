"""
Readiness node — Phase 5 (HITL).

Two steps:
1. Call the model to produce a ReadinessVerdict based on session grades.
2. Call interrupt() to surface the verdict to the human for approval.
   The human can approve it (verdict stands) or override it (supply a different verdict).

Reads:  state["grades"]
Writes: state["readiness_verdict"]  (dict from ReadinessVerdict.model_dump())
        state["verdict_approved"]   (True once the human signs off)

Analogy: a risk engine produces a credit score, then a loan officer approves or overrides
it before the decision is committed.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import interrupt

from loop.models import get_chat_model
from loop.observability import get_langfuse_callback
from loop.schemas import ReadinessVerdict

_SYSTEM = """\
You are a senior technical interview coach assessing a candidate's readiness.

Based on the mock-interview session grades provided, decide:
- "ready"     — the candidate can go into real interviews now.
- "not_ready" — the candidate needs more practice before interviewing.

Be honest but constructive. Base your verdict purely on the evidence in the grades.
"""

_HUMAN = """\
Session grades:
{grades_summary}

Produce a ReadinessVerdict."""

_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])


def readiness(state: dict) -> dict:
    grades = state.get("grades") or []

    if not grades:
        # No grades → default to not_ready with explanation
        default_verdict = ReadinessVerdict(
            verdict="not_ready",
            confidence=0.0,
            strengths=[],
            gaps=["No session grades available to assess readiness"],
            recommendation=(
                "Complete at least one graded mock-interview session before"
                " requesting a readiness verdict."
            ),
        )
        human_response = interrupt(
            {
                "action": "approve_verdict",
                "verdict": default_verdict.model_dump(),
            }
        )
        return _apply_response(default_verdict, human_response)

    grades_summary = "\n\n".join(
        f"Question {g['question_id']}: {g['score']}/10\n"
        f"Strengths: {', '.join(g.get('strengths', []))}\n"
        f"Gaps: {', '.join(g.get('improvements', []))}\n"
        f"Feedback: {g.get('overall_feedback', '')}"
        for g in grades
    )

    model = get_chat_model()
    structured_model = model.with_structured_output(ReadinessVerdict)
    chain = _PROMPT | structured_model

    cb = get_langfuse_callback()
    config = {"callbacks": [cb]} if cb else {}

    verdict: ReadinessVerdict = chain.invoke(
        {"grades_summary": grades_summary},
        config=config,
    )

    # ── HITL gate ─────────────────────────────────────────────────────────────
    # Pause here; the caller sees the verdict and decides: approve or override.
    # interrupt() saves graph state via the checkpointer, returns the human's
    # response dict when the graph is resumed via Command(resume=...).
    human_response = interrupt(
        {
            "action": "approve_verdict",
            "verdict": verdict.model_dump(),
        }
    )

    return _apply_response(verdict, human_response)


def _apply_response(verdict: ReadinessVerdict, human_response: dict) -> dict:
    """Turn the human's approval/override response into a state update."""
    if human_response.get("decision") == "override":
        # Human disagrees — record their override and reason
        final = {
            **verdict.model_dump(),
            "verdict": human_response["verdict"],  # "ready" or "not_ready"
            "override_reason": human_response.get("reason", ""),
        }
        return {"readiness_verdict": final, "verdict_approved": True}

    # Approved — store the model's verdict as-is
    return {"readiness_verdict": verdict.model_dump(), "verdict_approved": True}
