"""
Coach node — turns session grades into actionable feedback.

Reads:  state["grades"]
Writes: state["weak_areas"]  (list of topics to focus on next session)

The coach synthesizes all grades from the current session into a Feedback
object.  The weak_areas_update field feeds back into the planner in Phase 4,
where it will change the next session's prep plan.
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

    return {"weak_areas": feedback.weak_areas_update}
