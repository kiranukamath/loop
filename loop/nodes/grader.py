"""
Grader node — grades one candidate answer against its rubric.

Reads:  state["current_question_id"], state["answers"]
Writes: state["grades"]  (appends a Grade.model_dump() dict)

The rubric is fetched from the fixture question bank via loop.tools.
Phase 3: answers are pre-injected; Phase 5 they come from human interrupts.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from loop.models import get_chat_model
from loop.observability import get_langfuse_callback
from loop.schemas import Grade
from loop.tools import get_question_by_id, get_rubric

_SYSTEM = """\
You are an expert technical interviewer grading a candidate's answer.

You receive: the original question, the candidate's answer, and a rubric.
Grade honestly and constructively.
Each criterion score must be between 0 and its weight value (inclusive).
The overall score is the sum of criterion scores (max = rubric's max_score).
"""

_HUMAN = """\
Question: {question_prompt}

Candidate's answer:
{answer_text}

Rubric (max score: {max_score}):
{rubric_criteria}

Grade this answer."""

_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])


def grader(state: dict) -> dict:
    question_id = state["current_question_id"]

    # Find the answer for this specific question.
    answers = state.get("answers") or []
    answer = next((a for a in answers if a["question_id"] == question_id), None)
    if answer is None:
        raise ValueError(
            f"No answer in state for question_id={question_id!r}. "
            "Pre-populate state['answers'] before invoking the graph."
        )

    question = get_question_by_id(question_id)
    rubric = get_rubric(question_id)
    if rubric is None:
        raise ValueError(f"No rubric found for question_id={question_id!r}")

    # Format rubric criteria into readable text for the prompt.
    criteria_text = "\n".join(
        f"- {c['name']} (weight {c['weight']}): {c['description']}" for c in rubric["criteria"]
    )

    model = get_chat_model()
    structured_model = model.with_structured_output(Grade)
    chain = _PROMPT | structured_model

    cb = get_langfuse_callback()
    config = {"callbacks": [cb]} if cb else {}

    grade: Grade = chain.invoke(
        {
            "question_prompt": question["prompt"] if question else "",
            "answer_text": answer["text"],
            "max_score": rubric["max_score"],
            "rubric_criteria": criteria_text,
        },
        config=config,
    )

    existing = list(state.get("grades") or [])
    existing.append(grade.model_dump())
    return {"grades": existing}
