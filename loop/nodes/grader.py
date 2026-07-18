"""
Grader node — grades one candidate answer against its rubric.

Reads:  state["current_question_id"], state["answers"]
Writes: state["grades"]  (appends a Grade.model_dump() dict)

The rubric is fetched from the fixture question bank via loop.tools.
Phase 3: answers are pre-injected; Phase 5 they come from human interrupts.

Phase 8c: the grader is now RAG-grounded — it retrieves a reference/model
answer for the question and injects it into the prompt, so the model grades
against known-good evidence instead of purely its own judgment of correctness.
The reference is optional grounding, not a hard requirement: a question
without a reference_answers.json entry still grades normally (rubric alone).
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from loop.config import settings
from loop.models import get_chat_model, with_resilience
from loop.observability import get_langfuse_callback
from loop.schemas import Grade
from loop.tools import get_question_by_id, get_reference_answer, get_rubric

_SYSTEM = """\
You are an expert technical interviewer grading a candidate's answer.

You receive: the original question, a reference answer (if available), the
candidate's answer, and a rubric.  Use the reference answer as grounding for
what a strong response looks like — grade the candidate against it and the
rubric, not against your own unaided judgment of correctness.
Grade honestly and constructively.
Each criterion score must be between 0 and its weight value (inclusive).
The overall score is the sum of criterion scores (max = rubric's max_score).
"""

_HUMAN = """\
Question: {question_prompt}

Reference answer (for grounding — the candidate did not see this):
{reference_answer}

Candidate's answer:
{answer_text}

Rubric (max score: {max_score}):
{rubric_criteria}

Grade this answer."""

_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])

_NO_REFERENCE_TEXT = (
    "(no reference answer available for this question — grade against the rubric alone)"
)


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

    # Phase 8c: retrieve the reference answer for grounding.  Optional — falls
    # back to a clear placeholder rather than failing when a question has none.
    reference_answer = get_reference_answer(question_id) or _NO_REFERENCE_TEXT

    model = get_chat_model()
    structured_model = model.with_structured_output(Grade)
    chain = _PROMPT | structured_model

    # Phase 10a: retry the primary model; fall back to a secondary model (if
    # configured) after retries are exhausted.
    fallback_chain = None
    if settings.fallback_model_id:
        fallback_model = get_chat_model(settings.fallback_model_id)
        fallback_chain = _PROMPT | fallback_model.with_structured_output(Grade)
    chain = with_resilience(chain, fallback_chain)

    cb = get_langfuse_callback()
    config = {"callbacks": [cb]} if cb else {}

    grade: Grade = chain.invoke(
        {
            "question_prompt": question["prompt"] if question else "",
            "reference_answer": reference_answer,
            "answer_text": answer["text"],
            "max_score": rubric["max_score"],
            "rubric_criteria": criteria_text,
        },
        config=config,
    )

    # Return only the new grade — the _append_list reducer on state["grades"]
    # handles accumulation across sessions. Returning the full list here would
    # cause duplicates in multi-session runs (reducer appends to the existing list).
    return {"grades": [grade.model_dump()]}
