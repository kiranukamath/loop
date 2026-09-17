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

Phase 15a: an optional Reflexion self-critique pass — a second LLM call that
reviews the first grade against the rubric/reference and can revise it before
it's committed to state. Gated by settings.reflexion_enabled (default off).

Phase 15c: the grading prompt's system message is loaded from
fixtures/optimized_grader_prompt.txt when that artifact exists (written by the
offline `evals/optimize_grader.py` DSPy run), falling back to the hand-written
_SYSTEM prompt below otherwise. grader.py never imports dspy — it only reads
the plain-text artifact the optimizer produces.
"""

from __future__ import annotations

import pathlib

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

_NO_REFERENCE_TEXT = (
    "(no reference answer available for this question — grade against the rubric alone)"
)

# ── Phase 15c: optimized-prompt artifact loading ───────────────────────────────

_OPTIMIZED_PROMPT_PATH = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "fixtures"
    / "optimized_grader_prompt.txt"
)


def _load_system_prompt() -> str:
    """Return the grading system prompt: the DSPy-optimized artifact if it
    exists and is non-empty, otherwise the hand-written _SYSTEM above.

    This is the offline-optimization vs. runtime-serving split: evals/
    optimize_grader.py (a dev script, run manually, never in tests) is the
    only thing that writes _OPTIMIZED_PROMPT_PATH. The runtime here just
    reads whatever's on disk — same shape as loading a trained model's
    weights instead of retraining on every request.
    """
    if _OPTIMIZED_PROMPT_PATH.exists():
        text = _OPTIMIZED_PROMPT_PATH.read_text().strip()
        if text:
            return text
    return _SYSTEM


def _build_grading_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", _load_system_prompt()), ("human", _HUMAN)])


# ── Phase 15a: Reflexion self-critique ─────────────────────────────────────────

_CRITIQUE_SYSTEM = """\
You are auditing another interviewer's grade for scoring errors.

You receive the question, the reference answer, the candidate's answer, the
rubric, and the grade already assigned. Check for: criterion scores that
don't respect their rubric weight, an overall score that doesn't equal the
sum of the criterion scores, feedback that contradicts the score (e.g.
"strengths" outweighing "improvements" but a low score, or vice versa), or an
obviously miscounted score.

If the grade holds up, return it completely unchanged.
If you find a genuine error, return a corrected grade — do not change a grade
you merely would have written differently.
"""

_CRITIQUE_HUMAN = """\
Question: {question_prompt}

Reference answer:
{reference_answer}

Candidate's answer:
{answer_text}

Rubric (max score: {max_score}):
{rubric_criteria}

Original grade:
- score: {original_score}
- criterion_scores: {original_criterion_scores}
- strengths: {original_strengths}
- improvements: {original_improvements}
- overall_feedback: {original_feedback}

Critique this grade. Return it unchanged if correct, or a revised grade if not."""

_CRITIQUE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _CRITIQUE_SYSTEM), ("human", _CRITIQUE_HUMAN)]
)


def _self_critique(
    grade: Grade,
    rubric: dict,
    reference: str,
    answer_text: str,
    question_prompt: str,
) -> Grade:
    """Second LLM pass: critique `grade` against the rubric/reference and
    return either the same grade (if it holds up) or a revised one.

    Pure model call — offline-testable by stubbing get_chat_model() exactly
    like the primary grading call. Only invoked when settings.reflexion_enabled
    is True (see grader() below); disabled by default so today's single-pass
    behavior is unchanged.
    """
    criteria_text = "\n".join(
        f"- {c['name']} (weight {c['weight']}): {c['description']}" for c in rubric["criteria"]
    )

    model = get_chat_model()
    chain = with_resilience(_CRITIQUE_PROMPT | model.with_structured_output(Grade))

    cb = get_langfuse_callback()
    config = {"callbacks": [cb]} if cb else {}

    return chain.invoke(
        {
            "question_prompt": question_prompt,
            "reference_answer": reference,
            "answer_text": answer_text,
            "max_score": rubric["max_score"],
            "rubric_criteria": criteria_text,
            "original_score": grade.score,
            "original_criterion_scores": grade.criterion_scores,
            "original_strengths": grade.strengths,
            "original_improvements": grade.improvements,
            "original_feedback": grade.overall_feedback,
        },
        config=config,
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

    prompt = _build_grading_prompt()
    model = get_chat_model()
    structured_model = model.with_structured_output(Grade)
    chain = prompt | structured_model

    # Phase 10a: retry the primary model; fall back to a secondary model (if
    # configured) after retries are exhausted.
    fallback_chain = None
    if settings.fallback_model_id:
        fallback_model = get_chat_model(settings.fallback_model_id)
        fallback_chain = prompt | fallback_model.with_structured_output(Grade)
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

    # Phase 15a: optional second pass that can catch and fix a scoring error
    # in the grade above. Off by default -- today's single-pass behavior.
    if settings.reflexion_enabled:
        grade = _self_critique(
            grade,
            rubric,
            reference_answer,
            answer["text"],
            question["prompt"] if question else "",
        )

    # Return only the new grade — the _append_list reducer on state["grades"]
    # handles accumulation across sessions. Returning the full list here would
    # cause duplicates in multi-session runs (reducer appends to the existing list).
    return {"grades": [grade.model_dump()]}
