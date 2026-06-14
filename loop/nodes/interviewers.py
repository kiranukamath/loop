"""
Interviewer nodes — one per modality.

Each node:
1. Finds the first unanswered question for the current modality from the fixture bank.
2. Stores current_question_id in state.
3. Adds an AI message with the question text (which the candidate sees).

Phase 5 will interrupt the graph here to collect a real human answer.
Phase 3: the answer is pre-injected into state["answers"] by the runner/tests.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from loop.tools import get_questions_by_modality


def _ask_question(state: dict, modality: str) -> dict:
    """Shared logic: pick an unanswered question and add it as an AI message.

    Skips questions already present in state["answers"] so successive calls
    in the same session don't repeat a question.
    """
    questions = get_questions_by_modality(modality)
    if not questions:
        raise ValueError(f"No questions found for modality: {modality!r}")

    answered_ids = {a["question_id"] for a in (state.get("answers") or [])}

    # Pick the first question not yet answered; fall back to first if all answered.
    question = next((q for q in questions if q["id"] not in answered_ids), questions[0])

    msg = AIMessage(content=f"**{question['title']}**\n\n{question['prompt']}")
    return {
        "current_question_id": question["id"],
        "messages": [msg],  # add_messages reducer appends this
    }


def coding_interviewer(state: dict) -> dict:
    """Ask a coding question from the fixture bank."""
    return _ask_question(state, "coding")


def sd_interviewer(state: dict) -> dict:
    """Ask a system-design question."""
    return _ask_question(state, "system_design")


def beh_interviewer(state: dict) -> dict:
    """Ask a behavioral question."""
    return _ask_question(state, "behavioral")
