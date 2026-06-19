"""
Interviewer nodes — one per modality.

Each node:
1. Finds the first unanswered question for the current modality from the fixture bank.
2. Calls interrupt() to surface the question to the human and wait for their answer.
   The resume value is the answer text (a plain string).
3. Stores current_question_id + the human's answer in state.

Phase 7b: answers come from the human via interrupt/resume, not pre-injected.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from loop.tools import get_questions_by_modality


def _ask_question(state: dict, modality: str) -> dict:
    """Pick an unanswered question, interrupt for the human's answer, return both.

    interrupt() pauses the graph here and returns the resume value on the next call.
    The caller resumes with Command(resume=answer_text) where answer_text is a string.

    The _append_list reducer on state["answers"] accumulates answers across sessions,
    so answered_ids correctly skips questions the candidate already answered.
    """
    questions = get_questions_by_modality(modality)
    if not questions:
        raise ValueError(f"No questions found for modality: {modality!r}")

    answered_ids = {a["question_id"] for a in (state.get("answers") or [])}

    # Pick the first question not yet answered; fall back to first if all answered.
    question = next((q for q in questions if q["id"] not in answered_ids), questions[0])

    # Pause here: show the question, wait for the human's typed answer.
    # On resume, answer_text is whatever the caller passed to Command(resume=...).
    answer_text: str = interrupt(
        {
            "action": "answer_question",
            "question_id": question["id"],
            "question_title": question["title"],
            "question_prompt": question["prompt"],
        }
    )

    ai_msg = AIMessage(content=f"**{question['title']}**\n\n{question['prompt']}")
    human_msg = HumanMessage(content=answer_text)
    return {
        "current_question_id": question["id"],
        "messages": [ai_msg, human_msg],  # add_messages reducer appends
        "answers": [{"question_id": question["id"], "text": answer_text}],
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
