"""
Interviewer nodes — one per modality.

Each node:
1. Finds the best unanswered question for the current modality by semantic
   similarity to the session's focus (Phase 8b) — not just fixture order.
2. Calls interrupt() to surface the question to the human and wait for their answer.
   The resume value is the answer text (a plain string).
3. Stores current_question_id + the human's answer in state.

Phase 7b: answers come from the human via interrupt/resume, not pre-injected.
Phase 8b: question selection is semantic (search_questions) instead of exact-match
          (get_questions_by_modality[0]) — see loop/retrieval.py for the RAG pipeline.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from loop.guardrails import detect_injection, redact_pii
from loop.tools import get_questions_by_modality, search_questions


def _ask_question(state: dict, modality: str) -> dict:
    """Pick an unanswered question, interrupt for the human's answer, return both.

    interrupt() pauses the graph here and returns the resume value on the next call.
    The caller resumes with Command(resume=answer_text) where answer_text is a string.

    The _append_list reducer on state["answers"] accumulates answers across sessions,
    so answered_ids correctly skips questions the candidate already answered.

    Question selection (Phase 8b): search_questions() ranks the modality's questions
    by semantic similarity to the session's focus + topics.  k is set to the full
    size of the modality's question pool so we can still skip already-answered
    questions while ranking by relevance, rather than truncating the pool early.
    """
    focus = state.get("current_focus") or modality
    topics = state.get("current_topics") or []
    query = f"{focus}. Topics: {', '.join(topics)}" if topics else focus

    pool_size = len(get_questions_by_modality(modality))
    if pool_size == 0:
        raise ValueError(f"No questions found for modality: {modality!r}")

    candidates = search_questions(query, modality=modality, k=pool_size)

    answered_ids = {a["question_id"] for a in (state.get("answers") or [])}

    # Pick the most relevant question not yet answered; fall back to the top
    # candidate if all have been answered.
    question = next((q for q in candidates if q["id"] not in answered_ids), candidates[0])

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

    # Phase 10b: the candidate's answer is untrusted input — redact PII before
    # it's stored or sent to the grader, and flag (not block) suspected
    # prompt-injection attempts against the grader's system prompt.
    flagged = []
    if detect_injection(answer_text):
        flagged.append({"source": f"answer:{question['id']}", "reason": "prompt_injection_pattern"})
    answer_text = redact_pii(answer_text)

    ai_msg = AIMessage(content=f"**{question['title']}**\n\n{question['prompt']}")
    human_msg = HumanMessage(content=answer_text)
    return {
        "current_question_id": question["id"],
        "messages": [ai_msg, human_msg],  # add_messages reducer appends
        "answers": [{"question_id": question["id"], "text": answer_text}],
        "flagged_inputs": flagged,
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
