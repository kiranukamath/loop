"""
Question-bank and rubric lookups over static fixtures.

v1: reads from fixtures/ JSON files.
v2 seam: these function signatures are STABLE — swap the body to Postgres/API
without touching any caller.  Analogy: a Repository/DAO in Spring.

Usage:
    from loop.tools import get_questions_by_modality, get_rubric, get_question_by_id
"""

from __future__ import annotations

import json
import pathlib

_FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"


def _load_questions() -> list[dict]:
    return json.loads((_FIXTURES / "questions.json").read_text())["questions"]


def _load_rubrics() -> list[dict]:
    return json.loads((_FIXTURES / "rubrics.json").read_text())["rubrics"]


def _load_reference_answers() -> list[dict]:
    return json.loads((_FIXTURES / "reference_answers.json").read_text())["reference_answers"]


def get_questions_by_modality(modality: str) -> list[dict]:
    """Return all questions matching the given modality.

    Args:
        modality: one of "coding", "system_design", "behavioral"

    Returns:
        List of question dicts (id, modality, topic, difficulty, title, prompt).
        Empty list if modality has no questions.
    """
    return [q for q in _load_questions() if q["modality"] == modality]


def get_question_by_id(question_id: str) -> dict | None:
    """Return a question by its id, or None if not found."""
    for q in _load_questions():
        if q["id"] == question_id:
            return q
    return None


def get_rubric(question_id: str) -> dict | None:
    """Return the rubric for a question, or None if not found.

    The rubric dict contains: question_id, max_score, criteria (list of
    {name, weight, description}).
    """
    for r in _load_rubrics():
        if r["question_id"] == question_id:
            return r
    return None


def get_reference_answer(question_id: str) -> str | None:
    """Return the reference/model answer for a question, or None if not found.

    Phase 8c: grounds the grader in a known-good answer instead of the model's
    own unaided judgment.  Exact-match lookup by question_id (not semantic
    search) — the grader always already knows exactly which question it's
    grading, so there's no ambiguity to resolve by similarity.
    """
    for r in _load_reference_answers():
        if r["question_id"] == question_id:
            return r["reference"]
    return None


def search_questions(query: str, modality: str | None = None, k: int = 3) -> list[dict]:
    """Return up to k questions semantically closest to the query (Phase 8).

    Delegates to loop.retrieval — this function is the stable v2 interface;
    the body can swap from InMemoryVectorStore to pgvector without any caller
    (e.g. loop/nodes/interviewers.py) changing.

    Args:
        query:    Free-text description of the topic/skill to search for.
        modality: Optional hard filter — "coding", "system_design", or "behavioral".
        k:        Maximum number of results to return.

    Returns:
        List of question dicts ranked by semantic similarity, highest first.
    """
    from loop.retrieval import retrieve_questions

    return retrieve_questions(query, modality=modality, k=k)
