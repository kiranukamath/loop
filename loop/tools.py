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
