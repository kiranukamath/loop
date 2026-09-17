"""
Phase 18e tests — Postgres-backed question-bank DAO seam in loop/tools.py.

A real Postgres server is a *server* activity. These tests use a "fixture
adapter": a fake psycopg connection/cursor whose fetchall() returns rows
built directly from the same fixtures/*.json files the default (fixture)
path reads, so we can assert the Postgres path returns the IDENTICAL shape
as the fixture path -- without ever touching a real database.
"""

from __future__ import annotations

import json
import pathlib

_FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"


class _FakeCursor:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def execute(self, *args, **kwargs):
        pass

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConnection:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def cursor(self):
        return _FakeCursor(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _questions_rows() -> list[dict]:
    return json.loads((_FIXTURES / "questions.json").read_text())["questions"]


def _rubrics_rows() -> list[dict]:
    return json.loads((_FIXTURES / "rubrics.json").read_text())["rubrics"]


def _reference_answers_rows() -> list[dict]:
    return json.loads((_FIXTURES / "reference_answers.json").read_text())["reference_answers"]


def _configure_pg(monkeypatch, rows: list[dict]):
    from loop.config import settings

    monkeypatch.setattr(settings, "pgvector_conn_string", "postgresql://fake/db")
    monkeypatch.setattr("loop.tools._pg_connect", lambda: _FakeConnection(rows))


class TestPostgresQuestionsDAO:
    def test_load_questions_from_pg_matches_fixture_shape(self, monkeypatch):
        _configure_pg(monkeypatch, _questions_rows())

        from loop.tools import _load_questions

        result = _load_questions()
        assert result == _questions_rows()

    def test_get_questions_by_modality_uses_pg_backend(self, monkeypatch):
        _configure_pg(monkeypatch, _questions_rows())

        from loop.tools import get_questions_by_modality

        coding = get_questions_by_modality("coding")
        assert len(coding) > 0
        assert all(q["modality"] == "coding" for q in coding)

    def test_fixture_backend_still_default_when_unconfigured(self, monkeypatch):
        from loop.config import settings

        monkeypatch.setattr(settings, "pgvector_conn_string", "")
        from loop.tools import _load_questions

        assert _load_questions() == _questions_rows()


class TestPostgresRubricsDAO:
    def test_load_rubrics_from_pg_matches_fixture_shape(self, monkeypatch):
        _configure_pg(monkeypatch, _rubrics_rows())

        from loop.tools import _load_rubrics

        assert _load_rubrics() == _rubrics_rows()

    def test_get_rubric_uses_pg_backend(self, monkeypatch):
        rows = _rubrics_rows()
        _configure_pg(monkeypatch, rows)

        from loop.tools import get_rubric

        target = rows[0]
        rubric = get_rubric(target["question_id"])
        assert rubric == target


class TestPostgresReferenceAnswersDAO:
    def test_load_reference_answers_from_pg_matches_fixture_shape(self, monkeypatch):
        _configure_pg(monkeypatch, _reference_answers_rows())

        from loop.tools import _load_reference_answers

        assert _load_reference_answers() == _reference_answers_rows()

    def test_get_reference_answer_uses_pg_backend(self, monkeypatch):
        rows = _reference_answers_rows()
        _configure_pg(monkeypatch, rows)

        from loop.tools import get_reference_answer

        target = rows[0]
        assert get_reference_answer(target["question_id"]) == target["reference"]
