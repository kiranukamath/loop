"""
Phase 15c tests — grader.py's optimized-prompt artifact load/fallback.

These tests never import dspy and never run evals/optimize_grader.py (that
script makes real Bedrock calls and is a dev/offline activity per CLAUDE.md's
laptop-offline test gate). They only verify the runtime half of the split:
loop.nodes.grader._load_system_prompt() reads fixtures/optimized_grader_prompt.txt
when present, and falls back to the hand-written _SYSTEM prompt when absent.
"""

import pathlib

import loop.nodes.grader as grader_mod


class TestLoadSystemPrompt:
    def test_loads_repo_artifact_when_present(self):
        """fixtures/optimized_grader_prompt.txt ships in the repo (Phase 15c) --
        by default _load_system_prompt() must return its contents, not _SYSTEM."""
        assert grader_mod._OPTIMIZED_PROMPT_PATH.exists()
        loaded = grader_mod._load_system_prompt()
        artifact_text = grader_mod._OPTIMIZED_PROMPT_PATH.read_text().strip()
        assert loaded == artifact_text
        assert loaded != grader_mod._SYSTEM

    def test_falls_back_when_artifact_absent(self, monkeypatch, tmp_path):
        missing_path = tmp_path / "does_not_exist.txt"
        monkeypatch.setattr(grader_mod, "_OPTIMIZED_PROMPT_PATH", missing_path)
        assert grader_mod._load_system_prompt() == grader_mod._SYSTEM

    def test_falls_back_when_artifact_empty(self, monkeypatch, tmp_path):
        empty_path = tmp_path / "empty.txt"
        empty_path.write_text("   \n")
        monkeypatch.setattr(grader_mod, "_OPTIMIZED_PROMPT_PATH", empty_path)
        assert grader_mod._load_system_prompt() == grader_mod._SYSTEM

    def test_loads_custom_artifact_when_present(self, monkeypatch, tmp_path):
        custom_path = tmp_path / "custom.txt"
        custom_path.write_text("Grade strictly against the rubric weights.\n")
        monkeypatch.setattr(grader_mod, "_OPTIMIZED_PROMPT_PATH", custom_path)
        assert grader_mod._load_system_prompt() == "Grade strictly against the rubric weights."

    def test_build_grading_prompt_uses_loaded_system_text(self, monkeypatch, tmp_path):
        custom_path = tmp_path / "custom.txt"
        custom_path.write_text("A distinctive marker sentence for this test.\n")
        monkeypatch.setattr(grader_mod, "_OPTIMIZED_PROMPT_PATH", custom_path)

        prompt = grader_mod._build_grading_prompt()
        rendered = prompt.format(
            question_prompt="q",
            reference_answer="r",
            answer_text="a",
            max_score=10,
            rubric_criteria="c",
        )
        assert "A distinctive marker sentence for this test." in rendered


class TestArtifactPathPointsIntoFixtures:
    def test_path_is_under_fixtures_dir(self):
        expected = pathlib.Path(__file__).parent.parent / "fixtures" / "optimized_grader_prompt.txt"
        assert grader_mod._OPTIMIZED_PROMPT_PATH == expected.resolve()
