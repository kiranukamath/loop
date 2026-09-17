"""
Offline DSPy prompt optimization for Loop's grader (Phase 15c).

This is a SERVER / DEV script, not part of the runtime graph. It makes real
Bedrock model calls (many of them — DSPy's optimizers try and score several
candidate prompts) through DSPy's LM adapter. Per CLAUDE.md's laptop-offline
test gate, it must NEVER run inside `tests/` — tests only cover the artifact
load/fallback logic in loop/nodes/grader.py, which needs no dspy import at all.

Run it manually, with AWS Bedrock credentials configured exactly like the rest
of Loop:

    uv run python -m evals.optimize_grader

What it does, step by step:
  1. Defines a DSPy Signature (`GradeAnswer`) that mirrors the grader's job:
     given a question, rubric, reference answer, and candidate answer, predict
     a score. This is DSPy's "typed function signature" for an LLM call — the
     analogy is an interface, where DSPy fills in and tunes the implementation
     (the prompt) instead of you hand-writing it.
  2. Loads fixtures/grader_labels.json (the same Phase 6 labeled set
     evals/run_grader_eval.py scores the live grader against) into DSPy
     Examples.
  3. Wraps `evals.run_grader_eval.score_agreement` as the DSPy metric — no
     scoring logic is duplicated; DSPy just needs a 0/1 "did this example pass"
     signal per prediction, which score_agreement's "score_in_range" already is.
  4. Runs DSPy's MIPROv2 optimizer, which proposes and tests candidate grading
     instructions (system prompts) against the metric and keeps the best one.
  5. Writes the winning instruction text to fixtures/optimized_grader_prompt.txt.

Runtime split (the ML-ops "train vs. serve" pattern applied to prompts):
  loop/nodes/grader.py loads that file at grading time if it exists, falling
  back to its hand-written prompt otherwise (see grader.py's
  `_load_system_prompt()`). This script produces the artifact once, offline;
  the runtime never re-runs the optimizer, it just reads the winner off disk.
"""

from __future__ import annotations

import json
import pathlib

_FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"
_LABELS_PATH = _FIXTURES / "grader_labels.json"
_OUTPUT_PATH = _FIXTURES / "optimized_grader_prompt.txt"


# ── DSPy signature ──────────────────────────────────────────────────────────
# Defined lazily (inside a function) so this module can be *imported* without
# dspy installed -- only running the optimizer requires it. Nothing else in
# the codebase imports evals.optimize_grader.


def _build_signature():
    import dspy

    class GradeAnswer(dspy.Signature):
        """Grade a candidate's technical interview answer against a rubric,
        using a reference answer as grounding for what a strong response
        looks like. Grade honestly and constructively; the score must be
        between 0 and max_score."""

        question_prompt: str = dspy.InputField(desc="The interview question asked")
        reference_answer: str = dspy.InputField(desc="A known-good answer, for grounding")
        answer_text: str = dspy.InputField(desc="The candidate's answer")
        max_score: int = dspy.InputField(desc="Maximum possible score for this rubric")
        rubric_criteria: str = dspy.InputField(desc="Rubric criteria with weights and descriptions")
        score: int = dspy.OutputField(desc="Overall score, 0 to max_score")

    return GradeAnswer


# ── Data loading ──────────────────────────────────────────────────────────────


def _load_examples() -> list:
    """Turn fixtures/grader_labels.json into a list of dspy.Example, one per
    labeled item, with the same fields the live grader prompt fills in
    (loop/nodes/grader.py:grader()) so the optimized prompt transfers cleanly.
    """
    import dspy

    from loop.tools import get_question_by_id, get_reference_answer, get_rubric

    data = json.loads(_LABELS_PATH.read_text())
    examples = []
    for item in data["items"]:
        qid = item["question_id"]
        question = get_question_by_id(qid)
        rubric = get_rubric(qid)
        if rubric is None:
            continue
        reference = get_reference_answer(qid) or "(no reference answer available for this question)"
        criteria_text = "\n".join(
            f"- {c['name']} (weight {c['weight']}): {c['description']}" for c in rubric["criteria"]
        )
        example = dspy.Example(
            question_prompt=question["prompt"] if question else "",
            reference_answer=reference,
            answer_text=item["answer"],
            max_score=rubric["max_score"],
            rubric_criteria=criteria_text,
            expected_score=item["expected_score"],
            score_min=item["score_min"],
            score_max=item["score_max"],
        ).with_inputs(
            "question_prompt", "reference_answer", "answer_text", "max_score", "rubric_criteria"
        )
        examples.append(example)
    return examples


# ── Metric — reuses evals.run_grader_eval.score_agreement, no duplication ────


def _metric(example, pred, trace=None) -> float:
    """DSPy metric: 1.0 if pred.score falls in the labeled acceptable range,
    else 0.0. Delegates the actual comparison to score_agreement so the
    "what counts as agreement" definition lives in exactly one place.
    """
    from evals.run_grader_eval import score_agreement

    results = score_agreement(
        input={},
        output={"score": pred.score},
        expected_output={
            "score": example.expected_score,
            "score_min": example.score_min,
            "score_max": example.score_max,
        },
        metadata=None,
    )
    by_name = {r["name"]: r for r in results}
    return by_name["score_in_range"]["value"]


# ── LM configuration — same Bedrock model Loop already uses ─────────────────


def _make_lm():
    import dspy

    from loop.config import settings

    # DSPy's LM wraps litellm, which understands "bedrock/<model_id>" model
    # strings directly -- no separate Bedrock adapter class needed (verified
    # against dspy==3.3.1). Region comes from the same settings.aws_region
    # everything else in Loop uses.
    return dspy.LM(
        model=f"bedrock/{settings.bedrock_model_id}", aws_region_name=settings.aws_region
    )


# ── Optimizer run ─────────────────────────────────────────────────────────────


def optimize() -> str:
    """Run MIPROv2 against the labeled set and return the winning instruction text."""
    import dspy

    dspy.configure(lm=_make_lm())

    trainset = _load_examples()
    GradeAnswer = _build_signature()
    student = dspy.Predict(GradeAnswer)

    # auto="light" -- the labeled set is tiny (single digits of examples), so
    # the cheapest search budget is enough to see if any instruction rewrite
    # beats the hand-written baseline.
    optimizer = dspy.MIPROv2(metric=_metric, auto="light")
    optimized = optimizer.compile(student, trainset=trainset)

    predictors = optimized.predictors()
    if not predictors:
        return GradeAnswer.__doc__ or ""
    return predictors[0].signature.instructions


def main() -> None:
    instructions = optimize().strip()
    if not instructions:
        print("Optimizer returned no instructions -- not writing an empty artifact.")
        return
    _OUTPUT_PATH.write_text(instructions + "\n")
    print(f"Wrote optimized grader prompt ({len(instructions)} chars) to {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
