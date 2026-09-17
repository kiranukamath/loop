"""
Grader evaluation — score Loop's grader node against human-labeled examples.

Run with:
    uv run python -m evals.run_grader_eval

Requires:
  - Langfuse credentials in .env  (to read dataset + push scores)
  - AWS Bedrock credentials        (to call the grader model)
  - `evals/seed_dataset.py` run first to populate the dataset

What this does (step by step):
  1. Pull the "loop-grader-eval" dataset from Langfuse (all items).
  2. For each item, run Loop's grader node on the item's answer.
  3. Compare Loop's score to the human-labeled expected score.
  4. Push two scores per item back to Langfuse:
       - "score_in_range"  : 1.0 if |actual - expected| ≤ tolerance, else 0.0
       - "absolute_error"  : |actual - expected|   (lower = better)
  5. Push one aggregate score for the whole run:
       - "mean_absolute_error": average of all per-item absolute errors

Why use dataset.run_experiment() instead of a manual loop?
  run_experiment() handles threading, error recovery, linking traces to dataset
  items, and creating the run record in Langfuse UI — you'd need 50+ lines to
  replicate that manually.  It's the eval equivalent of JUnit's @ParameterizedTest:
  you write the what-to-test and how-to-score; the framework handles the wiring.
"""

from __future__ import annotations

import pathlib
from typing import Any

_FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"
_DATASET_NAME = "loop-grader-eval"


# ── Task function ─────────────────────────────────────────────────────────────
# Receives one DatasetItem; returns Loop's grader output for that item.
# Signature required by run_experiment: task(*, item, **kwargs) -> Any


def task(*, item: Any, **kwargs: Any) -> dict:
    """Run the grader node on one dataset item and return the Grade dict."""
    from loop.nodes.grader import grader

    qid = item.input["question_id"]
    answer_text = item.input["answer"]

    # Build the state the grader expects
    state = {
        "current_question_id": qid,
        "answers": [{"question_id": qid, "text": answer_text}],
        "grades": [],
    }

    result = grader(state)
    grades = result.get("grades") or []
    if not grades:
        raise ValueError(f"Grader returned no grades for question_id={qid!r}")
    return grades[0]  # Grade.model_dump() dict


# ── Item-level evaluator ──────────────────────────────────────────────────────
# Called once per dataset item after task() returns.
# Returns a list of Evaluation dicts — each becomes a score in Langfuse.


def score_agreement(
    *,
    input: dict,
    output: dict,
    expected_output: dict,
    metadata: dict | None,
    **kwargs: Any,
) -> list[dict]:
    """Compare Loop's grade to the human label.

    Returns two metrics:
    - score_in_range : 1 if Loop's score falls within [score_min, score_max], else 0
    - absolute_error : |Loop score − expected score|   (a contribution to MAE)
    """
    actual = output.get("score", 0)
    expected = expected_output.get("score", 0)
    score_min = expected_output.get("score_min", expected - 1)
    score_max = expected_output.get("score_max", expected + 1)

    in_range = score_min <= actual <= score_max
    abs_error = abs(actual - expected)

    return [
        {
            "name": "score_in_range",
            "value": 1.0 if in_range else 0.0,
            "comment": f"Loop={actual}, Human={expected} (acceptable {score_min}–{score_max})",
        },
        {
            "name": "absolute_error",
            "value": float(abs_error),
            "comment": f"|{actual} − {expected}| = {abs_error}",
        },
    ]


# ── Run-level evaluator ───────────────────────────────────────────────────────
# Called once after all items complete — receives every item's result.


def aggregate_mae(*, item_results: list[Any], **kwargs: Any) -> list[dict]:
    """Compute mean absolute error across all items in this run."""
    errors = []
    for r in item_results:
        if r.output and r.expected_output:
            actual = r.output.get("score", 0)
            expected = r.expected_output.get("score", 0)
            errors.append(abs(actual - expected))

    if not errors:
        return []

    mae = sum(errors) / len(errors)
    agreement_rate = sum(1 for e in errors if e <= 1) / len(errors)

    return [
        {
            "name": "mean_absolute_error",
            "value": round(mae, 2),
            "comment": f"Average |Loop score − human score| across {len(errors)} items",
        },
        {
            "name": "agreement_rate",
            "value": round(agreement_rate, 2),
            "comment": (
                f"Fraction of items where |error| ≤ 1"
                f"  ({sum(1 for e in errors if e <= 1)}/{len(errors)})"
            ),
        },
    ]


# ── Runner ────────────────────────────────────────────────────────────────────


def run_eval(run_name: str | None = None) -> Any:
    """Pull the dataset and run the experiment.

    Returns the `run_experiment()` result (so Phase 17a's live CI-gate tier
    can inspect result.run_evaluations for the agreement threshold check), or
    None if Langfuse isn't configured. Existing callers (just __main__ below)
    ignored the return value already, so adding one is not a breaking change.
    """
    from loop.observability import get_langfuse_client

    client = get_langfuse_client()
    if client is None:
        print("Langfuse not configured — set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY in .env")
        return None

    dataset = client.get_dataset(_DATASET_NAME)
    print(f"Dataset '{_DATASET_NAME}': {len(dataset.items)} items")

    # run_experiment() handles:
    #   - threading (max_concurrency controls parallelism)
    #   - linking each task run to the dataset item in Langfuse
    #   - creating a named dataset run for UI comparison
    result = dataset.run_experiment(
        name="grader-eval",
        run_name=run_name,  # e.g. "grader-eval-2026-06-15"
        description="Evaluate Loop's grader node against human-labeled answers.",
        task=task,
        evaluators=[score_agreement],
        run_evaluators=[aggregate_mae],
        max_concurrency=2,  # small — respects Bedrock rate limits
    )

    print(f"\nRun: {result.run_name}")
    print(f"Items processed: {len(result.item_results)}")

    # Print per-item summary
    for ir in result.item_results:
        actual = ir.output.get("score", "err") if ir.output else "err"
        expected = ir.expected_output.get("score", "?") if ir.expected_output else "?"
        evals_summary = ", ".join(f"{e['name']}={e['value']}" for e in (ir.evaluations or []))
        print(f"  {ir.item.id!r:35s}  Loop={actual:>3}  Human={expected:>3}  {evals_summary}")

    # Print run-level scores
    if result.run_evaluations:
        print("\nRun-level metrics:")
        for e in result.run_evaluations:
            print(f"  {e['name']}: {e['value']}  — {e.get('comment', '')}")

    client.flush()
    print("\nScores pushed to Langfuse.")
    return result


if __name__ == "__main__":
    import sys

    run_name = sys.argv[1] if len(sys.argv) > 1 else None
    run_eval(run_name=run_name)
