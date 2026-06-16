"""
Seed the Langfuse eval dataset from fixtures/grader_labels.json.

Run once (or re-run to add new items — Langfuse deduplicates by item id):

    uv run python -m evals.seed_dataset

Requires Langfuse credentials in .env (LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY).

What this creates in Langfuse:
  Dataset name : "loop-grader-eval"
  One item per entry in grader_labels.json:
    input          → {question_id, answer, rubric (loaded from fixtures)}
    expected_output → {score, score_min, score_max, label_notes}

Why store data in Langfuse instead of a local file?
  - You can compare eval runs side-by-side in the UI.
  - You can add items without touching code.
  - Every run is linked back to the dataset item, so you can drill into
    which specific examples got harder/easier over time.
  Analogy: like moving your JUnit test data from hard-coded constants to a
  database table — same tests, but richer history and easier to extend.
"""

from __future__ import annotations

import json
import pathlib
import sys

_FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"
_DATASET_NAME = "loop-grader-eval"


def _load_rubric(question_id: str) -> dict | None:
    rubrics_path = _FIXTURES / "rubrics.json"
    rubrics = json.loads(rubrics_path.read_text())
    for r in rubrics["rubrics"]:
        if r["question_id"] == question_id:
            return r
    return None


def seed(dry_run: bool = False) -> int:
    """Seed the Langfuse dataset.  Returns number of items upserted."""
    from loop.observability import get_langfuse_client

    client = get_langfuse_client()
    if client is None:
        print("Langfuse not configured — set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY in .env")
        return 0

    labels_path = _FIXTURES / "grader_labels.json"
    labels = json.loads(labels_path.read_text())

    # create_dataset is idempotent — safe to call on every run
    client.create_dataset(
        name=_DATASET_NAME,
        description=(
            "Human-labeled answer→grade pairs for evaluating Loop's grader node. "
            "Covers coding, system-design, and behavioral questions "
            "at strong/medium/weak quality levels."
        ),
    )
    print(f"Dataset '{_DATASET_NAME}' ready.")

    count = 0
    for item in labels["items"]:
        qid = item["question_id"]
        rubric = _load_rubric(qid)

        input_payload = {
            "question_id": qid,
            "answer": item["answer"],
            "rubric": rubric,
        }
        expected_payload = {
            "score": item["expected_score"],
            "score_min": item["score_min"],
            "score_max": item["score_max"],
            "label_notes": item.get("label_notes", ""),
        }

        if dry_run:
            print(
                f"  [dry-run] would upsert item {item['id']!r}"
                f"  expected_score={item['expected_score']}"
            )
        else:
            client.create_dataset_item(
                dataset_name=_DATASET_NAME,
                id=item["id"],  # stable id — Langfuse deduplicates on this
                input=input_payload,
                expected_output=expected_payload,
                metadata={"label_notes": item.get("label_notes", "")},
            )
            print(f"  upserted {item['id']!r}  q={qid}  expected_score={item['expected_score']}")
        count += 1

    print(f"\nSeeded {count} items into '{_DATASET_NAME}'.")
    return count


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    seed(dry_run=dry)
