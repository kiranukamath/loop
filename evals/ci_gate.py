"""
Eval-in-CI gate — Phase 17a.

Turns the Phase 6 eval scripts into a two-tier build gate, the same split
you'd use for a flaky-but-valuable integration suite: a fast deterministic
tier that runs on every commit, and a slower probabilistic tier that runs on
a schedule and pages someone when quality regresses.

Tier 1 — offline / deterministic (`check_offline_gate`):
  - Trajectory shape: build the graph with every model-calling node stubbed
    (same pattern as tests/test_evals.py's TestExtractTrajectory), drive it
    through all three interrupts, and assert the visited-node sequence via
    evals.trajectory_check.assert_trajectory (reused, not duplicated).
  - Grader determinism: for each fixtures/grader_labels.json item, stub the
    grader's model deterministically and call loop.nodes.grader.grader()
    twice on identical input — the two calls must return byte-identical
    output, and the (trivially-consistent-by-construction) score must land
    inside the fixture's labeled [score_min, score_max] band. This is a
    wiring/regression check (rubric lookup, reference-answer lookup, node
    plumbing) — it says nothing about whether a *live* model grades well;
    that's Tier 2's job.
  No live model calls. This is what `tests/test_ci_gate.py` runs, and what a
  per-commit CI job would run (see .github/workflows/eval-gate.yml).

Tier 2 — live (`check_live_gate`):
  Wraps evals.run_grader_eval.run_eval() (reused, not duplicated), reads the
  "agreement_rate" run-level score it pushes to Langfuse, and raises
  GateFailure if it's below `min_agreement`. Needs Bedrock + Langfuse
  credentials and hits the network — this is a NIGHTLY job, never the
  laptop/per-commit gate (see CLAUDE.md's "laptop test gate is absolute").

Run directly:
    uv run python -m evals.ci_gate --tier offline
    uv run python -m evals.ci_gate --tier live --min-agreement 0.7
"""

from __future__ import annotations

import json
import pathlib

from evals.run_grader_eval import run_eval
from evals.trajectory_check import assert_trajectory, extract_trajectory
from loop.nodes.grader import grader

_FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"


class GateFailure(AssertionError):
    """Raised when either eval-gate tier fails. A plain AssertionError
    subclass so `pytest.raises(AssertionError)` and `pytest.raises(GateFailure)`
    both work; __main__ below catches it to set a non-zero exit code."""


# ── Tier 1: offline / deterministic ────────────────────────────────────────────


def _build_stubbed_trajectory() -> list[str]:
    """Build the graph with every model-calling node stubbed, drive it through
    all three interrupts with canned resumes, and return the visited-node
    trajectory. Mirrors tests/test_evals.py's TestExtractTrajectory helper —
    kept here (not imported from tests/) so this gate has no dependency on
    the test suite and can run standalone as `python -m evals.ci_gate`.

    The real coding_interviewer node still runs (only its downstream grader
    is stubbed), and it calls into loop/retrieval.py's semantic search --
    under pytest, tests/conftest.py's autouse stub_embeddings/stub_reranker
    fixtures already fake that out for every test; running this module
    directly (no pytest) needs the same fakes applied by hand here so this
    stays a true Tier 1 (no live Bedrock call) either way.
    """
    from unittest import mock
    from unittest.mock import MagicMock

    from langchain_core.embeddings.fake import DeterministicFakeEmbedding
    from langchain_core.runnables import RunnableLambda
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    import loop.retrieval as retrieval_mod
    from loop.graph import build_graph
    from loop.reranker import FakeReranker
    from loop.schemas import Feedback, Grade, PrepPlan, ReadinessVerdict, Session
    from loop.state import initial_state

    stub_plan = PrepPlan(
        role_summary="ci-gate",
        total_sessions=1,
        sessions=[Session(session_number=1, modality="coding", topics=["x"], focus="y")],
        key_gaps=[],
        rationale="ci-gate",
    )
    stub_grade = Grade(
        question_id="cod-001",
        score=7,
        criterion_scores={
            "correctness": 3,
            "optimal_complexity": 2,
            "code_quality": 1,
            "communication": 1,
        },
        strengths=["ok"],
        improvements=[],
        overall_feedback="ok",
    )
    stub_verdict = ReadinessVerdict(
        verdict="ready", confidence=0.8, strengths=["ok"], gaps=[], recommendation="go"
    )
    stub_feedback = Feedback(summary="ok", action_items=[], weak_areas_update=[])

    def fake_model(rv):
        m = MagicMock()
        m.with_structured_output.return_value = RunnableLambda(lambda _: rv)
        return m

    with (
        mock.patch("loop.retrieval.get_embeddings", lambda: DeterministicFakeEmbedding(size=256)),
        mock.patch("loop.retrieval.get_reranker", lambda: FakeReranker()),
        mock.patch("loop.nodes.planner.get_chat_model", lambda: fake_model(stub_plan)),
        mock.patch("loop.graph.grader", lambda s: {"grades": [stub_grade.model_dump()]}),
        mock.patch("loop.nodes.coach.get_chat_model", lambda: fake_model(stub_feedback)),
        mock.patch("loop.nodes.readiness.get_chat_model", lambda: fake_model(stub_verdict)),
    ):
        retrieval_mod._build_index()
        app = build_graph().compile(checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "ci-gate-trajectory"}}
        state = initial_state()
        state["answers"] = [{"question_id": "cod-001", "text": "sliding window..."}]

        app.invoke(state, config=cfg)
        app.invoke(Command(resume={"decision": "approve"}), config=cfg)  # gate 1: plan
        app.invoke(Command(resume="sliding window approach"), config=cfg)  # gate 2: answer
        app.invoke(Command(resume={"decision": "approve"}), config=cfg)  # gate 3: readiness

        return extract_trajectory(app, cfg)


def check_trajectory_gate() -> None:
    """Tier 1a: the graph must visit the expected node sequence. Raises
    GateFailure (via assert_trajectory's AssertionError) on a regression."""
    trajectory = _build_stubbed_trajectory()
    try:
        assert_trajectory(trajectory)
    except AssertionError as exc:
        raise GateFailure(f"Trajectory gate failed: {exc}") from exc


def check_grader_determinism_gate(items: list[dict] | None = None) -> None:
    """Tier 1b: grader() must be deterministic given a fixed (stubbed) model,
    and every fixtures/grader_labels.json item's labeled score band must be
    internally consistent (score_min <= expected_score <= score_max).

    `items` lets tests inject an intentionally-broken fixture in-memory
    instead of writing a permanent bad-fixture file to disk.
    """
    from unittest import mock
    from unittest.mock import MagicMock

    from langchain_core.runnables import RunnableLambda

    from loop.schemas import Grade

    if items is None:
        items = json.loads((_FIXTURES / "grader_labels.json").read_text())["items"]

    for item in items:
        qid = item["question_id"]
        expected = item["expected_score"]
        score_min, score_max = item["score_min"], item["score_max"]

        # Regression #1: the fixture's own labeled band must be self-consistent.
        if not (score_min <= expected <= score_max):
            raise GateFailure(
                f"grader_labels item {item.get('id')!r}: expected_score={expected} "
                f"falls outside its own labeled band [{score_min}, {score_max}]"
            )

        # Stub the model deterministically at the expected score — this gate
        # is about node-wiring determinism (same input -> same output), not
        # live grading accuracy (that's check_live_gate's job).
        stub_grade = Grade(
            question_id=qid,
            score=expected,
            criterion_scores={"overall": expected},
            strengths=["stub"],
            improvements=[],
            overall_feedback="stub",
        )
        fake_model = MagicMock()
        fake_model.with_structured_output.return_value = RunnableLambda(lambda _: stub_grade)

        state = {
            "current_question_id": qid,
            "answers": [{"question_id": qid, "text": item["answer"]}],
            "grades": [],
        }
        with mock.patch("loop.nodes.grader.get_chat_model", lambda: fake_model):
            result_a = grader(state)
            result_b = grader(state)

        if result_a != result_b:
            raise GateFailure(
                f"grader() is non-deterministic for question_id={qid!r} given a "
                f"fixed model: {result_a} != {result_b}"
            )

        score = result_a["grades"][0]["score"]
        if not (score_min <= score <= score_max):
            raise GateFailure(
                f"grader() score {score} for question_id={qid!r} falls outside "
                f"labeled band [{score_min}, {score_max}]"
            )


def check_offline_gate(grader_labels_items: list[dict] | None = None) -> None:
    """Tier 1: run both offline checks. No live model calls."""
    check_trajectory_gate()
    check_grader_determinism_gate(grader_labels_items)


# ── Tier 2: live ────────────────────────────────────────────────────────────────


def check_live_gate(min_agreement: float = 0.7, run_name: str | None = None) -> float:
    """Tier 2: run the live grader eval and enforce a minimum agreement rate.

    Wraps evals.run_grader_eval.run_eval() — does not reimplement dataset
    scoring. Needs Bedrock + Langfuse credentials; hits the network. Meant
    for a nightly job, never `uv run pytest`.

    Returns the observed agreement_rate on success; raises GateFailure if
    Langfuse isn't configured, the dataset produced no results, or agreement
    has regressed below `min_agreement`.
    """
    result = run_eval(run_name=run_name)
    if result is None:
        raise GateFailure("Live eval gate could not run — Langfuse is not configured")

    run_evals = {e["name"]: e["value"] for e in (result.run_evaluations or [])}
    agreement = run_evals.get("agreement_rate")
    if agreement is None:
        raise GateFailure("Live eval gate: no 'agreement_rate' run-level score was produced")

    if agreement < min_agreement:
        raise GateFailure(
            f"Live eval gate FAILED: agreement_rate={agreement:.2f} "
            f"< min_agreement={min_agreement:.2f}"
        )
    return agreement


# ── GitHub Actions wiring (documented here; see .github/workflows/eval-gate.yml) ──
#
#   offline-eval job: on every push/PR, `uv run pytest tests/test_ci_gate.py`
#     (no secrets required — pure offline tier). Fails the build on regression.
#   nightly-eval job: on a cron schedule only, `uv run python -m evals.ci_gate
#     --tier live --min-agreement 0.7`, with Bedrock/Langfuse secrets injected
#     as env vars. Fails the scheduled run (and should page/notify) on
#     regression, but never blocks a commit or PR.


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Loop eval-in-CI gate")
    parser.add_argument("--tier", choices=["offline", "live"], default="offline")
    parser.add_argument("--min-agreement", type=float, default=0.7)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    try:
        if args.tier == "offline":
            check_offline_gate()
            print("Offline eval gate PASSED ✓")
        else:
            agreement = check_live_gate(min_agreement=args.min_agreement, run_name=args.run_name)
            print(f"Live eval gate PASSED ✓ (agreement_rate={agreement:.2f})")
    except GateFailure as exc:
        print(f"EVAL GATE FAILED: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
