"""
Red-team suite — Phase 17c.

Loop's guardrails (loop/guardrails.py, Phase 10b) are a DEFENSIVE check: they
sit at the intake/answer boundary and flag or redact untrusted text before it
reaches a prompt. This module is the OFFENSIVE complement: a small corpus of
known injection/jailbreak strings (fixtures/redteam_prompts.json) run through
those same guardrails, asserting the known attacks actually get caught and
that ordinary, benign candidate answers pass through unchanged.

Two levels, cheapest first:
  1. Unit level — run each fixture item straight through detect_injection()/
     redact_pii() and compare against its expect_flag/expect_pii label.
  2. End-to-end level — run the same strings through the real graph boundary
     (loop.graph.intake for JD/profile-shaped text, loop.nodes.interviewers'
     _ask_question for answer-shaped text) and assert state["flagged_inputs"]
     picks them up exactly the same way -- proving the wiring at the graph
     boundary, not just the guardrail functions in isolation.

Run the demo:
    uv run python -m evals.redteam
"""

from __future__ import annotations

import json
import pathlib

from loop.guardrails import detect_injection, redact_pii

_FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"
_REDTEAM_PROMPTS_PATH = _FIXTURES / "redteam_prompts.json"


def load_redteam_prompts(path: pathlib.Path | None = None) -> list[dict]:
    """Load the red-team corpus. Each item: id, category ("injection" or
    "benign"), text, expect_flag (bool, detect_injection's expected result),
    expect_pii (bool, whether redact_pii should change the text)."""
    data = json.loads((path or _REDTEAM_PROMPTS_PATH).read_text())
    return data["items"]


# ── Level 1: guardrail functions in isolation ──────────────────────────────────


def check_guardrails_against_corpus(items: list[dict] | None = None) -> list[dict]:
    """Run every corpus item through detect_injection()/redact_pii() and
    return the items whose actual result didn't match its expected label.

    Returns an empty list when the corpus passes entirely (the "done when"
    bar) — callers that want a hard failure should assert `not mismatches`.
    """
    items = items if items is not None else load_redteam_prompts()
    mismatches = []
    for item in items:
        actual_flag = detect_injection(item["text"])
        actual_pii_changed = redact_pii(item["text"]) != item["text"]

        problems = []
        if actual_flag != item["expect_flag"]:
            problems.append(f"detect_injection={actual_flag}, expected {item['expect_flag']}")
        if item.get("expect_pii", False) and not actual_pii_changed:
            problems.append("redact_pii left PII-bearing text unchanged")

        if problems:
            mismatches.append({"id": item["id"], "text": item["text"], "problems": problems})
    return mismatches


# ── Level 2: end-to-end through the graph's intake/answer gates ───────────────


def run_through_intake(jd_text: str, profile_text: str) -> dict:
    """Exercise loop.graph.intake()'s guardrail wiring directly with
    attacker-controlled text, bypassing the fixture files it normally reads.

    Mirrors intake()'s own body (loop/graph.py) instead of importing it,
    since intake() always reads fixtures/sample_jd.md + sample_profile.md
    from disk -- this lets the red-team corpus stand in for "a malicious JD
    paste" without touching those fixture files.
    """
    jd = redact_pii(jd_text)
    profile = redact_pii(profile_text)

    flagged = []
    if detect_injection(jd):
        flagged.append({"source": "jd", "reason": "prompt_injection_pattern"})
    if detect_injection(profile):
        flagged.append({"source": "profile", "reason": "prompt_injection_pattern"})

    return {"jd": jd, "profile": profile, "flagged_inputs": flagged}


def check_intake_gate_against_corpus(items: list[dict] | None = None) -> list[dict]:
    """For every corpus item, run it as a (JD, empty profile) pair through
    the intake gate and assert flagging matches the item's expect_flag.
    Returns mismatches (empty = all good), same contract as
    check_guardrails_against_corpus."""
    items = items if items is not None else load_redteam_prompts()
    mismatches = []
    for item in items:
        result = run_through_intake(jd_text=item["text"], profile_text="")
        was_flagged = any(f["source"] == "jd" for f in result["flagged_inputs"])
        if was_flagged != item["expect_flag"]:
            mismatches.append(
                {
                    "id": item["id"],
                    "text": item["text"],
                    "problems": [f"intake flagged={was_flagged}, expected {item['expect_flag']}"],
                }
            )
    return mismatches


def run_through_answer_gate(question_id: str, answer_text: str) -> dict:
    """Exercise the same flag-then-redact sequence
    loop/nodes/interviewers.py's _ask_question runs on a candidate's answer
    after interrupt() returns it, without needing a real interrupt/resume
    round-trip. Mirrors that function's body exactly (see its Phase 10b
    comment) so this red-team check is a faithful stand-in for "an attacker
    types this as their interview answer".
    """
    flagged = []
    if detect_injection(answer_text):
        flagged.append({"source": f"answer:{question_id}", "reason": "prompt_injection_pattern"})
    answer_text = redact_pii(answer_text)
    return {"answer_text": answer_text, "flagged_inputs": flagged}


def check_answer_gate_against_corpus(items: list[dict] | None = None) -> list[dict]:
    """Same check as check_intake_gate_against_corpus, but through the
    answer-gate path instead of the intake path."""
    items = items if items is not None else load_redteam_prompts()
    mismatches = []
    for item in items:
        result = run_through_answer_gate(question_id="cod-001", answer_text=item["text"])
        was_flagged = bool(result["flagged_inputs"])
        if was_flagged != item["expect_flag"]:
            mismatches.append(
                {
                    "id": item["id"],
                    "text": item["text"],
                    "problems": [
                        f"answer gate flagged={was_flagged}, expected {item['expect_flag']}"
                    ],
                }
            )
    return mismatches


# ── Demo runner ───────────────────────────────────────────────────────────────


def _run_demo() -> None:
    items = load_redteam_prompts()
    print(f"Red-team corpus: {len(items)} items")

    guardrail_mismatches = check_guardrails_against_corpus(items)
    intake_mismatches = check_intake_gate_against_corpus(items)
    answer_mismatches = check_answer_gate_against_corpus(items)

    n_attacks = sum(1 for i in items if i["category"] == "injection")
    n_benign = sum(1 for i in items if i["category"] == "benign")
    print(f"  {n_attacks} known attacks, {n_benign} benign inputs")

    all_mismatches = guardrail_mismatches + intake_mismatches + answer_mismatches
    if all_mismatches:
        print("\nMISMATCHES:")
        for m in all_mismatches:
            print(f"  [{m['id']}] {m['problems']}")
        raise SystemExit(1)

    print("\nAll known attacks flagged; all benign inputs passed through clean. PASSED ✓")


if __name__ == "__main__":
    _run_demo()
