"""
Graph assembly — Phase 5: interview loop + orchestration + memory + HITL.

Full topology:
  START → intake ─[conditional on state["company"]]─►
            research (if company set) → planner
            planner                    (if no company)
          → plan_approval [INTERRUPT] → session_router
    ─[conditional on current_modality]─►
      coding_interviewer  ─┐
      sd_interviewer       ├─► grader → coach → advance_session
      beh_interviewer     ─┘                          │
                                     ┌────────────────┘
                                     │ more sessions? → session_router (loop)
                                     └ all done?     → readiness [INTERRUPT] → END

Phase 7a additions:
  - advance_session node: increments session_index after each session
  - _route_after_session: loop back to session_router or proceed to readiness
  - session_router: now uses session_index to pick the right session (was always [0])

Phase 9b addition:
  - research node: a ReAct sub-agent (ONLY dynamic-control-flow node in the graph)
  - _route_after_intake: conditional edge — research only runs if state["company"] is set
  - intake() itself does NOT set company by default; the demo runner (main()) sets it
    explicitly from fixtures/sample_company.txt so all existing offline flows/tests
    are unaffected unless a company is deliberately provided.

Phase 13 additions (both flag-gated, default OFF — see build_graph()'s docstring):
  - panel_grading: grader → grade_dispatch (fan-out) / panel_grader / grade_aggregator
    (fan-in), a parallel "panel of graders" via the Send API (loop/nodes/panel.py).
  - orchestration_mode="supervisor": session_router / _route_by_modality /
    _route_after_session → interview_supervisor, one Command-handoff node that
    decides the next specialist (or readiness) at runtime (loop/nodes/supervisor.py).

Phase 15b addition (flag-gated, default OFF, "fixed" orchestration_mode only):
  - advance_session → _route_after_advance (replaces _route_after_session as the
    edge function) → "replan" when settings.replan_enabled and the just-finished
    session's grade diverges from the plan's assumptions, bounded by
    settings.replan_max_times → replan() re-invokes planner() on the remaining
    sessions only, then loops back to session_router like a normal "continue".

Phase 16b addition (flag-gated, default OFF):
  - readiness → reflect → END: reflect() consolidates recent episodic
    memories into semantic/procedural insights exactly once per curriculum
    run (the "curriculum boundary"), never per-session. With
    settings.reflection_enabled left at its default (False), reflect() is a
    no-op and this is byte-for-byte the Phase 15 graph.

Run with:  uv run python -m loop.graph
"""

import pathlib

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from loop.config import settings
from loop.guardrails import detect_injection, redact_pii
from loop.nodes.coach import coach
from loop.nodes.grader import grader
from loop.nodes.interviewers import beh_interviewer, coding_interviewer, sd_interviewer
from loop.nodes.panel import grade_aggregator, grade_dispatch, panel_grader
from loop.nodes.planner import planner
from loop.nodes.readiness import readiness
from loop.nodes.reflect import reflect
from loop.nodes.research import research
from loop.nodes.supervisor import interview_supervisor
from loop.observability import get_langfuse_callback
from loop.state import LoopState, initial_state

_FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"


# ── Intake node ───────────────────────────────────────────────────────────────


def intake(state: dict) -> dict:
    """Load JD and profile text into state — from a real upload if provided,
    else the static fixtures.

    Does NOT set state["company"] — company stays None (the initial_state()
    default) unless a caller sets it explicitly before invoking the graph.
    This keeps every existing flow (and every test that doesn't care about
    Phase 9) routing straight to the planner, unaffected by the new research
    node. See main() below for how the demo opts in via fixtures/sample_company.txt.

    Phase 10b: jd/profile are untrusted input — redact PII before it ever
    reaches a prompt, and flag (not block) suspected prompt-injection
    attempts so the human can see what was caught.

    Phase 18e: this is the JD-upload seam. A caller (e.g. a future
    POST /sessions body, or a test) can pre-populate state["jd"]/["profile"]
    with real uploaded text BEFORE invoking the graph; intake() uses it
    verbatim instead of the fixture files. Every existing caller leaves
    these fields at initial_state()'s None default, so the fixture path is
    unchanged unless a caller deliberately opts in.
    """
    jd_source = state.get("jd") or (_FIXTURES / "sample_jd.md").read_text()
    profile_source = state.get("profile") or (_FIXTURES / "sample_profile.md").read_text()
    jd = redact_pii(jd_source)
    profile = redact_pii(profile_source)

    flagged = []
    if detect_injection(jd):
        flagged.append({"source": "jd", "reason": "prompt_injection_pattern"})
    if detect_injection(profile):
        flagged.append({"source": "profile", "reason": "prompt_injection_pattern"})

    return {"jd": jd, "profile": profile, "flagged_inputs": flagged}


def _route_after_intake(state: dict) -> str:
    """Return 'research' if a target company is set, else 'planner'.

    This is the ONLY place a fixed workflow branches purely on whether Phase 9
    data is present — every other routing function (e.g. _route_by_modality)
    branches on data the graph itself always produces.
    """
    return "research" if state.get("company") else "planner"


# ── Session router node ───────────────────────────────────────────────────────


def session_router(state: dict) -> dict:
    """Pick the current session from the PrepPlan based on session_index.

    session_index starts at 0 and is incremented by advance_session after each
    session completes.  The loop continues until advance_session finds no more
    sessions and routes to readiness instead of back here.
    """
    plan = state.get("plan") or {}
    sessions = plan.get("sessions") or []
    if not sessions:
        raise ValueError("PrepPlan has no sessions — planner node must run first")
    idx = state.get("session_index") or 0
    if idx >= len(sessions):
        raise ValueError(f"session_index {idx} out of range for {len(sessions)} sessions")
    session = sessions[idx]
    return {
        "current_modality": session["modality"],
        "session_number": session["session_number"],
        "current_focus": session.get("focus"),
        "current_topics": session.get("topics"),
    }


# ── Session advance node + routing ───────────────────────────────────────────


def advance_session(state: dict) -> dict:
    """Increment session_index after a session completes.

    This node has one job: move the cursor forward by 1.
    The conditional edge after this node decides whether to loop back to
    session_router (more sessions) or proceed to readiness (all done).

    Analogy: the loop increment (i++) in a for-loop, followed by the
    condition check (i < n) that controls whether the loop body runs again.
    """
    return {"session_index": (state.get("session_index") or 0) + 1}


def _route_after_session(state: dict) -> str:
    """Return 'continue' if there are more sessions, 'done' if all are complete.

    Called by the conditional edge after advance_session.
    session_index has already been incremented by advance_session, so we compare
    the new index against the total number of sessions.
    """
    plan = state.get("plan") or {}
    sessions = plan.get("sessions") or []
    idx = state.get("session_index") or 0
    return "continue" if idx < len(sessions) else "done"


# ── Replanning (Phase 15b) ───────────────────────────────────────────────────


def _grade_divergence(state: dict) -> bool:
    """Deterministic divergence check: did the session that just finished score
    below settings.replan_score_threshold?

    Grades accumulate across the whole run (state["grades"] reducer appends),
    so the most recently appended grade is the one for the session
    advance_session just closed out. A real system might average several
    signals; this is intentionally the simplest check that makes "replan on
    divergence" demonstrable and offline-testable.
    """
    grades = state.get("grades") or []
    if not grades:
        return False
    return grades[-1]["score"] < settings.replan_score_threshold


def replan(state: dict) -> dict:
    """Re-invoke planner() on the remaining sessions after a live-grade divergence.

    Keeps every already-completed session untouched and replaces sessions from
    session_index onward with a fresh plan (planner() reads the latest
    weak_areas, so the replan reflects what just went wrong). Renumbers the
    replacement sessions to continue the existing sequence and caps them at
    the number of slots that were left, so the plan's total_sessions never
    grows past what was originally approved.

    Only ever reached via _route_after_advance, which bounds how many times
    this can run per session (settings.replan_max_times) — this node itself
    just does the splice and increments the counter.
    """
    plan = state.get("plan") or {}
    sessions = plan.get("sessions") or []
    idx = state.get("session_index") or 0
    remaining_slots = len(sessions) - idx

    new_plan = planner(state)["plan"]
    replacement = list(new_plan.get("sessions") or [])[:remaining_slots]
    for offset, session in enumerate(replacement):
        session["session_number"] = idx + offset + 1

    updated_sessions = sessions[:idx] + replacement
    updated_plan = {**plan, "sessions": updated_sessions, "total_sessions": len(updated_sessions)}

    return {
        "plan": updated_plan,
        "replan_count": (state.get("replan_count") or 0) + 1,
    }


def _route_after_advance(state: dict) -> str:
    """Edge function after advance_session: 'replan', 'continue', or 'done'.

    'replan' only when ALL of: replanning is enabled, there's at least one
    more session left (replanning after the last session is pointless —
    readiness runs next regardless), the just-finished session's grade
    diverges from plan assumptions, and the bound hasn't been hit yet.
    Otherwise this is byte-for-byte _route_after_session — with
    replan_enabled left at its default (False) this function always defers
    to _route_after_session, so the Phase 7a graph is unaffected.
    """
    plan = state.get("plan") or {}
    sessions = plan.get("sessions") or []
    idx = state.get("session_index") or 0
    more_sessions = idx < len(sessions)

    if (
        settings.replan_enabled
        and more_sessions
        and _grade_divergence(state)
        and (state.get("replan_count") or 0) < settings.replan_max_times
    ):
        return "replan"

    return _route_after_session(state)


# ── Plan approval node (HITL gate 1) ─────────────────────────────────────────


def plan_approval(state: dict, next_node: str = "session_router") -> Command:
    """Show the PrepPlan to the human and wait for approval.

    Calls interrupt() to pause the graph.  The human's response dict drives
    routing via Command(goto=...):

      {'decision': 'approve'}
          → Command(goto=next_node, update={'plan_approved': True})
      {'decision': 'edit', 'updated_plan': {...}}
          → Command(goto=next_node, update={'plan': ..., 'plan_approved': True})
      {'decision': 'reject'}
          → Command(goto=END, update={'plan_approved': False})

    IMPORTANT: this node has NO static edge defined — it always returns Command.
    A static edge would compete with Command(goto=END) and cause both paths to run.

    next_node (Phase 13b): "session_router" in orchestration_mode="fixed" (the
    default — every existing call site is unaffected), or "interview_supervisor"
    in orchestration_mode="supervisor". build_graph() binds this via a small
    closure so plan_approval itself stays orchestration-mode-agnostic — it
    doesn't need to know which mode it's running under, just where to hand off.

    Analogy: a pull-request that can be approved, edited, or closed without merge.
    """
    human_response: dict = interrupt(
        {
            "action": "approve_plan",
            "plan": state.get("plan"),
        }
    )

    decision = human_response.get("decision", "approve")

    if decision == "reject":
        return Command(goto=END, update={"plan_approved": False})

    if decision == "edit":
        return Command(
            goto=next_node,
            update={"plan": human_response["updated_plan"], "plan_approved": True},
        )

    # 'approve'
    return Command(goto=next_node, update={"plan_approved": True})


# ── Routing function (used by conditional edge) ───────────────────────────────


def _route_by_modality(state: dict) -> str:
    """Return the node name to route to based on state["current_modality"].

    This function is the 'switch statement' of the conditional edge.
    LangGraph calls it after session_router runs, then follows the returned key
    through the path_map to find the next node.
    """
    modality = state.get("current_modality", "")
    valid = {"coding", "system_design", "behavioral"}
    if modality not in valid:
        raise ValueError(f"Unknown modality: {modality!r}. Must be one of {valid}")
    return modality


# ── Graph definition ──────────────────────────────────────────────────────────


def build_graph(orchestration_mode: str = "fixed", panel_grading: bool = False) -> StateGraph:
    """Define the graph topology (blueprint — not yet runnable).

    Phase 13 parametrizes what was a single fixed topology through Phase 12.
    ONE build_graph() always wires the shared spine (intake → research/planner
    → plan_approval; interviewer → grade → coach → advance_session) and
    branches only the two regions Phase 13 touches — this avoids maintaining
    two near-duplicate graph-assembly functions:

      orchestration_mode:
        "fixed"      (default) — today's deterministic routing: session_router
                     reads the PrepPlan, _route_by_modality/_route_after_session
                     switch on it. Byte-for-byte the Phase 12 graph.
        "supervisor" — interview_supervisor (loop/nodes/supervisor.py) replaces
                     session_router + both routing functions with one LLM-driven
                     Command-handoff node.

      panel_grading:
        False (default) — today's single grader node.
        True             — grade_dispatch/panel_grader/grade_aggregator
                     (loop/nodes/panel.py) replace it with a parallel
                     fan-out/fan-in "panel of graders" via the Send API.

    Both flags default OFF so with no arguments this function reproduces the
    Phase 12 graph exactly — compile_graph() relies on that for its hard pin.
    """
    graph = StateGraph(LoopState)

    # ── Shared spine: nodes every mode/flag combination needs ────────────────
    graph.add_node("intake", intake)
    graph.add_node("research", research)  # Phase 9b — ReAct sub-agent, conditional
    graph.add_node("planner", planner)
    graph.add_node("coding_interviewer", coding_interviewer)
    graph.add_node("sd_interviewer", sd_interviewer)
    graph.add_node("beh_interviewer", beh_interviewer)
    graph.add_node("coach", coach)
    graph.add_node("advance_session", advance_session)
    graph.add_node("readiness", readiness)  # HITL gate 2
    graph.add_node("reflect", reflect)  # Phase 16b — curriculum-boundary consolidation

    # ── plan_approval (HITL gate 1) — hands off to the mode's entry point ───
    # plan_approval has NO static outgoing edge; it always returns Command(goto=...),
    # so a static edge would compete with it and run both paths (see its docstring).
    if orchestration_mode == "supervisor":
        graph.add_node(
            "plan_approval", lambda state: plan_approval(state, next_node="interview_supervisor")
        )
    else:
        graph.add_node("plan_approval", plan_approval)

    # ── Routing region: fixed routing functions OR the supervisor ───────────
    if orchestration_mode == "supervisor":
        graph.add_node("interview_supervisor", interview_supervisor)
        # interview_supervisor has NO static outgoing edge either — same reason
        # as plan_approval: it always returns Command(goto=<specialist|readiness>).
    else:
        graph.add_node("session_router", session_router)
        graph.add_conditional_edges(
            "session_router",
            _route_by_modality,
            path_map={
                "coding": "coding_interviewer",
                "system_design": "sd_interviewer",
                "behavioral": "beh_interviewer",
            },
        )

    # ── Grading region: single grader OR the parallel panel ─────────────────
    if panel_grading:
        graph.add_node("panel_grader", panel_grader)
        graph.add_node("grade_aggregator", grade_aggregator)
        # grade_dispatch is a conditional-edge path FUNCTION, not a node — it
        # returns list[Send] directly (no path_map needed; each Send names
        # its own target). Registered once per interviewer source node.
        graph.add_conditional_edges("coding_interviewer", grade_dispatch)
        graph.add_conditional_edges("sd_interviewer", grade_dispatch)
        graph.add_conditional_edges("beh_interviewer", grade_dispatch)
        graph.add_edge("panel_grader", "grade_aggregator")
        # grade_aggregator has NO static outgoing edge — like plan_approval,
        # it always returns Command(goto=...), either to "coach" (finalized)
        # or to more Sends (13c debate round). A static edge would compete.
    else:
        graph.add_node("grader", grader)
        graph.add_edge("coding_interviewer", "grader")
        graph.add_edge("sd_interviewer", "grader")
        graph.add_edge("beh_interviewer", "grader")
        graph.add_edge("grader", "coach")

    # ── Shared spine edges ────────────────────────────────────────────────────
    graph.add_edge(START, "intake")
    graph.add_conditional_edges(
        "intake",
        _route_after_intake,
        path_map={"research": "research", "planner": "planner"},
    )
    graph.add_edge("research", "planner")
    graph.add_edge("planner", "plan_approval")
    graph.add_edge("coach", "advance_session")

    # ── Multi-session loop: advance_session → (more) or (done) ──────────────
    # advance_session STAYS in both modes — it still just increments
    # session_index; only its outgoing edge changes.
    if orchestration_mode == "supervisor":
        graph.add_edge("advance_session", "interview_supervisor")
    else:
        # Phase 15b: "replan" is only ever returned when settings.replan_enabled
        # is True — with it left at the default (False), _route_after_advance
        # always defers to _route_after_session, so this is byte-for-byte the
        # Phase 7a routing unless a caller has explicitly opted in.
        graph.add_node("replan", replan)
        graph.add_edge("replan", "session_router")
        graph.add_conditional_edges(
            "advance_session",
            _route_after_advance,
            path_map={"replan": "replan", "continue": "session_router", "done": "readiness"},
        )

    graph.add_edge("readiness", "reflect")
    graph.add_edge("reflect", END)

    return graph


def compile_graph(orchestration_mode: str = "fixed", panel_grading: bool = False):
    """Compile the graph without checkpointer/store (used by tests).

    Defaults hard-pin the Phase 12 graph regardless of what's in .env — tests
    call compile_graph() bare and must get the identical graph every time.
    Tests stub nodes in loop.graph's namespace and call compile_graph() fresh
    each time — no thread_id required in the invoke config. Pass explicit
    args to exercise the Phase 13 variants (see tests/test_multiagent.py).
    """
    return build_graph(orchestration_mode=orchestration_mode, panel_grading=panel_grading).compile()


def compile_graph_with_memory():
    """Compile the graph with MemorySaver checkpointer + InMemoryStore.

    Unlike compile_graph(), this reads settings.orchestration_mode /
    settings.panel_grading — this is what the live server/demo runner uses,
    so opting into Phase 13 behavior is a .env change, no code change.

    Required for production use:
    - invoke must pass config={'configurable': {'thread_id': '...', 'user_id': '...'}}
    - thread_id isolates graph state per session
    - user_id keys the long-term store (weak_areas, session_count)
    """
    from loop.memory import compile_with_memory

    return compile_with_memory(
        build_graph(
            orchestration_mode=settings.orchestration_mode,
            panel_grading=settings.panel_grading,
        )
    )


# ── Module-level compiled graph (with memory for the runner) ──────────────────
compiled = compile_graph_with_memory()


# ── Manual runner ─────────────────────────────────────────────────────────────


_CANNED_ANSWERS = [
    {
        "question_id": "cod-001",
        "text": (
            "Use a sliding window with a hash set to track characters. "
            "Right pointer advances; on duplicate, advance left pointer until "
            "duplicate removed. Track max window size. O(n) time, O(k) space."
        ),
    },
    {
        "question_id": "cod-002",
        "text": (
            "Use ReentrantLock with two Conditions: notFull and notEmpty. "
            "put() checks capacity, awaits notFull; take() checks empty, awaits "
            "notEmpty. Always use while-loops not if for spurious wakeups."
        ),
    },
    {
        "question_id": "sys-001",
        "text": (
            "API gateway accepts payment with idempotency key. Key stored in Redis "
            "for dedup. Async queue (Kafka) to payment processor. Outbox pattern for "
            "at-least-once delivery. PostgreSQL event-sourced audit log. Partition by "
            "merchant_id for scale."
        ),
    },
    {
        "question_id": "sys-002",
        "text": (
            "Token bucket in Redis with Lua script for atomic check-and-decrement. "
            "Fail-open on Redis outage. Sliding window log for accuracy-critical endpoints."
        ),
    },
    {
        "question_id": "beh-001",
        "text": (
            "Team chose MongoDB for a financial ledger. I disagreed on ACID grounds, "
            "wrote an RFC comparing Mongo vs Postgres, presented it. We switched to "
            "Postgres — no consistency issues since."
        ),
    },
    {
        "question_id": "beh-002",
        "text": (
            "A critical reconciliation job was failing silently with no owner. "
            "I traced the root cause (timezone bug), fixed it, added alerting. "
            "Zero missed reconciliations since; team adopted the pattern."
        ),
    },
]


def _run_session_with_hitl(user_id: str, thread_id: str, company: str | None = None) -> dict:
    """Run one full graph session through both HITL gates and return final result.

    The graph pauses twice:
      1. plan_approval gate — after planner produces a PrepPlan
      2. readiness gate    — after coach synthesizes feedback

    This runner auto-approves both gates (demo mode).
    In a real UI the human would inspect the payloads and respond interactively.

    Phase 9b: pass company="Stripe" (or any name) to opt into the ReAct research
    node — this is a LIVE network + Bedrock demo, not part of the offline test
    gate. Leave company=None (default) to skip research entirely, exactly like
    every pre-Phase-9 session.
    """
    cb = get_langfuse_callback()
    callbacks = [cb] if cb else []

    state = initial_state()
    state["answers"] = list(_CANNED_ANSWERS)
    state["company"] = company

    config = {
        "configurable": {"thread_id": thread_id, "user_id": user_id},
        "callbacks": callbacks,
    }

    # ── First invoke — runs until the first interrupt ──────────────
    result = compiled.invoke(state, config=config)

    # ── Gate 1: plan_approval ──────────────────────────────────────
    if "__interrupt__" in result:
        ipt = result["__interrupt__"][0]
        payload = ipt.value
        print(f"\n  [GATE 1 — {payload['action']}]")
        plan = payload.get("plan") or {}
        print(
            f"  PrepPlan: {plan.get('total_sessions', '?')} sessions, "
            f"gaps: {plan.get('key_gaps', [])[:2]}"
        )
        print("  → Auto-approving plan (demo mode)")

        # Resume: approve the plan as-is
        result = compiled.invoke(
            Command(resume={"decision": "approve"}),
            config=config,
        )

    # ── Gate 2: readiness ──────────────────────────────────────────
    if "__interrupt__" in result:
        ipt = result["__interrupt__"][0]
        payload = ipt.value
        verdict = payload.get("verdict") or {}
        print(f"\n  [GATE 2 — {payload['action']}]")
        print(
            f"  Readiness verdict: {verdict.get('verdict', '?')} "
            f"(confidence {verdict.get('confidence', 0):.0%})"
        )
        print(f"  Gaps: {verdict.get('gaps', [])[:2]}")
        print("  → Auto-approving verdict (demo mode)")

        # Resume: approve the verdict as-is
        result = compiled.invoke(
            Command(resume={"decision": "approve"}),
            config=config,
        )

    return result


def main() -> None:
    """Phase 5 demo: two-gate HITL flow + cross-session memory.

    Session 1: cold start — two approval gates, auto-approved.
    Session 2: planner reads weak_areas stored by session 1's coach.
    """
    print("Loop — Phase 5 graph run (HITL demo)")
    print("=" * 60)

    # ── Session 1 ─────────────────────────────────────────────────
    print("\n[ SESSION 1 — cold start ]\n")
    r1 = _run_session_with_hitl(user_id="kiran", thread_id="loop-s1")

    print(f"\n  Modality:        {r1.get('current_modality')}")
    print(f"  Question ID:     {r1.get('current_question_id')}")
    print(f"  Grade score:     {r1['grades'][0]['score'] if r1.get('grades') else 'n/a'}/10")
    print(f"  Plan approved:   {r1.get('plan_approved')}")
    rv = r1.get("readiness_verdict") or {}
    print(f"  Readiness:       {rv.get('verdict', 'n/a')}")
    print(f"  Verdict approved:{r1.get('verdict_approved')}")
    print(f"  Weak areas:      {r1.get('weak_areas')}")

    from loop.memory import get_store_instance, semantic_namespace

    item = get_store_instance().get(semantic_namespace("kiran"), "weak_areas")
    if item:
        print(f"\n  Store after session 1: {item.value}")

    # ── Session 2 ─────────────────────────────────────────────────
    print("\n[ SESSION 2 — planner reads stored weak areas ]\n")
    r2 = _run_session_with_hitl(user_id="kiran", thread_id="loop-s2")

    print(f"\n  Modality:        {r2.get('current_modality')}")
    print(f"  Grade score:     {r2['grades'][0]['score'] if r2.get('grades') else 'n/a'}/10")
    rv2 = r2.get("readiness_verdict") or {}
    print(f"  Readiness:       {rv2.get('verdict', 'n/a')}")
    print(f"  Weak areas:      {r2.get('weak_areas')}")

    # ── Session 3 — Phase 9b: ReAct research demo (LIVE network + Bedrock) ──
    print("\n[ SESSION 3 — company research (Phase 9b demo) ]\n")
    company = (_FIXTURES / "sample_company.txt").read_text().strip()
    r3 = _run_session_with_hitl(user_id="kiran", thread_id="loop-s3", company=company)

    cr = r3.get("company_research") or {}
    print(f"\n  Company:          {cr.get('company', 'n/a')}")
    print(f"  Interview format: {cr.get('interview_format', 'n/a')}")
    print(f"  Focus areas:      {cr.get('focus_areas', [])}")
    print(f"  Sources:          {cr.get('sources', [])[:2]}")

    print("\nPhase 5 run complete.")


if __name__ == "__main__":
    main()
