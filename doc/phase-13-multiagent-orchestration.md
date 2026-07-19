# Phase 13 — Multi-agent orchestration

**Capability taught:** supervisor pattern · parallel fan-out/fan-in (Send) · Command
handoffs — crossing from "a fixed workflow with one dynamic node" into a real multi-agent
system, on Loop's own graph.

**Status:** 13a, 13b, and 13c (stretch) all done & tested. Phase 13 is fully done.

See [docs/16–19](../docs/README.md#phase-13--multi-agent-orchestration) for the underlying
concepts (fixed workflows vs. multi-agent, the Send API, Command handoffs + the supervisor
pattern, reducers + bounded agency) — this doc is the "what we actually built" walkthrough,
in the same spirit as [phase-11](phase-11-session-history.md) and
[phase-12](phase-12-mcp-interop.md).

## Why this phase exists

Every phase through Phase 12 has one dynamic-control-flow node (the Phase 9 ReAct
researcher), and otherwise a fixed graph: every edge is a permanent fact decided at
`build_graph()` time. Phase 13 is the first time control flow itself becomes something a
*model* decides at runtime — a supervisor node picking the next specialist — and the first
time one answer is scored by several independent model calls running **concurrently**
rather than sequentially. Both changes are flag-gated, default OFF: the existing
deterministic graph and all 254 Phase-12 tests keep passing byte-for-byte unchanged unless
you opt in.

## Verifying the API before writing any code (CLAUDE.md rule #7)

`Send`, `Command`, reducer-reset semantics, and the "a node can return
`Command(goto=[Send(...), ...])`" fan-out shape were all verified in throwaway scripts
against the installed `langgraph==1.2.5` before touching `loop/nodes/panel.py` or
`loop/nodes/supervisor.py` — the full transcripts live in [docs 17–19](../docs/README.md).
The two facts that most shaped the design, only found by actually running code:

- **`Send.arg` REPLACES the node's input; it does not merge with graph state.** A
  `Send`-invoked node receives exactly `arg` — nothing else. This is why `grade_dispatch`
  has to pre-fetch the question/rubric/reference-answer and pack everything into each
  `Send`'s payload, rather than letting `panel_grader` look anything up itself.
- **`{"channel": []}` does NOT clear an append-reducer channel** — `(left or []) + ([] or
  []) = left`, unchanged. Only a reducer that treats a *different* sentinel (Loop uses
  `None`) as "clear" can actually reset a channel. This directly shaped
  `_reset_or_append` (`loop/state.py`).

## 13a — Parallel fan-out/fan-in: a panel of graders

`loop/nodes/panel.py` replaces the single `grader` node with three functions implementing
a map-reduce shape, gated by `settings.panel_grading` (default `False`):

```python
def grade_dispatch(state: dict) -> list[Send]:
    ...
    context = _build_context(question_id, answer["text"])
    return [
        Send("panel_grader", {"persona": persona, "question_id": question_id,
                               "round": 0, "peer_scores": None, **context})
        for persona in settings.grader_personas   # default: correctness, communication, depth
    ]
```

`grade_dispatch` is registered via `add_conditional_edges(source, grade_dispatch)` — no
`path_map`, since each `Send` already names `panel_grader` as its target. It's registered
once per interviewer source node (`coding_interviewer`, `sd_interviewer`,
`beh_interviewer`), replacing the static `interviewer → grader` edges when
`panel_grading=True`.

`panel_grader(payload)` reads only `payload`, never `state` — it can't see the other two
personas' invocations or anything else in the graph. It scores the **full** rubric from
one persona's lens (personas are an emphasis, not a 1:1 map to rubric criterion names,
since criteria vary per question), and stitches `persona`/`question_id`/`round` from the
payload rather than trusting the model to echo them back:

```python
def panel_grader(payload: dict) -> dict:
    ...
    result: PersonaGrade = chain.invoke({...})
    partial = {"persona": persona, "question_id": payload["question_id"],
               "round": payload["round"], **result.model_dump()}
    return {"panel_grades": [partial]}
```

`grade_aggregator(state)` is the fan-in, reached by a static `panel_grader →
grade_aggregator` edge (verified: 3 `Send`s into one aggregator → the aggregator runs
exactly once, not three times). It averages each criterion across personas (not sums —
each persona scores the *full* rubric, so summing would blow past `max_score`), unions
strengths/improvements, and produces a `Grade` with the **identical shape** the sequential
`grader` node produces — `coach` and everything downstream never know panel grading ran.

**`grade_aggregator` always returns `Command`, never a plain dict** — same reason
`plan_approval` (Phase 5) does. It routes to `"coach"` when finalizing, or (13c) to more
`Send`s when a debate round is due. A static outgoing edge would compete with either
`Command`, so `graph.py` registers none.

## 13b — Supervisor + handoffs

`loop/nodes/supervisor.py::interview_supervisor` replaces `session_router` +
`_route_by_modality` + `_route_after_session` as a unit, gated by
`settings.orchestration_mode == "supervisor"` (default `"fixed"`):

```python
def interview_supervisor(state: dict) -> Command[Literal[
    "coding_interviewer", "sd_interviewer", "beh_interviewer", "readiness"
]]:
    plan = state.get("plan") or {}
    sessions = plan.get("sessions") or []
    idx = state.get("session_index") or 0

    if not sessions or idx >= min(len(sessions), settings.max_sessions):
        return Command(goto="readiness")            # bound hit — no model call

    session = sessions[idx]
    decision: SupervisorDecision = chain.invoke({...})   # model DECIDES next modality
    return Command(
        goto=_MODALITY_TO_NODE[decision.next_modality],
        update={"current_modality": decision.next_modality, ...},
    )
```

`advance_session` (Phase 7a) **stays exactly as-is** — it still just increments
`session_index`. Only its outgoing edge changes: `advance_session → interview_supervisor`
(unconditional) in supervisor mode, vs. the Phase 12 conditional edge
(`session_router`/`readiness`) in fixed mode.

`plan_approval` (Phase 5) needed one small change to support this: an optional `next_node`
parameter (default `"session_router"`, unchanged for every existing call site), so
`build_graph()` can bind it to `"interview_supervisor"` via a closure in supervisor mode —
`plan_approval` itself stays orchestration-mode-agnostic:

```python
if orchestration_mode == "supervisor":
    graph.add_node("plan_approval", lambda state: plan_approval(state, next_node="interview_supervisor"))
else:
    graph.add_node("plan_approval", plan_approval)
```

`interview_supervisor` has no static outgoing edge either — same "always returns Command"
discipline as `plan_approval` and `grade_aggregator`.

**`ONE build_graph()`, not two variants.** `build_graph(orchestration_mode="fixed",
panel_grading=False)` always wires the shared spine (`intake → research/planner →
plan_approval`, `interviewer → grade → coach → advance_session`) and branches only the two
regions Phase 13 touches, with plain `if`. `compile_graph()` hard-pins both flags to their
Phase-12 defaults regardless of `.env`, so the existing test suite is pinned to exactly
the old graph; `compile_graph_with_memory()` reads `settings`, so the live server can opt
in with an env var change and no code change.

## 13c (stretch) — a panel debate round

`settings.panel_debate` (default `False`, only meaningful when `panel_grading=True`)
extends `grade_aggregator`: seeing only round-0 partials, it fans out ONE more round —
each persona sees the *other* personas' round-0 scores and revises once — before reducing
whichever round is latest:

```python
def grade_aggregator(state: dict) -> Command:
    ...
    if settings.panel_debate and latest_round == 0:
        sends = [Send("panel_grader", {..., "round": 1, "peer_scores": [...]}) for p in current_round]
        return Command(goto=sends)          # fan out again — no state update yet

    grade = _reduce_to_grade(question_id, current_round)
    return Command(goto="coach", update={"grades": [grade.model_dump()], "panel_grades": None})
```

`panel_grades` accumulates BOTH rounds (the reducer just appends), and `grade_aggregator`
filters to `latest_round` before reducing — round-0 partials are never silently
double-counted. Verified this two-round shape (fan-in that conditionally re-fans-out)
works correctly with a minimal graph before writing it into `panel.py` — see
[doc 17](../docs/17-send-api-map-reduce.md).

## Testing strategy

Per the Phase 13 offline test strategy: `loop.nodes.supervisor.get_chat_model` /
`loop.nodes.panel.get_chat_model` are stubbed with `RunnableLambda`s, exactly like every
prior phase. Two different stub shapes were needed:

- **Constant stubs** (`lambda _: STUB_VALUE`, ignoring input) — the existing
  conftest.py planner-stub pattern, used everywhere the test doesn't need the model's
  *output* to depend on its input (panel grading math, the bound check, the recursion
  backstop).
- **An actual function of input**, for one specific test: "a stubbed supervisor reproduces
  today's modality sequence for a 2-session plan." Since `chain = _PROMPT | structured_model`
  pipes the *rendered* prompt into the stub (not the raw input dict), the stub parses the
  rendered human message back out (`re.search(r"suggested modality for this session:
  (\w+)", text)`) to echo whatever modality the plan suggested — reproducing the exact
  sequence `_route_by_modality` would have produced, while still going through a genuine
  model-shaped call.

`tests/test_multiagent.py` (18 tests) covers:

- `grade_dispatch` — fan-out shape, payload contents, configurable personas, missing-answer error.
- `panel_grader` — payload fields win over model output; handles round-1 peer scores.
- `grade_aggregator` — averaging math, strengths/improvements dedup, reset sentinel,
  missing-partials error, the 13c debate fan-out, and reducing only the latest round.
- The full compiled graph with `panel_grading=True` — the aggregator runs exactly once
  (not once per persona), and the final grade + reset scratch channel are correct end to
  end.
- `interview_supervisor` — stops at the bound without ever calling the model; maps a
  decision to the right interviewer node.
- The full compiled graph with `orchestration_mode="supervisor"` — reproduces the
  plan's modality sequence, and the model is never called a 3rd time once the bound is hit.
- `GraphRecursionError` as the real backstop, against a deliberately broken supervisor
  stub that ignores `session_index` entirely.

**Mutation-tested, not just written and trusted:** before calling this done, two
regressions were deliberately introduced and confirmed to fail the right tests — disabling
`interview_supervisor`'s bound check (broke both bound-related tests, as expected) and
adding a static `grade_aggregator → coach` edge alongside its `Command` (this one did
**not** get caught by the existing tests, since `panel_debate=False` in that test means the
static edge and the `Command` agree on the same destination — a genuine gap, documented
rather than silently left implied. The risk itself was independently verified via the
throwaway scripts in [doc 17](../docs/17-send-api-map-reduce.md#a-node-can-fan-out-too--not-just-a-conditional-edge),
so the design is still correct; the test suite just doesn't exercise the *specific* failure
mode of someone re-adding that edge in the non-debate case).

## Result

18 new tests in `tests/test_multiagent.py`, all offline; 272 total passing; `ruff check`
clean. All four `(orchestration_mode, panel_grading)` combinations compile with exactly
the expected node sets, confirmed directly:

```
default graph nodes:          [..., 'grader', ..., 'session_router']
panel-only graph nodes:       [..., 'grade_aggregator', 'panel_grader', ..., 'session_router']
supervisor-only graph nodes:  [..., 'grader', ..., 'interview_supervisor']
supervisor+panel graph nodes: [..., 'grade_aggregator', 'panel_grader', ..., 'interview_supervisor']
```
