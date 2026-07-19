# 18 — Command handoffs and the supervisor pattern

Phase 13b (`loop/nodes/supervisor.py`) replaces three fixed-workflow functions
(`session_router`, `_route_by_modality`, `_route_after_session`) with one node,
`interview_supervisor`, that decides the next specialist at runtime and **hands off** to
it. This doc covers the handoff mechanism (`Command(goto=...)`) and the supervisor pattern
it enables — Loop already used one handoff before Phase 13 even started, which turns out
to be the best way to understand the new one.

## `Command` — a node's other return type

Every node so far (`grader`, `coach`, `session_router`, ...) returns a plain `dict`:
"here's what changed in state." LangGraph merges it in and follows whatever static/
conditional edges are declared for that node. `Command` is a **third way to return** that
carries its own routing instruction, bypassing the graph's declared edges entirely:

```python
class Command(Generic[N], ToolOutputMixin):
    graph: str | None = None
    update: Any | None = None
    resume: dict[str, Any] | Any | None = None
    goto: Send | Sequence[Send | N] | N = ()
```

(Verified via `inspect.getsource(Command)` against the installed `langgraph==1.2.5`.)

`update` is the same "state changes" a plain dict would carry. `goto` is the new part: the
name of the node to run next (or a `Send`, or a list of either — see
[doc 17](17-send-api-map-reduce.md) for the `Send` half). A node returning `Command`
**decides its own next step**, rather than the graph's static edges deciding it.

## Loop already does this once — `plan_approval`

Before Phase 13, `plan_approval` (`graph.py:147`, Phase 5) is the one place in Loop's
graph where a node's next step depends on a runtime decision rather than a fixed edge:

```python
def plan_approval(state: dict, next_node: str = "session_router") -> Command:
    human_response = interrupt({"action": "approve_plan", "plan": state.get("plan")})
    decision = human_response.get("decision", "approve")

    if decision == "reject":
        return Command(goto=END, update={"plan_approved": False})
    if decision == "edit":
        return Command(goto=next_node, update={"plan": human_response["updated_plan"], ...})
    return Command(goto=next_node, update={"plan_approved": True})
```

The human's decision (approve / edit / reject) determines whether the graph proceeds to
`next_node` or jumps straight to `END`. **A human** made this decision via `interrupt()`.
Phase 13b's `interview_supervisor` is the exact same shape — a node that returns
`Command(goto=...)` based on a decision — except **a model** makes the decision, and it's
choosing between more than two destinations.

Seeing this parallel is the whole "aha" of Phase 13b: handoffs aren't a new concept
introduced by multi-agent systems, they're the *same* mechanism Loop's HITL gates already
used, generalized from "human decides yes/no/edit" to "model decides which of N
specialists."

## The no-static-edge rule

`plan_approval`'s docstring has always carried this warning:

> IMPORTANT: this node has NO static edge defined — it always returns Command. A static
> edge would compete with Command(goto=END) and cause both paths to run.

Phase 13 adds two more nodes that follow the same rule: `interview_supervisor` and
`grade_aggregator`. The reasoning is identical in both cases — a node that sometimes needs
to route to different destinations at runtime must have **no** static outgoing edge in the
graph, because LangGraph doesn't treat a `Command`'s `goto` as an override of a static
edge; it treats it as an *additional* instruction. If both exist, both fire in the same
superstep.

This was verified directly (not just inferred from the docstring) while building
`grade_aggregator`'s Phase 13c debate round — see
[doc 17's fan-out section](17-send-api-map-reduce.md#a-node-can-fan-out-too--not-just-a-conditional-edge)
for the passing verification script, and the mutation-tested proof that removing this
discipline doesn't reliably fail existing tests either (it's a design rule enforced by
code review / docstrings, not something every possible regression is automatically caught
by — see `doc/phase-13-multiagent-orchestration.md`'s testing section for the honest
caveat).

## The supervisor pattern

A "supervisor" in the multi-agent literature is just: one node whose entire job is
picking which specialist runs next, based on state, using a model instead of a switch
statement. `interview_supervisor` replaces THREE fixed-workflow pieces at once:

| Phase 12 (fixed) | Phase 13b (supervisor) |
|---|---|
| `session_router` — reads `plan["sessions"][idx]`, sets `current_modality`/`current_focus`/`current_topics`/`session_number` | `interview_supervisor` reads the same plan data (plus `weak_areas`) and *decides* the modality/focus/topics via a structured-output model call |
| `_route_by_modality` — `dict` lookup: modality string → interviewer node name | `interview_supervisor` maps its own `SupervisorDecision.next_modality` → node name directly, in the same function |
| `_route_after_session` — `idx < len(sessions)` ? `"continue"` : `"done"` | `interview_supervisor`'s own bound check, evaluated BEFORE calling the model (see [doc 19](19-reducers-deep-dive-and-bounded-agency.md#bounded-agency-reprise)) |

```python
def interview_supervisor(state: dict) -> Command[Literal[
    "coding_interviewer", "sd_interviewer", "beh_interviewer", "readiness"
]]:
    plan = state.get("plan") or {}
    sessions = plan.get("sessions") or []
    idx = state.get("session_index") or 0

    if not sessions or idx >= min(len(sessions), settings.max_sessions):
        return Command(goto="readiness")           # bound hit — no model call at all

    session = sessions[idx]
    decision: SupervisorDecision = chain.invoke({...})   # model call — the real decision

    return Command(
        goto=_MODALITY_TO_NODE[decision.next_modality],
        update={"current_modality": decision.next_modality, ...},
    )
```

Two things worth noticing:

1. **The bound check runs first, unconditionally, before any model call.** This isn't a
   style choice — it means a finished plan never spends tokens/latency asking a model
   "should we stop?" The model is only ever consulted when there's a genuine decision to
   make.
2. **The model can deviate from the plan.** Unlike `_route_by_modality`, which blindly
   trusts `session["modality"]`, the supervisor's prompt explicitly allows it to pick a
   *different* modality than the plan suggested, if `weak_areas` warrants it. This is the
   one place in Loop where the plan is a *suggestion* a model can override, not a fixed
   script — see the `_SYSTEM` prompt in `loop/nodes/supervisor.py`.

## `Command[Literal[...]]` — documenting the possible destinations

`interview_supervisor`'s return type is annotated
`Command[Literal["coding_interviewer", "sd_interviewer", "beh_interviewer", "readiness"]]`.
`Command` is generic (`Command(Generic[N], ...)`), so this subscription is valid Python —
verified directly:

```python
>>> Command[Literal["a", "b"]]
langgraph.types.Command[typing.Literal['a', 'b']]
```

This is documentation, not enforcement — LangGraph doesn't use the annotation to validate
`goto` at runtime. Its value is the same as any type hint's: a reader (or an IDE) can see
every destination this node might hand off to, without having to trace the model call and
the `_MODALITY_TO_NODE` dict to reconstruct that list themselves. `plan_approval`, by
contrast, is annotated as a plain `Command` (no `Literal`) — its destinations
(`next_node`, `END`) are simpler and already spelled out in its docstring, so the extra
annotation wasn't worth the noise there. Use the annotation when a node's possible
handoffs are non-obvious from a quick read; skip it when they're already clear.

## Where to look in Loop's code

- `loop/graph.py::plan_approval` — the original (Phase 5) handoff, now parametrized with
  `next_node` so it can hand off to either `session_router` (fixed mode) or
  `interview_supervisor` (supervisor mode) without knowing which mode is active.
- `loop/nodes/supervisor.py::interview_supervisor` — the new handoff node.
- `loop/graph.py::build_graph` — wires `plan_approval`'s `next_node` and
  `advance_session`'s outgoing edge based on `orchestration_mode`; registers
  `interview_supervisor` with no static outgoing edge.
- `tests/test_multiagent.py::TestSupervisorGraph` — the full graph reproducing today's
  plan-based sequence with a stubbed (but input-dependent) model.
