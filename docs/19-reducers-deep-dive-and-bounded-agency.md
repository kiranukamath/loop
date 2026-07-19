# 19 — Reducers as the only way to mutate a channel (+ bounded agency, reprised)

Two lessons Phase 13 makes concrete that were only implicit before: (1) a state channel's
reducer is the *entire* rule for how it changes — including clearing it — and (2) any
system where a model can influence control flow needs an explicit bound, checked in code,
not just implied by good prompting.

## Reducers, briefly (recap)

Every LangGraph state field is a "channel." When a node returns `{"grades": [new_grade]}`,
LangGraph doesn't just overwrite `state["grades"]` — it calls that channel's **reducer**
function: `reducer(old_value, new_value) -> merged_value`. `loop/state.py` already had two
reducers before Phase 13:

- `add_messages` (LangGraph built-in) — appends `BaseMessage`s.
- `_append_list` — Loop's own: `(left or []) + (right or [])`, used for `grades`,
  `answers`, `flagged_inputs`. Treats `None` as "nothing to append" — a genuine no-op.

Phase 13a needs a **third** behavior these two don't cover: a channel that fills up during
one question's grading (`panel_grades`, one entry per persona), then needs to be **wiped
clean** before the next question's grading starts. Neither existing reducer can do this.

## Why `{"panel_grades": []}` doesn't clear anything

This is the counterintuitive part, and it's worth seeing fail before seeing the fix.
`_append_list`'s formula is `(left or []) + (right or [])`. If a node returns
`{"panel_grades": []}`, the reducer receives `right = []`, and computes:

```
(left or []) + ([] or [])  =  left + []  =  left       # UNCHANGED
```

Returning an empty list to an append-reducer channel is a **no-op**, not a reset — you're
asking it to append nothing, and it does exactly that: nothing. Verified directly:

```python
def reset_node(state):
    return {"scratch": []}
# ... 2 workers wrote [0, 1] into scratch via _append_list ...
# RESULT after empty-list reset attempt: {'scratch': [0, 1]}     <- NOT cleared
```

## The fix: a reducer where `None` means "clear," not "no-op"

`_append_list` treats `None` as "nothing to append" (also a no-op, functionally identical
to `[]` for that reducer). `_reset_or_append` (`loop/state.py`) instead makes `None` mean
**clear**:

```python
def _reset_or_append(left: list | None, right: list | None) -> list:
    if right is None:
        return []
    return (left or []) + right
```

Verified this actually clears:

```python
def reset_node(state):
    return {"scratch": None}
# ... 2 workers wrote [0, 1] into scratch via _reset_or_append ...
# RESULT after None-reset: {'scratch': []}                        <- cleared
```

This is safe specifically because `panel_grades` has exactly two writers, and they never
collide in meaning: `panel_grader` always appends a real partial (a dict, never `None`),
and `grade_aggregator` sends `None` exactly when — and only when — it means "I've reduced
this question's partials into a Grade, wipe the scratch space." No other reducer in Loop's
state uses this semantics, and none of them need to — `grades`/`answers`/`flagged_inputs`
only ever grow across a session; nothing ever needs to clear them.

**The rule this generalizes:** a channel's reducer is the *entire* contract for how that
channel can change, in every direction — appending, no-op, AND clearing. If a channel
needs a "reset" behavior, that behavior has to live in the reducer itself; there is no
plain node return that can bypass it.

## Bounded agency, reprised

Phase 9's ReAct research agent introduced Loop's first bounded-agency lesson:
`research_max_iterations` caps how many tool-call rounds the agent can run before
LangGraph's `recursion_limit` would kick in anyway. Phase 13b needs the exact same
discipline for a different reason — `interview_supervisor` is a node whose next
destination depends on a model's judgment, and "the model will probably say readiness
eventually" is not a guarantee.

**Two layers, doing two different jobs:**

1. **The supervisor's own bound** — `idx >= min(len(sessions), settings.max_sessions)`,
   checked in code, *before* the model is ever called. This is the real guarantee: no
   matter what the model would say, a session count past the bound always routes to
   `readiness`. `settings.max_sessions` is reused rather than adding a new config knob —
   it's the same cap `_route_after_session` already enforced in fixed mode.
2. **`recursion_limit`** (LangGraph's own supersteps-per-invoke cap, default 25, raising
   `langgraph.errors.GraphRecursionError`) — the **backstop**, for when layer 1 itself has
   a bug (or someone swaps in a genuinely broken supervisor node that ignores the bound
   entirely).

Verified that the backstop actually fires, not just that it's documented to:

```python
def loopy(state):
    return Command(goto="loopy", update={"n": state.get("n", 0) + 1})
# compiled.invoke({"n": 0}, config={"recursion_limit": 5})
# -> GraphRecursionError: Recursion limit of 5 reached without hitting a stop condition.
```

`tests/test_multiagent.py::TestSupervisorRecursionBackstop` reproduces this with a
deliberately broken `interview_supervisor` stub — one that always hands off to
`coding_interviewer`, never checking `session_index` at all — and asserts
`GraphRecursionError` is raised rather than the process hanging. This is the honest
version of "what if the supervisor never converges": layer 1 (session bound) is
*bypassed entirely* by the broken stub, so only layer 2 (recursion_limit) is left standing
— and it holds.

**The general principle**, restated from Phase 9 and now proven twice: any node whose
outgoing edge is a *decision* rather than a *fact about the graph* needs its own
termination condition, checked in code, independent of what the decision-maker (model or
otherwise) says. The graph-level recursion limit is real, but treating it as your only
safety net means a single misbehaving node can burn 25 supersteps (which, unlike a bounded
`while` loop, includes real model calls, tokens, and latency) before anything stops it.

## Where to look in Loop's code

- `loop/state.py::_reset_or_append` — the reducer, with the full "why `None`, not `[]`"
  reasoning inline.
- `loop/nodes/panel.py::grade_aggregator` — the only writer that ever sends the reset
  sentinel.
- `loop/nodes/supervisor.py::interview_supervisor` — the bound check, evaluated before any
  model call.
- `tests/test_multiagent.py::TestSupervisorRecursionBackstop::test_graph_recursion_error_is_the_real_backstop`
  — the backstop, proven against a real broken node, not just a design doc.
