# 17 — The Send API: parallel fan-out/fan-in (map-reduce for agents)

This doc covers the mechanism behind Phase 13a (`loop/nodes/panel.py`): how to invoke the
same node N times concurrently with N different inputs, then reduce their results back
into one value. Every claim below was verified against the installed
`langgraph==1.2.5` in a throwaway script before any product code was written (CLAUDE.md
rule #7) — the verification transcripts are included so you can re-run them yourself.

## The problem Send solves

A normal edge (`graph.add_edge("a", "b")`) or conditional edge
(`graph.add_conditional_edges("a", path_fn, path_map={...})`) always sends the **same
state** to **one** node. There's no built-in way to say "run this node three times, once
per persona, each with different input" — until `Send`.

```python
from langgraph.types import Send

Send(node: str, arg: Any, *, timeout=None)
```

Verified signature (`inspect.signature(Send.__init__)` against `langgraph==1.2.5`):

```
(self, /, node: 'str', arg: 'Any', *, timeout: 'float | timedelta | TimeoutPolicy | None' = None) -> 'None'
```

The docstring is explicit about the intended use case:

> One such example is a "map-reduce" workflow where your graph invokes the same node
> multiple times in parallel with different states, before aggregating the results back
> into the main graph's state.

## `Send.arg` REPLACES state — it does not merge with it

This is the load-bearing fact that shapes `grade_dispatch`'s entire design. A normal node
receives the graph's `LoopState`. A `Send`-invoked node receives **exactly `arg`** —
nothing from the rest of state leaks in, and nothing you don't pack into `arg` is
available to the node.

Verified directly:

```python
def worker(payload):
    return {"scratch": [payload["i"] * 10]}

# Send("worker", {"i": 5}) → worker receives {"i": 5}. Not LoopState. Not "i": 5
# merged into a copy of state — LITERALLY {"i": 5}, full stop.
```

This is why `panel_grader(payload: dict)` in `loop/nodes/panel.py` explicitly documents
"reads `payload`, NEVER `state`" — and why `grade_dispatch` has to look up the question,
rubric, and reference answer **before** building each `Send`, packing all of it into
`arg`, rather than letting `panel_grader` fetch anything itself:

```python
def grade_dispatch(state: dict) -> list[Send]:
    ...
    context = _build_context(question_id, answer["text"])
    return [
        Send("panel_grader", {"persona": persona, "question_id": question_id,
                               "round": 0, "peer_scores": None, **context})
        for persona in settings.grader_personas
    ]
```

## Registering a fan-out: a conditional edge, not a node

`grade_dispatch` is never passed to `graph.add_node()`. It's registered exactly like any
other routing function — as the `path` argument to `add_conditional_edges` — except
instead of returning a node-name string, it returns `list[Send]`:

```python
graph.add_conditional_edges("coding_interviewer", grade_dispatch)
```

Verified signature (`inspect.signature(StateGraph.add_conditional_edges)`):

```
(self, source: 'str', path: 'Callable[..., Hashable | Sequence[Hashable]] | ...', path_map: 'dict[Hashable, str] | list[str] | None' = None) -> 'Self'
```

`path_map` is optional and can be omitted entirely when `path` returns `Send` objects —
each `Send` already names its own destination node, so there's nothing for a `path_map` to
translate. Verified with a minimal graph:

```python
def dispatch(state):
    return [Send("worker", {"i": i}) for i in range(3)]

g.add_conditional_edges("entry", dispatch)   # no path_map
```

## The fan-in barrier: reached once, not once per Send

The node downstream of a fan-out — reached via a normal static edge from the fanned-out
node — runs **exactly once per superstep**, regardless of how many `Send`s landed on it.
This is LangGraph's "superstep" model: every task scheduled in a round must complete
before the next node in the topology runs, so three concurrent `panel_grader` calls all
finish before `grade_aggregator` starts, and `grade_aggregator` itself only starts once.

Verified:

```python
def worker(payload):
    return {"scratch": [payload["i"] * 10]}

def collect(state):
    print("collect saw scratch=", state.get("scratch"))
    return {"done": True}

g.add_edge("worker", "collect")
# ... 3 Sends into worker ...
# OUTPUT: collect saw scratch= [0, 10, 20]      <- printed exactly once
```

If `collect` had run once per `Send`, you'd see the print three times, each showing a
different partial `scratch`. It printed once, showing all three results merged — that's
the barrier and the reducer working together (see
[doc 19](19-reducers-deep-dive-and-bounded-agency.md) for the reducer half).

## A node can fan out too — not just a conditional edge

The examples above register `grade_dispatch` as a conditional-edge *path function*. But
`grade_aggregator` (the fan-**in** node) sometimes needs to fan out *again* — Phase 13c's
debate round, where round-0 results trigger one more round of `Send`s before finalizing.
Conditional edges are a fixed part of the topology, declared at `build_graph()` time — but
`grade_aggregator` needs to decide *at runtime, based on what it saw*, whether to fan out
again or stop. A `Command` returned from a **regular node** can do this:

```python
def grade_aggregator(state: dict) -> Command:
    ...
    if settings.panel_debate and latest_round == 0:
        return Command(goto=[Send("panel_grader", payload) for ... ])   # fan out again
    return Command(goto="coach", update={"grades": [...], "panel_grades": None})
```

Verified this exact shape works — a plain node (not a conditional-edge function) returning
`Command(goto=[Send(...), Send(...)])`:

```python
def fan_out(state):
    return Command(goto=[Send("worker", {"i": i}) for i in range(3)])
# ... same fan-in barrier behavior as the conditional-edge case: `collect` runs once.
```

And the two-round debate shape (fan-in that sometimes re-fans-out, sometimes finalizes)
was verified end to end before writing `grade_aggregator`:

```python
def aggregator(state):
    entries = state.get("scratch") or []
    latest = max(e["round"] for e in entries)
    if latest == 0:
        return Command(goto=[Send("worker", {"round": 1, "i": i}) for i in range(2)])
    return Command(goto="finish", update={"scratch": None, "round_seen": latest})
# OUTPUT: FINISH scratch= [] round_seen= 1
```

This is *why* `grade_aggregator` has **no static outgoing edge** in `graph.py` — exactly
like `plan_approval` (Phase 5), a node that sometimes returns `Command(goto=X)` and
sometimes `Command(goto=Y)` must never also have a static edge to either destination, or
both paths fire in the same superstep. See
[doc 18](18-command-handoffs-and-supervisor-pattern.md#the-no-static-edge-rule) for the
full explanation and the concrete bug this would cause.

## Where to look in Loop's code

- `loop/nodes/panel.py::grade_dispatch` — the map step (conditional-edge path function).
- `loop/nodes/panel.py::panel_grader` — the per-persona worker (reads `payload`, not `state`).
- `loop/nodes/panel.py::grade_aggregator` — the reduce step, including the 13c re-fan-out.
- `loop/graph.py::build_graph` — registers `grade_dispatch` via `add_conditional_edges`
  (no `path_map`) and `panel_grader → grade_aggregator` via a plain static edge.
- `tests/test_multiagent.py::TestPanelGradingGraph::test_aggregator_runs_once_not_once_per_persona`
  — the fan-in-once guarantee, asserted through the real compiled graph, not just the
  throwaway scripts above.
