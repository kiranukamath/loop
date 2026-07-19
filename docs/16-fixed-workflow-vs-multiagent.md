# 16 — Fixed workflows vs. multi-agent systems

Every phase through Phase 12 built what's honestly called a **fixed workflow with one
dynamic node**. This doc is the concept-level reset before diving into Phase 13's three
mechanisms (Send/map-reduce, Command handoffs, the supervisor pattern) — it answers "what
actually changes" before "how do I build it."

## Who owns control flow?

That's the entire question. Two systems can run the exact same set of steps (call a
planner, run an interviewer, grade the answer, coach, repeat) and differ only in **who
decides which step runs next**:

- **Fixed workflow:** the *graph's topology* decides. `graph.add_edge("grader", "coach")`
  is a permanent fact about the program, decided at `build_graph()` time, before any input
  ever arrives. A conditional edge like `_route_by_modality` adds a *little* runtime
  flexibility — it inspects state and picks one of a few pre-declared paths — but the
  **set of possible paths is still fixed at build time**. There is no path
  `_route_by_modality` can return that isn't already baked into its `path_map`.
- **Multi-agent system:** a *model* decides. `interview_supervisor` (Phase 13b) doesn't
  pick from a switch statement — it reads state, reasons about it, and returns
  `Command(goto=...)` naming whichever node it decided on. The graph's static edges never
  encode "session_router → coding_interviewer"; that fact only exists as text the model
  generated at runtime.

Loop's graph through Phase 12 has exactly **one** node that already works this way:
`research()`, the Phase 9 ReAct sub-agent. Its internal tool-calling loop is genuinely
dynamic — the model decides which tool to call and when to stop — but from the *parent
graph's* point of view, `research` is still just one more fixed node with a static
outgoing edge to `planner`. Phase 13 is the first time dynamic control flow escapes a
single node and starts shaping the graph's own topology.

## The backend analogy

If you've built backend services, you've drawn this exact line before, just with a
different name:

| Loop concept | Backend analogy |
|---|---|
| Fixed workflow (`add_edge`, `add_conditional_edges` with a `path_map`) | A hard-coded orchestration DAG — Step 2 always follows Step 1, maybe with an `if/else` for two known branches. Think a deployment pipeline's YAML: the set of stages is fixed; only *which* stage runs given `if: branch == main` is conditional. |
| Supervisor node (`interview_supervisor`) | A coordinator **service** that decides call order at runtime — e.g. a saga orchestrator that inspects an order's current state and business rules to decide "call the payment service next" vs. "call the refund service," rather than a BPMN diagram someone drew in advance. |
| Parallel fan-out/fan-in (`grade_dispatch` → `panel_grader` × K → `grade_aggregator`) | A map-reduce job, or a fan-out of N async calls (`CompletableFuture.allOf(...)` / `Promise.all(...)`) followed by a join that only proceeds once all N have returned. |

The reasoning-supervisor idea is *not* free — it trades determinism for adaptability. A
fixed workflow is trivial to reason about (you can read the edges and know every possible
path). A supervisor's possible paths depend on what the model decides, which depends on
the prompt, the data, and (with a real LLM instead of a stub) some amount of run-to-run
variance. That's exactly why Phase 13's bounded-agency lesson (see
[doc 19](19-reducers-deep-dive-and-bounded-agency.md#bounded-agency-reprise)) matters as
much as the mechanism itself: the moment a graph's control flow depends on a model's
judgment, something else has to guarantee it terminates.

## What Phase 13 actually adds, concretely

Both are **flag-gated, default OFF** — Loop's graph is byte-for-byte the Phase 12 graph
unless you opt in (see [doc 17](17-send-api-map-reduce.md) and
[doc 18](18-command-handoffs-and-supervisor-pattern.md) for the mechanics):

1. **`panel_grading=True`** — the single `grader` node becomes a **parallel** panel of K
   persona-graders (`grade_dispatch` → `panel_grader` × K → `grade_aggregator`). This
   isn't really "the model decides control flow" — the fan-out is still declared by the
   graph — but it's the map-reduce half of "multi-agent": several independent model calls
   running concurrently, then reduced into one answer.
2. **`orchestration_mode="supervisor"`** — `session_router` / `_route_by_modality` /
   `_route_after_session` (three fixed-workflow functions) are replaced by ONE node,
   `interview_supervisor`, that reasons about what to run next and hands off via
   `Command(goto=...)`. This IS "the model decides control flow."

Read next: [doc 17](17-send-api-map-reduce.md) for the Send/map-reduce mechanics, then
[doc 18](18-command-handoffs-and-supervisor-pattern.md) for Command handoffs and the
supervisor pattern, then [doc 19](19-reducers-deep-dive-and-bounded-agency.md) for the
reducer internals and the bounded-agency guarantee that makes any of this safe to ship.
