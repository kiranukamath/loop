# PLAN.md — Loop build plan & progress tracker

This is the **single source of truth** for what to build, in what order, and how far we've
gotten. Each phase below is written to be **self-contained** so a fresh session (including a
Sonnet session) can pick up a phase cold. Read `CLAUDE.md` first for the working contract.

**Golden rule:** build one phase per turn, teach before coding, pause for approval at the
end of each phase, and update the **Progress tracker** + **Changelog** below when a phase
completes.

---

## Progress tracker

| Phase | Title | Capability taught | Status | Approved by owner |
|------:|-------|-------------------|--------|-------------------|
| 0 | Foundation | setup / model seam / observability | ✅ done & approved | pending |
| 1 | State & minimal graph | LangGraph mental model | ✅ done & approved | pending |
| 2 | Planning | structured output, planning | ✅ done & approved | pending |
| 3 | Interview loop & orchestration | orchestration, sub-agents | ✅ done & approved | — |
| 4 | Memory | short- + long-term memory | ✅ done & approved | — |
| 5 | HITL | interrupts & resume | ✅ done & approved | — |
| 6 | Eval & observability | agent evaluation | ✅ done & approved | — |
| 7 | Make it usable (web UI) | streaming + real HITL + durable state | ✅ done & approved | — |

Status legend: ⬜ not started · 🟡 in progress · ✅ done & approved · ⏸️ blocked

**Phase 7 complete ✅ (158 tests, 0 lint errors). All sub-steps 7a–7e done.**
Loop is fully usable: `uv run uvicorn loop.api:app --port 8000 --reload` → open http://localhost:8000

**Phase 0 decisions (owner, 2026-06-13):**
- **Two environments:** this laptop = minimal *dev box* — install deps, run `ruff` + unit
  `pytest` only. The **server** (has Langfuse, AWS creds) is where live Bedrock calls,
  tracing, and full graph runs happen. **Invariant for every phase:** everything under
  `tests/` MUST pass offline on the laptop (no network/Bedrock/Langfuse — stub the model);
  anything needing a real model or a trace is a *server* activity, not part of the laptop
  test gate.
- **Python 3.14** (matches system python; owner updated CLAUDE.md). `uv python pin 3.14`.
- **Bedrock:** creds AVAILABLE on laptop — a **long-term Bedrock API key** (bearer token,
  env `AWS_BEARER_TOKEN_BEDROCK`, not the older access-key/secret pair). So hello-world can
  make a real call locally. Config must read this key + region + model id from env. Verify
  `ChatBedrockConverse` picks up the bearer-token env var during Phase 0 (check installed
  `langchain-aws`/`boto3` — fall back to access-key/secret if needed).
- **Langfuse:** NOT present (anywhere) → wire `observability.py` as a seam that is a **no-op
  when unconfigured** (no keys/host). Stand up Langfuse (Docker) later, before/at Phase 6.
- **Tests stay offline regardless:** unit tests stub the model — they never call Bedrock,
  even though creds now exist. Live hello-world call is a manual/dev check, not the test gate.
- **Fixtures:** use generic-but-realistic samples (senior backend role); owner can swap in
  real JD/profile anytime.

---

## How to run any phase (the loop)

1. Read `CLAUDE.md` + this phase's section here.
2. **Teach first:** explain what/why/concepts/tradeoffs (see "Concepts to teach").
3. **Verify APIs** for any new library used this phase (`uv pip show`, read installed
   source or official docs) — do not trust memorized APIs.
4. Implement only the files in "Files touched". Keep it minimal.
5. **Walk through the code** file by file; comment tricky parts inline.
6. Run the "Done when" checks. Show output.
7. Update Progress tracker + Changelog here. **Pause for approval.**

---

## Phase 0 — Foundation  *(Capability: none yet — setup, model seam, observability)*

**Goal:** a runnable skeleton that proves the model factory and Langfuse tracing work, with
static fixtures in place. No agent graph yet.

**Concepts to teach:**
- `uv` projects & lockfiles vs. pip/venv (analogy to Maven/Gradle for a Java dev).
- The **model factory** pattern: why we hide `ChatBedrockConverse` behind a function so the
  model is swappable (Bedrock now, Ollama later) — like a Spring `@Bean`/interface seam.
- **Observability from day one**: what Langfuse traces, why a "callback handler" is the
  hook LangChain calls on every model invocation (analogy: a servlet filter / interceptor).
- `pydantic-settings` for typed config from env.

**Task checklist:**
- [ ] Confirm/install `uv`; pin Python 3.14 (`uv python pin 3.14`).
- [ ] `uv init` the project; create `pyproject.toml` with deps: `langgraph`, `langchain`,
      `langchain-aws`, `langfuse`, `pydantic`, `pydantic-settings`, `python-dotenv`;
      dev deps: `ruff`, `pytest`. (Verify latest compatible versions during install.)
- [ ] Configure `ruff` + `pytest` in `pyproject.toml`.
- [ ] `loop/config.py` — `Settings` (model id, AWS region, Langfuse host/keys) via env.
- [ ] `loop/models.py` — `get_chat_model()` factory returning `ChatBedrockConverse`, with a
      clear extension point (provider switch) for the v2 Ollama swap. Do NOT implement Ollama.
- [ ] `loop/observability.py` — Langfuse client + LangChain callback handler wiring.
- [ ] `docker-compose.yml` for self-hosted Langfuse; document start-up.
- [ ] `.env.example` listing required vars (never commit a real `.env`).
- [ ] `fixtures/`: `sample_jd.md`, `sample_profile.md`, `questions.json` (a few per
      modality), `rubrics.json`. Keep small but realistic.
- [ ] A "hello world" script that calls the model once and shows the trace in Langfuse.
- [ ] One smoke test in `tests/` (e.g. factory returns a model; config loads).

**Files touched:** `pyproject.toml`, `loop/{config,models,observability,__init__}.py`,
`docker-compose.yml`, `.env.example`, `fixtures/*`, `tests/test_smoke.py`, `README.md`.

**Done when (revised per decisions above):** the project installs via `uv sync`; the model
factory + config + observability seam are in place; the hello-world script is wired (runs a
live Bedrock call *if* creds exist, else fails with a clear "configure Bedrock" message);
Langfuse wiring is a safe no-op when unconfigured; fixtures exist; `uv run ruff check .` and
`uv run pytest` pass. (Live Bedrock call + real Langfuse trace verified later, once enabled.)

**Skills needed (for a Sonnet session):** Python packaging with `uv`; reading
`langchain-aws` / `langfuse` installed APIs; Docker Compose basics; pydantic-settings.
No agentic concepts required yet.

---

## Phase 1 — State & a minimal graph  *(Capability: the LangGraph mental model)*

**Goal:** define `LoopState` and build a trivial graph: `intake → one node → END`. Compile
and invoke it. This is purely to internalize the graph mechanics.

**Concepts to teach:**
- **State** = the shared, typed dict that flows through the graph (analogy: the request
  context / accumulator passed between handlers). What a **reducer** is and why
  `add_messages`-style reducers exist (append vs. overwrite).
- **Node** = a function `state -> partial state update`. **Edge** = wiring between nodes.
  **Entry point**, **END**. **compile()** turns the definition into a runnable; **invoke()**
  runs it (analogy: building vs. executing a pipeline).

**Task checklist:**
- [ ] `loop/state.py` — `LoopState` (TypedDict or Pydantic): JD, profile, messages,
      session metadata, placeholders for plan / grades / weak areas.
- [ ] `loop/graph.py` — `StateGraph`, one trivial node (e.g. `intake` that loads fixtures
      into state), wire `intake -> END`, `compile()`.
- [ ] A runner (`uv run python -m loop.graph` or a script) that invokes the graph and prints
      resulting state. Trace via Langfuse.
- [ ] `tests/test_graph.py` — invoking the compiled graph yields expected state shape.

**Files touched:** `loop/state.py`, `loop/graph.py`, `tests/test_graph.py`.

**Done when:** graph compiles, invokes, returns populated state; test + lint pass; trace
visible in Langfuse.

**Skills needed:** LangGraph `StateGraph`/node/edge/compile/invoke; TypedDict/Pydantic state;
reducers (esp. `add_messages`). Verify against installed `langgraph` version.

---

## Phase 2 — Planning  *(Capability: PLANNING)*

**Goal:** a `planner` node that turns JD + profile (+ later, weak areas) into a structured
**PrepPlan** using Pydantic structured output.

**Concepts to teach:**
- **Structured output**: forcing the model to return a typed object (analogy: deserializing
  a response into a DTO/POJO instead of parsing strings). `.with_structured_output(Schema)`.
- Prompt design for planning: system role, grounding in JD+profile, constraints.
- Planning as an agent capability: decompose goal → curriculum → per-session plan.

**Task checklist:**
- [ ] `loop/schemas.py` — `PrepPlan` (topics, per-session modality mix, focus areas,
      rationale), plus sub-models as needed.
- [ ] `loop/nodes/planner.py` — planner node: builds prompt from state, calls model with
      structured output, writes `PrepPlan` into state.
- [ ] Wire planner into the graph after intake.
- [ ] `tests/test_planner.py` — given fixture JD+profile, planner returns a valid `PrepPlan`
      (stub/msock the model or assert schema validity).

**Files touched:** `loop/schemas.py`, `loop/nodes/planner.py`, `loop/graph.py`, tests.

**Done when:** invoking the graph produces a schema-valid `PrepPlan` grounded in the JD;
lint + tests pass; trace shows the structured call.

**Skills needed:** Pydantic v2 models; LangChain `with_structured_output` (verify provider
support on Bedrock Converse); prompt templating. Builds on Phases 0–1.

---

## Phase 3 — Interview loop & orchestration  *(Capability: ORCHESTRATION)*

**Goal:** sub-agents per modality (coding / system-design / behavioral) + a grader + a
feedback coach; a **lead** node routes by modality; implement the ask → answer → grade loop.

**Concepts to teach:**
- **Orchestration / sub-agents**: a lead node delegating to specialist nodes (analogy:
  a coordinator service calling domain services). Why separate prompts/roles per modality.
- **Conditional edges / routing**: choosing the next node based on state (the agent's
  control flow). The **agent loop**: ask → collect answer → grade → decide continue/stop.
- Reading questions + rubrics via `loop/tools.py` over fixtures (stable interface for v2).

**Task checklist:**
- [ ] `loop/tools.py` — fetch question(s) by modality/topic and rubric by question id from
      fixtures; v2-stable signatures.
- [ ] `loop/nodes/interviewers.py` — coding / system-design / behavioral interviewer nodes.
- [ ] `loop/nodes/grader.py` — grades an answer against its rubric (structured `Grade`).
- [ ] `loop/nodes/coach.py` — turns grades into actionable feedback.
- [ ] `loop/schemas.py` — add `Grade`, `Feedback`, `Answer` models.
- [ ] Lead/router node + conditional edges implementing the loop over the plan's sessions.
- [ ] Tests for routing logic and the grade loop (deterministic with stubbed model).

**Files touched:** `loop/tools.py`, `loop/nodes/{interviewers,grader,coach}.py`,
`loop/schemas.py`, `loop/graph.py`, tests.

**Done when:** graph runs a full mock session (pick modality → ask → answer → grade →
feedback) over fixtures; routing covered by tests; lint + tests pass; trace shows sub-agent
calls.

**Skills needed:** LangGraph conditional edges & routing; multi-node graphs; structured
output reuse; tool functions. Heaviest phase — may warrant splitting across two turns.

---

## Phase 4 — Memory  *(Capability: MEMORY — short- & long-term)*

**Goal:** add a **checkpointer** (short-term, resume a session) and a **store** (long-term:
weak areas, history, target role) such that long-term memory **changes the next plan**.

**Concepts to teach:**
- **Short-term memory = checkpointer**: persists graph state per thread so you can pause and
  resume (analogy: HTTP session / saved transaction state). `thread_id` as the key.
- **Long-term memory = store**: cross-session facts keyed by user, queried by nodes
  (analogy: a user-profile table / cache that outlives a request). v1 in-memory; v2 Postgres.
- **Semantic recall**: retrieving relevant past items (weak areas / questions) by similarity.
- The feedback loop: grades → update weak areas in store → planner reads them next session.

**Task checklist:**
- [ ] `loop/memory.py` — checkpointer (in-memory/SQLite for v1) + store wiring; v2 seam for
      Postgres documented.
- [ ] Compile graph with checkpointer; run with a `thread_id`; demonstrate resume.
- [ ] After grading, persist weak areas to the store keyed by user.
- [ ] Planner (Phase 2) reads weak areas from the store → adapts the next `PrepPlan`.
- [ ] Tests: resume restores state; second-session plan reflects stored weak areas.

**Files touched:** `loop/memory.py`, `loop/graph.py`, `loop/nodes/planner.py`,
`loop/nodes/grader.py`, tests.

**Done when:** a session can be resumed by `thread_id`; a second session's plan demonstrably
changes based on stored weak areas; tests + lint pass.

**Skills needed:** LangGraph checkpointers (`MemorySaver`/SQLite) & `BaseStore`; thread/
config plumbing; (optional) embeddings for semantic recall. Verify store API on installed
`langgraph`.

---

## Phase 5 — HITL  *(Capability: HUMAN-IN-THE-LOOP)*

**Goal:** interrupt the graph to let a human **approve the prep plan** and the **readiness
verdict**, then resume with the human's decision.

**Concepts to teach:**
- **Interrupts**: pausing a graph mid-run to get human input, then resuming (analogy: a
  workflow that blocks on a manual approval step). `interrupt()` / `Command(resume=...)` and
  why this requires a checkpointer (from Phase 4).
- Designing approval gates: what state is shown, what decisions resume the graph.

**Task checklist:**
- [ ] Add an approval interrupt after the planner (approve/edit/reject the `PrepPlan`).
- [ ] Add a `readiness` node producing a "ready / not ready" verdict + an interrupt for the
      human to approve/override it.
- [ ] Runner demonstrates: run → hits interrupt → human responds → resume to completion.
- [ ] Tests for both interrupt points (resume with approve and with reject/override).

**Files touched:** `loop/graph.py`, `loop/nodes/` (readiness), runner, tests.

**Done when:** graph pauses at both gates and resumes correctly on human input; tests + lint
pass; interrupts/resumes visible in traces.

**Skills needed:** LangGraph `interrupt`/`Command` API (verify on installed version — this
API has changed across releases); checkpointer-backed resume from Phase 4.

---

## Phase 6 — Eval & observability  *(Capability: EVAL)*

**Goal:** evaluate both the **product** (are the user's answers scored well?) and the **agent
itself** (does Loop's grading agree with human labels?). Track improvement; add a trajectory
check.

**Concepts to teach:**
- **Eval datasets**: labeled examples (answer → expected grade) stored in Langfuse.
- **Scoring the grader**: compare Loop's grade vs. human label (agreement / error).
- **Trajectory check**: did the agent take a sensible path (right modality, asked → graded →
  adapted)? Distinct from output scoring.
- Why agent eval differs from unit tests: probabilistic, needs aggregate metrics over a set.

**Task checklist:**
- [ ] `evals/` — a small labeled dataset (answers + expected grades) loaded to Langfuse.
- [ ] Eval script: run the grader over the dataset, score agreement, push scores to Langfuse.
- [ ] A trajectory check over a recorded run (assert expected node sequence / decisions).
- [ ] Document how to read results in Langfuse and how to track improvement over time.

**Files touched:** `evals/*`, possibly `loop/observability.py`, tests/docs.

**Done when:** running the eval produces grader-vs-human agreement scores visible in
Langfuse; a trajectory check passes; documented how to interpret + track over time.

**Skills needed:** Langfuse datasets + scores API (verify on installed `langfuse` — major
API changes between v2 and v3); designing eval metrics; LangGraph run introspection.

---

## Phase 7 — Make it usable (FastAPI + web UI)  *(Capability: streaming + real HITL + durable state)*

**Goal:** turn Loop from a test-only graph into something a human can actually sit down and
use in a browser. v1 faked every human touchpoint (auto-approved gates, pre-injected answers,
one session, in-memory state). Phase 7 makes all of those *real*, driven from a clean web UI.

**Owner decisions (2026-06-17):** UI stack = **FastAPI backend + single Tailwind HTML page**
(no build step, vanilla JS `fetch`/`EventSource`). Scope = **full usable bundle** (streaming +
real answers/approvals + SQLite persistence + multi-session loop), built incrementally.

**The graph is the unchanged domain service; FastAPI is just a second "driver" beside the tests.**
This is the lesson: if the architecture is clean, adding a web layer doesn't touch the graph's core.

**New deps to add (verified absent 2026-06-17):**
`langgraph-checkpoint-sqlite` (SqliteSaver — separate package from core langgraph),
`fastapi`, `uvicorn[standard]`, `sse-starlette`.

**Concepts to teach (as each sub-step needs them):**
- **Streaming** (`graph.stream(..., stream_mode=...)`) — emit node/token events as they happen
  instead of one blocking `invoke()`. The core agentic-UX pattern v1 never used.
- **SSE (Server-Sent Events)** — one-way server→browser push over plain HTTP. Simpler than
  WebSockets; perfect for streaming agent progress. Analogy: a long-lived HTTP response that
  keeps flushing chunks (like a Spring `SseEmitter`).
- **interrupt() over HTTP** — the LangGraph pause/resume gates become request/response pairs:
  the interrupt payload is the HTTP response; `Command(resume=...)` is the next request body.
- **Durable checkpointer** — `SqliteSaver` instead of `MemorySaver`, so sessions + resume
  survive a process restart (the laptop-friendly middle step before the v2 Postgres swap).
- **Multi-session orchestration** — loop the graph over *all* planned sessions, not just one.

**Sub-steps (each = one turn: teach → code → test → pause):**

- **7a — Multi-session loop (graph only, no UI).** Add a `session_index` to state.
  `session_router` picks `sessions[session_index]`. After `coach`, a conditional edge routes
  back to `session_router` if sessions remain, else on to `readiness`. Tests for the loop.
  *Why first:* don't build a UI on a one-shot graph.
- **7b — Real answer gate.** Add a third interrupt: interviewer asks → `interrupt()` for the
  human's answer → resume → grader grades the real answer. Canned answers kept only as a
  test fallback. Tests for the new gate.
- **7c — SQLite persistence.** Add `langgraph-checkpoint-sqlite`; swap `MemorySaver` →
  `SqliteSaver` behind the existing `loop/memory.py` seam (store decision verified at build
  time). Prove a session resumes after a simulated restart. Tests.
- **7d — FastAPI backend (no page yet).** `loop/api.py`: `POST /sessions` (run to first gate),
  `POST /sessions/{id}/resume` (approve/edit/reject/answer), `GET /sessions/{id}/stream` (SSE
  node progress). Returns interrupt payloads as JSON. Tests via FastAPI `TestClient` (offline,
  stubbed model).
- **7e — Tailwind page.** One static HTML page served by FastAPI: start → review/approve plan →
  answer questions (streamed) → see feedback → approve/override readiness verdict. Looks clean,
  zero build step.

**Files touched (across sub-steps):** `loop/graph.py`, `loop/state.py`, `loop/memory.py`,
`loop/api.py` (new), `loop/static/index.html` (new), `pyproject.toml`, `tests/*`, `PLAN.md`.

**Done when:** a person can open a browser, get a plan, approve it, answer interview questions
with live streaming, and receive a readiness verdict they can approve/override — and the whole
session survives a server restart. Tests stay offline (stub the model); lint + pytest pass.

**Skills needed:** FastAPI (routing, `TestClient`, static files); SSE / `sse-starlette`;
LangGraph `.stream()` + streaming modes; `SqliteSaver`; a little vanilla JS + Tailwind CDN.

---

## v2 / future enhancements (NOT in v1)

- Real data sources (live JD ingest, real question bank) replacing fixtures.
- Postgres checkpointer + store (swap the Phase 4 seam).
- Ollama / open-source model via the Phase 0 model factory seam.
- Richer, larger, semantically-indexed question bank.

---

## Changelog

> Append one entry per completed phase: date, phase, what was built, key decisions, what the
> owner learned. Keep newest at top.

### Phase 7c+7d+7e — 2026-06-21
**Built (7c — SQLite persistence):** `db_path` setting in `config.py`; `_make_checkpointer()` in
`memory.py` reads config and returns `SqliteSaver(conn)` (with `setup()` + `mkdir -p` on first run)
or `MemorySaver` when `db_path` is empty. `DB_PATH` set in `.env` → `db/loop_state.sqlite`. `db/`
gitignored. 3 new tests in `TestCheckpointerFactory` including persistence-across-reconnect test.

**Built (7d — FastAPI backend):** `loop/api.py` — `POST /sessions`, `POST /sessions/{id}/resume`,
`GET /sessions/{id}/stream` (SSE). `_pending` dict holds the Command between resume POST and next
stream GET. `_safe_payload()` strips large text fields; only scores/summaries go over the wire.
`StaticFiles` mount + root redirect to `/static/index.html`. 20 new tests in `test_api.py`.
`pyproject.toml` updated with `fastapi`, `uvicorn[standard]`, `sse-starlette`,
`langgraph-checkpoint-sqlite`. `.claude/launch.json` created for preview tool.

**Built (7e — Tailwind UI):** `loop/static/index.html` — single-page dark-theme interview UI.
Landing screen → progress steps (Plan / Interview / Assessment) → live activity log (SSE node events
stream in as each agent node completes) → Gate 1 card (PrepPlan review, approve/edit/reject) →
Gate 2 card (question + textarea, Ctrl+Enter shortcut) → Gate 3 card (readiness verdict,
approve/override) → Done screen. Verified end-to-end with real Bedrock: 6-session plan generated,
answer scored 9/10, multi-session loop advancing to session 2.

**Key decisions / lessons:**
- `SqliteSaver(conn)` takes a raw `sqlite3.Connection` directly; `check_same_thread=False` is required
  for FastAPI (same thread handles multiple requests). `setup()` is idempotent — safe to call on every startup.
- One SSE connection per graph segment (open → stream nodes → close on interrupt/done). Browser
  re-opens EventSource after each resume POST. Simpler than long-lived connections with async queues.
- `graph.stream(stream_mode="updates")` yields `{node_name: state_delta}` dicts. `__interrupt__`
  is a special key: value is a tuple of `Interrupt` objects with `.value` = the interrupt payload.
- FastAPI sync routes run in a thread pool automatically — no `asyncio.run_in_executor` boilerplate needed.
- All three capabilities (SQLite, FastAPI, SSE) verified live: real Bedrock call, real grading, real
  multi-session loop visible in the browser.

### Phase 7a+7b — 2026-06-19
**Built (7a — multi-session loop):** `session_index` field in `LoopState`; `_append_list` reducer on
`answers` + `grades`; `advance_session` node; `_route_after_session` routing function; conditional edge
`advance_session → session_router` (continue) or `→ readiness` (done). `tests/test_multisession.py`
(18 tests). 128 tests, 0 lint errors.

**Built (7b — real answer gate):** `interrupt()` call inside `_ask_question` in `interviewers.py`;
interviewer now returns AI question message + human answer message + answer delta for the reducer;
grader refactored to return only the new grade (reducer handles accumulation). Three-gate flow:
plan_approval → answer_question → approve_verdict; `_run_session_with_hitl` in `graph.py` updated
to dispatch on interrupt `action` key. All tests updated for 3-gate flow (test_hitl.py redesigned,
test_memory.py interviewer stubs added). 135 tests, 0.70s, 0 lint errors.

**Key decisions / lessons:**
- **Loop-back edges:** LangGraph directed graphs can have cycles. `advance_session` increments index,
  `_route_after_session` decides continue vs done. Clean separation of concerns — like a Spring Batch
  `RepeatStatus.CONTINUABLE` pattern.
- **Reducers are essential for accumulating state across loop iterations.** Without `_append_list` on
  `grades`, session 2 overwrites session 1's grades. The reducer fires: `left ++ right`.
- **"Return only the delta" rule:** when a field has a reducer, node must return ONLY the new item,
  not the full accumulated list. Returning the full list causes the reducer to double-append.
- **`interrupt()` for input, not just decisions.** Gate 2 uses interrupt to collect the candidate's
  answer text — the graph pauses mid-interviewer-node and resumes with the string.
- **Stub pattern for interrupt in unit tests:** `monkeypatch.setattr("loop.nodes.interviewers.interrupt", lambda payload: answer)`.
  Calling an interviewer function directly outside a graph context fails because interrupt() requires
  a live graph run with a checkpointer.

### Phase 6 — 2026-06-16
**Built:** `fixtures/grader_labels.json` (9 human-labeled answer→grade pairs across coding,
system-design, behavioral modalities at strong/medium/weak quality); `evals/__init__.py`;
`evals/seed_dataset.py` (seeds Langfuse dataset, idempotent via stable item ids, dry-run mode);
`evals/run_grader_eval.py` (task function, `score_agreement` per-item evaluator, `aggregate_mae`
run-level evaluator, `run_eval()` using `dataset.run_experiment()`); `evals/trajectory_check.py`
(`extract_trajectory` from checkpointed state history, `assert_trajectory` with prefix/interviewer/suffix checks);
`loop/observability.py` updated (added `get_langfuse_client()`); `tests/test_evals.py` (25 new tests).
110/110 tests, 0.70s. 0 lint errors.

**Key decisions / lessons:**
- Eval datasets in Langfuse = parameterized fixtures stored externally. Like JUnit `@ParameterizedTest`
  data, but with run-history tracking across deploys.
- `dataset.run_experiment()` (Langfuse v4) handles threading, trace linking, and run creation — replaces
  the manual `item.link()` loop from v2/v3. Accepts `task`, `evaluators` (per-item), `run_evaluators` (aggregate).
- Two complementary eval dimensions: **output quality** (did grader score correctly?) and **trajectory**
  (did the agent visit nodes in the right order?). These are independent — a wrong path can produce a right answer.
- `extract_trajectory()`: `get_state_history()` returns snapshots newest-first; reverse to chronological; collect
  `snap.next[0]` filtering system nodes (`__start__`, `__end__`, `__resume__`).
- Stub the entire grader node function (not just its model) in trajectory tests — grader validates `state["answers"]`
  before calling the model, so patching the model alone still raises.
- `get_langfuse_client()` returns None when unconfigured so eval scripts degrade gracefully on dev machines.

### Phase 5 — 2026-06-15
**Built:** `loop/nodes/readiness.py` (model call → ReadinessVerdict schema → interrupt for human
approval/override); `loop/graph.py` updated (plan_approval inline node with interrupt; readiness
wired after coach; graph topology now has two HITL gates; main() demonstrates the two-gate
auto-approve flow); `loop/schemas.py` updated (ReadinessVerdict schema added); `loop/state.py`
updated (readiness_verdict: Optional[dict]); `tests/test_hitl.py` (16 new tests: gate 1 pause,
payload content, next-nodes, approve/edit/reject paths; gate 2 pause, approve/override paths;
full two-gate flow, state persistence, thread isolation). 85/85 tests, 0.66s.

**Key decisions / lessons:**
- `interrupt(payload)` does NOT raise — it returns early with `__interrupt__` in the state
  dict. The node resumes from the interrupt() call when `app.invoke(Command(resume=...), cfg)`
  is called with the same thread_id.
- `Command(goto=END)` does NOT override a static edge. If a node has a static outgoing edge
  AND returns `Command(goto=END)`, both paths run. Fix: give plan_approval NO static edge —
  it always returns `Command(goto=...)` to make routing explicit.
- `Command(goto='X', update={...})` is returned FROM a node to control routing dynamically.
  `Command(resume=value)` is passed TO `invoke()` from the caller side to resume a paused
  graph — they are different uses of the same dataclass.
- All non-HITL tests stub `plan_approval` (returning `Command(goto='session_router', ...)`)
  and `readiness` (returning a canned verdict dict) so interrupt gates are skipped.

### Phase 4 — 2026-06-14
**Built:** `loop/memory.py` (MemorySaver + InMemoryStore singletons, compile_with_memory());
`loop/nodes/planner.py` updated (_get_stored_weak_areas reads from store, merges with state weak_areas);
`loop/nodes/coach.py` updated (_persist_weak_areas writes to store after each session);
`loop/graph.py` updated (compile_graph_with_memory(), two-session demo in main());
`tests/test_memory.py` (14 new tests covering store basics, planner reads, coach writes,
thread isolation, cross-session feedback loop). 69/69 tests, 0.38s.

**Key decisions:**
- `get_store()` and `get_config()` both raise RuntimeError outside a graph context.
  Both are wrapped in try/except so planner/coach work when called in unit tests directly.
- `get_store()` returns None inside a graph compiled WITHOUT a store — nodes check `if store is None`.
- compile_graph() (no memory) stays for tests; compile_graph_with_memory() is for production.
- Module-level singletons in memory.py ensure the store persists across calls in the same process.
- user_id passed via config["configurable"]["user_id"] — separate from thread_id so one user can
  have many sessions but one store entry. Default: "default".
- v2 seam: swap MemorySaver → AsyncPostgresSaver, InMemoryStore → PostgresStore.

---

### Phase 3 — 2026-06-14
**Built:** `loop/tools.py` (get_questions_by_modality, get_question_by_id, get_rubric — v2-stable DAO pattern);
`loop/nodes/interviewers.py` (coding_interviewer, sd_interviewer, beh_interviewer — shared _ask_question helper);
`loop/nodes/grader.py` (grades answer against rubric via with_structured_output(Grade));
`loop/nodes/coach.py` (synthesizes grades into Feedback, updates weak_areas);
`loop/schemas.py` updated (Answer model added); `loop/graph.py` updated (8-node graph with conditional routing);
`tests/test_phase3.py` (27 new tests); conftest + test_graph + test_planner updated.
55/55 tests, 0.31s.

**Key decisions:**
- Conditional edges: `add_conditional_edges("session_router", _route_by_modality, path_map={...})`.
  The routing function is just `(state) -> str` — like a switch statement expressed as graph topology.
- Stub strategy: conftest patches `loop.graph.grader` / `loop.graph.coach` (node function names
  in graph.py's namespace), not just the models. After `compile_graph()` re-runs each test, it picks
  up the stubbed lambdas. Patching just the model is insufficient because the grader raises before
  reaching the model when answers are missing.
- `importlib.reload(graph_mod)` in test_planner wipes conftest patches — that test stubs grader/coach
  via `monkeypatch.setattr(graph_mod, "grader", ...)` after the reload.
- Phase 3 answers are pre-injected into state["answers"]. Phase 5 will replace this with a real
  human interrupt.

---

### Phase 2 — 2026-06-13
**Built:** `loop/schemas.py` (PrepPlan, Session, Grade, Feedback); `loop/nodes/planner.py`
(ChatPromptTemplate + with_structured_output chain); `graph.py` updated to intake → planner
→ END; `tests/test_planner.py` (8 tests); `tests/conftest.py` (autouse stub_planner fixture
so graph tests stay offline). 28/28 tests, 0.40s.

**Key decisions:**
- `RunnableLambda` required instead of bare `MagicMock` for stubs — LangChain's pipe `|`
  wraps non-Runnables as callables, so `.invoke()` never fires on a plain mock.
- `conftest.py` `autouse` fixture stubs the planner for all `test_graph.py` tests, keeping
  them offline; `test_planner.py` manages its own patching and is excluded by filename check.
- Weak-areas captured via closure on `RunnableLambda` — post-prompt messages, not raw dict.
- `plan` stored as `model_dump()` dict in state (not a Pydantic instance) — JSON-serialisable
  and safe to pass through LangGraph state.

---

### Phase 1 — 2026-06-13
**Built:** `loop/state.py` (LoopState + initial_state); `loop/graph.py` (StateGraph: intake →
END, build_graph, compile_graph, main runner); `tests/test_graph.py` (9 tests — structure,
intake node, state shape, reducer behaviour). 20/20 tests passing.

**Key decisions:**
- LoopState extends `dict` (not TypedDict) for Python 3.14 + LangGraph compatibility.
- `messages` uses `Annotated[list[BaseMessage], add_messages]` reducer — append, not overwrite.
- All phase 2–5 fields declared as `Optional` now so state schema is stable across phases.
- `build_graph()` / `compile_graph()` split: tests call `compile_graph()` fresh each time;
  module-level `compiled` singleton used by the runner.

---

### Phase 0 — 2026-06-13
**Built:** uv project (Python 3.14, hatchling flat layout); `pyproject.toml` with all runtime
+ dev deps; `loop/config.py` (pydantic-settings singleton); `loop/models.py` (factory →
ChatBedrockConverse, Ollama seam); `loop/observability.py` (Langfuse v4 CallbackHandler,
no-op when unconfigured); `loop/hello.py` (live smoke-run script); fixtures (`sample_jd.md`,
`sample_profile.md`, `questions.json`, `rubrics.json`); `tests/test_smoke.py` (11 offline tests).

**Key decisions:**
- Switched build backend from uv_build (expects src/) to hatchling (flat loop/ layout).
- Langfuse v4 uses `CallbackHandler(public_key=...)` — different from v2/v3 (secret_key+host).
- `ChatBedrockConverse` takes `bedrock_api_key` for the bearer token (verified via model_fields).
- Monkeypatch ordering: reload config → reload models → THEN patch, so patch survives the reload.
- Two-env split: laptop = ruff + offline pytest gate; live Bedrock call via `loop/hello.py`.

**Installed versions:** langgraph 1.2.5, langchain-aws 1.5.1, langfuse 4.7.1, pydantic-settings 2.14.1.
