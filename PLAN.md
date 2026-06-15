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
| 5 | HITL | interrupts & resume | ⬜ not started | — |
| 6 | Eval & observability | agent evaluation | ⬜ not started | — |

Status legend: ⬜ not started · 🟡 in progress · ✅ done & approved · ⏸️ blocked

**Current phase:** Phase 5 — awaiting owner approval of Phase 4.

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

## v2 / future enhancements (NOT in v1)

- Real data sources (live JD ingest, real question bank) replacing fixtures.
- Postgres checkpointer + store (swap the Phase 4 seam).
- Ollama / open-source model via the Phase 0 model factory seam.
- Richer, larger, semantically-indexed question bank.

---

## Changelog

> Append one entry per completed phase: date, phase, what was built, key decisions, what the
> owner learned. Keep newest at top.

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
