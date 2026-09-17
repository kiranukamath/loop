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
| 8 | Retrieval / RAG | semantic search & grounding | ✅ done & approved (8a, 8b, 8c) | — |
| 9 | Tool-use research agent | dynamic tool-calling (ReAct) | ✅ done & approved (9a, 9b) | — |
| 10 | Production hardening | resilience, safety, cost | ✅ done & approved (10a, 10b, 10c) | — |
| 11 | Session history UI | reading checkpoint state / replay | ✅ done & approved (11a, 11b) | — |
| — | *— v2 "frontier track" begins below (Phases 12+) —* | | | |
| 12 | MCP & interoperability | interop / Model Context Protocol | ✅ done & approved (12a, 12b) | — |
| 13 | Multi-agent orchestration | supervisor · parallel (Send) · handoffs | ✅ done & approved (13a, 13b, 13c) | — |
| 14 | Advanced RAG (Track C) | reranking · CRAG · query rewriting | ✅ done & approved (14a, 14b, 14c) | — |
| 15 | Self-improvement (Track E) | Reflexion · DSPy · replanning | ✅ done & approved (15a, 15b, 15c) | — |
| 16 | Advanced memory (Track D) | reflection · episodic/semantic/procedural | ⬜ not started (v2) — spec'd | — |
| 17 | Eval-in-CI (Track H) | regression gate · agent simulation · red-team | ⬜ not started (v2) — spec'd | — |
| 18 | Production infra & real data (Track G) | pgvector · Postgres · Ollama · real search · real data | ⬜ not started (v2) — spec'd | — |
| 19 | Voice / multimodal (Track F) | STT+TTS · diagram grading · code sandbox | ⬜ backlog (v2, optional) | — |

Status legend: ⬜ not started · 🟡 in progress · ✅ done & approved · ⏸️ blocked
**"spec'd"** = full phase section written below; **"backlog — not spec'd"** = lives only in the
[v2 roadmap](#v2-roadmap--the-frontier-track-phases-12) table, detailed when we reach it.

**v2 roadmap locked in (owner, 2026-07-18):** v1 (0–7) + production track (8–11) are complete.
The **v2 "frontier track"** starts at Phase 12 with **MCP & interoperability** (owner-selected
first). The full brainstorm of candidate v2 tracks is captured in the
[v2 roadmap](#v2-roadmap--the-frontier-track-phases-12) section below; only Phase 12 is spec'd
in detail so far.

**Phase 7 complete ✅ (158 tests, 0 lint errors). All sub-steps 7a–7e done.**
Loop is fully usable: `uv run uvicorn loop.api:app --port 8000 --reload` → open http://localhost:8000

**Phase 8a complete ✅ (173 tests, 0 lint errors).**
Embeddings seam (`loop/embeddings.py`) + InMemoryVectorStore retrieval module (`loop/retrieval.py`) + expanded question bank (24 questions, 24 rubrics) + 15 offline retrieval tests.

**Phase 8b complete ✅ (176 tests, 0 lint errors).**
`search_questions()` added to `loop/tools.py`; interviewers now pick questions by semantic
similarity to `session.focus`/`topics` (via `current_focus`/`current_topics` in state, set by
`session_router`) instead of fixture order. Global `stub_embeddings` autouse fixture added to
`tests/conftest.py` so every test — not just `test_retrieval.py` — stays offline.

**Phase 9a complete ✅ (189 tests, 0 lint errors).**
Search seam (`loop/research/search.py`, `ddgs` keyless provider, Tavily v2 seam) + `@tool`
wrapper (`loop/research/tools.py::search_web`) + `CompanyResearch` schema + `company`/
`company_research` state fields + `fixtures/sample_company.txt`. **Deviation from the original
plan text:** using `langchain.agents.create_agent` in 9b instead of the PLAN-specified
`langgraph.prebuilt.create_react_agent`, which is now deprecated in the installed
`langgraph-prebuilt==1.1.0` — verified per CLAUDE.md rule #7.

**Phase 9b complete ✅ (202 tests, 0 lint errors). Phase 9 is fully done.**
`loop/nodes/research.py` — the ReAct research node, wired conditionally after `intake` via
`_route_after_intake` (research only runs if `state["company"]` is set; otherwise routes straight
to `planner`, exactly like every pre-Phase-9 flow). `intake()` deliberately does NOT set `company`
by default — the demo runner (`main()`) opts in explicitly from `fixtures/sample_company.txt` —
so no existing test or flow was affected by adding the new node. `planner.py` now folds
`company_research` into its prompt via `_format_company_research()`, with a clear placeholder when
absent.

**Phase 10 complete ✅ (237 tests, 0 lint errors). Phase 10 is fully done.**
`loop/models.py::with_resilience()` wraps the *already-built* structured-output
chain (`prompt | model.with_structured_output(Schema)`) with `.with_fallbacks()`
then `.with_retry()` — deviation from the PLAN's literal "wrap the returned
model" wording: verified on installed `langchain-core==1.4.7` that
`model.with_retry()` returns a `RunnableRetry` which does NOT proxy
`.with_structured_output()` (checked via `hasattr`), so retry/fallback must
wrap the *chain*, not the bare model, or every structured-output node breaks.
`loop/guardrails.py` (`redact_pii`, `detect_injection`) wired into `intake()`
(JD/profile) and the interviewer answer gate; flags surface via
`state["flagged_inputs"]` (new `_append_list`-reduced field) and the SSE
payload. `loop/budget.py` (`SessionBudget`, `BudgetCallbackHandler`) hooks in
as a LangChain callback at the top-level `graph.stream()` call in `api.py`
(NOT inside any node) — `on_llm_end` sees every model call's `usage_metadata`
regardless of which node made it, so no grader/planner/coach/readiness code
needed to change for cost tracking. `BudgetExceeded` is caught in `api.py`'s
SSE generator and surfaces as a `{"type": "error"}` event instead of a raw 500.

**Phase 11 complete ✅ (242 tests, 0 lint errors). Phase 11 is fully done. All 12 phases (0–11) are now complete.**
`GET /sessions` queries the SqliteSaver's `checkpoints` table directly for
`(thread_id, MAX(checkpoint_id))` — `checkpoint_id` is a time-sortable UUID6,
so `MAX()` gives the latest checkpoint per thread without a separate
timestamp column (verified by inspecting a live SqliteSaver's schema: no `ts`
column exists on `checkpoints`, but `StateSnapshot.created_at` does, read via
`graph.get_state(config)` per thread for the actual ISO timestamp shown in
the UI). Returns `{"sessions": [], "persistence": "none"}` on MemorySaver
(matches the dev-box default when `DB_PATH` is unset) — no durable history
exists to list. `GET /sessions/{id}/history` reads only the newest
`get_state_history()` snapshot (LangGraph merges every node's delta into the
full state, so the final snapshot already has everything). Simplification
vs. the original PLAN sample: `sessions[]` omits a per-session
`weak_areas_after` breakdown (would require replaying every snapshot, not
just the final one) — the detail response instead exposes one top-level
`weak_areas` (the final accumulated list). `loop/static/sessions.html` is a
vanilla-JS list/detail page (same dark Tailwind theme as `index.html`);
manually verified in-browser against the real `db/loop_state.sqlite` file
from an earlier phase's demo run — list view, detail view (plan card, graded
Q&A card with score bar, gaps), and back-navigation all render correctly.

**Phase 13 complete ✅ (272 tests, 0 lint errors). Phase 13 is fully done (13a, 13b, 13c).**
`loop/nodes/panel.py` — `grade_dispatch`/`panel_grader`/`grade_aggregator`, a parallel
"panel of graders" via the Send API, gated by `settings.panel_grading` (default `False`).
`loop/nodes/supervisor.py` — `interview_supervisor`, a Command-handoff node replacing
`session_router`/`_route_by_modality`/`_route_after_session` as a unit, gated by
`settings.orchestration_mode == "supervisor"` (default `"fixed"`). `loop/graph.py`'s
`build_graph()` is now parametrized (`orchestration_mode`, `panel_grading`) and branches
only the two regions Phase 13 touches — `compile_graph()` hard-pins both flags to their
Phase-12 defaults so the existing 254 tests are pinned byte-for-byte; `compile_graph_with_
memory()` reads `settings`, so the live server opts in via `.env`, no code change.
`loop/state.py` gained `panel_grades` (new `_reset_or_append` reducer — `None` means
"clear", not "append nothing", since a plain `{"x": []}` return can't clear an append-only
channel) and `supervisor_decisions` (traceability). `loop/schemas.py` gained
`PersonaGrade`/`SupervisorDecision`. `tests/test_multiagent.py` — 18 new offline tests,
mutation-tested (deliberately broke the supervisor's bound check and confirmed the right
tests failed) before being called done. Docs: `docs/16-19` (fixed-workflow-vs-multiagent,
the Send API, Command handoffs + supervisor pattern, reducers + bounded agency) +
`doc/phase-13-multiagent-orchestration.md` (the build walkthrough).

**Phase 12b complete ✅ (254 tests, 0 lint errors). Phase 12 is fully done.**
`loop/research/mcp_client.py::load_mcp_tools()` builds a
`MultiServerMCPClient` from `settings.mcp_server_configs` (new, empty-dict
default — an explicit allow-list) and returns its tools as `BaseTool`s;
empty config (the default) short-circuits to `[]` before importing
`langchain_mcp_adapters` or spawning anything. `loop/nodes/research.py`
merges them into the tool list: `tools=[search_web, *load_mcp_tools()]`.
**Real bug found and fixed, not just a design choice made:** the PLAN
framed the sync/async crossing as two equally-valid options; empirically
testing both showed `agent.invoke()` (the original Phase 9 call) raises
`NotImplementedError: StructuredTool does not support sync invocation` the
moment an MCP-loaded (async-only) tool is actually called, while making
`research()` `async def` raises `TypeError: No synchronous function
provided` the moment the (sync-invoked) parent graph reaches it. Fixed by
keeping `research()` sync but swapping its internal call to
`asyncio.run(agent.ainvoke(...))` — verified this doesn't change Phase 9's
existing behavior (a pure-sync tool list runs identically through
`ainvoke()`). `tests/fixtures/mock_mcp_server.py` (a second tiny `FastMCP`
server, one canned tool) is spawned as a **real subprocess** by the real
`MultiServerMCPClient` in `tests/test_mcp.py` — a genuine MCP-over-stdio
round trip, still offline (stdio is a local subprocess, not a network
call). Docs: `docs/11-mcp-transports-and-adapters.md` rewritten with the
empirical sync/async trace (not the hypothetical two-options framing);
`doc/phase-12-mcp-interop.md` extended with the full 12b walkthrough.

**Phase 12a complete ✅ (250 tests, 0 lint errors).**
`loop/mcp_server.py` — a `FastMCP("loop")` server exposing four thin
`@mcp.tool()` wrappers (`list_questions`, `search_questions`, `get_rubric`,
`get_reference_answer`), each a one-line delegation to an already-tested
`loop/tools.py` function — no logic duplicated. New deps `mcp==1.28.1` +
`langchain-mcp-adapters==0.3.0` (verified against installed versions per
CLAUDE.md rule #7: `FastMCP.tool()`/`.run()` signatures inspected directly,
not trusted from memory; found that `list_tools()`/`call_tool()` are `async`
even though `@mcp.tool()` registration is sync — flagged as the sync/async
question 12b must resolve for the Phase 9 research node).
`tests/test_mcp.py` (8 tests, fully offline — stdio is a local subprocess,
not a network call, so it fits the laptop gate) asserts both tool
registration (name/description/input-schema) and that calling a tool
through MCP returns identical data to calling the underlying `tools.py`
function directly. Manually verified `uv run python -m loop.mcp_server`
starts cleanly under stdio. Docs: `docs/09–11-mcp-*.md` (protocol concepts:
JDBC/LSP analogy, host/client/server roles, tools/resources/prompts,
transports, the `langchain-mcp-adapters` bridge, bounded agency for external
tools) + `doc/phase-12-mcp-interop.md` (the 12a build walkthrough).

**Phase 8c complete ✅ (178 tests, 0 lint errors). Phase 8 is fully done.**
`fixtures/reference_answers.json` (24 short model answers, one per question) + `get_reference_answer()`
in `loop/tools.py` (exact-match lookup by question_id — grading always knows the exact question,
so no embedding search needed here) + grader prompt updated to inject the reference answer as
grounding evidence. Falls back to a placeholder when no reference exists, so grading never breaks.

**Phases 8–10 — the "resume / production track" (planned 2026-06-28, not started).**
v1 (Phases 0–7) covered the five core agentic capabilities. Phases 8–10 extend Loop into the
skills that turn a LangGraph demo into production-grade agentic AI: **Retrieval (RAG)**,
**dynamic tool-calling (ReAct)**, and **production hardening (resilience / safety / cost)**.
Same non-negotiable contract: one phase per turn, teach before coding, pause for approval.
Same laptop test gate: every new capability must be testable offline (fake embeddings,
stubbed search, simulated errors — no network in the test suite).

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

## Phase 8 — Retrieval / RAG  *(Capability: RETRIEVAL — semantic search & grounding)*

**Goal:** replace exact-match fixture lookups with **semantic retrieval**. Build an embeddings
seam + a vector store, ingest an expanded question bank, and have the interviewer pick the best
question for a planned topic by *meaning*, not by array index. Optionally ground the grader in a
retrieved reference answer so grading is evidence-based, not vibes.

**Why it exists / what it teaches:** RAG is the single most-requested agentic skill in job
postings. The lesson is the **retrieve → augment → generate** loop and *why semantic search
beats keyword/exact match*. It fits the existing v2-stable tool seam perfectly: we swap the
body of `tools.py`, and the graph never changes.

**Owner decisions / stack (APIs verified present 2026-06-28 against installed versions):**
- **Vector store:** `langchain_core.vectorstores.InMemoryVectorStore` — zero extra deps, rebuilt
  at process startup from fixtures. *v2 seam:* swap to **pgvector** (a `PGVector` vector store)
  without touching callers.
- **Embeddings (real / server):** `langchain_aws.BedrockEmbeddings` (Amazon Titan), behind a
  factory `loop/embeddings.py` that mirrors `loop/models.py` (same provider-switch seam).
- **Embeddings (tests / laptop):** `langchain_core.embeddings.fake.DeterministicFakeEmbedding`
  — deterministic vectors, no network. This is how the laptop test gate stays offline.
- **No new heavyweight deps** (no FAISS/Chroma) — InMemoryVectorStore keeps it laptop-friendly.

**Concepts to teach:**
- **Embedding** = turning text into a vector of numbers so "close meaning" = "close vector"
  (analogy: a hash that preserves *similarity* instead of destroying it). **Cosine similarity**
  as the distance measure.
- **Vector store** = an index of (vector, text, metadata) you can query by nearest-neighbour
  (analogy: a Lucene/Elasticsearch index, but keyed on meaning vectors not tokens).
- **Top-k retrieval** and **metadata filtering** (filter by modality, then rank by similarity).
- **Chunking** (mention only — our question docs are short, so 1 doc = 1 chunk; explain why
  long docs must be split).
- **The RAG triad: retrieve → augment (stuff into the prompt) → generate.** Why this beats
  fine-tuning for a changing knowledge base.
- **Embeddings factory as a seam** — same lesson as the model factory: hide the provider so
  Bedrock-now / pgvector-later is a one-file change.

**Sub-steps (each = one turn: teach → code → test → pause):**

- **8a — Embeddings seam + retrieval module + expanded question bank.**
  - `loop/embeddings.py`: `get_embeddings()` → `BedrockEmbeddings`, with the same
    `model_provider` switch shape as `models.py`. (Add `bedrock_embed_model_id` to config,
    default a Titan embeddings model id; verify the exact id is enabled in the account/region.)
  - `loop/retrieval.py`: build an `InMemoryVectorStore` from `fixtures/questions.json` (embed
    `title + prompt + topic` as the document text; keep `id`, `modality`, `topic`, `difficulty`
    as metadata). Expose `retrieve_questions(query: str, modality: str | None = None, k: int = 3)
    -> list[dict]` returning the original question dicts ranked by similarity. Build the index
    once as a module-level singleton (rebuilt at import).
  - Expand `fixtures/questions.json` to ~24 questions (≈8 per modality, varied topics +
    difficulties) so retrieval has real signal to rank.
  - `tests/test_retrieval.py` — **offline**, monkeypatching `get_embeddings` to return
    `DeterministicFakeEmbedding(size=...)`: index builds; `retrieve_questions` returns ≤k items;
    modality filter excludes other modalities; a query semantically near a known question ranks
    it highly.
- **8b — Wire retrieval into the interview flow.**
  - Add `search_questions(query, modality, k)` to `loop/tools.py` (v2-stable signature) that
    delegates to `loop/retrieval.py`; keep the old fixture functions for back-compat.
  - In the interviewer / `session_router`, pick the question by semantic match to the session's
    `focus` + `topics` (e.g. `search_questions(focus, modality, k=1)[0]`) instead of
    `get_questions_by_modality(modality)[0]`.
  - Tests: interviewer selects a topic-relevant question; deterministic with fake embeddings.
- **8c — (optional, stretch) RAG-grounded grading.**
  - Add short reference / model-answer snippets to fixtures (e.g. `reference_answers.json`).
  - In the grader, retrieve the reference for the question being graded and inject it into the
    grader prompt so the model grades against ground truth, not its own guess.
  - Tests: grader prompt includes the retrieved reference; grade still schema-valid.

**Task checklist:**
- [ ] `loop/config.py` — add `bedrock_embed_model_id` (+ keep the `model_provider` switch).
- [ ] `loop/embeddings.py` — `get_embeddings()` factory (Bedrock now, provider seam for v2).
- [ ] `loop/retrieval.py` — InMemoryVectorStore index + `retrieve_questions(...)`.
- [ ] `fixtures/questions.json` — expand to ~24 questions across modalities.
- [ ] `loop/tools.py` — add `search_questions(...)` (stable signature) over retrieval.
- [ ] Wire semantic selection into interviewer / session_router.
- [ ] (optional) reference answers + RAG-grounded grader.
- [ ] `tests/test_retrieval.py` + interviewer-selection tests, all offline (fake embeddings).

**Files touched:** `loop/{config,embeddings,retrieval,tools}.py`, `loop/nodes/interviewers.py`
(and `grader.py` if 8c), `fixtures/questions.json` (+ optional `reference_answers.json`),
`tests/test_retrieval.py` (+ updates to interviewer tests), `PLAN.md`.

**Done when:** the interviewer picks questions by semantic similarity to the planned topic
(provably different from index-0 lookup); retrieval is laptop-testable with fake embeddings;
lint + pytest pass offline; (server) a live Bedrock-embeddings run is sanity-checked.

**Skills needed:** embeddings + vector stores (`InMemoryVectorStore`, `BedrockEmbeddings`,
`DeterministicFakeEmbedding` — verify on installed versions); cosine similarity intuition;
reusing the factory-seam pattern from `models.py`.

---

## Phase 9 — Tool-calling research agent  *(Capability: TOOL USE / dynamic agency)*

**Goal:** add Loop's **first *dynamic* agent** — a ReAct research sub-agent that, given a company
name, *decides on its own* which tools to call (web search, page fetch) to produce a structured
`CompanyResearch` brief that grounds the planner. The existing graph is a *fixed workflow*; this
node hands control flow to the LLM.

**Why it exists / what it teaches:** shows you understand the difference between a hard-coded
workflow (every node + edge fixed in advance) and an actual **agent** (the model chooses which
tool to call, and when, in a loop). Function calling + ReAct + bounded agency are core
interview/resume talking points.

**Owner decisions / stack (APIs verified present 2026-06-28):**
- **Agent:** `langgraph.prebuilt.create_react_agent` — a pre-built ReAct graph. Embed it as a
  **single node** inside the bigger Loop graph (a sub-graph used as a node). Verify its current
  signature + structured-output option (`response_format=`) on the installed version before use.
- **Tools:** `@tool`-decorated functions (`langchain_core.tools.tool`).
- **Search seam:** `loop/research/search.py` exposes `web_search(query, k)` behind a **swappable
  provider** (same philosophy as the model factory):
  - v1 default: a **keyless** provider (e.g. DuckDuckGo via a small lib) **or** Tavily when
    `TAVILY_API_KEY` is set — chosen by config. Pick one to implement; wire the seam for both.
  - **Offline tests stub the search tool** to return canned fixture results → laptop gate stays
    offline. Real search is a *server* activity, like live Bedrock calls.
- **New deps (verify + add at build time):** the chosen search client (e.g. `ddgs`/
  `duckduckgo-search`, or `langchain-tavily`). Add to `pyproject.toml` in 9a.

**Concepts to teach:**
- **Tool / function calling:** the model emits a structured *tool call* → the runtime executes
  the Python function → the result is fed back to the model → repeat (analogy: the model is a
  caller that can invoke your service methods and read the responses).
- **The ReAct loop:** *Reason → Act (call a tool) → Observe (read result) → repeat → Answer.*
- **Agent-as-a-node:** a whole sub-agent compiled and dropped into the parent graph as one step.
- **Bounded agency (safety):** recursion/iteration limit, a **tool allow-list**, and a timeout —
  so a dynamic agent can't loop forever or call something it shouldn't.
- **Why dynamic ≠ fixed workflow:** when to reach for an agent vs. a deterministic graph.

**Sub-steps (each = one turn: teach → code → test → pause):**

- **9a — Search seam + tools + schema + state.**
  - `loop/research/__init__.py`, `loop/research/search.py`: `web_search(query, k)` with a
    provider switch (keyless default; clear error if the provider/key is missing).
  - `loop/research/tools.py`: `@tool` wrappers (e.g. `search_web`, optionally `fetch_url`).
  - `loop/schemas.py`: add `CompanyResearch` (`company`, `interview_format`, `focus_areas`,
    `tech_stack`, `recent_news`, `sources`).
  - `loop/state.py`: add `company: str` and `company_research: dict | None`; update
    `initial_state()`; add a sample company to fixtures/profile.
  - Tests: search seam returns the documented shape (stubbed); tools are callable in isolation.
- **9b — ReAct research node wired before the planner.**
  - `loop/nodes/research.py`: build `create_react_agent(model, tools)`, run it with a research
    prompt for `state["company"]`, and coerce the final output into `CompanyResearch` (via the
    agent's structured-output option, or a follow-up `with_structured_output` pass). Enforce a
    bounded iteration limit.
  - Wire `intake → research → planner`, but **conditionally**: run research only when
    `state["company"]` is set, else skip straight to planner (a conditional edge — the routing
    lesson from Phase 3 reused).
  - `loop/nodes/planner.py`: consume `state["company_research"]` — add a `{company_research}`
    section to the planner prompt so the plan is grounded in real company signal.
  - Tests (offline, stub model + tools): research populates `company_research`; planner prompt
    includes it; the no-company path skips research; the iteration bound holds.

**Task checklist:**
- [ ] `pyproject.toml` — add the chosen search client; `uv sync`.
- [ ] `loop/research/{__init__,search,tools}.py` — seam + `@tool`s.
- [ ] `loop/schemas.py` — `CompanyResearch`.
- [ ] `loop/state.py` — `company` + `company_research`; update `initial_state()` + fixtures.
- [ ] `loop/nodes/research.py` — `create_react_agent` node, bounded.
- [ ] `loop/graph.py` — conditional `intake → research → planner` wiring.
- [ ] `loop/nodes/planner.py` — consume company research in the prompt.
- [ ] `tests/test_research.py` — offline (stubbed model + tools), incl. the skip path + bound.

**Files touched:** `loop/research/*` (new), `loop/nodes/{research,planner}.py`,
`loop/{schemas,state,graph}.py`, `pyproject.toml`, fixtures, `tests/test_research.py`, `PLAN.md`.

**Done when:** with a company set, the graph runs a ReAct research step that calls tools and
produces a schema-valid `CompanyResearch` that demonstrably shapes the plan; with no company it
skips cleanly; the agent is iteration-bounded; tests pass offline (stubbed tools); (server) one
live web-search run is sanity-checked.

**Skills needed:** `create_react_agent` (verify signature + `response_format`), `@tool`,
tool-calling/ReAct mental model, sub-graph-as-node, conditional edges (Phase 3), a search client.

---

## Phase 10 — Production hardening  *(Capability: PRODUCTION — resilience, safety, cost)*

**Goal:** make Loop safe and operable: **model-level resilience** (retry + fallback), **input/
output guardrails** (PII redaction + prompt-injection detection), and **token/cost budgeting**
with a per-session ceiling surfaced in the UI. This is the *backend-engineer-applied-to-AI*
phase — the highest-paying intersection of your existing skills with agentic AI.

**Why it exists / what it teaches:** the production concerns most ML demos skip. "I put guardrails
+ cost budgets + retries/fallbacks around an LLM app" is a senior signal almost nobody shows.
Plays directly to your backend/AWS strengths.

**Concepts to teach:**
- **Resilience:** LangChain Runnables compose `.with_retry(...)` (backoff on transient errors)
  and `.with_fallbacks([...])` (switch to a cheaper/other model when the primary fails) — the
  Resilience4j/Hystrix pattern, but for model calls.
- **Guardrails:** input sanitisation (**PII redaction**, deterministic regex) and **prompt-
  injection detection** (the JD/answer text is *untrusted user input* — it must not be able to
  hijack the system prompt). The analogy: validating/escaping request bodies before they hit
  your service.
- **Cost & token budgeting:** every model response carries a **usage** block (input/output
  tokens). Accumulate it, price it, and **enforce a per-session ceiling** so a runaway loop
  can't burn the budget — like a rate-limiter / quota on an expensive downstream call.

**Sub-steps (each = one turn: teach → code → test → pause):**

- **10a — Resilience in the model seam.**
  - `loop/models.py`: wrap the returned model with `.with_retry(...)` and
    `.with_fallbacks([fallback_model])`. Config: `retry_max_attempts`, `fallback_model_id`
    (empty = no fallback). Verify the exact Runnable retry/fallback API on the installed
    `langchain-core` before wiring.
  - Tests (offline): a stub that raises a transient error once then succeeds proves retry fires;
    a stub whose primary always fails proves the fallback is used.
- **10b — Guardrails.**
  - `loop/guardrails.py`: `redact_pii(text) -> str` (regex: email, phone, long digit runs) and
    `detect_injection(text) -> bool` (heuristic patterns like "ignore previous instructions",
    role-override attempts; leave a seam for an optional LLM-based check).
  - Wire: sanitise JD + profile at **intake**; check/redact candidate **answers** at the answer
    gate before they're stored or sent to the grader.
  - Surface a "⚠ flagged input" signal in the API payload + UI when injection is detected.
  - Tests (deterministic, offline): known PII is redacted; known injection strings are flagged;
    clean text passes through unchanged.
- **10c — Cost & token budget.**
  - `loop/budget.py`: read the usage block from model response metadata (Bedrock Converse
    `usage`), accumulate `tokens_in/out` + `cost_usd` into state, and enforce
    `max_session_tokens`. On breach, stop the session gracefully with a clear message instead
    of erroring.
  - `loop/config.py`: a small price table (per-1k input/output by model id) + `max_session_tokens`.
  - Surface `tokens_used` + `cost_usd` in the SSE payload (`_safe_payload` in `api.py`) and show
    a live counter in `loop/static/index.html`.
  - Tests (offline): the accountant sums usage correctly; the ceiling trips at the threshold;
    the cost math matches the price table.

**Task checklist:**
- [ ] `loop/models.py` — `.with_retry()` + `.with_fallbacks()`; config knobs.
- [ ] `loop/guardrails.py` — `redact_pii` + `detect_injection`; wire intake + answer gate.
- [ ] `loop/budget.py` — usage accountant + per-session ceiling.
- [ ] `loop/config.py` — `retry_max_attempts`, `fallback_model_id`, price table,
      `max_session_tokens`.
- [ ] `loop/api.py` + `loop/static/index.html` — surface flagged-input + tokens/cost.
- [ ] `tests/test_guardrails.py`, `tests/test_budget.py`, resilience tests in `test_models.py`.

**Files touched:** `loop/{models,guardrails,budget,config,api}.py`,
`loop/nodes/{interviewers}.py` (answer-gate redaction), `loop/static/index.html`,
`tests/test_{guardrails,budget,models}.py`, `PLAN.md`.

**Done when:** model calls retry on transient failure and fall back to a secondary model; pasted
JDs/answers are PII-redacted and injection-flagged; a session enforces a token/cost ceiling with
the live counter visible in the browser; all new guards are deterministic and tested offline;
lint + pytest pass.

**Skills needed:** Runnable `.with_retry`/`.with_fallbacks` (verify on installed `langchain-core`);
regex-based redaction; prompt-injection threat model; reading Bedrock Converse `usage` metadata;
wiring counters through SSE + UI (Phase 7).

---

## Phase 11 — Session history UI  *(Capability: checkpoint replay + observability)*

**Goal:** add a **session history page** where you can browse every past interview, see what
the model asked, what you answered, how you were scored, and what the readiness verdict was.
All data already exists in the **SQLite checkpoint** (LangGraph persisted the full state after
every node). Phase 11 surfaces it through two new API endpoints and a new page in the UI.

**Why it exists / what it teaches:**
LangGraph's `get_state_history()` is one of its most powerful production features — you can
replay or inspect any prior run without storing anything extra. This phase teaches you how to
*read back* from the checkpoint, which is the same mechanism that powers time-travel debugging
and audit logs in production agentic systems. It also reinforces the checkpoint mental model
from Phase 4 and how state accumulates across nodes.

**Complementary to Langfuse:** Langfuse already shows raw LLM prompts + completions (already
working for you). Phase 11 shows the *structured interview data* (plan, Q&A pairs, grade
breakdowns, readiness verdict) in your own UI — a different view of the same session, not a
duplicate of Langfuse.

**Owner decisions / design choices:**
- **Data source:** SQLite checkpoint via `graph.get_state_history(config)`. The final snapshot
  (newest in the history) holds the fully-accumulated state: plan, all answers, all grades,
  weak areas, readiness verdict. We read only the final snapshot — no need to replay every
  intermediate node snapshot.
- **Session listing:** LangGraph has no built-in "list all threads" API. We query the SQLite
  `checkpoints` table directly (`SELECT DISTINCT thread_id, MAX(ts) ...`) to get thread IDs
  and timestamps. The compiled graph's checkpointer connection is reused — no second connection.
- **Langfuse trace link (optional):** if Langfuse is configured, each session card can show
  a "View in Langfuse" link constructed from the `thread_id`. This is cosmetic — if Langfuse
  is not configured the link is simply omitted.
- **No new deps** — everything uses what is already installed.

**Concepts to teach (before coding):**
- **`get_state_history(config)`** — returns a generator of `StateSnapshot` objects, newest-first.
  Each snapshot has `.values` (the full state at that moment), `.next` (which node ran next),
  and `.metadata` (step number, timestamp). Analogy: git log — each commit is a snapshot of
  the full tree at that point. The most recent commit (first in the list) is HEAD — the final
  state of the session.
- **Why the final snapshot has everything:** each node returns only a *delta*, but the
  checkpoint stores the *merged* state after every node runs. So the last snapshot has the
  fully accumulated `answers`, `grades`, `weak_areas`, and `readiness_verdict` from every
  session that ran. No need to replay all snapshots to reconstruct the timeline.
- **Querying SQLite directly for thread listing:** `SqliteSaver` writes to a `checkpoints`
  table. We `SELECT DISTINCT thread_id, MAX(ts)` to get all known sessions and their last
  activity time. This is safe because we own both the writer (LangGraph) and the reader (us).
- **Serving multiple HTML pages from FastAPI:** `StaticFiles` already mounts `loop/static/`.
  A second `.html` file in that directory is automatically served — no extra route needed.

**Sub-steps (each = one turn: teach → code → test → pause):**

- **11a — Two new API endpoints.**
  - `GET /sessions` — list all past sessions. Query the SQLite `checkpoints` table for
    `(thread_id, MAX(ts))`, join with the final state to pull `readiness_verdict.verdict`
    and `plan.total_sessions`. Return a JSON list sorted newest-first:
    ```json
    [
      {
        "thread_id": "abc-123",
        "started_at": "2026-06-28T10:42:00Z",
        "verdict": "ready",
        "sessions_completed": 2
      },
      ...
    ]
    ```
    If `db_path` is empty (MemorySaver), return `[]` with a `"persistence": "none"` flag —
    MemorySaver holds no durable history.
  - `GET /sessions/{thread_id}/history` — return the full structured interview timeline for
    one session. Call `_graph.get_state_history(config)`, take the *first* snapshot (newest =
    final), and shape the response:
    ```json
    {
      "thread_id": "abc-123",
      "started_at": "...",
      "plan": { "role_summary": "...", "total_sessions": 2, "key_gaps": [...] },
      "sessions": [
        {
          "session_number": 1,
          "modality": "system_design",
          "question": { "id": "sys-001", "title": "...", "prompt": "..." },
          "answer": "Token bucket in Redis...",
          "grade": { "score": 8, "strengths": [...], "improvements": [...], "overall_feedback": "..." },
          "weak_areas_after": ["distributed rate limiting"]
        },
        ...
      ],
      "readiness_verdict": { "verdict": "ready", "confidence": 0.82, "gaps": [...], "recommendation": "..." }
    }
    ```
    Build `sessions[]` by zipping `state["answers"]` with `state["grades"]` on `question_id`.
    Look up the question title/prompt from `tools.get_question_by_id()` (already exists).
  - Tests in `tests/test_api.py` (add to existing file):
    - `GET /sessions` returns a list (empty list when MemorySaver; non-empty after a stubbed
      session run via TestClient).
    - `GET /sessions/{id}/history` returns the correct shape with plan + sessions + verdict.
    - `GET /sessions/nonexistent/history` returns 404.
    - All offline (stubbed model, MemorySaver — same pattern as the 20 existing API tests).

- **11b — Sessions history page.**
  - `loop/static/sessions.html` — new page, same dark Tailwind theme as `index.html`.
  - **Session list view:** a table / card list of past sessions. Each row shows:
    - Date + time
    - Verdict badge (green "ready" / amber "not ready" / grey "incomplete")
    - Number of sessions completed
    - Key gaps (first 2, truncated)
    - "View details" button
  - **Session detail view (same page, JS toggle):** when "View details" is clicked, fetch
    `GET /sessions/{id}/history` and render:
    - **Plan card:** role summary, key gaps, total sessions planned.
    - **Q&A cards (one per answered session):** modality badge, question text (collapsible),
      your answer (collapsible), score bar (e.g. `████████░░ 8/10`), strengths list,
      improvements list, overall feedback paragraph.
    - **Readiness verdict card:** verdict badge, confidence percentage, gaps list,
      recommendation paragraph.
    - Optional: "View in Langfuse →" link if `LANGFUSE_HOST` is configured (link to
      `{langfuse_host}/traces?search={thread_id}`).
  - A "← Back to sessions" link returns to the list view without a page reload.
  - Add a "📋 History" link in `index.html`'s header that opens `/static/sessions.html`.
  - No build step — vanilla JS + Tailwind CDN, same as `index.html`.

**Task checklist:**
- [ ] `loop/api.py` — `GET /sessions` endpoint (SQLite query or MemorySaver fallback).
- [ ] `loop/api.py` — `GET /sessions/{thread_id}/history` endpoint (get_state_history + shaping).
- [ ] `tests/test_api.py` — 4+ new tests for both endpoints (offline, MemorySaver-backed).
- [ ] `loop/static/sessions.html` — session list + detail view, same Tailwind dark theme.
- [ ] `loop/static/index.html` — add "History" nav link in the header.
- [ ] Verify: lint + pytest pass; manual smoke with a real session in the browser.

**Files touched:** `loop/api.py`, `loop/static/sessions.html` (new),
`loop/static/index.html` (header link), `tests/test_api.py`, `PLAN.md`.

**Done when:** opening `http://localhost:8000/static/sessions.html` shows a list of past
sessions pulled from SQLite; clicking any session shows the full Q&A timeline with scores,
feedback, and the readiness verdict; a session with no history shows a graceful empty state;
lint + pytest pass.

**Skills needed:** `graph.get_state_history()` (verify against installed `langgraph` — check
the `StateSnapshot` field names; they can change across versions); direct SQLite query for
thread listing; shaping a nested JSON response in FastAPI; vanilla JS `fetch` + DOM rendering
(same pattern as `index.html`).

---

## v2 roadmap — the frontier track (Phases 12+)

> **Purpose of this section:** capture the full brainstorm of "what would make Loop competitive
> on the 2026 agentic-AI job market" so no idea is lost, and record the sequence. Phases **12–18
> are now spec'd in full below** (tracks B/A/C/E/D/H/G); only **Phase 19 (voice — Track F)**
> remains backlog, spec'd when reached. Same non-negotiable contract as v1: one phase per turn,
> teach → code → test → pause, laptop-offline test gate. (Specifying 14–18 ahead of build was an
> explicit owner request; each still defers exact API calls to build-time verification per rule #7.)
>
> **Framing:** v1 is a *mostly-fixed workflow* with one dynamic node (the Phase 9 ReAct
> researcher). The tracks below cross the line into what the market actually pays for: real
> multi-agent systems, MCP interop, deeper RAG/memory, self-improvement loops, and
> AWS-grade productionization (the owner's edge).

**Candidate tracks (ranked; each maps to one or more future phases):**

| Track | Theme | Headline ideas | Why it makes you competitive |
|-------|-------|----------------|------------------------------|
| **B** | **MCP & interop** *(Phase 12 — locked in)* | Expose Loop as an MCP **server**; **consume** external MCP servers; (stretch) A2A protocol | The 2025–26 standard everyone hires for; highest competitiveness-per-effort; standalone |
| **A** | **Real multi-agent** *(Phase 13 — spec'd)* | Supervisor + agent **handoffs** (`Command(goto=…)`); **parallel** fan-out/fan-in (Send API); panel-of-judges debate | The defining "you actually understand agents" signal; reworks the graph you know best |
| **C** | **Advanced RAG** *(Phase 14 — spec'd)* | Hybrid search + **reranking**; query rewriting / HyDE; **corrective-RAG (CRAG)** / Self-RAG; contextual retrieval | Most-requested RAG depth; builds on Phase 8 |
| **D** | **Advanced memory** *(Phase 16 — spec'd)* | **Reflection/consolidation** (Generative-Agents pattern); episodic/semantic/procedural typing; semantic store search; decay/conflict | Hot research problem; differentiator; builds on Phase 4 |
| **E** | **Self-improvement loops** *(Phase 15 — spec'd)* | **Reflexion** self-critique; **DSPy** prompt optimization against the eval set; replanning; adaptive difficulty (CAT) | Ties eval (Phase 6) to optimization; a rare, high-signal skill |
| **F** | **Multimodal & realtime** *(Phase 19 — backlog)* | **Voice** interviews (STT+TTS); whiteboard/diagram grading (vision); real code execution sandbox | Portfolio wow-factor; teaches realtime streaming |
| **G** | **Production & platform** *(Phase 18 — spec'd; = the old "current v2" infra backlog)* | Deploy to **AWS** (ECS/Lambda/LangGraph Platform); **Postgres + pgvector**; **prompt caching**; queueing/rate-limits; OTel + cost dashboards; auth/multi-tenancy | The owner's AWS/backend edge — highest-paying intersection |
| **H** | **Eval & safety at prod grade** *(Phase 17 — spec'd)* | **Eval-in-CI** regression gate; **agent simulation** testing (synthetic candidate); automated red-teaming / Llama Guard | Treats agent quality like a test suite — standout senior signal; backend CI/CD strength |

**Recommended sequence** (front-load universally-demanded skills, finish on the AWS edge):
**12 = B (MCP)** → 13 = A (multi-agent) → 14 = C (advanced RAG) → 15 = E (self-improvement) →
16 = D (advanced memory) → 17 = H (eval-in-CI + simulation) → 18 = G (AWS deploy + pgvector +
caching) → 19 = F (voice, optional capstone). Sequence is advisory — re-pick per phase.

> The **infrastructure-swap** items (pgvector, Postgres, Ollama, real search, real data) are
> *plumbing swaps behind existing seams*, not learning phases in themselves — so they are grouped
> into one production phase: **Phase 18 — Production infrastructure & real data** (Track G, spec'd
> below). Placed after the concept-heavy phases (owner-delegated ordering decision); pullable
> earlier if you decide to deploy.

---

## Phase 12 — MCP & interoperability  *(Capability: INTEROP — Model Context Protocol)*

**Goal:** make Loop speak **MCP in both directions** — (a) **expose** Loop's question-bank /
rubric / reference / grader logic as an **MCP server** any client (Claude Desktop, Cursor,
another agent) can call, and (b) **consume** external MCP servers' tools inside Loop's ReAct
research agent (Phase 9). Same tool *logic*, now behind the industry-standard protocol instead
of bespoke Python calls — the decoupling lesson, taken to its conclusion.

**Why it exists / what it teaches:** MCP (Anthropic, late 2024; now the de-facto standard) is
the *"USB-C for LLM tools"* — one protocol so any client can use any server's tools/resources/
prompts without custom glue. Authoring an MCP server and consuming MCP servers is now
table-stakes in agentic-AI hiring. It also proves the seam philosophy end-to-end: the *exact
same* Phase 3 `tools.py` + Phase 8 `retrieval.py` code becomes reusable by tools you don't own,
and the Phase 9 `create_agent` node gains external tools without a rewrite.

**Owner decisions / stack (⚠ VERIFY at build time per CLAUDE.md rule #7 — these packages are
NOT installed yet; MCP APIs move fast):**
- **Server SDK:** the official **`mcp`** Python SDK's **`FastMCP`** (`from mcp.server.fastmcp
  import FastMCP`) — decorator-based (`@mcp.tool()`), feels like FastAPI. *Verify at build:*
  whether the current/maintained path is the SDK's bundled FastMCP or the standalone
  **`fastmcp`** (v2) package, and the exact import + run API. Pin the verified version.
- **Client/adapter:** **`langchain-mcp-adapters`** (`MultiServerMCPClient`) — loads an MCP
  server's tools as `langchain-core` `BaseTool`s, which the existing `create_agent` research
  node **already accepts**. *Verify at build:* `get_tools()` is **async** — the Phase 9
  `research()` node is **sync**, so decide the sync/async boundary (an `asyncio.run(...)`
  wrapper vs. making the node async — check what `create_agent` + `graph.stream()` expect).
- **Transport:** **stdio** (client spawns the server as a subprocess, talks over stdin/stdout)
  for *both* the server we publish and the servers we consume — fully offline, exactly how
  Claude Desktop launches local servers. **Streamable-HTTP/SSE transport = v2 seam** (same
  "seam exists, not implemented" treatment as Tavily search and the Ollama model).
- **New deps (add + `uv sync` in 12a):** `mcp`, `langchain-mcp-adapters`. Verify + pin versions.
- **Offline test gate:** MCP over stdio is a **local subprocess, not a network call** → it fits
  the laptop gate. Tests spin up a tiny local stdio MCP server *fixture* and round-trip through
  the real client. No external server, no keys, deterministic. (Registering Loop's server in a
  real Claude Desktop is a *server/manual activity*, like live Bedrock calls.)

**Concepts to teach:**
- **What MCP is:** an open protocol standardizing how apps give LLMs tools + data. *Analogy:*
  **JDBC/ODBC** (one driver interface, many databases) or **LSP** (one protocol, every editor
  talks to every language server) — write the tool once, every MCP client can use it. Contrast
  with a LangChain `@tool`, which only LangChain can consume.
- **MCP primitives:** **tools** (model-invoked functions — our focus), **resources** (readable
  context/data addressed by URI), **prompts** (reusable prompt templates a client can surface).
  We expose tools; mention where Loop's rubrics could be *resources* and its interview prompts
  could be *prompts*.
- **Host / client / server roles:** the **host** (Claude Desktop, an IDE, your agent runtime)
  runs an MCP **client** that connects to MCP **servers**. Loop plays **both** — server (12a)
  and client (12b).
- **Transports:** **stdio** (local subprocess, zero network) vs **streamable HTTP/SSE** (remote).
  Why stdio is the natural laptop/offline choice and what changes for a remote server.
- **`langchain-mcp-adapters` as the bridge:** MCP tools → LangChain `BaseTool`, so Phase 9's
  `create_agent` needs **no rewrite** — the same decoupling lesson as the model/embeddings/
  search factories.
- **Bounded agency reprise:** consuming *external* tools re-raises the Phase 9 safety points —
  an **allow-list** of which MCP servers/tools the agent may load (config-driven), and the
  recursion bound still applies. Untrusted external tools are a prompt-injection surface
  (ties back to Phase 10 guardrails).

**Sub-steps (each = one turn: teach → code → test → pause):**

- **12a — Loop as an MCP server.**
  - `loop/mcp_server.py`: a `FastMCP("loop")` instance exposing existing fixture logic as
    `@mcp.tool()`s — e.g. `list_questions(modality)`, `search_questions(query, modality, k)`
    (delegates to `loop.tools.search_questions` / `loop.retrieval`), `get_rubric(question_id)`,
    `get_reference_answer(question_id)`. Each tool is a **thin wrapper over already-tested
    Phase 3/8 functions — NO logic duplication.** Runnable via `uv run python -m loop.mcp_server`
    (stdio entrypoint).
  - Document how to register it in an MCP client (Claude Desktop config JSON / the MCP
    Inspector) — a *server/manual activity*, not part of the laptop gate.
  - Tests (offline): import the server module, assert the tools are registered (name /
    description / input schema) and that each tool wrapper returns the same shape as the
    underlying `tools.py` function. (Full stdio round-trip is exercised in 12b.)

- **12b — Loop consumes external MCP servers (research agent).**
  - `loop/research/mcp_client.py`: `load_mcp_tools()` builds a `MultiServerMCPClient` from
    config and returns the loaded `BaseTool`s. **Empty config → returns `[]`** (feature-off
    default, like an empty `fallback_model_id`). Handle the async `get_tools()` from the sync
    research node (verified boundary from the stack decision above).
  - `loop/config.py`: add `mcp_server_configs` (default empty) — the **allow-list** of servers
    the research agent may consume: `name → {command, args, transport: "stdio"}`.
  - `loop/nodes/research.py`: append MCP-loaded tools to the existing `[search_web]` list
    before building `create_agent`. **With no MCP config, behavior is byte-for-byte Phase 9**
    (zero regressions — same "opt-in, off by default" design as `company` in 9b, so no existing
    test starts making MCP/subprocess calls).
  - Tests (offline): a tiny fixture MCP server (`tests/fixtures/mock_mcp_server.py`, a `FastMCP`
    exposing one canned tool over stdio) is launched by the **real** `MultiServerMCPClient`;
    assert (1) `load_mcp_tools()` returns the fixture's tool as a `BaseTool`, (2) invoking it
    round-trips the canned result through actual MCP-over-stdio (a subprocess — deterministic,
    offline), (3) empty config → `[]` and the research node's tool list is unchanged.

**Task checklist:**
- [ ] `pyproject.toml` — add `mcp`, `langchain-mcp-adapters`; `uv sync`. Verify + pin versions.
- [ ] `loop/mcp_server.py` — `FastMCP` server wrapping `tools.py` / `retrieval.py`; `__main__`
      stdio entrypoint.
- [ ] `loop/research/mcp_client.py` — `load_mcp_tools()` via `MultiServerMCPClient` (config-driven,
      empty → `[]`; sync/async boundary handled).
- [ ] `loop/config.py` — `mcp_server_configs` (default empty allow-list).
- [ ] `loop/nodes/research.py` — merge MCP tools into the research agent's tool list.
- [ ] `tests/fixtures/mock_mcp_server.py` — minimal stdio MCP server fixture.
- [ ] `tests/test_mcp.py` — server tool-registration + client stdio round-trip + empty-config
      no-op, all offline.
- [ ] Docs (README / `doc/`): how to register Loop's server in Claude Desktop / MCP Inspector.

**Files touched:** `loop/mcp_server.py` (new), `loop/research/mcp_client.py` (new),
`loop/config.py`, `loop/nodes/research.py`, `pyproject.toml`,
`tests/fixtures/mock_mcp_server.py` (new), `tests/test_mcp.py` (new), `PLAN.md`.

**Done when:** `uv run python -m loop.mcp_server` serves Loop's tools over MCP (manually verified
in Claude Desktop / MCP Inspector — a server activity); the research agent can load + call tools
from an external MCP server via config; with no MCP config the flow is byte-for-byte the Phase 9
flow; a local stdio MCP round-trip is tested offline; lint + pytest pass.

**Skills needed:** `FastMCP` server authoring (`@mcp.tool()`, stdio transport);
`langchain-mcp-adapters` `MultiServerMCPClient` (async `get_tools()`); MCP primitives &
transports; reusing the factory/seam pattern; handling the sync/async boundary between the
LangGraph node and the async MCP client — **all verified against installed versions at build.**

---

## Phase 13 — Multi-agent orchestration  *(Capability: MULTI-AGENT — supervisor · parallel · handoffs)*

**Goal:** cross the line from a *fixed workflow* into a *real multi-agent system* by teaching the
three core patterns on Loop's own graph: a **supervisor** that reasons about who acts next,
**handoffs** between agents via `Command(goto=…)`, and **parallel fan-out/fan-in** via the
**Send API**. Every change is **flag-gated, default-off** so the existing deterministic graph and
all ~242 offline tests keep passing unchanged (the Phase 9b "opt-in, off by default" discipline).

**Why it exists / what it teaches:** "supervisor + handoffs + parallel agents" is the defining
"you actually understand agents, not just prompt chains" signal, and reworks the graph the owner
already knows deeply. v1's graph decides control flow; here the *model* decides it.

**Owner decisions / stack (APIs empirically verified against installed `langgraph==1.2.5` /
`langchain-core==1.4.7` during planning; re-verify at build per rule #7):**
- **`Send` import:** `from langgraph.types import Send` — **NOT** the deprecated
  `langgraph.constants` re-export (emits `LangGraphDeprecatedSinceV10`). Consistent with the
  existing `Command`/`interrupt` imports from `langgraph.types` in `graph.py`.
- **`Send(node, arg)` semantics:** `arg` becomes the invoked node's **input — it is NOT merged
  with graph state.** So the dispatcher must *pack* everything the persona needs into `arg`; the
  persona node reads `payload`, not `state`. (Proven during planning: a `Send`-invoked node
  received exactly the `arg`, not `LoopState`.)
- **Fan-in runs once:** three `Send` branches into one aggregator ⇒ the aggregator executes
  **1×** (superstep barrier), so appending one final `Grade` won't duplicate. (Verified.)
- **Un-clearable append channel:** a channel with an append reducer **cannot be reset** by a
  normal node return — teach a small **custom reset-sentinel reducer** `_reset_or_append(left,
  right)` (returns `[]` on a `None`/sentinel `right`, else `(left or []) + (right or [])`). This
  is the honest fix for "reset the panel after aggregation" and a genuinely good reducer lesson.
  (Verified: `InvalidUpdateError` on un-reduced parallel writes; `{"panel": []}` does NOT clear an
  `_append_list` channel.)
- **Bound:** the real supervisor bound lives in its **own logic** (decide `done → readiness` when
  `session_index >= min(len(plan["sessions"]), settings.max_sessions)`); the graph-level
  **`recursion_limit`** (default 25 supersteps, raises `GraphRecursionError` from
  `langgraph.errors`) is only the backstop.

**Concepts to teach:**
- **Fixed workflow vs. multi-agent:** who owns control flow — the graph (today) vs. a reasoning
  supervisor. Analogy: a hard-coded orchestration DAG vs. a coordinator service that decides call
  order at runtime.
- **Supervisor pattern:** one node that picks the next specialist from state — a reasoning
  replacement for the routing functions `session_router` / `_route_by_modality` /
  `_route_after_session`.
- **Handoffs:** `Command(goto=…, update=…)` returned from a node to route dynamically — the
  codebase **already does this once** in `plan_approval` (graph.py:147), so this generalizes a
  pattern the owner has seen. Annotate the return `Command[Literal[…destinations…]]` so LangGraph
  knows the possible targets without static edges.
- **Parallel fan-out/fan-in (Send API):** dispatch N concurrent branches, then reduce their
  results — map-reduce for agents. Reducers (Phase 7a's `_append_list`) are how the fan-in
  collects results; a reset reducer is how the fan-in *clears* the scratch channel.
- **Reducers as the ONLY way to mutate a channel (incl. reset):** the lesson above, made concrete.
- **Bounded agency (reprise):** a supervisor that loops needs a bound — same safety lesson as the
  Phase 9 ReAct iteration bound.

**Sub-steps (each = one turn: teach → code → test → pause):**

- **13a — Parallel fan-out/fan-in: a "panel of graders".** Alongside the single `grader`
  (grader.py:60), add (new `loop/nodes/panel.py`): a **`grade_dispatch`** fan-out function
  returning `list[Send]` — one `Send("panel_grader", payload)` per persona (*correctness /
  communication / depth*), packing persona name + answer text + question_id + rubric slice into
  `payload`; a single parametrized **`panel_grader(payload)`** node that scores from its payload
  (reads `payload`, **not** `state`) and writes a partial into a NEW transient `panel_grades`
  channel (reducer `_reset_or_append`); and a **`grade_aggregator(state)`** fan-in node (reached by
  a static edge `panel_grader → grade_aggregator`) that reduces the personas into ONE final `Grade`
  — output shape identical to today's grader `{"grades":[Grade.model_dump()]}` so `coach` and every
  downstream test are unchanged — and emits the reset sentinel to clear `panel_grades`. Wiring is
  behind `panel_grading: bool = False`: when on, the three interviewer→grader edges (graph.py:251-253)
  route into `grade_dispatch`, and `grade_aggregator → coach`; when off, today's `grader` path is
  untouched. The answer-collection `interrupt()` stays in the **sequential** interviewer upstream —
  **no `interrupt()` inside the fan-out** (personas are pure `answer → score`, so resume never
  re-enters the grading superstep mid-flight; `with_resilience` covers transient persona failures).
- **13b — Supervisor + handoffs.** Add `loop/nodes/supervisor.py::interview_supervisor(state) ->
  Command[Literal["coding_interviewer","sd_interviewer","beh_interviewer","readiness"]]`: an LLM
  (or stubbed) decision that, given plan + progress + live `weak_areas`, picks the next specialist
  (or `readiness`) via `Command(goto=…, update={current_modality/current_focus/current_topics/
  session_number})`; specialists may hand off via `Command(goto=…)`. **`advance_session`
  (graph.py:118) STAYS** — it still increments `session_index`; only its outgoing edge changes to
  `advance_session → interview_supervisor`. The supervisor **replaces the three routing functions**,
  not the increment node. Bounded by its own `session_index >= min(len(sessions), max_sessions)`
  guard; `recursion_limit` is the backstop.
- **13c — (stretch) panel debate round.** Extend 13a so personas see each other's round-1 scores
  and revise once before aggregation (multi-agent debate → better eval quality). Add a `round`
  field to each partial so the aggregator selects the **latest** round, not a blind reduce of both.

**Graph assembly (avoids the two-variant maintenance trap):** ONE
`build_graph(orchestration_mode: str = settings.orchestration_mode, panel_grading: bool =
settings.panel_grading)` that always wires the shared spine (`intake → research/planner →
plan_approval`, `interviewer → grade → coach → advance_session`) and branches only the two small
regions (routing block; grader-vs-panel block) with `if`. `compile_graph(mode="fixed",
panel_grading=False)` hard-pins the existing ~242 tests to today's behavior regardless of env;
`compile_graph_with_memory()` reads `settings`. Adding optional params to the existing
`compile_graph()` / `compile_graph_with_memory()` (graph.py:270,279) is backward-compatible.

**New state / config:** `state.py` — `panel_grades: Annotated[Optional[list[dict]],
_reset_or_append]` (+ the new reducer; init `None`); optional `supervisor_decisions`
(traceability). `config.py` — `panel_grading: bool = False`, `grader_personas: list[str] =
["correctness","communication","depth"]`, `orchestration_mode: Literal["fixed","supervisor"] =
"fixed"`; **reuse `max_sessions`** as the supervisor bound (no new bound knob).

**Task checklist:**
- [x] `loop/state.py` — `panel_grades` channel + `_reset_or_append` reducer.
- [x] `loop/config.py` — `panel_grading`, `grader_personas`, `orchestration_mode` (+ `panel_debate` for 13c).
- [x] `loop/nodes/panel.py` — `grade_dispatch` (fan-out) + `panel_grader` (persona) + `grade_aggregator` (fan-in).
- [x] `loop/nodes/supervisor.py` — `interview_supervisor` (Command-handoff routing).
- [x] `loop/graph.py` — parametrized `build_graph(...)`; flag-gated wiring of both regions.
- [x] `loop/schemas.py` — `PersonaGrade` + `SupervisorDecision`.
- [x] `tests/test_multiagent.py` — offline, deterministic (see strategy below). 18 tests.

**Files touched:** `loop/graph.py`, `loop/nodes/panel.py` (new), `loop/nodes/supervisor.py` (new),
`loop/state.py`, `loop/config.py`, `loop/schemas.py`, `tests/test_multiagent.py` (new), `PLAN.md`.

**Done when:** with flags off, the graph is byte-for-byte the Phase 12 graph and all existing
tests pass; with `panel_grading=True`, one answer is graded by K personas concurrently and
aggregated into a single `Grade`, with `panel_grades` provably reset by the sentinel reducer
(tested offline with stubbed persona models — `grade_dispatch` returns N `Send`s; aggregator runs
once); with `orchestration_mode="supervisor"`, a stubbed supervisor reproduces today's modality
sequence for a 2-session plan then lands on `readiness`, and an "always keep going" stub still
terminates via the `max_sessions` guard (plus a test proving `GraphRecursionError` is the
backstop); lint + pytest pass; (server) one live supervisor + one live panel run sanity-checked.

**Offline test strategy:** stub `loop.nodes.supervisor.get_chat_model` /
`loop.nodes.panel.get_chat_model` with `RunnableLambda`s that are **pure functions of
state/payload** (the `conftest.py` planner-stub pattern), so both new paths are deterministic; all
tests use `compile_graph(...)` (no checkpointer) with interviewers stubbed as today (no interrupts
fire offline).

**Skills needed:** LangGraph `Send` (fan-out/fan-in via a conditional edge returning `list[Send]`),
`Command(goto=…, update=…)` handoffs with `Command[Literal[…]]` return annotations, supervisor
pattern, `recursion_limit` / `GraphRecursionError`, custom reset reducers — the load-bearing
mechanics were empirically confirmed during planning; re-verify exact signatures at build.

---

## Phase 14 — Advanced RAG  *(Capability: RETRIEVAL depth — hybrid · rerank · corrective)*

**Goal:** upgrade Phase 8's single-shot top-k cosine lookup into a production retrieval pipeline —
**hybrid search + reranking**, **query rewriting / HyDE**, and a **corrective-RAG (CRAG)** loop
that grades retrieved questions and re-retrieves (or falls back to web search) when they're weak —
all **behind the unchanged `retrieve_questions()` / `search_questions()` signature** (retrieval.py:88,
tools.py:79), so no caller changes.

**Why it exists / what it teaches:** RAG *depth* — especially reranking — is the most-requested
retrieval skill. Phase 8 taught retrieve→augment→generate; Phase 14 teaches *why naive top-k
cosine misses*, and the standard fixes.

**Owner decisions / stack (verify at build per rule #7):**
- **Reranker:** Bedrock Rerank (e.g. `amazon.rerank` / `cohere.rerank`) behind a `get_reranker()`
  factory mirroring `embeddings.py`; a deterministic **fake reranker** offline. Verify the
  `langchain-aws` rerank API vs. calling `bedrock-runtime` rerank directly.
- **Hybrid search:** BM25 (keyword) fused with the existing dense embeddings via **Reciprocal
  Rank Fusion (RRF)**. BM25 via `rank-bm25` (small, pure-Python, offline) or LangChain
  `BM25Retriever`. Verify the retriever/ensemble API.
- **Query rewriting / HyDE:** an LLM step (reuses the model factory) that expands `focus`/`topics`
  into multiple queries, or embeds a hypothetical answer (HyDE).
- **CRAG fallback:** reuse the Phase 9 `search_web` seam (research/tools.py) when the bank has no
  relevant question.
- **Offline gate:** fake embeddings + fake reranker + deterministic BM25 + stubbed relevance grader.

**Concepts to teach:** why cosine top-k over-retrieves near-duplicates and misses lexical matches
(→ hybrid + RRF); a **cross-encoder reranker** vs. a bi-encoder embedding (accuracy/latency
trade); **query transformation** (multi-query, HyDE) as recall boosters; **corrective RAG** — a
self-correcting retrieval loop (a mini-agent around retrieval); the reranker as another **factory
seam**, same lesson as `models.py`/`embeddings.py`.

**Sub-steps (each = one turn: teach → code → test → pause):**
- **14a — Hybrid search + reranking.** Add a BM25 retriever over the 24-question bank; fuse with
  dense retrieval via RRF; rerank the fused candidates via `get_reranker()`. Swap the *internals*
  of `retrieve_questions()` (retrieval.py:88) to dense+BM25 → RRF → rerank → top-k; signature
  unchanged. Config: `rerank_enabled`, `rerank_model_id`, `hybrid_enabled`. Tests: fake reranker
  reorders deterministically; hybrid returns the lexical+semantic union; signature/shape stable.
- **14b — Query rewriting / HyDE.** Add `rewrite_query(focus, topics) -> list[str]` before
  retrieval; the interviewer's query (interviewers.py `_ask_question`) becomes a richer query set.
  Config: `query_rewrite_mode: off|multiquery|hyde`. Tests: stubbed rewriter expands 1 → N;
  retrieval consumes all and de-dupes.
- **14c — Corrective RAG (CRAG).** Add `grade_retrieval(question, focus) -> relevance` + a bounded
  re-retrieve loop; on persistent low relevance, fall back to `search_web`. Config: `crag_enabled`,
  `crag_min_relevance`, `crag_max_retries`. Tests: low-relevance stub triggers one re-retrieve then
  fallback; the retry bound holds.

**Task checklist:** `loop/reranker.py` (new factory); `loop/retrieval.py` (hybrid+RRF+rerank in
`_build_index`/`retrieve_questions`); `loop/tools.py` (query-rewrite hook, signature stable);
`loop/nodes/interviewers.py` (query set); `loop/config.py` (rerank/hybrid/rewrite/CRAG knobs);
`pyproject.toml` (`rank-bm25`); `tests/test_retrieval.py` + new tests.

**Files touched:** `loop/{reranker,retrieval,tools,config}.py`, `loop/nodes/interviewers.py`,
`loop/research/tools.py` (CRAG fallback reuse), `pyproject.toml`, `tests/*`, `PLAN.md`.

**Done when:** retrieval runs hybrid+rerank behind the unchanged `retrieve_questions()`/
`search_questions()` signature; query rewriting + CRAG are flag-gated and bounded; every path is
offline (fake embeddings/reranker/BM25/relevance-grader); lint + pytest pass; (server) one live
Bedrock-rerank run sanity-checked.

**Skills needed:** BM25 + RRF fusion, cross-encoder reranking (Bedrock Rerank), query
transformation (multi-query/HyDE), corrective-RAG loops, reranker factory seam — verified at build.

---

## Phase 15 — Self-improvement loops  *(Capability: SELF-IMPROVEMENT — reflexion · replan · DSPy)*

**Goal:** make Loop improve its own outputs and prompts: a **Reflexion self-critique** pass where
the grader (and optionally planner) critiques and revises its own output before committing;
**replanning** so the planner adapts the *remaining* curriculum mid-run from live grades; and
**automatic prompt optimization (DSPy)** that tunes the grader prompt against the Phase 6 labeled
set (`fixtures/grader_labels.json`).

**Why it exists / what it teaches:** self-refinement (verifier↔generator loops) and *programmatic*
prompt optimization are rare, high-signal skills — and this is where Phase 6's eval finally feeds
back into making the agent better, not just measuring it.

**Owner decisions / stack (verify at build per rule #7):**
- **Reflexion:** a second LLM pass ("critique your own grade against rubric+reference; revise if
  warranted") — pure model calls, offline-testable with stubs.
- **DSPy:** `dspy` optimizes the grader prompt (a DSPy module) against `grader_labels.json` with a
  metric = **`score_agreement`** (reuse `evals/run_grader_eval.py:69`). Optimization is a
  **server/dev activity** (needs model calls, like live Bedrock) that emits an **optimized-prompt
  artifact**; the runtime just *loads* it. Verify `dspy` version + its Bedrock LM adapter.
- **Replanning:** reuse `planner()` (planner.py:103) + the multi-session loop; a bounded re-plan of
  remaining sessions.
- **Offline gate:** stub all model calls; tests load a **fixture** optimized-prompt (never run DSPy
  in `tests/`).

**Concepts to teach:** the **Reflexion** generate→critique→revise loop and why a second pass
catches first-pass errors; **replanning / plan-and-execute** (adapt the plan when reality diverges
from assumptions); **DSPy** — prompts as *optimizable programs* tuned against a labeled metric
instead of hand-crafted; separating an **offline optimization step** (produces an artifact) from
the **runtime** that consumes it (the model-training/serving split, familiar from ML ops).

**Sub-steps (each = one turn: teach → code → test → pause):**
- **15a — Reflexion self-critique.** Add `_self_critique(grade, rubric, reference) -> grade` to
  `grader()` (grader.py); optional for `planner()`. Flag `reflexion_enabled` (default off →
  today's behavior). Tests: a stubbed critic flips a wrong score; disabled path unchanged.
- **15b — Replanning / adaptive curriculum.** Add a bounded `replan` decision after `coach`/
  `advance_session`: if live grades diverge from plan assumptions, re-invoke `planner` on the
  remaining sessions. Config: `replan_enabled`, `replan_score_threshold`, `replan_max_times`.
  Tests: stubbed low grades trigger one re-plan that changes remaining modalities; no divergence →
  no replan; the bound holds.
- **15c — DSPy prompt optimization (dev/offline artifact).** `evals/optimize_grader.py` DSPy-
  optimizes the grader prompt against `grader_labels.json` with `score_agreement` as the metric,
  writing `fixtures/optimized_grader_prompt.txt`; `grader()` loads that artifact if present, else
  the hand-written prompt. Tests: grader loads a fixture optimized prompt; falls back cleanly when
  absent. (Running the optimizer = server/dev activity.)

**Task checklist:** `loop/nodes/grader.py` (self-critique + artifact load); `loop/nodes/planner.py`
(optional critique); `loop/graph.py` (bounded `replan` edge); `loop/config.py` (reflexion/replan
knobs); `evals/optimize_grader.py` (new); `fixtures/optimized_grader_prompt.txt` (artifact);
`tests/*`.

**Files touched:** `loop/nodes/{grader,planner}.py`, `loop/graph.py`, `loop/config.py`,
`evals/optimize_grader.py` (new), `fixtures/optimized_grader_prompt.txt` (new), `tests/*`, `PLAN.md`.

**Done when:** reflexion revises weak grades (flag-gated, offline-tested); the planner replans
remaining sessions on live divergence (bounded, offline-tested); the grader loads a DSPy-optimized
prompt artifact with graceful fallback; lint + pytest pass; (server) one live DSPy optimization run
+ a before/after `aggregate_mae` delta sanity-checked.

**Skills needed:** Reflexion (generate/critique/revise), plan-and-execute/replanning, DSPy prompt
optimization + Bedrock LM adapter, the offline-artifact vs. runtime split — verified at build.

---

## Phase 16 — Advanced memory  *(Capability: MEMORY architecture — typing · reflection · semantic recall)*

**Goal:** upgrade Phase 4's flat weak-areas store into a real memory architecture — **memory
typing** (episodic / semantic / procedural), a **reflection/consolidation** agent that summarizes
sessions into higher-level insights, **semantic memory search** (recall by meaning, not dump-all),
and **decay / conflict resolution** — all on the existing `InMemoryStore` seam (memory.py:70).

**Why it exists / what it teaches:** memory is the hottest hard problem in agents. Phase 4 taught
checkpointer-vs-store; Phase 16 teaches memory as an **architecture**, not a dict.

**Owner decisions / stack (verify at build per rule #7):**
- Reuse `loop/memory.py` `InMemoryStore` + the `("loop","users")[user_id]` namespace
  (coach.py:66, planner.py:96). Expand from `{weak_areas, session_count}` to typed sub-namespaces:
  `("loop","users",user_id,"episodic"|"semantic"|"procedural")`, with **backward-compatible reads**
  of the old flat shape.
- **Semantic store search:** LangGraph store index/embeddings search (`store.search` with an index
  config) — verify the `InMemoryStore` semantic-search API on installed `langgraph`; use fake
  embeddings offline (the Phase 8 `get_embeddings` seam + `DeterministicFakeEmbedding`).
- **Reflection:** a `reflect` node (LLM) that reads recent **episodic** memories and writes
  consolidated **semantic/procedural** insights; runs at a session/curriculum boundary, bounded.
- **Decay/conflict:** a **deterministic** policy (session-count staleness TTL; newer/higher-
  confidence wins on contradiction) so it stays laptop-testable.

**Concepts to teach:** **episodic** (what happened) vs **semantic** (durable facts about the user)
vs **procedural** (how to coach this user) memory; **reflection/consolidation** (the Generative-
Agents pattern — memory that *thinks*, turning raw episodes into insight); **semantic recall**
(retrieve relevant memories by embedding, not load-everything); **memory lifecycle** (decay +
conflict resolution) as first-class concerns.

**Sub-steps (each = one turn: teach → code → test → pause):**
- **16a — Memory typing.** Refactor the store schema into episodic / semantic / procedural
  namespaces. `coach._persist_weak_areas` (coach.py:43) → typed writes; `planner._get_stored_weak_areas`
  (planner.py:77) → typed reads; keep back-compat reads of the old `{weak_areas}` shape. Tests:
  typed write/read round-trips; migration from the flat shape.
- **16b — Reflection / consolidation.** Add `loop/nodes/reflect.py`: read episodic memories, write
  consolidated semantic+procedural insights via an LLM; wire at the curriculum boundary, bounded;
  `planner` reads the consolidated semantic memory. Config: `reflection_enabled`. Tests: a stubbed
  reflection turns 3 episodes into 1 insight; the planner prompt consumes it.
- **16c — Semantic recall + decay/conflict.** Embed semantic memories; recall top-k by similarity
  (`store.search`) instead of dumping all. Add the deterministic decay/conflict policy. Config:
  `memory_recall_k`, `memory_ttl_sessions`. Tests: semantic recall ranks the relevant memory (fake
  embeddings); stale memory decays out; contradictory facts resolve to the newer.

**Task checklist:** `loop/memory.py` (typed namespaces + `store.search` helper); `loop/nodes/coach.py`
(typed writes); `loop/nodes/planner.py` (typed + semantic reads); `loop/nodes/reflect.py` (new);
`loop/schemas.py` (memory-item models); `loop/config.py` (reflection/recall/ttl knobs);
`tests/test_memory.py` + new.

**Files touched:** `loop/memory.py`, `loop/nodes/{coach,planner,reflect}.py`, `loop/schemas.py`,
`loop/config.py`, `tests/*`, `PLAN.md`.

**Done when:** memories are typed (episodic/semantic/procedural) with back-compat; a reflection
agent consolidates episodes into insights that demonstrably shape the next plan; recall is semantic
(fake-embeddings offline); decay/conflict is deterministic and tested; lint + pytest pass; (server)
a live reflection run sanity-checked.

**Skills needed:** memory typing, reflection/consolidation, LangGraph store semantic search, memory
lifecycle (decay/conflict) — verified at build against installed `langgraph`.

---

## Phase 17 — Eval-in-CI, agent simulation & red-teaming  *(Capability: EVAL/SAFETY at prod grade)*

**Goal:** turn the Phase 6 eval scripts into a **CI regression gate**, add **agent-simulation
testing** (an LLM "synthetic candidate" plays full sessions end-to-end), and an **automated
red-team suite** (jailbreak/injection) that extends the Phase 10 guardrails.

**Why it exists / what it teaches:** treating agent quality like a test suite — with a gate that
fails the build when grading agreement regresses — is a standout senior signal and plays straight
to the owner's CI/CD strength. Agent simulation and red-teaming are techniques almost nobody
demonstrates.

**Owner decisions / stack (verify at build per rule #7):**
- **Two-tier eval gate** (resolves "evals are model-dependent vs. offline gate"): **(1) offline
  deterministic tier** in `tests/` — stubbed model, asserts the node trajectory via
  `evals/trajectory_check.assert_trajectory` (trajectory_check.py:71) + grader determinism on
  fixtures — runs on **every CI commit**; **(2) live tier** — `evals/run_grader_eval.run_eval`
  (run_grader_eval.py:144) scores agreement/`aggregate_mae` against `grader_labels.json` and exits
  non-zero if agreement drops below a threshold — runs **nightly on the server**, not the laptop
  gate. `evals/ci_gate.py` orchestrates + thresholds.
- **Agent simulation:** a `SimulatedCandidate` that auto-answers the interviewer's `interrupt()`s,
  driving the whole graph unattended. **Offline:** a deterministic scripted candidate (canned
  answers per modality). **Live:** an LLM candidate (server).
- **Red-team:** a corpus run through `guardrails.detect_injection` + end-to-end through the intake/
  answer gates, asserting flag/redact. Deterministic, offline. Optional **Llama Guard** seam.

**Concepts to teach:** an **eval regression gate** (agent quality as a build check) and why it must
be split into a deterministic offline tier + a probabilistic live tier; **agent simulation** (an
LLM drives your agent end-to-end to surface flow bugs no unit test catches); **red-teaming**
(offensive testing of guardrails) as the complement to Phase 10's defensive redaction/flagging.

**Sub-steps (each = one turn: teach → code → test → pause):**
- **17a — Eval-in-CI gate.** `evals/ci_gate.py`: the **offline deterministic tier**
  (`assert_trajectory` + grader-on-fixtures, no model) as a pytest-runnable gate; the **live tier**
  wraps `run_eval` with a `min_agreement` threshold that exits non-zero on regression. Document the
  GitHub Actions wiring (offline tier every commit; live tier nightly). Tests: the gate passes on
  good fixtures and fails on an injected regression fixture.
- **17b — Agent simulation.** `evals/simulate_session.py`: a `SimulatedCandidate` that resumes the
  graph's answer interrupts automatically; run a full multi-session sim end-to-end. Offline scripted
  candidate; assert the full trajectory + that grades/verdict are produced. Tests: a deterministic
  sim completes a 2-session run offline.
- **17c — Red-team suite.** `evals/redteam.py` + `fixtures/redteam_prompts.json`: run
  injection/jailbreak strings through `detect_injection` and end-to-end; assert flagging. Optional
  Llama Guard seam in `guardrails.py`. Tests: known attacks flagged; clean inputs pass unchanged.

**Task checklist:** `evals/ci_gate.py` (new); `evals/simulate_session.py` (new); `evals/redteam.py`
(new); `fixtures/redteam_prompts.json` (new); `loop/guardrails.py` (Llama Guard seam);
`tests/test_evals.py` + new; `.github/workflows/*` (documented, not necessarily run here).

**Files touched:** `evals/{ci_gate,simulate_session,redteam}.py` (new), `fixtures/redteam_prompts.json`
(new), `loop/guardrails.py`, `tests/*`, `.github/workflows/` (doc), `PLAN.md`.

**Done when:** an offline deterministic eval gate runs in CI on every commit and fails on a seeded
regression; a live agreement-threshold gate is wired for nightly; an agent simulation drives full
sessions unattended (scripted offline / LLM live); a red-team suite flags known attacks and passes
clean inputs; lint + pytest pass; (server) the live tiers sanity-checked.

**Skills needed:** CI eval gates (deterministic + probabilistic tiers), agent simulation harnesses,
red-teaming / guardrail testing, reusing `run_grader_eval`/`trajectory_check` — verified at build.

---

## Phase 18 — Production infrastructure & real data  *(Capability: PRODUCTION — Track G core)*

**Goal:** replace the v1 in-memory / keyless / fixture backends with production backends, **each
behind the seam the project already established**, so the graph and node code never change. This
is the "make it real / deploy it" phase — the owner's AWS/backend edge. **This phase formalizes
the old "v2 / future enhancements" backlog into an executable phase.**

**Ordering decision (owner-delegated, 2026-07-18):** this phase runs **AFTER** the concept-heavy
frontier phases (12 MCP, 13 multi-agent, and tracks C/D/E/H), **not before Phase 12.** The infra
items are *plumbing swaps behind seams that already exist* — solid engineering, but almost zero
*new* agentic-AI concepts — and nothing depends on them either way (MCP and multi-agent run fine
on `InMemoryVectorStore` / `SqliteSaver` / Bedrock). They are the backbone of the production track
(**Track G**) and earn their keep at deploy time. **Overridable:** pull this phase forward without
touching any other phase if you decide to deploy early — that's the whole point of the seams.

**Why it exists / what it teaches:** productionizing an agent (durable multi-user state, a real
vector DB, real external data, a self-hostable model) is the highest-paying intersection of the
owner's existing skills with agentic AI, and proves the seam discipline end-to-end.

**Concepts to teach:** connection-string config & pooling; durable checkpointer/store semantics
vs. in-memory; pgvector as a real ANN index vs. `InMemoryVectorStore`; a DAO swap from fixtures to
a real store without touching callers; why the `api.py` pending-command dict must also move to a
durable store for true cross-restart resume.

**Sub-steps (each = one turn: teach → code → test → pause) — one seam per step:**

- **18a — Postgres checkpointer + store.** Swap `memory.py::_make_checkpointer()` / the `_store`
  singleton (memory.py:69-70; v2 comment at memory.py:68) to `PostgresSaver` / `PostgresStore`,
  reading a new `pg_conn_string` config knob (no such field exists yet — must add). Also move
  `api.py`'s process-local `_pending` / `_budgets` dicts (api.py:91-98 caveat: "v2 fix: store
  pending commands in Redis or a DB table") to a durable store so resume survives a restart.
  Tests: durable resume across a simulated restart (offline — a temp DB or the existing SQLite
  path as the laptop stand-in; Postgres itself is a server activity).
- **18b — pgvector.** Swap `InMemoryVectorStore` → `PGVector` in `retrieval.py:74` **only** (seam
  comment retrieval.py:18-19); `retrieve_questions()` signature + all callers unchanged. Reuses the
  `get_embeddings()` factory. Tests: retrieval still returns the documented shape (offline — fake
  embeddings; a real pgvector run is a server activity).
- **18c — Real search (Tavily).** Implement `_tavily_search()` (search.py:58, currently
  `NotImplementedError` at search.py:65) behind the existing auto-switch on `tavily_api_key`
  (config.py:62). Tests: stub the Tavily client (offline), same discipline as the `ddgs` stub.
- **18d — Ollama model + embeddings.** Fill the two Ollama seams: `models.py:34` and
  `embeddings.py:31` (both raise `NotImplementedError` today), selected by `model_provider`
  (config.py:44). Tests: factory returns the Ollama model when `model_provider="ollama"` (offline
  — assert construction, don't call it).
- **18e — Real data (JD ingest + large question bank).** Swap the `tools.py` fixture DAO
  (tools.py:4-6 seam) to read questions / rubrics / reference answers from Postgres (pairs with
  18b's pgvector index); add a real JD-upload path at the seam noted in graph.py:62-65 (the
  `intake` redact/flag path already treats JD/profile as untrusted input). Tests: DAO returns the
  same shapes from the new backend (offline with a seeded temp DB / fixture adapter).

**Files touched:** `loop/{memory,api,retrieval,models,embeddings,tools,config}.py`,
`loop/research/search.py`, `pyproject.toml` (new deps: `langgraph-checkpoint-postgres`,
`langchain-postgres`/pgvector, `langchain-ollama`, `langchain-tavily` — verify + pin at build),
`tests/*`, `PLAN.md`.

**Done when:** each backend swap is done behind its existing seam with **no change to the graph or
any calling node**; every swap is laptop-testable offline (stubbed clients / temp DB / fake
embeddings); lint + pytest pass; live Postgres / pgvector / Tavily / Ollama runs are sanity-checked
as server activities.

**Skills needed:** `PostgresSaver` / `PostgresStore`, `PGVector`, `langchain-ollama`,
`langchain-tavily` (all verified at build); connection config; the DAO/repository swap pattern.

---

## Changelog

> Append one entry per completed phase: date, phase, what was built, key decisions, what the
> owner learned. Keep newest at top.

### Phase 15a+15b+15c — 2026-09-17
**Built:** `loop/nodes/grader.py` — `_self_critique(grade, rubric, reference, answer_text,
question_prompt) -> Grade` (15a): a second `get_chat_model()` pass, structured-output on the
same `Grade` schema, given the original grade plus rubric/reference/answer and asked to
return it unchanged or revised; `grader()` calls it only when `settings.reflexion_enabled`
is `True` (default `False` — today's single-pass path is byte-for-byte unchanged). Also
`_load_system_prompt()`/`_build_grading_prompt()` (15c): read
`fixtures/optimized_grader_prompt.txt` if present and non-empty, else the hand-written
`_SYSTEM` — `grader()` now builds its prompt through this each call. `loop/graph.py` (15b) —
`_grade_divergence(state)` (last-appended grade's score below `replan_score_threshold`),
`replan(state)` (re-invokes `planner()`, splices its fresh sessions in from `session_index`
onward, renumbers them, caps at however many slots were left so `total_sessions` never
grows, increments `replan_count`), and `_route_after_advance(state)` — the new edge function
after `advance_session` in `orchestration_mode="fixed"` (returns `"replan"`/`"continue"`/
`"done"`; supervisor mode is untouched, replanning isn't wired for it). `loop/config.py` —
`reflexion_enabled`, `replan_enabled`, `replan_score_threshold` (default 5),
`replan_max_times` (default 1), all off/bounded by default. `loop/state.py` —
`replan_count` field (plain overwrite, defaults to 0). `evals/optimize_grader.py` (new,
dev-only) — a DSPy `Signature`/`Predict` module graded against
`fixtures/grader_labels.json` via `MIPROv2`, with `evals.run_grader_eval.score_agreement`
wrapped (not duplicated) as the DSPy metric; writes the winning instruction text to
`fixtures/optimized_grader_prompt.txt`. `fixtures/optimized_grader_prompt.txt` (new) — a
hand-written "already optimized" prompt (independent-per-criterion scoring, explicit
partial-credit guidance, a sum-check reminder) checked in as the realistic artifact so
`grader()`'s load path has something real to load; a live DSPy run can overwrite it later.
`pyproject.toml` — `dspy>=3.3.1` (installs cleanly; `dspy.LM(model="bedrock/<model_id>", ...)`
via litellm, no separate Bedrock adapter class needed). `tests/test_reflexion.py`,
`tests/test_replan.py`, `tests/test_prompt_optimization.py` (new) — 322/322 total, 0 lint
errors.

**Key decisions / lessons:**
- **Planner self-critique left unimplemented (spec marked it optional):** PLAN.md says
  "optional for `planner()`" — grader.py already demonstrates the generate→critique→revise
  pattern end-to-end and offline-tested; adding a second copy in planner.py for the same
  lesson would be duplication without new teaching value, so it was skipped per CLAUDE.md's
  "clarity over cleverness, no premature abstraction."
- **Divergence check is a single deterministic signal, on purpose:** `_grade_divergence`
  looks only at the most recently appended grade (the session `advance_session` just closed
  out) vs. `replan_score_threshold`. A production system might average several signals; the
  spec asked for "a simple deterministic divergence check," and a single, inspectable
  threshold is easier to teach and to unit-test than a composite score.
- **`replan()` reuses `planner()` rather than a bespoke prompt:** the spec explicitly says
  "re-invoke `planner()` on the remaining sessions" — `planner()` already reads the latest
  `weak_areas`, so a fresh call naturally reflects what just went wrong. `replan()`'s own job
  is purely the splice/renumber/cap around that call, which is also why it's easy to unit-test
  without stubbing a new prompt.
- **The optimized-prompt artifact is loaded on every `grader()` call, not cached at import
  time:** a small `pathlib.Path.read_text()` per call is cheap and keeps
  `_load_system_prompt()` independently testable (monkeypatch the path constant, no module
  reload needed) — matches how `tests/test_prompt_optimization.py` verifies both the
  present- and absent-artifact branches without ever importing `dspy`.
- **`dspy` installed cleanly via `uv add dspy`** (package name is `dspy`, not `dspy-ai`, as
  of `dspy==3.3.1` — verified per CLAUDE.md rule #7 rather than assumed) and its Bedrock
  path goes through `litellm`'s `bedrock/<model_id>` model-string convention; the optimizer
  script imports `dspy` only inside functions (never at module top level, and never inside
  `loop/nodes/grader.py`), so nothing in the runtime or test suite requires it to be
  importable at collection time beyond the dependency simply being installed.

### Phase 14a+14b+14c — 2026-09-17
**Built:** `loop/reranker.py` (new) — `get_reranker()` factory mirroring `embeddings.py`:
`BedrockRerank` (`langchain_aws.document_compressors.rerank`, constructed from a
`model_arn` built from `rerank_model_id` + `aws_region` — not re-exported from the
package `__init__`, imported from its submodule) for the live path, `FakeReranker`
(deterministic word-overlap scoring) for offline/tests. `loop/retrieval.py` — swapped
`retrieve_questions()`'s internals to dense + BM25 (`rank-bm25`'s `BM25Okapi`, built
alongside the vector store in `_build_index()`) fused via Reciprocal Rank Fusion
(`_reciprocal_rank_fusion`, `k=60`, rank-position-only so cosine similarity and BM25's
unbounded scores never need normalising against each other) → reranked (`get_reranker()`
on the fused shortlist) → top-k; the public signature is byte-for-byte unchanged, and with
`hybrid_enabled=False`/`rerank_enabled=False` it's exactly the Phase 8 cosine-only path.
Added `rewrite_query(focus, topics) -> list[str]` (14b — off/multiquery/HyDE via the
existing model factory, sharing one `QueryRewrite` schema for both non-off modes) and
`grade_retrieval(question, focus) -> float` + `crag_search(query, modality, k, focus)`
(14c — a bounded corrective loop: grade the top hit, re-retrieve with a broadened query up
to `crag_max_retries` times, then fall back to the Phase 9 `search_web` tool for one final
grounded retrieval pass). `loop/nodes/interviewers.py::_ask_question` now calls
`rewrite_query()` then `crag_search()` per query, merging + de-duping candidates by id
instead of one direct `search_questions()` call. `loop/config.py` —
`hybrid_enabled`/`rerank_enabled`/`rerank_model_id` (on by default — this is the retrieval
upgrade itself, not an opt-in), `query_rewrite_mode` (default `"off"`), `crag_enabled`
(default `False`, adds an LLM call per question pick), `crag_min_relevance`,
`crag_max_retries`. `loop/schemas.py` — `QueryRewrite`, `RetrievalGrade`. `pyproject.toml`
— `rank-bm25`. `tests/test_reranker.py`, `tests/test_query_rewrite.py`, `tests/test_crag.py`
(new) + 12 new tests appended to `tests/test_retrieval.py` — 299/299 total, 0 lint errors.

**Key decisions / lessons:**
- **Verified before coding (CLAUDE.md rule #7):** `langchain_aws.document_compressors`'s
  `__init__.py` does not re-export `BedrockRerank` — it must be imported from
  `langchain_aws.document_compressors.rerank` directly, and it takes a `model_arn` (not a
  bare `model_id` like `ChatBedrockConverse`/`BedrockEmbeddings`), so `get_reranker()`
  builds the ARN from `rerank_model_id` + `aws_region`. `BedrockRerank(...)` construction
  itself makes no network call (only `.rerank()` does), confirmed by constructing one with
  no AWS credentials present — this is why `get_reranker()` can be exercised structurally
  in tests without stubbing construction, only the network-touching `.rerank()` call
  (handled globally by `stub_reranker`, a new autouse `conftest.py` fixture mirroring
  `stub_embeddings`, since `hybrid_enabled`/`rerank_enabled` default **on**).
- **Hybrid + rerank default ON, unlike every other Phase 12–13 flag (default off):** those
  flags gate genuinely optional behaviour; hybrid+rerank *is* the Phase 14a retrieval
  pipeline, so leaving it off by default would mean the phase shipped disabled. The existing
  272 Phase 0–13 tests still pass unmodified because the fake embeddings + fake reranker
  reproduce a stable, deterministic ranking that the old cosine-only assertions still hold
  under (verified by running the full suite, not assumed).
- **`query_rewrite_mode` default `"off"` and `crag_enabled` default `False`:** both attach
  an LLM call per question pick when turned on, changing runtime behaviour/cost — off by
  default follows the same "flag-gated, byte-identical when off" contract as
  `panel_grading`/`orchestration_mode="supervisor"` in Phase 13.
- **CRAG's bound is structural, not a counter that could be miscounted:** `crag_search` is
  a plain Python `for _ in range(crag_max_retries)` loop plus exactly one grade-check before
  it and one web-search fallback pass after it — at most `crag_max_retries + 2` calls to
  `grade_retrieval`/`retrieve_questions` combined, regardless of what the LLM grader
  returns; a test with a grader stubbed to always return the lowest possible relevance
  score proves the retry count stays bounded and `search_web` is called exactly once.
- **`rewrite_query("off")` reconstructs the exact pre-Phase-14 query string** used by
  `interviewers.py` before this phase, so the default path (single query, `crag_enabled`
  off) is a byte-for-byte no-op wrapper — confirmed by the full existing test suite passing
  with zero changes to its assertions.

---

### Phase 13a+13b+13c — 2026-07-19
**Built:** `loop/nodes/panel.py` — `grade_dispatch(state) -> list[Send]` (fan-out, one
persona per `settings.grader_personas`), `panel_grader(payload)` (per-persona scoring,
reads `payload` never `state`), `grade_aggregator(state) -> Command` (fan-in — averages
criterion scores across personas, unions strengths/improvements, emits the reset sentinel).
`loop/nodes/supervisor.py::interview_supervisor` — a `Command`-handoff node replacing
`session_router`/`_route_by_modality`/`_route_after_session` as a unit, bounded by
`session_index >= min(len(sessions), settings.max_sessions)` checked BEFORE any model call.
`loop/graph.py::build_graph(orchestration_mode="fixed", panel_grading=False)` — ONE function
wiring the shared spine plus two flag-gated regions; `compile_graph()` hard-pins both flags
so the existing 254 tests are pinned to the exact Phase 12 graph; `compile_graph_with_
memory()` reads `settings`. `plan_approval` gained an optional `next_node` param (default
`"session_router"`, unchanged for every existing caller) so it can hand off to
`interview_supervisor` in supervisor mode without knowing which mode is active.
`loop/state.py` — `panel_grades` (new `_reset_or_append` reducer) + `supervisor_decisions`
(traceability). `loop/config.py` — `panel_grading`, `grader_personas`, `panel_debate`
(13c stretch), `orchestration_mode`. `loop/schemas.py` — `PersonaGrade`, `SupervisorDecision`.
`tests/test_multiagent.py` — 18 new offline tests (272/272 total, 0 lint errors). Docs:
`docs/16-19` (fixed-workflow-vs-multiagent, the Send API, Command handoffs + supervisor
pattern, reducers + bounded agency) + `doc/phase-13-multiagent-orchestration.md`.

**Key decisions / lessons:**
- **Verified before coding (CLAUDE.md rule #7), not trusted from the PLAN's prose:** three
  facts that shaped the design, each confirmed with a throwaway script against
  `langgraph==1.2.5` before touching product code — (1) `Send(node, arg)`'s `arg` REPLACES
  the target node's input, it does not merge with graph state; (2) `{"panel_grades": []}`
  does NOT clear an append-reducer channel (`(left or []) + ([] or []) = left`, unchanged)
  — only a reducer that treats a *different* sentinel (we use `None`) as "clear" can reset
  one; (3) a plain node (not just a conditional-edge path function) CAN return
  `Command(goto=[Send(...), ...])` to fan out — this is what makes the 13c debate round's
  "fan-in that sometimes re-fans-out" shape possible from a single `grade_aggregator` node.
- **`grade_aggregator` and `interview_supervisor` both register with NO static outgoing
  edge**, same rule `plan_approval` (Phase 5) already followed: a node that sometimes
  returns `Command(goto=X)` and sometimes `Command(goto=Y)` must never also have a static
  edge to either destination, or both paths fire in the same superstep. Verified this isn't
  just a docstring warning — deliberately adding a competing static edge in a mutation test
  did NOT get caught by the non-debate-path tests (both routes agreed on `"coach"` in that
  configuration), a real gap in coverage that's now documented rather than silently assumed
  covered.
- **Mutation-tested before calling it done**, not just written and trusted: disabled
  `interview_supervisor`'s bound check and confirmed the two bound-related tests failed
  (and only those two); confirmed `GraphRecursionError` fires against a deliberately broken
  supervisor stub that ignores `session_index` entirely, proving the recursion-limit
  backstop is real, not just documented.
- **Offline supervisor test needed a non-constant model stub**, unlike every prior phase's
  `lambda _: STUB_VALUE` pattern: since `chain = _PROMPT | structured_model` pipes the
  *rendered* prompt into the stub (not the raw input dict), "reproduce today's plan-based
  modality sequence" required a stub that parses the rendered human message back out via
  regex to echo the plan's suggested modality — a genuine function of input, not a constant.
- Full write-up: [`doc/phase-13-multiagent-orchestration.md`](doc/phase-13-multiagent-orchestration.md).

### Phase 12a+12b — 2026-07-19
**Built:** `loop/mcp_server.py` — a `FastMCP("loop")` server exposing four thin `@mcp.tool()`
wrappers (`list_questions`, `search_questions`, `get_rubric`, `get_reference_answer`), each a
one-line delegation to an already-tested `loop/tools.py` function. `loop/research/mcp_client.py`
— `load_mcp_tools()` builds a `MultiServerMCPClient` from the new `settings.mcp_server_configs`
allow-list (empty by default) and returns its tools as `BaseTool`s; empty config short-circuits
to `[]` before importing `langchain_mcp_adapters` or spawning anything. `loop/nodes/research.py`
merges them into the Phase 9 ReAct agent's tool list: `tools=[search_web, *load_mcp_tools()]`.
New deps `mcp==1.28.1` + `langchain-mcp-adapters==0.3.0`. `tests/test_mcp.py` (12 tests) +
`tests/fixtures/mock_mcp_server.py` (a second tiny `FastMCP` fixture server, spawned as a real
subprocess). 254/254 tests, 0 lint errors. Docs: `docs/09-15` (MCP protocol, roles/primitives,
transports, wire protocol, FastMCP internals, the async event loop, security/bounded agency) +
`doc/phase-12-mcp-interop.md` (build walkthrough).

**Key decisions / lessons:**
- **Real bug found, not just a design choice made (CLAUDE.md rule #7 in action):** the PLAN
  framed the sync/async crossing as two equally-valid options. Empirically testing both showed
  only one works: `agent.invoke()` (the original Phase 9 call) raises `NotImplementedError:
  StructuredTool does not support sync invocation` the moment an MCP-loaded (async-only) tool is
  actually called, because LangGraph's `ToolNode` tries its sync path first and does not fall
  back to async; making `research()` itself `async def` raises `TypeError: No synchronous
  function provided` the instant the (sync-invoked) parent graph reaches it. Fixed by keeping
  `research()` sync but swapping its internal call to `asyncio.run(agent.ainvoke(...))` —
  verified this doesn't change Phase 9's existing behavior (a pure-sync tool list runs
  identically through `ainvoke()`).
- MCP-over-stdio is a **local subprocess**, not a network call — so unlike Tavily search or a
  live Bedrock call, it stays fully inside the laptop offline test gate. 12a tests call the
  `FastMCP` instance's own async methods in-process; 12b's `tests/fixtures/mock_mcp_server.py` is
  spawned as a genuine subprocess by the real `MultiServerMCPClient`, proving the whole wire
  protocol round-trips, not just the Python-level plumbing.
- `mcp_server_configs` follows the same "opt-in, feature-off by default" pattern as Phase 10's
  `fallback_model_id` and Phase 9b's `company` — with no servers configured, the research node's
  tool list stays byte-for-byte the Phase 9 flow.
- Full write-up: [`doc/phase-12-mcp-interop.md`](doc/phase-12-mcp-interop.md).

### Phase 11a+11b — 2026-07-18
**Built:** `loop/api.py` — `GET /sessions` (lists past sessions, newest first) and
`GET /sessions/{thread_id}/history` (full structured interview timeline for one session).
`loop/static/sessions.html` — list + detail view, same dark Tailwind theme as `index.html`;
`index.html` gained a "📋 History" header link. `tests/test_api.py` — 5 new tests (242/242
total, 0 lint errors).

**Key decisions / lessons:**
- LangGraph has **no built-in "list every thread" API** — a checkpointer only knows
  save/load for a thread_id you already have. `GET /sessions` queries the `SqliteSaver`'s
  own `checkpoints` table directly: `SELECT thread_id, MAX(checkpoint_id) ... GROUP BY
  thread_id`. `checkpoint_id` turned out to be a time-sortable UUID6 (verified by
  inspecting a live checkpoint row), so plain string `MAX()` gives the latest checkpoint
  per thread — no separate timestamp column needed for ordering. The actual ISO
  timestamp shown in the UI comes from `StateSnapshot.created_at` via `graph.get_state()`.
- `GET /sessions/{id}/history` reads only the **newest** `get_state_history()` snapshot —
  each node returns a delta, but the checkpoint stores the *merged* state, so the last
  snapshot already has every accumulated `answer`/`grade`/`weak_area` from the whole run.
  No need to replay the full history.
- Returns `{"sessions": [], "persistence": "none"}` on `MemorySaver` (the dev-box default
  when `DB_PATH` is unset) rather than a bare empty list — an honest signal, not a silent
  "no sessions yet."
- **Simplification vs. the PLAN.md sample:** dropped the per-session `weak_areas_after`
  breakdown (would require replaying every snapshot, not just the final one) in favor of
  one top-level `weak_areas` field on the detail response.
- Manually verified in-browser against the real `db/loop_state.sqlite` left over from an
  earlier phase's demo run — list view, detail view, and back-navigation all confirmed
  working with real (not synthetic) session data.
- Full write-up: [`doc/phase-11-session-history.md`](../doc/phase-11-session-history.md).

### Phase 10a+10b+10c — 2026-07-18
**Built:** `loop/models.py` — `with_resilience(chain, fallback_chain)` (retry + optional
fallback wrapped around a *finished* structured-output chain); `get_chat_model(model_id=None)`
gained a model-id override. `loop/guardrails.py` — `redact_pii()` + `detect_injection()`,
wired into `intake()` (JD/profile) and the interviewer answer gate; new `state["flagged_inputs"]`
field. `loop/budget.py` — `SessionBudget` + `BudgetCallbackHandler`, attached as a LangChain
callback at the top-level `graph.stream()` call in `api.py`; enforces `max_session_tokens`,
surfaces `tokens_used`/`cost_usd` in the SSE payload, stops a session gracefully on breach.
`loop/config.py` — `retry_max_attempts`, `fallback_model_id`, `max_session_tokens`,
`MODEL_PRICES_PER_1K`. `tests/test_models.py`, `tests/test_guardrails.py`, `tests/test_budget.py`
— 35 new tests (237/237 total at end of Phase 10, 0 lint errors).

**Key decisions / lessons:**
- **Deviated from PLAN.md's literal text** ("wrap the returned model with `.with_retry()`/
  `.with_fallbacks()`"). Verified directly against the installed `langchain-core==1.4.7`:
  `model.with_retry()` returns a `RunnableRetry` that does **not** proxy
  `.with_structured_output()` through to the wrapped model (`hasattr(..., "with_structured_output")`
  is `False`). Every node needs `.with_structured_output()`, so resilience must wrap the
  *finished chain* (`prompt | model.with_structured_output(Schema)`), not the bare model —
  going the other direction silently breaks every structured-output node.
- Guardrails **flag, not block**, suspected prompt injection — a human is already in the
  loop downstream (grader, plan approval, readiness gate), so flagging preserves the
  "model proposes, human disposes" contract instead of risking a false-positive silently
  dropping a legitimate answer.
- Budget tracking is a **LangChain callback (`on_llm_end`)** attached once at the
  `api.py` graph-invoke level — not code added to every node. `on_llm_end` fires for
  every model call regardless of which node made it (same mechanism `get_langfuse_callback()`
  already used), so zero changes were needed in `grader.py`/`coach.py`/`planner.py`/
  `readiness.py` to track their cost.
- Known simplification: the budget callback prices every call against the primary
  `model_id` (can't reliably tell which model served a call from inside a callback) —
  matters only if a Phase 10a fallback actually fires; documented, not silently assumed.
- Full write-up: [`doc/phase-10-production-hardening.md`](../doc/phase-10-production-hardening.md).

### Phase 8a — 2026-06-29
**Built:** `loop/embeddings.py` — `get_embeddings()` factory returning `BedrockEmbeddings(model_id, region_name)`;
mirrors the model factory seam from Phase 0 (`get_chat_model()`). `bedrock_embed_model_id` added to `loop/config.py`
(default: `amazon.titan-embed-text-v2:0`). `loop/retrieval.py` — lazy-initialised `InMemoryVectorStore` singleton;
`_build_index()` reads `fixtures/questions.json`, wraps each question as a `Document` (page_content = title + prompt +
topic; metadata = id/modality/topic/difficulty + `_source` for zero-cost retrieval), embeds via `get_embeddings()`.
`retrieve_questions(query, modality, k)` applies a `Callable[[Document], bool]` filter (not a dict — verified against
langchain-core==1.4.7) before top-k cosine ranking. `fixtures/questions.json` expanded from 6 → 24 questions
(8 per modality, varied topics + difficulties). `fixtures/rubrics.json` expanded to 24 matching rubrics.
`tests/test_retrieval.py` — 15 offline tests using `DeterministicFakeEmbedding(size=256)` monkeypatched via the
factory seam; covers index build, top-k constraint, modality filter correctness, result shape, and deterministic ranking.
173/173 tests, 0 lint errors.

**Key decisions / lessons:**
- `InMemoryVectorStore.similarity_search` accepts `filter` as `Callable[[Document], bool]`, NOT a dict.
  Must pass a lambda: `filter=lambda doc: doc.metadata.get("modality") == modality`.
- Lazy singleton (`_vector_store = None` → built on first `retrieve_questions()` call) allows tests to
  monkeypatch `get_embeddings` THEN call `_build_index()` to rebuild with the fake — no import-order race.
- `DeterministicFakeEmbedding` is hash-based: identical text → identical vector → cosine = 1.0.
  The exact-match retrieval test reconstructs the document's `page_content` string precisely to exploit this.
- `BedrockEmbeddings` all credential fields are `Optional` — boto3 credential chain picks up
  `AWS_BEARER_TOKEN_BEDROCK` automatically, same as `ChatBedrockConverse`.

### Phase 9a — 2026-07-06
**Built:** `loop/config.py` — `tavily_api_key` (empty = keyless DuckDuckGo default) and
`research_max_iterations` (agent iteration bound). `loop/research/search.py::web_search(query, k)`
— provider-switch seam mirroring `models.py`/`embeddings.py`; `_duckduckgo_search` via `ddgs`
(the maintained successor to `duckduckgo-search`), `_tavily_search` raises `NotImplementedError`
as a deliberate v2 seam. `loop/research/tools.py::search_web` — `@tool`-wrapped, formats results
as readable text for the model. `loop/schemas.py::CompanyResearch`. `loop/state.py` — `company`
and `company_research` fields + `initial_state()` update. `fixtures/sample_company.txt` ("Stripe").
`pyproject.toml` — added `ddgs>=9.14.4` via `uv add`. `tests/test_research.py` — 11 offline tests;
stubs `ddgs.DDGS` directly (never touches the network) plus dedicated tests for the Tavily seam,
the tool wrapper, schema validity, and state defaults. 189/189 tests, 0 lint errors.

**Key decisions / lessons:**
- **Deviated from PLAN.md's literal text:** the plan specified `langgraph.prebuilt.create_react_agent`,
  but the installed `langgraph-prebuilt==1.1.0` docstring flags it as deprecated in favor of
  `langchain.agents.create_agent` (present in the already-installed `langchain==1.3.9`). Verified
  both signatures directly (`inspect.signature` + `help()`) before choosing — `create_agent`'s
  `response_format=` writes to `state["structured_response"]`, an equivalent mechanism. This is
  exactly the scenario CLAUDE.md rule #7 anticipates: plans go stale against fast-moving libraries;
  verify installed APIs, don't trust memorized or previously-written text.
- `duckduckgo-search` was renamed to `ddgs` on PyPI — installing the old name would have pulled a
  deprecated package. Confirmed `DDGS().text(query, max_results=k)` with a live call before locking
  in the interface (dev environment has network access; the laptop test gate does not need it).
- The web search seam is the ONE place in the whole v1 project that reaches live external data —
  same "server activity, not laptop test gate" treatment as live Bedrock calls. Tests stub `ddgs.DDGS`
  itself, not just `web_search`, to prove the seam's internals are also correctly wired.
- Tavily gets the identical "seam exists, raises NotImplementedError" treatment as the Ollama model
  provider in `models.py` — consistent v2-seam pattern across the whole codebase.

### Phase 9b — 2026-07-06
**Built:** `loop/nodes/research.py::research(state)` — builds a `create_agent` ReAct sub-agent
(model + `search_web` tool + `response_format=CompanyResearch`), invokes it with
`recursion_limit=settings.research_max_iterations`, and returns
`{"company_research": CompanyResearch.model_dump()}` from `result["structured_response"]`.
`loop/graph.py` — `research` node registered; `_route_after_intake(state)` conditional edge
(`"research"` if `state["company"]` truthy, else `"planner"`); `research → planner` fixed edge;
`intake()` unchanged (still never sets `company`). `_run_session_with_hitl()` gained an optional
`company` param; `main()` demonstrates a third session with `company` read from
`fixtures/sample_company.txt` ("Stripe") to exercise the research path live. `loop/nodes/planner.py`
— `_format_company_research()` helper + new `{company_research}` prompt section, with
`_NO_COMPANY_RESEARCH_TEXT` placeholder when absent. 13 new tests in `tests/test_research.py`
(routing, node internals via a stubbed `create_agent`, iteration-bound assertion, tool-registration
assertion, planner prompt-capture tests) + `research` added to `test_graph.py`'s node-registration
assertion. 202/202 tests, 0 lint errors. **Phase 9 (Tool-calling research agent) is now fully
complete — 9a, 9b.**

**Key decisions / lessons:**
- **The offline-safety design is the main lesson of this sub-step:** rather than have `intake()`
  set `company` from a fixture (which would make EVERY existing test — `test_hitl.py`,
  `test_multisession.py`, `test_api.py`, `test_evals.py` — silently attempt a real ReAct/Bedrock/
  network call the moment the conditional edge was added), `company` stays `None` by default from
  `initial_state()`, and only the demo runner opts in explicitly. This meant zero of the ~30
  pre-existing tests needed new stubs for the research node — verified by running the full suite
  immediately after wiring the graph, before writing a single new test.
- Testing a real ReAct tool-calling loop offline would require a fake chat model that emits
  tool-call messages in sequence — disproportionate effort for testing OUR wiring code rather than
  `create_agent`'s internals (a well-tested library function, not our code). Instead, `create_agent`
  itself is monkeypatched to return a minimal fake object with just `.invoke(input, config)`,
  which is enough to verify: the tool list passed in, the message content built from `company`,
  and the `recursion_limit` propagated from config — the actual contract our node is responsible for.
- `create_agent`'s `response_format=<PydanticModel>` writes the validated instance to
  `result["structured_response"]` — confirmed via `typing.get_type_hints(AgentState)` on the
  installed `langchain==1.3.9`, not from memory.

### Phase 8c — 2026-06-29
**Built:** `fixtures/reference_answers.json` — one short reference/model answer per question
(24 entries, keyed by question_id). `loop/tools.py::get_reference_answer(question_id)` — exact
lookup (same DAO pattern as `get_rubric`); returns `None` if missing rather than raising, since
the reference is optional grounding, not a hard requirement like the rubric. `loop/nodes/grader.py`
— `_HUMAN` prompt template gains a `{reference_answer}` section; `_NO_REFERENCE_TEXT` placeholder
used when a question has no reference; system prompt instructs the model to grade against the
reference + rubric rather than its own unaided judgment. 2 new tests in `TestGrader`: one captures
the formatted `ChatPromptValue` via `.to_string()` and asserts the retrieved reference text is
present in it; one confirms grading still succeeds with `get_reference_answer` returning `None`.
178/178 tests, 0 lint errors. **Phase 8 (Retrieval/RAG) is now fully complete — 8a, 8b, 8c.**

**Key decisions / lessons:**
- 8c's "retrieval" is deliberately NOT a semantic/embedding search — the grader always knows the
  exact `question_id` it's grading, so an exact dict lookup (like `get_rubric`) is correct and
  simpler than forcing a vector search where there's no ambiguity to resolve by similarity.
  The "RAG" framing in PLAN.md refers to grounding the grader in retrieved evidence broadly, not
  literally requiring cosine similarity at every retrieval step.
- `ChatPromptValue.to_string()` (verified on langchain-core==1.4.7) is the clean way to inspect
  what a `ChatPromptTemplate | model` chain actually sent to the model in a test — capture the
  prompt value in the stubbed model's lambda before returning the canned structured output.
- The reference answer is optional grounding: missing an entry degrades to "grade against rubric
  alone" rather than raising, keeping the fixture bank easy to extend incrementally.

### Phase 8b — 2026-06-29
**Built:** `loop/tools.py::search_questions(query, modality, k)` — thin v2-stable wrapper
delegating to `loop.retrieval.retrieve_questions`; kept `get_questions_by_modality` for
back-compat (still used to compute pool size). `loop/state.py` — added `current_focus` and
`current_topics` fields. `loop/graph.py::session_router` — now also returns
`current_focus`/`current_topics` from the active session. `loop/nodes/interviewers.py::_ask_question`
— builds a query from `current_focus` (+ topics, joined), calls `search_questions(query, modality,
k=pool_size)` to get the FULL modality pool ranked by relevance, then applies the existing
answered-skip logic on the ranked list (so no-repeat behavior survives the switch from fixture
order to semantic order). `tests/conftest.py` — new global `stub_embeddings` autouse fixture
(patches `get_embeddings` → `DeterministicFakeEmbedding`, rebuilds the index) so every test that
runs a real interviewer node through the graph (`test_hitl.py`, `test_multisession.py`,
`test_api.py`, ...) stays offline, not just `test_retrieval.py`. 4 new tests in
`tests/test_phase3.py::TestInterviewers` verifying focus-driven selection, no-focus fallback, and
answered-skip-under-semantic-ranking. 176/176 tests, 0 lint errors.

**Key decisions / lessons:**
- `k = pool_size` (not a fixed small k) for the interviewer's `search_questions` call — ranking
  needs the FULL modality pool so the answered-skip filter has something to fall through to;
  a small k like 1 or 3 would silently exclude unanswered questions that just didn't rank in
  the top-k, causing incorrect fallback-to-repeat behavior.
- The embeddings offline-test fixture had to move from being local to `test_retrieval.py` to a
  global `conftest.py` autouse fixture the moment a production code path (interviewers) started
  calling `get_embeddings()` — any test exercising the real graph would otherwise attempt a live
  Bedrock call, breaking the laptop test gate silently until first run without AWS creds.
- Exact-text reconstruction (`f"{title}. {prompt} Topic: {topic}"`) is the reliable way to force
  `DeterministicFakeEmbedding` to rank a specific question first in tests — same technique as
  Phase 8a's `test_known_question_retrieved_by_exact_text`.

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
