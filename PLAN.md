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
| 10 | Production hardening | resilience, safety, cost | ⬜ not started | — |
| 11 | Session history UI | reading checkpoint state / replay | ⬜ not started | — |

Status legend: ⬜ not started · 🟡 in progress · ✅ done & approved · ⏸️ blocked

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

## v2 / future enhancements (NOT in v1 — and NOT in the Phase 8–11 track)

> Phases 8–10 deliberately pull *semantic retrieval over fixtures* (8) and *real web search* (9)
> forward as learning extensions. The items below remain out of scope even after that track —
> they swap the *infrastructure* behind seams the project already establishes.

- **pgvector** behind the Phase 8 retrieval seam (replace `InMemoryVectorStore`).
- tool for web search
- Postgres checkpointer + store (swap the Phase 4 seam).
- Ollama / open-source model via the Phase 0 model factory seam.
- Real live JD ingest + a genuinely large external question bank (Phase 8 stays over fixtures).

---

## Changelog

> Append one entry per completed phase: date, phase, what was built, key decisions, what the
> owner learned. Keep newest at top.

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
