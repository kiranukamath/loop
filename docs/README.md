# Loop — Concept Docs

Reference documentation for the concepts introduced in each phase of the Loop project.
Read these alongside `PLAN.md` and `CLAUDE.md`.

---

## Phase 8 — Retrieval / RAG

The seven concepts you need to understand before implementing semantic question retrieval.
Read them in order on first pass; use them as a reference after that.

| # | Doc | One-line summary |
|---|-----|-----------------|
| 1 | [Embeddings](01-embeddings.md) | Text → vector of numbers that preserves semantic similarity |
| 2 | [Cosine Similarity](02-cosine-similarity.md) | How "closeness" between two vectors is measured |
| 3 | [Vector Store](03-vector-store.md) | The index that stores (vector, text, metadata) and answers nearest-neighbour queries |
| 4 | [Top-k and Filtering](04-topk-and-filtering.md) | Controlling how many results come back and pre-filtering by metadata |
| 5 | [Chunking](05-chunking.md) | Why long documents must be split before embedding (and why we don't need it here) |
| 6 | [The RAG Triad](06-rag-triad.md) | Retrieve → Augment → Generate: the full pipeline and why it beats alternatives |
| 7 | [Embeddings Factory Seam](07-embeddings-factory-seam.md) | The provider-swappable `get_embeddings()` factory — same pattern as `get_chat_model()` |
| 8 | [End-to-End Flow](08-end-to-end-flow.md) | Full session trace: server startup → plan → interview → grading → verdict, with both embedding moments called out |

---

## Phase 12 — MCP & interoperability

What MCP is, the roles/primitives it defines, and how Loop plays both server
(12a) and client (12b) roles without duplicating any tool logic. Docs 9–11
are the concept-level pass (read these first); docs 12–15 are the deep dive
— verified against installed source, with the real bug Phase 12b hit traced
line-by-line.

| # | Doc | One-line summary |
|---|-----|-----------------|
| 9 | [What MCP is, and why](09-mcp-protocol.md) | The JDBC/LSP analogy — one protocol instead of N×M client/tool glue |
| 10 | [MCP roles and primitives](10-mcp-roles-and-primitives.md) | Host/client/server; tools vs. resources vs. prompts; `FastMCP` basics |
| 11 | [Transports and the LangChain bridge](11-mcp-transports-and-adapters.md) | stdio vs. HTTP/SSE; `langchain-mcp-adapters`; the sync/async boundary (resolved); bounded agency |
| 12 | [The MCP wire protocol](12-mcp-wire-protocol.md) | JSON-RPC 2.0, the `initialize` handshake, stdio framing, why stdout must stay protocol-only |
| 13 | [`FastMCP` internals](13-fastmcp-internals.md) | How `@mcp.tool()` builds a JSON schema from type hints (dynamic Pydantic model), docstring → description, the `call_tool()` return shape |
| 14 | [Async event loop deep dive](14-async-event-loop-deep-dive.md) | What an event loop is; both real stack traces from the sync/async bug, traced to the exact source lines; why the fix is provably safe |
| 15 | [MCP security & bounded agency](15-mcp-security-and-bounded-agency.md) | Tool poisoning/rug-pull, confused deputy, indirect prompt injection via tool output, the per-call session cost tradeoff |

---

## Phase 13 — Multi-agent orchestration

Crossing from "a fixed workflow with one dynamic node" into a real multi-agent system:
who owns control flow, the Send API for parallel fan-out/fan-in, Command handoffs and the
supervisor pattern, and the reducer + bounded-agency mechanics that make any of it safe.
Read 16 first (the concept reset), then 17–18 (the two new mechanisms), then 19 (the
internals + the safety guarantee).

| # | Doc | One-line summary |
|---|-----|-----------------|
| 16 | [Fixed workflow vs. multi-agent](16-fixed-workflow-vs-multiagent.md) | Who decides the next step — the graph's topology, or a model's judgment — and the backend analogy for each |
| 17 | [The Send API: map-reduce](17-send-api-map-reduce.md) | `Send(node, arg)` replaces state, not merges with it; the fan-in barrier runs once; a node can fan out too |
| 18 | [Command handoffs & the supervisor pattern](18-command-handoffs-and-supervisor-pattern.md) | `Command(goto=...)`, the no-static-edge rule, `plan_approval` as the handoff Loop already had, `interview_supervisor` as the generalization |
| 19 | [Reducers deep dive + bounded agency](19-reducers-deep-dive-and-bounded-agency.md) | Why `{"x": []}` doesn't clear a channel, the reset-sentinel reducer, and the two-layer termination guarantee (own bound + `recursion_limit` backstop) |

---

## Quick reading guide

**First time through Phase 8:** read 1 → 2 → 3 → 6 → 7. Skip 4 and 5 on first read.

**Before writing `retrieval.py`:** re-read 3 and 4.

**Before writing tests:** re-read 7 (the factory seam section on `DeterministicFakeEmbedding`).

**Interview prep (explaining RAG):** read 6, then 1, then 2.

**First time through Phase 12 (concepts):** read 9 → 10 → 11 in order — each
one builds on the last (protocol → roles/primitives → transports/safety).

**Going deeper on Phase 12 (mechanics + the real bug):** read 12 → 13 → 14 →
15 after 9–11. 14 is the one to read closely before touching any code that
mixes sync LangGraph nodes with async tool sources — it's a real failure
mode, not a hypothetical.

**First time through Phase 13:** read 16 → 17 → 18 → 19 in order. 17 and 18 are
the two mechanisms `loop/nodes/panel.py` and `loop/nodes/supervisor.py` are
each built from; 19 is what makes either one safe to actually ship.

---

## Other docs in this directory

- [commands.md](commands.md) — common shell commands for running, testing, and linting Loop
