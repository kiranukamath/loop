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

## Quick reading guide

**First time through Phase 8:** read 1 → 2 → 3 → 6 → 7. Skip 4 and 5 on first read.

**Before writing `retrieval.py`:** re-read 3 and 4.

**Before writing tests:** re-read 7 (the factory seam section on `DeterministicFakeEmbedding`).

**Interview prep (explaining RAG):** read 6, then 1, then 2.

---

## Other docs in this directory

- [commands.md](commands.md) — common shell commands for running, testing, and linting Loop
