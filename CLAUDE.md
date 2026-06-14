# CLAUDE.md — Loop

> Read this first in every session. It defines how to work on this repo, what Loop is,
> and where each phase's detailed instructions live (`PLAN.md`).

## What Loop is

Loop is an **agentic technical-interview coach** built to *teach* agentic AI patterns.
A user pastes a job description (JD) + their profile; Loop plans a prep curriculum, runs
mock interviews (coding / system design / behavioral), grades answers against rubrics,
remembers weak areas across sessions, and adapts the next session. A human approves the
prep plan and the final "ready / not ready" readiness verdict.

The project exists to exercise the **five agent capabilities**, one emphasized per phase:
**Planning, Orchestration, Memory, HITL (human-in-the-loop), Eval.**

## Working contract (NON-NEGOTIABLE — this is a learning project)

The owner is a **senior backend engineer (Java/Spring Boot, AWS)** who is **new to Python
LLM tooling and agentic AI**. They learn by building incrementally. Every session MUST:

1. **One phase at a time.** Never scaffold the whole repo at once.
2. **Explain before coding.** Open each phase with: what we're building, why it exists,
   the concepts involved, design choices + tradeoffs.
3. **Teach new concepts plainly.** When a LangGraph/LangChain/Langfuse idea first appears
   (state, node, edge, checkpointer, store, tool, structured output, callback, interrupt,
   reducer), explain it in plain language with a small analogy, *then* show it in our code.
   Assume deep backend knowledge, zero familiarity with these libraries.
4. **Walk through code after writing it** — file by file; explain non-obvious functions;
   comment the tricky parts in the code itself.
5. **Pause at end of each phase.** Summarize what was learned, update `PLAN.md` progress,
   then WAIT for explicit confirmation before the next phase.
6. **Clarity over cleverness.** Minimal, readable code. No premature abstraction.
7. **Verify library APIs** against installed versions / current docs — do NOT trust
   memorized LangGraph/Langfuse APIs; they change fast. Check `uv pip show <pkg>` and the
   installed package source / official docs before using an API.

If the owner says "go deeper" or "why" — expand the explanation before continuing.

## Tech stack & hard constraints

- **Python 3.14**, managed with **`uv`**. (System python is 3.14 — always use the uv venv.)
- **`ruff`** for lint+format. **`pytest`** for tests.
- **LangGraph** = agent graph. **LangChain** = models/prompts/tools. **Langfuse**
  (self-hosted via Docker) = tracing + eval.
- **Model:** default to **AWS Bedrock** via `langchain-aws` `ChatBedrockConverse`, behind a
  **swappable model factory** (`loop/models.py`). Ollama swap is **v2** — wire the seam
  now (factory + config), do NOT implement Ollama yet.
- **v1 uses static fixtures** in `fixtures/` (canned questions, rubrics, sample JD+profile).
  The goal is the agentic flow, not data plumbing.
- **Keep tool interfaces stable** so v2 can swap backends (Postgres, real question bank,
  Ollama) without touching the graph.

### Explicitly v2 / out of scope for v1
Real data sources, Postgres checkpointer/store, Ollama model, richer/dynamic question bank.
Do not build these in v1 unless the owner re-scopes.

## Repo layout (target — built incrementally, not all at once)

```
loop/
  pyproject.toml          # uv project, deps, ruff + pytest config
  .env.example            # required env vars (AWS, Langfuse keys) — never commit real .env
  README.md
  CLAUDE.md               # this file
  PLAN.md                 # phased plan + live progress tracker (UPDATE EVERY PHASE)
  docker-compose.yml      # self-hosted Langfuse (added when Phase 0 needs it)
  loop/                   # the package
    __init__.py
    config.py             # settings (pydantic-settings): model id, region, Langfuse keys
    models.py             # swappable model factory -> ChatBedrockConverse (Ollama seam)
    observability.py      # Langfuse client/callback wiring
    state.py              # LoopState (Phase 1+)
    graph.py              # graph assembly: nodes + edges + compile (Phase 1+)
    nodes/                # one file per node/sub-agent (Phase 2+)
      planner.py          # Phase 2
      interviewers.py     # Phase 3 (coding / system-design / behavioral)
      grader.py           # Phase 3
      coach.py            # Phase 3 (feedback)
    schemas.py            # Pydantic models for structured output (PrepPlan, Grade, ...)
    memory.py             # checkpointer + store wiring (Phase 4)
    tools.py              # question-bank / rubric tools over fixtures (stable interface)
  fixtures/               # static JD, profile, questions, rubrics (Phase 0)
  evals/                  # Langfuse datasets + eval scripts (Phase 6)
  tests/                  # pytest, mirrors loop/
```
Treat this as the destination. Each phase adds only the files it needs.

## Commands (fill in / verify during Phase 0)

```bash
uv sync                       # install deps from pyproject/uv.lock
uv run python -m loop.<...>   # run a module in the venv
uv run ruff check . && uv run ruff format .
uv run pytest
docker compose up -d          # start self-hosted Langfuse (once compose exists)
```

## Conventions

- Config via `pydantic-settings`, read from env / `.env`. Never hardcode model ids,
  regions, or keys. Never commit secrets.
- All LLM calls go through `loop/models.py` — nodes never construct a model directly.
- All LLM calls are traced via Langfuse from Phase 0 onward (observability from day one).
- Structured output via Pydantic schemas in `loop/schemas.py` + `.with_structured_output()`.
- Tools read from `fixtures/` in v1 but expose a v2-stable signature.
- Tests are fast and deterministic; mock/stub the LLM where the test isn't about the model.

## Where to get phase instructions

`PLAN.md` holds the detailed, self-contained plan and the live progress tracker. Before
starting work, read the relevant phase section there — it lists the goal, concepts to
teach, the task checklist, files touched, the "done when" bar, and the skills needed.
After finishing a phase, update its status and the changelog in `PLAN.md`.
