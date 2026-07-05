# End-to-End Flow: Where Embeddings Fit in Loop

This doc traces a full interview session from server startup to readiness verdict,
showing exactly where the **embedding model** (Titan v2) and the **chat model**
(Claude Haiku) are each called, and what happens in between.

**The one-line summary:** embeddings run twice — once to index the question bank
at startup, once per session to pick a relevant question. Everything else (asking,
grading, coaching, verdicts) is the chat model.

---

## The two embedding moments

```
MOMENT 1 — at process startup (once, ~24 calls)
    Every question in fixtures/questions.json is embedded and stored in RAM.
    This is the "index build" — loop/retrieval.py::_build_index().

MOMENT 2 — at query time (once per interview session, per modality)
    The session's `focus` text is embedded.
    Compared via cosine similarity against the 24 stored vectors.
    The most similar question is returned.
```

Two model types are involved, never confused with each other:

| | Embedding model (Titan v2) | Chat model (Claude Haiku) |
|---|---|---|
| Role | Finds the right question by *meaning* | Phrases, asks, grades, coaches |
| Input → Output | text → 1536 floats | text → text |
| Called via | `loop/embeddings.py::get_embeddings()` | `loop/models.py::get_chat_model()` |
| Deterministic? | Yes — same text, same vector, always | No — generative, varies by prompt |

---

## Phase 1 — Process starts (EMBEDDING MOMENT 1)

```
uv run uvicorn loop.api:app --reload
        ↓
FastAPI starts
        ↓
loop/retrieval.py is imported
        ↓
_build_index() runs
        ↓
reads fixtures/questions.json  (24 questions)
        ↓
for each question:
    page_content = "title. prompt. Topic: topic"
    calls BedrockEmbeddings.embed_query(page_content)
    → POST to AWS Bedrock Titan v2 (an "inference" call, not a chat completion)
    → returns [0.12, -0.34, 0.88, ...]  (1536 floats)
        ↓
InMemoryVectorStore stores:
    {
      "uuid-1": (vector_for_cod001, Document(page_content, metadata)),
      "uuid-2": (vector_for_cod002, Document(page_content, metadata)),
      ...24 entries total...
    }
        ↓
Server is ready. Vectors stay in RAM for the process lifetime.
```

Analogy: like a Spring Boot `@PostConstruct` — runs once at startup, populates
an in-memory structure. The vector store is a `HashMap<String, float[]>` that
also knows how to do cosine math.

---

## Phase 2 — User opens the browser

```
Browser → GET http://localhost:8000
        ↓
FastAPI serves loop/static/index.html
        ↓
User sees: JD text box + Profile text box + Start button
```

---

## Phase 3 — User submits JD + profile

```
Browser → POST /sessions
          body: { jd: "...", profile: "..." }
        ↓
loop/api.py: creates a thread_id (UUID), builds initial state:
    {
      jd: "Senior Backend Engineer at Stripe...",
      profile: "5 years Java/Spring Boot, AWS...",
      session_index: 0,
      messages: [],
      plan: None,
      ...
    }
        ↓
graph.stream(initial_state, config={thread_id: "abc123"})
        ↓
LangGraph begins executing nodes...
```

---

## Phase 4 — intake → planner

```
NODE: intake
    Loads JD + profile into state.
    Checks InMemoryStore for stored weak areas from past sessions.
        ↓
NODE: planner
    Builds a prompt:
        "Given this JD and profile, create a 3-session prep plan..."
        + any stored weak areas from memory
        ↓
    Calls get_chat_model()  (Claude Haiku via Bedrock)
        → LLM generates a PrepPlan (structured output):
            sessions: [
              { modality: "coding",        focus: "algorithms and complexity",  topics: [...] },
              { modality: "system_design", focus: "distributed consistency",    topics: [...] },
              { modality: "behavioral",    focus: "ownership and conflict",     topics: [...] },
            ]
        ↓
    PrepPlan stored in state["plan"]
        ↓
INTERRUPT: plan_approval
    Graph pauses. Returns interrupt payload to FastAPI.
    FastAPI sends the PrepPlan to the browser as JSON.
```

---

## Phase 5 — User reviews and approves the plan

```
Browser shows: Gate 1 — PrepPlan review card
User clicks Approve (or edits topics)
        ↓
Browser → POST /sessions/abc123/resume
          body: { action: "approve", data: {} }
        ↓
graph resumes with Command(resume={"action": "approve"})
        ↓
NODE: session_router
    reads state["plan"]["sessions"][state["session_index"]]
    → current session: { modality: "coding", focus: "algorithms and complexity" }
        ↓
    routes to: coding_interviewer
```

---

## Phase 6 — Interviewer selects a question (EMBEDDING MOMENT 2)

```
NODE: coding_interviewer
    reads session.focus = "algorithms and complexity"
    reads session.modality = "coding"
        ↓
    calls search_questions("algorithms and complexity", modality="coding", k=1)
        ↓
    loop/tools.py → loop/retrieval.py → retrieve_questions(...)
        ↓
    EMBEDDING MOMENT 2:
        calls BedrockEmbeddings.embed_query("algorithms and complexity")
        → POST to AWS Bedrock Titan v2
        → returns query_vector = [0.45, -0.12, ...]  (1536 floats)
        ↓
    InMemoryVectorStore computes cosine similarity against all 8 coding vectors:
        cos(query_vector, vector_for_cod001) = 0.71   ← sliding window
        cos(query_vector, vector_for_cod002) = 0.68   ← thread-safe queue
        cos(query_vector, vector_for_cod003) = 0.91   ← binary search  ✓ highest
        cos(query_vector, vector_for_cod004) = 0.74   ← tree LCA
        cos(query_vector, vector_for_cod005) = 0.88   ← dynamic programming
        cos(query_vector, vector_for_cod006) = 0.79   ← graphs BFS/DFS
        cos(query_vector, vector_for_cod007) = 0.65   ← merge intervals
        cos(query_vector, vector_for_cod008) = 0.70   ← fast power
        ↓
    Returns: [question cod-003: "Binary search and its variants"]
    (the modality filter already excluded all system_design/behavioral
     questions BEFORE ranking — see 04-topk-and-filtering.md)
        ↓
    coding_interviewer calls get_chat_model() (Claude Haiku):
        "You are a coding interviewer. Ask this question to the candidate:
         {cod-003 prompt}"
        → LLM phrases the question naturally
        ↓
INTERRUPT: answer_question
    Graph pauses. Returns the question text to the browser.
```

---

## Phase 7 — Candidate answers

```
Browser shows: Gate 2 — question + textarea
User types their answer, presses Ctrl+Enter
        ↓
Browser → POST /sessions/abc123/resume
          body: { action: "answer", data: "I would implement binary search using..." }
        ↓
graph resumes with the candidate's answer text
```

---

## Phase 8 — Grader scores the answer

```
NODE: grader
    reads state["answers"][-1]  (the candidate's answer)
    reads the question id from state
    calls get_rubric("cod-003")  → loads rubric from rubrics.json
        ↓
    calls get_chat_model() (Claude Haiku):
        "Grade this answer against the rubric.
         Rubric criteria: correctness (4pts), complexity_analysis (3pts), ...
         Candidate answer: I would implement binary search using..."
        → structured output: Grade(score=8, max_score=10, feedback="...", criteria_scores={...})
        ↓
    Grade stored in state["grades"]
```

---

## Phase 9 — Coach gives feedback, session advances

```
NODE: coach
    reads state["grades"][-1]
    calls get_chat_model():
        "Turn this grade into actionable coaching feedback..."
        → Feedback(summary="...", weak_areas=["complexity analysis", ...])
        ↓
    Writes weak_areas to InMemoryStore (long-term memory):
        store.put(("user", "default"), "weak_areas", ["complexity analysis", ...])
        ↓
NODE: advance_session
    state["session_index"] += 1
        ↓
_route_after_session:
    session_index < total sessions? → back to session_router (next session)
    session_index == total?          → go to readiness
```

---

## Phase 10 — Remaining sessions repeat Phases 6–9

Same flow, different modality — and a fresh embedding query each time. For
session 2 (system_design, focus="distributed consistency"):

```
retrieve_questions("distributed consistency", modality="system_design", k=1)
        ↓
cosine similarity among the 8 system_design vectors only
        ↓
returns: sys-003 "CAP theorem and consistency trade-offs"  ← semantically closest
```

No new questions are embedded here — the 24 question vectors were already
computed once at startup (Moment 1). Only the query text gets embedded fresh
each time (Moment 2).

---

## Phase 11 — Readiness verdict

```
NODE: readiness
    reads all grades from state["grades"]
    calls get_chat_model():
        "Based on these 3 session grades, is this candidate ready?"
        → ReadinessVerdict(verdict="ready", confidence=0.78, reasoning="...")
        ↓
INTERRUPT: approve_verdict
    Browser shows Gate 3 — verdict card
    User clicks Approve or Override
        ↓
    Graph ends. Session complete.
```

---

## Where the embedding vectors physically live

```
┌─────────────────────────────────────────────────────┐
│                  RAM (process memory)               │
│                                                     │
│  InMemoryVectorStore._store = {                     │
│    "uuid-1": (                                      │
│       vector=[0.12, -0.34, 0.88, ...],  ← 1536 f32 │
│       Document(                                     │
│         page_content="Binary search...",            │
│         metadata={                                  │
│           id: "cod-003",                            │
│           modality: "coding",                       │
│           topic: "algorithms",                      │
│           difficulty: "easy",                       │
│           _source: { full question dict }           │
│         }                                           │
│       )                                             │
│    ),                                               │
│    ... 23 more entries ...                          │
│  }                                                  │
│                                                     │
│  Rebuilt on every process restart (in-memory only). │
│  v2: this moves to pgvector (persisted on disk).    │
└─────────────────────────────────────────────────────┘
```

---

## The complete data flow in one picture

```
questions.json
      │
      │ at startup (Moment 1 — 24 embed calls)
      ▼
BedrockEmbeddings (Titan v2)  ──►  InMemoryVectorStore  (stays in RAM)
                                          │
                                          │ at query time (Moment 2 — 1 embed call/modality)
                    session.focus ──►  embed_query()
                                          │
                                          ▼
                                    cosine similarity
                                          │
                                          ▼
                                   best question dict
                                          │
                                          ▼
                               coding_interviewer node
                                          │
                                          ▼
                               Claude Haiku (chat model)
                                          │
                                          ▼
                               question shown to user
```

Two separate models, two separate purposes:
- **Titan v2** (embedding model) — finds the right question by meaning
- **Claude Haiku** (chat model) — phrases it, grades it, gives feedback

## See also

- [06-rag-triad.md](06-rag-triad.md) — the retrieve → augment → generate pattern this flow implements
- [07-embeddings-factory-seam.md](07-embeddings-factory-seam.md) — how `get_embeddings()` decides which provider handles Moment 1 and Moment 2
- [03-vector-store.md](03-vector-store.md) — the lifecycle of the `InMemoryVectorStore` shown here
