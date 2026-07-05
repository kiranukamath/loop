# The RAG Triad: Retrieve → Augment → Generate

## What is RAG?

**RAG** stands for **Retrieval-Augmented Generation**. It is an architecture pattern where a language model's response is grounded in documents retrieved at query time, rather than relying solely on what the model memorized during training.

```
User query
    ↓
[RETRIEVE]  — embed the query, search the vector store, get relevant documents
    ↓
[AUGMENT]   — inject the retrieved documents into the prompt as context
    ↓
[GENERATE]  — the LLM produces a response grounded in the retrieved context
    ↓
Response
```

## Step by step

### 1. Retrieve

```python
query = "distributed systems consistency and consensus"
results = retrieve_questions(query, modality="system_design", k=3)
# → [
#     {"title": "Design a distributed database with strong consistency", ...},
#     {"title": "Explain the CAP theorem trade-offs", ...},
#     {"title": "How does Raft achieve consensus?", ...},
# ]
```

The query is embedded, and the vector store returns the most semantically similar documents. No keywords had to match — "consistency and consensus" retrieved "CAP theorem" and "Raft" because they're semantically close.

### 2. Augment

The retrieved documents are inserted into the LLM's prompt:

```
System: You are a technical interviewer. Pick the most relevant question below
        for a candidate focusing on distributed systems consistency.

Retrieved questions:
1. Design a distributed database with strong consistency (hard)
2. Explain the CAP theorem trade-offs (medium)
3. How does Raft achieve consensus? (hard)

Session focus: {session.focus}
Candidate weak areas: {weak_areas}

Pick the best question and ask it.
```

The model now has **grounding** — it picks from real options rather than hallucinating a question.

### 3. Generate

The LLM reads the retrieved context and produces a response grounded in it:

```
"Let's talk about the CAP theorem. Given a distributed system that must handle
network partitions, how would you choose between consistency and availability,
and what real-world systems have you seen make different trade-offs here?"
```

The model adapted the retrieved question to the conversation, but it started from real data — not its training-time memory.

## Why RAG beats alternatives

### vs. Pure generation (no retrieval)

Without retrieval, the model generates from training memory:

- **Hallucination risk:** the model may invent plausible-sounding but wrong questions, outdated content, or questions that don't match the rubric format.
- **No update path:** when the question bank changes, you'd need to retrain or fine-tune the model.
- **No transparency:** you can't audit what the model "knows" about your question bank.

### vs. Fine-tuning

Fine-tuning bakes the question bank into model weights:

| Concern | Fine-tuning | RAG |
|---|---|---|
| Adding new questions | Re-train (expensive, slow) | Add a document to the store |
| Removing outdated questions | Re-train | Delete from the store |
| Cost | High (GPU training) | Low (just embedding + storage) |
| Transparency | Black box weights | You can inspect retrieved documents |
| Catastrophic forgetting | Risk of overwriting prior knowledge | Isolated from model weights |
| Hallucination | Still possible | Grounded in retrieved text |

RAG wins for a dynamic knowledge base (question banks change; rubrics evolve). Fine-tuning wins when you need to change the model's *behavior* (writing style, reasoning patterns) — not its *knowledge*.

### vs. Stuffing all documents in the prompt

You could skip retrieval and put all 24 questions in every prompt:

- Works at 24 questions. Fails at 2400 — context windows have limits and per-token costs.
- RAG scales; full-context stuffing doesn't.
- RAG is also more focused: the model reasons over 3 relevant documents, not 24 noisy ones.

## RAG in this project

**Phase 8a (current):** build the retrieval layer (`retrieve_questions`). No graph change yet.

**Phase 8b (next):** wire retrieval into the interviewer. Replace:

```python
# Before (Phase 3 — exact match):
question = get_questions_by_modality(session.modality)[0]

# After (Phase 8b — semantic retrieval):
question = search_questions(session.focus, session.modality, k=1)[0]
```

**Phase 8c (stretch):** RAG-grounded grading. Retrieve a reference answer for the question being graded and inject it into the grader prompt so grading is evidence-based:

```
Grade the candidate's answer against this reference:
[retrieved reference answer]

Candidate's answer: {answer}
```

## The RAG contract in Loop

```
Loop's RAG triad for question selection:

RETRIEVE  → retrieve_questions(session.focus, session.modality, k=3)
              ↑ InMemoryVectorStore + BedrockEmbeddings (Titan v2)
              ↑ DeterministicFakeEmbedding in tests

AUGMENT   → the interviewer node inserts retrieved questions into its prompt

GENERATE  → the interviewer LLM selects and phrases the question for the candidate
```

The graph never changes. The retrieval logic lives in `loop/retrieval.py`, called via `loop/tools.py`. The seam is stable — pgvector in v2 slots in without touching the graph or the nodes.

## See also

- [01-embeddings.md](01-embeddings.md) — how queries and documents become vectors
- [02-cosine-similarity.md](02-cosine-similarity.md) — how similarity is scored during retrieval
- [03-vector-store.md](03-vector-store.md) — where documents are indexed
- [04-topk-and-filtering.md](04-topk-and-filtering.md) — controlling what retrieval returns
- [07-embeddings-factory-seam.md](07-embeddings-factory-seam.md) — the provider-swappable embeddings seam
