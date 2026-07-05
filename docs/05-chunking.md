# Chunking

## What is chunking?

Chunking is the process of splitting a long document into smaller, overlapping pieces ("chunks") before embedding and indexing them.

```
Long document (5000 tokens)
        ↓
Chunk 1: tokens   0–500
Chunk 2: tokens 400–900    ← 100-token overlap with Chunk 1
Chunk 3: tokens 800–1300   ← 100-token overlap with Chunk 2
...
        ↓
Each chunk is embedded separately and stored as its own document
```

## Why does chunking exist?

Embedding models have a **context window limit** — they can only encode a fixed number of tokens at once (e.g. Titan v2 supports up to 8192 tokens). If your document is longer, you must split it.

But even within the limit, there's a more fundamental problem: **averaging over too much text loses local meaning**.

Imagine embedding an entire 20-page technical report. The resulting vector is an average of all the concepts in the report — databases, networking, security, deployment. A query about "database indexing" might still match this document, but less precisely than if you had indexed the specific section about databases as its own chunk.

Chunking solves this: each chunk is a focused unit of meaning, producing a more precise and retrieval-friendly vector.

## The overlap

Chunks overlap by a configurable number of tokens (typically 10–20% of chunk size). This prevents information from being lost at chunk boundaries:

```
Without overlap:
  Chunk 1: "The system uses a write-ahead log"
  Chunk 2: "to ensure durability after a crash."

With overlap:
  Chunk 1: "The system uses a write-ahead log to"
  Chunk 2: "write-ahead log to ensure durability after a crash."
```

Without overlap, the sentence is split between two chunks and neither chunk alone makes complete sense. With overlap, both chunks contain the full idea.

## Why this project does NOT chunk

Our question bank entries are short:

```json
{
  "title": "Binary search implementation",
  "prompt": "Implement binary search on a sorted array. Analyze time and space complexity.",
  "topic": "algorithms"
}
```

This is ~30 tokens. Well within the embedding model's context window, and short enough that a single vector captures the full meaning accurately.

**The rule of thumb:** if a document fits comfortably in 1–2 sentences (under ~200 tokens), embed it whole. Chunking adds complexity without benefit at that size.

In this project: **1 question = 1 document = 1 embedding = 1 vector store entry**.

## When you would need chunking (v2 / future scenarios)

| Scenario | Why chunking is needed |
|---|---|
| Ingest a full job description (3–5 pages) | Too long for a single meaningful embedding |
| Index engineering blog posts or docs | Each section has different content; averaging loses specificity |
| Reference answer corpus (paragraphs of explanation) | Each concept needs its own chunk to retrieve precisely |
| PDF resume parsing | Sections (experience, education, skills) should be separate chunks |

The LangChain `RecursiveCharacterTextSplitter` and `TokenTextSplitter` handle this — they're out of scope for this project's v1 but are the standard tools.

## Chunking strategy affects retrieval quality

Chunk too large → vectors are too general, retrieval is imprecise.  
Chunk too small → vectors lack context, retrieval returns fragments without enough meaning.  
Overlap too small → boundary information is lost.  
Overlap too large → redundant chunks, slower indexing, higher storage.

The "right" chunk size depends on the domain, the embedding model, and the downstream use. In practice, 400–600 tokens with 10–15% overlap is a common starting point.

## Summary

Chunking is a necessary preprocessing step for long documents, but it's transparent in this project because our fixture questions are already small enough to embed whole. Understanding it matters for:

- Explaining why RAG systems split documents before indexing (common interview question)
- Knowing when to add chunking if the question bank grows to include long reference materials
- Phase 8c (stretch goal): if reference answers are multi-paragraph, each answer should be chunked

## See also

- [01-embeddings.md](01-embeddings.md) — what embedding produces from each chunk
- [03-vector-store.md](03-vector-store.md) — where chunks (as Documents) are stored
- [06-rag-triad.md](06-rag-triad.md) — the full pipeline chunking feeds into
