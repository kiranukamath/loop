# Vector Store

## What is a vector store?

A vector store is a data structure (and its surrounding tooling) for storing, indexing, and **querying documents by semantic similarity**. It stores triples of:

```
(vector, text, metadata)
   ↑         ↑       ↑
embedding  original  arbitrary key-value tags
           text      (modality, topic, difficulty, id, ...)
```

You query it not with keywords but with a **query vector** — and it returns the documents whose stored vectors are most similar (by cosine similarity) to your query.

## Elasticsearch analogy

| Elasticsearch (keyword search) | Vector store (semantic search) |
|---|---|
| Builds an inverted index: token → document list | Builds an ANN index: vector → nearest document list |
| Query: tokenize → look up posting lists | Query: embed → find nearest neighbors |
| "database" matches docs containing that word | "database" matches docs about storage, indices, B-trees |
| Exact match, fast | Approximate match, semantic |

Both serve the same fundamental purpose — "find relevant documents fast" — but the definition of "relevant" is different. Keyword search is lexical; vector search is semantic.

## The two operations

### 1. Index / ingest

At startup, you embed every document and store `(vector, text, metadata)`:

```python
from langchain_core.documents import Document

docs = [
    Document(
        page_content="Explain the time complexity of quicksort",
        metadata={"id": "q1", "modality": "coding", "topic": "algorithms", "difficulty": "medium"}
    ),
    # ... more documents
]

vector_store = InMemoryVectorStore(embedding=get_embeddings())
vector_store.add_documents(docs)
```

For each document, the store calls `embed(page_content)` to get the vector, then stores `(vector, page_content, metadata)` together.

### 2. Query / retrieve

At query time, you embed the query and ask for the top-k nearest documents:

```python
results = vector_store.similarity_search(
    "how does merge sort work",
    k=3,
    filter={"modality": "coding"}   # metadata pre-filter
)
# → [Document(page_content="...", metadata={...}), ...]
```

The store embeds `"how does merge sort work"`, computes cosine similarity against every stored vector, applies the metadata filter, and returns the top-3.

## InMemoryVectorStore (what we use in v1)

`langchain_core.vectorstores.InMemoryVectorStore` is the simplest implementation:

- **Storage:** a plain Python list/dict in RAM
- **Search:** brute-force — computes cosine similarity against every stored vector
- **Cost:** O(n × d) per query, where n = number of documents, d = vector dimension
- **Good for:** n ≤ ~100,000 documents (our question bank has 24)
- **Zero dependencies:** no Docker, no FAISS, no Chroma — just Python

For a question bank of 24 documents, brute-force is indistinguishable from an optimized index. The performance difference only becomes visible at millions of documents.

## The v2 seam: pgvector

When the question bank grows (v2), we swap `InMemoryVectorStore` for `PGVector` (pgvector extension on Postgres). The swap happens in one place — `loop/retrieval.py` — and no caller code changes:

```python
# v1 (current):
vector_store = InMemoryVectorStore(embedding=get_embeddings())

# v2 (future):
vector_store = PGVector(embedding=get_embeddings(), connection_string=settings.pg_url, ...)
```

Both expose the same `similarity_search(query, k, filter)` interface. This is the same seam philosophy as `get_chat_model()` and `get_embeddings()`.

## Lifecycle in this project

```
Process starts
    ↓
loop/retrieval.py imports
    ↓
_build_index() runs once (module-level singleton)
    ↓
reads fixtures/questions.json
    ↓
wraps each question as a LangChain Document
    ↓
calls vector_store.add_documents(docs)
    ↓
InMemoryVectorStore embeds each doc, stores (vector, text, metadata)
    ↓
Index is ready — stays in memory for the process lifetime

Interviewer needs a question
    ↓
retrieve_questions("distributed systems consistency", modality="system_design", k=3)
    ↓
embeds query, cosine-searches, filters by modality
    ↓
returns list of matching question dicts
```

The index is rebuilt at every process restart (because it's in-memory). For 24 questions with Titan v2, this takes < 1 second. For v2 with pgvector, the index persists on disk.

## In tests

```python
# monkeypatch get_embeddings → DeterministicFakeEmbedding
# DeterministicFakeEmbedding produces deterministic (but semantically meaningless) vectors
# The store still builds, add_documents still works, similarity_search still ranks
# Just the rankings have no semantic meaning — which is fine for structural tests
```

## See also

- [01-embeddings.md](01-embeddings.md) — how document text becomes a vector
- [02-cosine-similarity.md](02-cosine-similarity.md) — how similarity is scored
- [04-topk-and-filtering.md](04-topk-and-filtering.md) — controlling k and metadata filters
- [06-rag-triad.md](06-rag-triad.md) — the vector store's role in the full RAG pipeline
