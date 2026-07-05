# Top-k Retrieval and Metadata Filtering

## Top-k retrieval

After computing cosine similarity between the query vector and every stored document vector, the vector store returns the **k most similar documents** — the top-k.

```python
results = vector_store.similarity_search("explain binary search", k=3)
# → returns the 3 questions most semantically similar to "explain binary search"
```

`k` is a hyperparameter you control:

| k | Trade-off |
|---|---|
| 1 | Fastest, highest precision, no alternatives if the top match is wrong |
| 3 | Good default — gives the caller options; used in `retrieve_questions` |
| 10+ | Higher recall, but more noise; useful when you'll re-rank downstream |

In this project, `k=3` by default and `k=1` when the interviewer needs exactly one question to ask.

## Why not just return everything above a similarity threshold?

A threshold like "return all documents with similarity > 0.7" is brittle:

- Thresholds are sensitive to the embedding model and data distribution — what's "high" for one model is "medium" for another.
- A fixed threshold can return 0 results (nothing passes) or 50 results (everything passes).
- Top-k always returns exactly k results (or fewer if the store has fewer documents), making caller behavior predictable.

In practice, top-k + metadata filtering is the standard approach.

## Metadata filtering

Vector stores let you attach arbitrary key-value metadata to each document. Filtering applies a hard constraint **before** the similarity ranking: only documents matching the filter are considered.

```python
# Without filter: searches all 24 questions
results = vector_store.similarity_search("explain binary search", k=3)

# With filter: searches only coding questions (≈8 of the 24)
results = vector_store.similarity_search(
    "explain binary search",
    k=3,
    filter={"modality": "coding"}
)
```

Think of it as a SQL `WHERE` clause applied before `ORDER BY similarity DESC LIMIT k`:

```sql
SELECT * FROM questions
WHERE modality = 'coding'            -- metadata filter applied first
ORDER BY cosine_similarity DESC      -- then rank by similarity
LIMIT 3;                             -- then take top-k
```

## Why filter before ranking?

Without the modality filter, a query for "explain binary search" might return:
- "Walk me through a binary search implementation" ✅ (coding — correct)
- "Tell me about a time you made a tough technical decision" ❌ (behavioral — wrong)
- "Design a URL shortening service" ❌ (system_design — wrong)

The behavioral answer might score high because the word "technical" is vaguely related. The filter prevents cross-modality contamination entirely — it's a hard constraint, not a soft signal.

## In this project

```python
# loop/retrieval.py
def retrieve_questions(
    query: str,
    modality: str | None = None,
    k: int = 3,
) -> list[dict]:
    filter_dict = {"modality": modality} if modality else None
    docs = _vector_store.similarity_search(query, k=k, filter=filter_dict)
    # convert Documents back to the original question dicts
    return [doc.metadata["_source"] for doc in docs]
```

`modality` is optional: pass `None` to search across all modalities (useful for free-form queries). Pass `"coding"` / `"system_design"` / `"behavioral"` to constrain.

## Metadata attached to each document

```json
{
  "id": "sd-consistency-001",
  "modality": "system_design",
  "topic": "distributed systems",
  "difficulty": "hard",
  "_source": { ...the full original question dict... }
}
```

`_source` stores the full question dict so retrieval can return it without a second lookup. The other fields (`modality`, `topic`, `difficulty`) are filterable tags.

## Compound filters (v2 / future)

Most vector stores support compound filters:

```python
filter={"modality": "coding", "difficulty": "medium"}
```

This returns only coding questions of medium difficulty. The PLAN.md v2 seam (pgvector) would support this natively. `InMemoryVectorStore` in v1 supports it too — the filter dict is interpreted as AND.

## See also

- [03-vector-store.md](03-vector-store.md) — what the vector store is and how it's built
- [02-cosine-similarity.md](02-cosine-similarity.md) — how similarity is scored before top-k selection
- [06-rag-triad.md](06-rag-triad.md) — how retrieval fits the full pipeline
