# Cosine Similarity

## The problem: how do you measure "closeness" between two vectors?

After embedding two texts, you have two lists of 1536 floating-point numbers. You need a single score that says "how similar are these two vectors?" There are several options:

| Distance metric | Measures | Problem for text |
|---|---|---|
| Euclidean distance | Straight-line distance between two points | Penalizes long documents — a long and short text about the same topic land far apart by magnitude |
| Dot product | Product of magnitudes × angle | Biased toward high-magnitude (long-text) vectors |
| **Cosine similarity** | **Angle between two vectors** | **Scale-invariant — works well for text regardless of length** |

## What is cosine similarity?

Cosine similarity measures the **cosine of the angle θ between two vectors**, regardless of their lengths.

```
cosine_similarity(A, B) = (A · B) / (|A| × |B|)

where:
  A · B  = dot product (sum of element-wise products)
  |A|    = magnitude of vector A (square root of sum of squares)
```

The result is always between -1 and +1:

```
cos(θ) =  1.0  → same direction  → identical meaning
cos(θ) =  0.0  → perpendicular   → unrelated
cos(θ) = -1.0  → opposite        → opposite meaning (rare in practice)
```

For most text pairs you'll see scores in the 0.6–1.0 range for related content and 0.0–0.4 for unrelated content.

## Why angle, not distance?

Imagine two vectors in 2D:

```
A = [1, 2]   (short version of "database indexing")
B = [3, 6]   (long version of the same concept, just scaled up)
```

Euclidean distance: `sqrt((3-1)² + (6-2)²) = sqrt(20) ≈ 4.5` — they look "far apart"

But they point in exactly the same direction: `B = 3 × A`. Same angle → same meaning.

Cosine similarity: `(1×3 + 2×6) / (sqrt(5) × sqrt(45)) = 15 / 15 = 1.0` — correctly identifies them as identical in meaning.

This is why cosine similarity is the standard for text: a long answer and a short answer about the same topic should be considered similar, even though the long one has a larger magnitude.

## Concrete example

```
query  = embed("how does database indexing work?")
doc_A  = embed("B-tree index speeds up SELECT queries")
doc_B  = embed("tell me about your leadership style")

cosine_similarity(query, doc_A)  →  0.91  (high — related)
cosine_similarity(query, doc_B)  →  0.12  (low — unrelated)
```

The vector store computes this score for every stored vector and returns the top-k highest scores.

## Java / Spring analogy

Think of cosine similarity as the scoring function inside a `Comparator<Document>`:

```java
// Conceptually what the vector store does at query time:
documents.stream()
    .map(doc -> Map.entry(doc, cosineSimilarity(queryVector, doc.vector())))
    .sorted(Map.Entry.<Document, Double>comparingByValue().reversed())
    .limit(k)
    .collect(toList());
```

You never write this yourself — `InMemoryVectorStore.similarity_search()` does it internally — but this is exactly what's happening.

## In this project

```python
# loop/retrieval.py — under the hood, InMemoryVectorStore uses cosine similarity
results = vector_store.similarity_search(query_text, k=3, filter={"modality": "coding"})
```

You configure `k` (how many results) and the metadata filter. The cosine scoring is automatic.

## See also

- [01-embeddings.md](01-embeddings.md) — what vectors are and how they're produced
- [03-vector-store.md](03-vector-store.md) — how the index is built and queried
- [04-topk-and-filtering.md](04-topk-and-filtering.md) — top-k retrieval and metadata filters
