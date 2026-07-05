# Embeddings

## What is an embedding?

An embedding is a function that converts text (or any data) into a **fixed-length list of floating-point numbers** called a **vector**. The key property: **similar meanings produce similar vectors**.

```
embed("database indexing")       → [0.12, -0.34, 0.88, ...]   # 1536 numbers
embed("B-tree index structure")  → [0.11, -0.31, 0.90, ...]   # very close
embed("behavioral interview")    → [-0.72, 0.55, -0.14, ...]  # far away
```

This is the opposite of a hash function like MD5:

| | Hash (MD5) | Embedding |
|---|---|---|
| Similar inputs | Completely different output | Similar (nearby) output |
| Purpose | Uniqueness / fingerprint | Similarity / meaning |
| Output size | Fixed (128 bits) | Fixed (e.g. 1536 floats) |
| Property | Destroys similarity | **Preserves similarity** |

## Java analogy

Think of it as a `Comparator<String>` that understands semantics rather than lexicographic order — except instead of returning `-1 / 0 / 1`, it returns a **point in space**. Two strings with similar meanings land close together; unrelated strings land far apart.

## How are embedding models trained?

You don't need to train one — you call a pre-trained model (Amazon Titan Embeddings v2 in this project). But understanding the training helps build the mental model:

The model is trained on massive text corpora with a contrastive objective: pairs of sentences known to be related (a question and its answer, two paraphrases) are pushed together in vector space; unrelated pairs are pushed apart. After training, the geometry of the space encodes semantic relationships.

## Dimensionality

Titan Embeddings v2 produces **1536-dimensional** vectors. You can think of each dimension as a learned "semantic axis" — though they're not human-interpretable. What matters is that the full 1536-number fingerprint captures enough about the meaning of a sentence to measure similarity reliably.

## What embedding is NOT

- It is **not** keyword matching. "DB index" and "B-tree" will be close even with no shared tokens.
- It is **not** perfect. Embeddings are approximate. Rare domains, code, or highly technical jargon can fool them.
- It is **not** lossless. You cannot reconstruct the original text from its embedding.

## In this project

```
loop/embeddings.py   # get_embeddings() factory → BedrockEmbeddings(Titan v2)
loop/retrieval.py    # calls get_embeddings() to build the question index at startup
tests/               # uses DeterministicFakeEmbedding instead (no network)
```

The embedding model is hidden behind a factory (`get_embeddings()`) — same pattern as `get_chat_model()`. Callers never construct `BedrockEmbeddings` directly, so swapping to a local model later requires changing one file.

## Embedding in action (pseudocode)

```python
embeddings = get_embeddings()

# At index-build time:
vector = embeddings.embed_query("What is a B-tree index?")
# → [0.12, -0.34, 0.88, ...]  (1536 floats, always the same for the same text)

# At query time:
query_vector = embeddings.embed_query("explain database indexing strategies")
# → [0.10, -0.30, 0.85, ...]  (close to the B-tree vector above)
```

## See also

- [02-cosine-similarity.md](02-cosine-similarity.md) — how we measure "closeness" between two vectors
- [03-vector-store.md](03-vector-store.md) — how we index and search vectors
- [06-rag-triad.md](06-rag-triad.md) — the full retrieve → augment → generate pipeline
