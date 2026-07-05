# The Embeddings Factory Seam

## What is a factory seam?

A **factory seam** is a design pattern where a single function (the factory) owns the decision of which concrete implementation to instantiate. All callers ask the factory for "an embeddings model" — they never construct one directly. The factory reads configuration and decides which provider to use.

This mirrors the **model factory** already in `loop/models.py`:

```python
# loop/models.py — the model factory (Phase 0)
def get_chat_model() -> BaseChatModel:
    if settings.model_provider == "ollama":     # v2 seam
        raise NotImplementedError("Ollama swap is v2")
    return ChatBedrockConverse(
        model_id=settings.bedrock_model_id,
        region_name=settings.aws_region,
        bedrock_api_key=settings.aws_bearer_token_bedrock,
    )
```

`loop/embeddings.py` follows the identical pattern:

```python
# loop/embeddings.py — the embeddings factory (Phase 8)
def get_embeddings() -> Embeddings:
    if settings.model_provider == "ollama":     # v2 seam
        raise NotImplementedError("Ollama embeddings swap is v2")
    return BedrockEmbeddings(
        model_id=settings.bedrock_embed_model_id,
        region_name=settings.aws_region,
    )
```

## Spring analogy

In Spring, you'd express this as a `@Bean` factory method on a `@Configuration` class:

```java
@Configuration
public class EmbeddingsConfig {

    @Bean
    @ConditionalOnProperty(name = "model.provider", havingValue = "bedrock", matchIfMissing = true)
    public EmbeddingsService bedrockEmbeddings(Settings settings) {
        return new BedrockEmbeddingsService(settings.getEmbedModelId(), settings.getRegion());
    }

    @Bean
    @ConditionalOnProperty(name = "model.provider", havingValue = "ollama")
    public EmbeddingsService ollamaEmbeddings(Settings settings) {
        return new OllamaEmbeddingsService(settings.getOllamaHost());
    }
}
```

Callers inject `EmbeddingsService` — they never know whether they got Bedrock or Ollama. The `get_embeddings()` function is the Python equivalent: callers call the function, not the constructor.

## Why this matters

Without the seam, you'd have `BedrockEmbeddings(...)` scattered across `retrieval.py`, test files, and anywhere else that needs embeddings. When v2 arrives:

**Without seam:** find every `BedrockEmbeddings(...)` call in the codebase and update it. Error-prone, easy to miss one.

**With seam:** change one line in `loop/embeddings.py`. Done.

## The seam in practice

```
┌─────────────────────┐
│   loop/retrieval.py │  ──► get_embeddings() ──► BedrockEmbeddings (v1)
└─────────────────────┘                      └──► OllamaEmbeddings  (v2)
                                             └──► pgvector native   (v2)
         ▲
         │  only this file changes in v2
         │
┌─────────────────────┐
│  loop/embeddings.py │   ← the seam lives here
└─────────────────────┘
```

`retrieval.py` calls `get_embeddings()`. It does not import `BedrockEmbeddings`. It does not know or care which provider is active. The seam is the boundary.

## The test override

Because all embedding usage goes through `get_embeddings()`, tests can replace the entire embeddings implementation with a single monkeypatch:

```python
# tests/test_retrieval.py
from langchain_core.embeddings.fake import DeterministicFakeEmbedding

@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(
        "loop.retrieval.get_embeddings",
        lambda: DeterministicFakeEmbedding(size=256)
    )
```

Every test that triggers `retrieve_questions(...)` now uses fake embeddings — deterministic, no network, instantly fast. The rest of the code is unchanged because it only ever calls `get_embeddings()`.

**Without the seam**, you'd need to patch `BedrockEmbeddings` directly — harder to target, more fragile, and you'd risk patching at the wrong import path.

## DeterministicFakeEmbedding

`langchain_core.embeddings.fake.DeterministicFakeEmbedding` produces a vector for each input string by hashing the string into a fixed-size array of deterministic floats. Same text always gets the same vector — which means the vector store can build, add_documents, and similarity_search all work normally. The similarity scores are meaningless (hash-based, not semantic), but the mechanics work, which is all the tests need.

```python
fake = DeterministicFakeEmbedding(size=256)
fake.embed_query("hello world")  # → [0.23, -0.11, 0.87, ...]  (always the same)
fake.embed_query("hello world")  # → [0.23, -0.11, 0.87, ...]  (identical — deterministic)
fake.embed_query("goodbye")      # → [different values — different hash]
```

## Configuration

```python
# loop/config.py
class Settings(BaseSettings):
    # ... existing fields ...
    bedrock_embed_model_id: str = "amazon.titan-embed-text-v2:0"
    # model_provider already exists — "bedrock" (default) or "ollama" (v2)
```

The embed model id has a sensible default so the project works out of the box with Bedrock, and can be overridden via `.env` if the account uses a different Titan variant or a custom model.

## Files involved

| File | Role |
|---|---|
| `loop/config.py` | `bedrock_embed_model_id` setting |
| `loop/embeddings.py` | `get_embeddings()` factory — the seam |
| `loop/retrieval.py` | Only caller of `get_embeddings()` in production |
| `tests/test_retrieval.py` | Patches `loop.retrieval.get_embeddings` → `DeterministicFakeEmbedding` |

## See also

- [01-embeddings.md](01-embeddings.md) — what embeddings are
- [03-vector-store.md](03-vector-store.md) — where the factory output is used
- [06-rag-triad.md](06-rag-triad.md) — the full pipeline the factory feeds into
- `loop/models.py` — the identical factory pattern for the chat model (reference implementation)
