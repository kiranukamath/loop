"""
Memory wiring for Loop — short-term (checkpointer) and long-term (store).

Short-term — checkpointer:
  Saves complete graph state per thread_id after every node.
  Analogy: HTTP session / database savepoint.
  Required for Phase 5 interrupts (graph must know where to resume).

  Two implementations selected by config:
    db_path = ""            → MemorySaver  (in-process, lost on restart)
    db_path = "loop.sqlite" → SqliteSaver  (file on disk, survives restarts)

  Usage: compile(checkpointer=get_checkpointer())

Long-term — InMemoryStore:
  Cross-session key-value store keyed by (namespace, user_id).
  Analogy: a user-profile table that outlives a single request.
  v1: in-memory (lost on process restart).
  v2 seam: swap InMemoryStore → PostgresStore without touching node code.
  Usage: compile(store=get_store_instance())

Nodes access the store at runtime via:
    from langgraph.config import get_store, get_config
    store = get_store()           # returns None if no store wired
    user_id = get_config()['configurable'].get('user_id', 'default')

── Phase 16a: memory typing ─────────────────────────────────────────────────
Through Phase 4, everything about a user lived in one flat bucket:
    ("loop", "users")  /  key = user_id
    value = {"weak_areas": [...], "session_count": n}

Phase 16 splits that into three sub-namespaces per user — the classic agent-
memory taxonomy:
    ("loop", "users", user_id, "episodic")    — what happened. One immutable
        record per session (key "session-N"), written by coach.py right
        after grading. Raw log, never consolidated by itself.
    ("loop", "users", user_id, "semantic")    — durable facts. Key
        "weak_areas" holds the same merged list Phase 4 tracked (now
        migrated here); keys "insight-*" hold reflect()'s (Phase 16b)
        consolidated, individually-embeddable facts about the candidate,
        recalled by similarity (Phase 16c) instead of dumped whole.
    ("loop", "users", user_id, "procedural")  — durable coaching strategy.
        Key "coaching_notes" holds reflect()'s guidance on HOW to coach this
        candidate (pacing, format, what lands) — distinct from WHAT their
        gaps are.

Back-compat: get_weak_areas_state() reads the new "semantic" namespace first
and falls back to the old flat ("loop", "users") shape when nothing typed
exists yet, so weak_areas written before this migration are never silently
lost — every write migrates the record forward into the typed shape.

── Phase 16c: semantic recall + decay/conflict ──────────────────────────────
The store singleton is built WITH an index config (dims + an embeddings
function + which fields to embed), which is what unlocks store.search(...,
query=...) doing real cosine-similarity ranking instead of a plain listing
(verified against the installed langgraph==1.2.5 InMemoryStore/BaseStore
source — langgraph/store/base/__init__.py's IndexConfig + BaseStore.search
docstrings, and langgraph/store/memory/__init__.py's InMemoryStore.__init__).
_LazyEmbeddings defers the actual get_embeddings() factory call until the
first real embed — the store singleton below is built at import time
(loop.graph's `compiled = compile_graph_with_memory()` runs at module load),
long before a test's autouse fixture gets a chance to monkeypatch
get_embeddings() with a fake. Deferring the call means whichever factory is
current AT USE TIME wins (Bedrock in prod, DeterministicFakeEmbedding in
tests) — the same "seam, not a singleton" discipline as models.py/embeddings.py.

recall_semantic_memories() layers a deterministic decay/conflict policy on
top of that similarity search — no model call, so it's fully offline-testable:
  - decay: an insight whose created_session_count is more than
    settings.memory_ttl_sessions sessions old is treated as stale and dropped.
  - conflict: resolve_conflict() (used by put_semantic_insight, the write
    path) keeps whichever of an existing/candidate insight is NEWER
    (higher created_session_count), breaking ties by higher confidence — so a
    contradiction always resolves to the more recent read on the candidate.
"""

from __future__ import annotations

import re
import sqlite3

from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.store.base import SearchItem
from langgraph.store.memory import InMemoryStore

from loop.config import settings
from loop.embeddings import get_embeddings

# ── Namespaces (Phase 16a: typed sub-namespaces; legacy flat shape kept for
#    back-compat reads only — nothing writes there any more) ─────────────────

_LEGACY_NAMESPACE = ("loop", "users")  # Phase 4 flat shape: key=user_id


def episodic_namespace(user_id: str) -> tuple[str, ...]:
    """What happened: one immutable record per session."""
    return ("loop", "users", user_id, "episodic")


def semantic_namespace(user_id: str) -> tuple[str, ...]:
    """Durable facts about the candidate: weak_areas + reflect()'s insights."""
    return ("loop", "users", user_id, "semantic")


def procedural_namespace(user_id: str) -> tuple[str, ...]:
    """Durable facts about how to coach this candidate."""
    return ("loop", "users", user_id, "procedural")


# ── Checkpointer (short-term) ─────────────────────────────────────────────────


def _make_checkpointer() -> MemorySaver:
    """Build a checkpointer based on config.

    pg_conn_string set → PostgresSaver  (durable, real Postgres — Phase 18a).
    db_path set        → SqliteSaver    (durable, file-backed).
    neither set        → MemorySaver    (in-memory, tests/dev).

    SqliteSaver.setup() / PostgresSaver.setup() create the schema tables on
    first call. check_same_thread=False is required for SQLite because
    FastAPI runs nodes on the same thread as the event loop — SQLite's
    default thread check would raise otherwise.

    Phase 18a: PostgresSaver.from_conn_string() is a @contextmanager (it
    closes the connection on exit) — unsuitable for a process-level
    singleton that must outlive this function call. Instead we open the
    psycopg connection directly (autocommit + dict_row, the exact settings
    from_conn_string uses internally — verified by reading the installed
    langgraph-checkpoint-postgres==3.1.2 source) and construct PostgresSaver
    from it, same pattern as the existing sqlite3.connect() path below.
    A real Postgres server is a *server* activity — tests only exercise this
    dispatch with psycopg's Connection.connect mocked out.
    """
    if settings.pg_conn_string:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg import Connection
        from psycopg.rows import dict_row

        conn = Connection.connect(
            settings.pg_conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
        )
        saver = PostgresSaver(conn)
        saver.setup()
        return saver  # type: ignore[return-value]

    if settings.db_path:
        import pathlib

        from langgraph.checkpoint.sqlite import SqliteSaver

        pathlib.Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(settings.db_path, check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        return saver  # type: ignore[return-value]
    return MemorySaver()


# ── Store (long-term) ─────────────────────────────────────────────────────────


class _LazyEmbeddings(Embeddings):
    """Defers the get_embeddings() factory call until the first real embed.

    See the module docstring's Phase 16c section for why this exists: the
    store singleton is constructed at import time, before test fixtures can
    monkeypatch loop.embeddings/loop.memory's get_embeddings with a fake.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return get_embeddings().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return get_embeddings().embed_query(text)


def _make_store():
    """Build the long-term store, indexed for semantic search (Phase 16c).

    fields=["text"] means only items that HAVE a top-level "text" field get
    embedded — that's exactly reflect()'s semantic insights (put via
    put_semantic_insight), and nothing else. The "weak_areas" record and
    procedural notes have no "text" field, so they're stored un-embedded,
    same cost as a plain InMemoryStore() put.

    Phase 18a: pg_conn_string set → PostgresStore, built the same way as
    PostgresSaver above (a direct psycopg connection, not the
    from_conn_string() context manager, so the store outlives this function).
    PostgresStore takes the same index config shape as InMemoryStore
    (verified against installed langgraph-checkpoint-postgres==3.1.2's
    PostgresIndexConfig) so _LazyEmbeddings works identically either way.
    """
    if settings.pg_conn_string:
        from langgraph.store.postgres import PostgresStore
        from psycopg import Connection
        from psycopg.rows import dict_row

        conn = Connection.connect(
            settings.pg_conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
        )
        store = PostgresStore(
            conn,
            index={
                "dims": settings.embedding_dims,
                "embed": _LazyEmbeddings(),
                "fields": ["text"],
            },
        )
        store.setup()
        return store

    return InMemoryStore(
        index={
            "dims": settings.embedding_dims,
            "embed": _LazyEmbeddings(),
            "fields": ["text"],
        }
    )


# Module-level singletons so all graph runs in the same process share state.
_checkpointer = _make_checkpointer()
_store = _make_store()


def get_checkpointer() -> MemorySaver:
    """Return the process-level checkpointer (MemorySaver, SqliteSaver, or
    PostgresSaver)."""
    return _checkpointer


def get_store_instance():
    """Return the process-level long-term store (InMemoryStore or
    PostgresStore)."""
    return _store


def compile_with_memory(graph: StateGraph):
    """Compile a graph with both checkpointer and store attached.

    The compiled graph requires thread_id in every invoke config:
        graph.invoke(state, config={'configurable': {'thread_id': 'abc', 'user_id': 'kiran'}})

    user_id is optional (defaults to 'default') but must be consistent across
    sessions for the store to accumulate weak_areas correctly.
    """
    return graph.compile(
        checkpointer=get_checkpointer(),
        store=get_store_instance(),
    )


# ── Phase 16a: weak_areas read/write helper (typed, with back-compat read) ───


def get_weak_areas_state(store, user_id: str) -> tuple[list[str], int]:
    """Read (weak_areas, session_count) for user_id.

    Reads the typed semantic namespace first (Phase 16a shape). Falls back to
    the old flat ("loop", "users") namespace (Phase 4 shape) only when no
    typed entry exists yet, so data written before this migration is never
    silently dropped. Never writes itself — coach._persist_weak_areas
    migrates the record forward into the typed shape on its next write.
    """
    item = store.get(semantic_namespace(user_id), "weak_areas")
    if item is not None:
        return list(item.value.get("weak_areas", [])), item.value.get("session_count", 0)

    legacy = store.get(_LEGACY_NAMESPACE, user_id)
    if legacy is not None:
        return list(legacy.value.get("weak_areas", [])), legacy.value.get("session_count", 0)

    return [], 0


def put_weak_areas_state(store, user_id: str, weak_areas: list[str], session_count: int) -> None:
    """Write the merged weak_areas list into the typed semantic namespace."""
    store.put(
        semantic_namespace(user_id),
        "weak_areas",
        {"weak_areas": weak_areas, "session_count": session_count},
    )


def put_episode(store, user_id: str, session_number: int, new_weak_areas: list[str]) -> None:
    """Write one immutable episodic record for a completed session."""
    store.put(
        episodic_namespace(user_id),
        f"session-{session_number}",
        {"session_number": session_number, "new_weak_areas": new_weak_areas},
    )


def get_recent_episodes(store, user_id: str, session_count: int, window: int) -> list[dict]:
    """Return up to `window` most recent episodic records, oldest first.

    Reads by explicit key (session-N) rather than store.search()/list() —
    that keeps recency deterministic regardless of the store's internal
    iteration order, and bounds reflect() to a fixed-size window regardless
    of how long a user's history grows ("bounded", per PLAN.md).
    """
    namespace = episodic_namespace(user_id)
    start = max(1, session_count - window + 1)
    episodes = []
    for n in range(start, session_count + 1):
        item = store.get(namespace, f"session-{n}")
        if item is not None:
            episodes.append(item.value)
    return episodes


# ── Phase 16b/16c: semantic insights + procedural notes ──────────────────────


def _slugify(text: str) -> str:
    """Turn free text into a short, stable dict key (a "topic" for an insight).

    Two insights that slugify to the same key are treated as the SAME fact
    re-asserted — that's what makes resolve_conflict() meaningful (an update
    to an existing topic, not just another unrelated entry).
    """
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "insight"


def resolve_conflict(existing: dict | None, candidate: dict) -> dict:
    """Deterministic conflict policy for two semantic-memory values sharing a
    topic key: newer session wins; ties broken by higher confidence.

    No model call — this is what keeps memory writes cheap and fully
    offline-testable, per PLAN.md's "decay/conflict is deterministic" bar.
    """
    if existing is None:
        return candidate
    existing_session = existing.get("created_session_count", 0)
    candidate_session = candidate.get("created_session_count", 0)
    if candidate_session != existing_session:
        return candidate if candidate_session > existing_session else existing
    return (
        candidate if candidate.get("confidence", 0) >= existing.get("confidence", 0) else existing
    )


def put_semantic_insight(
    store, user_id: str, *, topic: str, text: str, confidence: float, session_count: int
) -> None:
    """Write one consolidated semantic insight, resolving conflicts by topic.

    index=["text"] tells the store to embed just the "text" field of THIS
    item (only meaningful when the store itself was built with an index
    config — see _make_store() — otherwise it's a no-op, same as a plain put).
    """
    namespace = semantic_namespace(user_id)
    key = _slugify(topic)
    existing = store.get(namespace, key)
    candidate = {
        "text": text,
        "topic": topic,
        "confidence": confidence,
        "created_session_count": session_count,
    }
    winner = resolve_conflict(existing.value if existing else None, candidate)
    store.put(namespace, key, winner, index=["text"])


def put_procedural_note(store, user_id: str, *, notes: list[str], session_count: int) -> None:
    """Overwrite the single evolving 'coaching_notes' procedural-memory entry.

    Unlike semantic insights (many independent per-topic facts), procedural
    memory is one evolving strategy for this candidate — always the latest
    reflect() run's guidance, not something to keep multiple versions of.
    """
    store.put(
        procedural_namespace(user_id),
        "coaching_notes",
        {"notes": notes, "session_count": session_count},
    )


def get_procedural_notes(store, user_id: str) -> list[str]:
    item = store.get(procedural_namespace(user_id), "coaching_notes")
    return list(item.value.get("notes", [])) if item else []


def recall_semantic_memories(
    store,
    user_id: str,
    *,
    query: str,
    session_count: int,
    k: int | None = None,
    ttl: int | None = None,
) -> list[dict]:
    """Recall the top-k semantic insights relevant to `query`, decayed.

    Ranking: store.search(namespace, query=...) — cosine similarity when the
    store was built with an index config (see _make_store()); degrades to an
    unranked listing (still correct, just not similarity-ordered) when it
    wasn't, e.g. a bare InMemoryStore() built by a test that doesn't care
    about ranking. Either way, results are then filtered by the deterministic
    decay policy: drop anything older than `ttl` sessions.

    "weak_areas" (no "text" field) is naturally excluded — callers filter on
    value.get("text"), and the un-indexed fields path never sets one.
    """
    k = k if k is not None else settings.memory_recall_k
    ttl = ttl if ttl is not None else settings.memory_ttl_sessions
    namespace = semantic_namespace(user_id)

    # Over-fetch: some hits may be stale and get dropped below, so ask for
    # more than k up front to still return up to k fresh results.
    results: list[SearchItem] = store.search(namespace, query=query, limit=max(k * 3, k))

    fresh = [
        r.value
        for r in results
        if r.value.get("text") and session_count - r.value.get("created_session_count", 0) <= ttl
    ]
    return fresh[:k]
