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

Store namespace convention:
    ("loop", "users")  /  key = user_id
    value = {"weak_areas": [...], "session_count": n}
"""

from __future__ import annotations

import sqlite3

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.store.memory import InMemoryStore

from loop.config import settings


def _make_checkpointer() -> MemorySaver:
    """Build a checkpointer based on config.

    db_path set → SqliteSaver (durable, file-backed).
    db_path empty → MemorySaver (in-memory, tests/dev).

    SqliteSaver.setup() creates the schema tables on first call.
    check_same_thread=False is required because FastAPI runs nodes on
    the same thread as the event loop — SQLite's default thread check
    would raise otherwise.
    """
    if settings.db_path:
        import pathlib

        from langgraph.checkpoint.sqlite import SqliteSaver

        pathlib.Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(settings.db_path, check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        return saver  # type: ignore[return-value]
    return MemorySaver()


# Module-level singletons so all graph runs in the same process share state.
# v2: replace with AsyncPostgresSaver / PostgresStore, reading conn string from config.
_checkpointer = _make_checkpointer()
_store: InMemoryStore = InMemoryStore()


def get_checkpointer() -> MemorySaver:
    """Return the process-level checkpointer (MemorySaver or SqliteSaver)."""
    return _checkpointer


def get_store_instance() -> InMemoryStore:
    """Return the process-level long-term store."""
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
