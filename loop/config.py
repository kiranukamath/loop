"""
Settings loaded from environment variables (or a .env file).

pydantic-settings reads each field from the matching env var name (uppercased).
If a required field is missing it raises a clear error at startup — exactly like
Spring's @ConfigurationProperties failing fast on a missing property.
"""

from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env into os.environ early so boto3 / ChatBedrockConverse can read
# AWS_BEARER_TOKEN_BEDROCK directly.  pydantic-settings populates Settings
# fields but does NOT set os.environ — boto3 needs it there.
load_dotenv(override=False)


class Settings(BaseSettings):
    # ── AWS / Bedrock ────────────────────────────────────────────────────────
    # NOTE: the Bedrock API bearer token (AWS_BEARER_TOKEN_BEDROCK) is NOT read
    # here.  ChatBedrockConverse reads it directly from the environment via its
    # own default_factory — passing it through Settings would duplicate it and
    # cause a process-wide env-var side effect warning.  Just set it in .env.

    aws_region: str = "us-east-1"

    # Chat model ID — must be enabled in your account + region.
    # Haiku 4.5 is the cheapest current Claude model on Bedrock.
    bedrock_model_id: str = "anthropic.claude-haiku-4-5-20251001-v1:0"

    # Embeddings model ID (Phase 8) — Amazon Titan Embeddings v2.
    # Produces 1536-dimensional vectors; must be enabled in your account + region.
    bedrock_embed_model_id: str = "amazon.titan-embed-text-v2:0"

    # ── Langfuse (optional — no-op when all three are absent/empty) ─────────
    # All three are required for a self-hosted instance.
    # Leave all empty to disable tracing entirely.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # ── Model provider switch (v2 seam) ─────────────────────────────────────
    # "bedrock" is the only implemented provider in v1.
    model_provider: str = "bedrock"

    # ── Session cap ──────────────────────────────────────────────────────────
    # Maximum number of interview sessions per run.  The planner may suggest
    # more; this hard-caps the list so the graph never exceeds this many questions.
    # Override with MAX_SESSIONS=N in .env or the environment.
    max_sessions: int = 2

    # ── Persistence ──────────────────────────────────────────────────────────
    # Path to the SQLite file used by SqliteSaver.  Empty = use MemorySaver
    # (in-memory, lost on restart — fine for tests and quick dev runs).
    # Set to a real path (e.g. "loop_state.sqlite") for durable persistence.
    db_path: str = ""

    # ── Web search seam (Phase 9) ─────────────────────────────────────────────
    # Keyless DuckDuckGo (via `ddgs`) is the v1 default — no signup required.
    # Setting TAVILY_API_KEY switches to Tavily automatically (v2 — not yet
    # implemented; see loop/research/search.py for the seam).
    tavily_api_key: str = ""

    # Iteration bound for the ReAct research agent — prevents runaway tool-call
    # loops.  Passed as LangGraph's recursion_limit when invoking the agent.
    research_max_iterations: int = 6

    # ── Resilience (Phase 10a) ────────────────────────────────────────────────
    # Max attempts (including the first) before a model call gives up.
    # LangChain's Runnable.with_retry() backs off exponentially between attempts —
    # same idea as Resilience4j's Retry decorator around a flaky downstream call.
    retry_max_attempts: int = 3

    # Optional fallback model id, tried only if every retry against the primary
    # model fails. Empty = no fallback (raise the original error after retries).
    fallback_model_id: str = ""

    # ── Cost & token budget (Phase 10c) ───────────────────────────────────────
    # Hard ceiling on total tokens (input + output) per session. 0 = unlimited.
    # Enforced by loop/budget.py via a callback attached to each graph run.
    max_session_tokens: int = 50_000

    # ── External MCP servers (Phase 12b) ──────────────────────────────────────
    # Allow-list of external MCP servers the research agent may load tools
    # from, keyed by a name you choose: {"command": ..., "args": [...],
    # "transport": "stdio"}. Empty (the default) = feature off -- the research
    # agent's tool list stays byte-for-byte the Phase 9 flow, and
    # loop/research/mcp_client.py never spawns a subprocess.
    # Only stdio is supported in v1 -- remote HTTP/SSE servers are a v2 seam,
    # same treatment as the Tavily search provider and the Ollama model swap.
    mcp_server_configs: dict[str, dict[str, object]] = {}

    # ── Multi-agent orchestration (Phase 13) ──────────────────────────────────
    # Panel grading (13a): fan an answer out to K persona-graders in parallel
    # via LangGraph's Send API, then reduce their partials into one Grade.
    # Off by default — loop/nodes/grader.py's sequential path is untouched
    # either way; build_graph() only wires the panel region when this is True.
    panel_grading: bool = False
    grader_personas: list[str] = ["correctness", "communication", "depth"]

    # Panel debate (13c, stretch): when panel_grading is also on, each
    # persona sees the others' round-0 scores and revises once before
    # grade_aggregator reduces the LATEST round only. Off by default.
    panel_debate: bool = False

    # Orchestration mode (13b): "fixed" keeps today's deterministic routing
    # functions (session_router / _route_by_modality / _route_after_session).
    # "supervisor" replaces all three with one LLM-driven interview_supervisor
    # node that decides the next specialist (or readiness) via Command(goto=...)
    # at runtime — the graph stops owning control flow; the model does.
    orchestration_mode: Literal["fixed", "supervisor"] = "fixed"

    # ── Advanced RAG (Phase 14) ────────────────────────────────────────────────
    # 14a: hybrid search (dense + BM25 via Reciprocal Rank Fusion) and
    # cross-encoder reranking, both on by default -- this is the retrieval
    # pipeline upgrade itself, not an optional extra. Tests stub get_reranker()
    # with FakeReranker (see conftest.py) so no live Bedrock rerank call runs.
    hybrid_enabled: bool = True
    rerank_enabled: bool = True
    rerank_model_id: str = "amazon.rerank-v1:0"

    # 14b: query rewriting before retrieval. "off" keeps today's single-query
    # behaviour byte-for-byte; "multiquery" asks the model for extra phrasings
    # of the same need; "hyde" asks the model to write a hypothetical matching
    # question and embeds that instead of the raw focus/topics.
    query_rewrite_mode: Literal["off", "multiquery", "hyde"] = "off"

    # 14c: Corrective RAG -- grade the top retrieved question against the
    # session's focus and re-retrieve (bounded by crag_max_retries) when it
    # scores below crag_min_relevance; falls back to search_web as a last
    # resort. Off by default -- it adds an LLM call to every question pick.
    crag_enabled: bool = False
    crag_min_relevance: float = 0.5
    crag_max_retries: int = 2

    # ── Self-improvement loops (Phase 15) ───────────────────────────────────────
    # 15a: a second LLM pass that critiques (and can revise) the grader's own
    # output before it's committed to state — a Reflexion-style generate→
    # critique→revise loop. Off by default: today's single-pass grading is
    # byte-for-byte unchanged unless explicitly opted in.
    reflexion_enabled: bool = False

    # 15b: bounded replanning. After a session's grade comes back, if it
    # diverges from the plan's assumptions (score below replan_score_threshold),
    # re-invoke planner() on the remaining sessions only. replan_max_times
    # bounds the number of times this can happen in one run — without a bound,
    # a candidate who keeps scoring low would trigger a replan forever.
    replan_enabled: bool = False
    replan_score_threshold: int = 5
    replan_max_times: int = 1

    # ── Advanced memory (Phase 16) ────────────────────────────────────────────
    # 16c: dimensionality of the vectors loop/memory.py's long-term store
    # indexes semantic insights with. Must match whatever get_embeddings()
    # actually returns (Titan v2 defaults to 1024-dim output).
    embedding_dims: int = 1024

    # 16b: reflect() consolidates recent episodic memories (per-session
    # weak-area updates) into durable semantic/procedural insights, wired at
    # the end of a curriculum run (graph.py: readiness -> reflect -> END).
    # Off by default -- with it off, reflect() is a no-op and weak_areas
    # keeps working exactly as it did through Phase 4/16a.
    reflection_enabled: bool = False

    # 16c: top-k semantic insights recalled by embedding similarity
    # (loop/memory.py's recall_semantic_memories) for the planner prompt,
    # instead of dumping every stored insight in.
    memory_recall_k: int = 3

    # 16c: a semantic insight older than this many sessions (by the
    # session_count it was written at) is treated as stale and excluded from
    # recall -- a simple, deterministic decay policy, no model call.
    memory_ttl_sessions: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Module-level singleton — import this everywhere instead of constructing Settings().
settings = Settings()

# USD per 1,000 tokens, as (input_price, output_price) — used by loop/budget.py
# to price a model call's usage. Extend this when adding a new bedrock_model_id
# or fallback_model_id. Prices are illustrative estimates, not live AWS pricing —
# do not treat cost_usd as billing-accurate without checking current Bedrock rates.
MODEL_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "anthropic.claude-haiku-4-5-20251001-v1:0": (0.001, 0.005),
    "anthropic.claude-sonnet-5-20251101-v1:0": (0.003, 0.015),
}
