"""
Settings loaded from environment variables (or a .env file).

pydantic-settings reads each field from the matching env var name (uppercased).
If a required field is missing it raises a clear error at startup — exactly like
Spring's @ConfigurationProperties failing fast on a missing property.
"""

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
