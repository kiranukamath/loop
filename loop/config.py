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

    # Model ID — must be enabled in your account + region.
    # Haiku 4.5 is the cheapest current Claude model on Bedrock.
    bedrock_model_id: str = "anthropic.claude-haiku-4-5-20251001-v1:0"

    # ── Langfuse (optional — no-op when absent) ──────────────────────────────
    langfuse_public_key: str = ""

    # ── Model provider switch (v2 seam) ─────────────────────────────────────
    # "bedrock" is the only implemented provider in v1.
    model_provider: str = "bedrock"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Module-level singleton — import this everywhere instead of constructing Settings().
settings = Settings()
