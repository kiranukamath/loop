"""
Swappable model factory.

Every node in the graph calls get_chat_model() — no node ever imports
ChatBedrockConverse directly.  This is the single place to change the model
provider.  In v2, adding Ollama here is all that's needed.

Analogy: a Spring @Configuration class with a @Bean method backed by an
interface.  Callers depend on the interface (BaseChatModel); the factory
decides the implementation.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from loop.config import settings


def get_chat_model(model_id: str | None = None) -> BaseChatModel:
    """Return the configured chat model.

    model_id overrides settings.bedrock_model_id — used by Phase 10a to build
    a fallback model (a different, usually cheaper/more-available, model id)
    without adding a second factory function.

    Currently supports provider="bedrock" only.
    The v2 Ollama path raises NotImplementedError as a deliberate seam —
    it will be filled in without touching any calling node.
    """
    if settings.model_provider == "bedrock":
        return _make_bedrock_model(model_id or settings.bedrock_model_id)

    # v2 seam — not implemented yet
    if settings.model_provider == "ollama":
        raise NotImplementedError(
            "Ollama provider is a v2 feature.  Set MODEL_PROVIDER=bedrock in your .env for now."
        )

    raise ValueError(f"Unknown model provider: {settings.model_provider!r}")


def _make_bedrock_model(model_id: str) -> BaseChatModel:
    """Construct a ChatBedrockConverse for the given model id.

    AWS_BEARER_TOKEN_BEDROCK is NOT passed explicitly — ChatBedrockConverse
    reads it from the environment automatically via its own default_factory
    (verified in langchain-aws==1.5.1 source).  Setting it in .env is enough.
    """
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(
        model_id=model_id,
        region_name=settings.aws_region,
    )


def with_resilience(chain: Runnable, fallback_chain: Runnable | None = None) -> Runnable:
    """Wrap an already-built chain with retry + optional fallback (Phase 10a).

    Call this on the FINAL chain (e.g. `prompt | model.with_structured_output(Schema)`),
    not on the bare model. Verified on the installed langchain-core==1.4.7:
    `.with_retry()`/`.with_fallbacks()` are generic Runnable methods, so they work
    on any chain shape. Going the other direction does NOT work — wrapping a bare
    model first (`model.with_retry()`) returns a RunnableRetry, which drops
    `.with_structured_output()` entirely (it doesn't proxy to the wrapped model).

    fallback_chain must be built from the SAME prompt/schema, just a different
    model — see loop/nodes/planner.py for the calling pattern.

    Analogy: Resilience4j's Retry + Fallback decorators stacked around one call,
    but composed as plain Runnable wrappers instead of annotations.
    """
    if fallback_chain is not None:
        chain = chain.with_fallbacks([fallback_chain])
    return chain.with_retry(stop_after_attempt=settings.retry_max_attempts)
