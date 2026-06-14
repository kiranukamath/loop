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

from loop.config import settings


def get_chat_model() -> BaseChatModel:
    """Return the configured chat model.

    Currently supports provider="bedrock" only.
    The v2 Ollama path raises NotImplementedError as a deliberate seam —
    it will be filled in without touching any calling node.
    """
    if settings.model_provider == "bedrock":
        return _make_bedrock_model()

    # v2 seam — not implemented yet
    if settings.model_provider == "ollama":
        raise NotImplementedError(
            "Ollama provider is a v2 feature.  Set MODEL_PROVIDER=bedrock in your .env for now."
        )

    raise ValueError(f"Unknown model provider: {settings.model_provider!r}")


def _make_bedrock_model() -> BaseChatModel:
    """Construct a ChatBedrockConverse.

    AWS_BEARER_TOKEN_BEDROCK is NOT passed explicitly — ChatBedrockConverse
    reads it from the environment automatically via its own default_factory
    (verified in langchain-aws==1.5.1 source).  Setting it in .env is enough.
    """
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(
        model_id=settings.bedrock_model_id,
        region_name=settings.aws_region,
    )
