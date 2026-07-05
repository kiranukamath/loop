"""
Swappable embeddings factory.

Mirrors loop/models.py exactly: every caller calls get_embeddings() and never
imports BedrockEmbeddings directly.  Swapping to Ollama embeddings in v2 means
changing this one file.

Analogy: the same Spring @Configuration/@Bean pattern used in models.py, but
for the embeddings provider instead of the chat model provider.

Verified against langchain-aws==1.5.1:
    BedrockEmbeddings(model_id=..., region_name=...)
    All credential fields are Optional — boto3's standard credential chain
    (env vars, ~/.aws/credentials, instance profile) is used automatically.
"""

from langchain_core.embeddings import Embeddings

from loop.config import settings


def get_embeddings() -> Embeddings:
    """Return the configured embeddings model.

    Currently supports provider="bedrock" only.
    The v2 Ollama path raises NotImplementedError as a deliberate seam.
    """
    if settings.model_provider == "bedrock":
        return _make_bedrock_embeddings()

    # v2 seam — not implemented yet
    if settings.model_provider == "ollama":
        raise NotImplementedError(
            "Ollama embeddings is a v2 feature.  Set MODEL_PROVIDER=bedrock for now."
        )

    raise ValueError(f"Unknown model provider: {settings.model_provider!r}")


def _make_bedrock_embeddings() -> Embeddings:
    """Construct a BedrockEmbeddings for Amazon Titan v2.

    AWS credentials are NOT passed explicitly — BedrockEmbeddings reads them
    from the boto3 credential chain (same as ChatBedrockConverse in models.py).
    Setting AWS_BEARER_TOKEN_BEDROCK in .env is sufficient.
    """
    from langchain_aws import BedrockEmbeddings

    return BedrockEmbeddings(
        model_id=settings.bedrock_embed_model_id,
        region_name=settings.aws_region,
    )
