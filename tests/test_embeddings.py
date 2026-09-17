"""
Phase 18d tests — Ollama embeddings seam.

Mirrors tests/test_models.py's TestOllamaModelSeam: a real Ollama server is a
*server* activity, so these tests assert only construction (right
model_id/base_url), never a real embed call.
"""

from __future__ import annotations

import pytest


class TestOllamaEmbeddingsSeam:
    def test_ollama_provider_returns_ollama_embeddings(self, monkeypatch):
        from langchain_ollama import OllamaEmbeddings

        from loop.config import settings
        from loop.embeddings import get_embeddings

        monkeypatch.setattr(settings, "model_provider", "ollama")
        monkeypatch.setattr(settings, "ollama_embed_model_id", "nomic-embed-text")
        monkeypatch.setattr(settings, "ollama_base_url", "http://localhost:11434")

        embeddings = get_embeddings()
        assert isinstance(embeddings, OllamaEmbeddings)
        assert embeddings.model == "nomic-embed-text"
        assert embeddings.base_url == "http://localhost:11434"

    def test_bedrock_is_still_the_default(self, monkeypatch):
        from loop.config import settings

        assert settings.model_provider == "bedrock"

    def test_unknown_provider_raises(self, monkeypatch):
        from loop.config import settings
        from loop.embeddings import get_embeddings

        monkeypatch.setattr(settings, "model_provider", "not-a-real-provider")
        with pytest.raises(ValueError, match="Unknown model provider"):
            get_embeddings()
