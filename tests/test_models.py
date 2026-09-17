"""
Phase 10a tests — resilience in the model seam (retry + fallback).

All offline: no real ChatBedrockConverse or Bedrock call is exercised.
`with_resilience()` is tested directly against fake Runnables built with
RunnableLambda, so the retry/fallback mechanics are verified independent of
any real model or node.
"""

from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableLambda


class _CountingFailer:
    """Callable that raises `fail_times` times, then returns `result`.

    Used as the function inside a RunnableLambda so `.invoke()` triggers it.
    """

    def __init__(self, fail_times: int, result):
        self.calls = 0
        self.fail_times = fail_times
        self.result = result

    def __call__(self, _input):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient failure")
        return self.result


class _AlwaysFailer:
    def __init__(self):
        self.calls = 0

    def __call__(self, _input):
        self.calls += 1
        raise RuntimeError("permanent failure")


# ── get_chat_model model_id override ──────────────────────────────────────────


class TestGetChatModelOverride:
    def test_uses_settings_default_when_no_override(self, monkeypatch):
        from loop.config import settings
        from loop.models import get_chat_model

        captured = {}
        monkeypatch.setattr(
            "loop.models._make_bedrock_model",
            lambda model_id: captured.setdefault("model_id", model_id),
        )
        get_chat_model()
        assert captured["model_id"] == settings.bedrock_model_id

    def test_override_model_id_is_used(self, monkeypatch):
        from loop.models import get_chat_model

        captured = {}
        monkeypatch.setattr(
            "loop.models._make_bedrock_model",
            lambda model_id: captured.setdefault("model_id", model_id),
        )
        get_chat_model("some-fallback-id")
        assert captured["model_id"] == "some-fallback-id"


# ── with_resilience: retry ────────────────────────────────────────────────────


class TestWithResilienceRetry:
    def test_retries_and_succeeds_after_transient_failure(self, monkeypatch):
        from loop.config import settings
        from loop.models import with_resilience

        monkeypatch.setattr(settings, "retry_max_attempts", 3)
        failer = _CountingFailer(fail_times=1, result="ok")
        chain = RunnableLambda(failer)

        result = with_resilience(chain).invoke({})

        assert result == "ok"
        assert failer.calls == 2  # failed once, succeeded on the retry

    def test_gives_up_after_max_attempts_with_no_fallback(self, monkeypatch):
        from loop.config import settings
        from loop.models import with_resilience

        monkeypatch.setattr(settings, "retry_max_attempts", 3)
        failer = _AlwaysFailer()
        chain = RunnableLambda(failer)

        with pytest.raises(RuntimeError, match="permanent failure"):
            with_resilience(chain).invoke({})

        assert failer.calls == 3


# ── with_resilience: fallback ─────────────────────────────────────────────────


class TestWithResilienceFallback:
    def test_falls_back_when_primary_fails(self, monkeypatch):
        from loop.config import settings
        from loop.models import with_resilience

        monkeypatch.setattr(settings, "retry_max_attempts", 2)
        primary = RunnableLambda(_AlwaysFailer())
        fallback_failer = _CountingFailer(fail_times=0, result="fallback-result")
        fallback = RunnableLambda(fallback_failer)

        result = with_resilience(primary, fallback).invoke({})

        assert result == "fallback-result"
        assert fallback_failer.calls == 1

    def test_no_fallback_used_when_primary_succeeds(self, monkeypatch):
        from loop.config import settings
        from loop.models import with_resilience

        monkeypatch.setattr(settings, "retry_max_attempts", 2)
        primary = RunnableLambda(lambda _: "primary-result")
        fallback_failer = _AlwaysFailer()
        fallback = RunnableLambda(fallback_failer)

        result = with_resilience(primary, fallback).invoke({})

        assert result == "primary-result"
        assert fallback_failer.calls == 0


# ── Phase 18d: Ollama model seam ──────────────────────────────────────────────
#
# A real Ollama server is a *server* activity (needs `ollama serve` running
# locally). These tests assert only CONSTRUCTION — that get_chat_model()
# builds a ChatOllama with the right model_id/base_url when
# model_provider="ollama" — never that it can actually be invoked.


class TestOllamaModelSeam:
    def test_ollama_provider_returns_chat_ollama(self, monkeypatch):
        from langchain_ollama import ChatOllama

        from loop.config import settings
        from loop.models import get_chat_model

        monkeypatch.setattr(settings, "model_provider", "ollama")
        monkeypatch.setattr(settings, "ollama_model_id", "llama3.1")
        monkeypatch.setattr(settings, "ollama_base_url", "http://localhost:11434")

        model = get_chat_model()
        assert isinstance(model, ChatOllama)
        assert model.model == "llama3.1"
        assert model.base_url == "http://localhost:11434"

    def test_ollama_provider_respects_model_id_override(self, monkeypatch):
        from loop.config import settings
        from loop.models import get_chat_model

        monkeypatch.setattr(settings, "model_provider", "ollama")
        model = get_chat_model("mistral")
        assert model.model == "mistral"

    def test_unknown_provider_raises(self, monkeypatch):
        from loop.config import settings
        from loop.models import get_chat_model

        monkeypatch.setattr(settings, "model_provider", "not-a-real-provider")
        with pytest.raises(ValueError, match="Unknown model provider"):
            get_chat_model()
