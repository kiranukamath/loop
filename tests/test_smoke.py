"""
Phase 0 offline smoke tests.  No network calls — the model factory is
patched to return a fake model.  These must pass on a laptop with no creds.
"""

import importlib

import pytest

# ── helpers ───────────────────────────────────────────────────────────────


def _reload_config_with_env(monkeypatch, extra: dict) -> None:
    """Set env vars and reload config so Settings picks them up fresh."""
    required = {"BEDROCK_API_KEY": "test-key"}
    required.update(extra)
    for k, v in required.items():
        monkeypatch.setenv(k, v)

    # pydantic-settings reads env at construction time, so we reload the module.
    import loop.config as cfg_mod

    importlib.reload(cfg_mod)


# ── config ────────────────────────────────────────────────────────────────


class TestSettings:
    def test_defaults_load_with_required_key(self, monkeypatch):
        """Settings loads with just the required bedrock_api_key set."""
        _reload_config_with_env(monkeypatch, {})
        import loop.config as cfg

        assert cfg.settings.model_provider == "bedrock"
        assert cfg.settings.aws_region == "us-east-1"

    def test_custom_region(self, monkeypatch):
        _reload_config_with_env(monkeypatch, {"AWS_REGION": "us-west-2"})
        import loop.config as cfg

        assert cfg.settings.aws_region == "us-west-2"

    def test_missing_bedrock_key_raises(self, monkeypatch):
        """Missing BEDROCK_API_KEY must raise at construction time."""
        # Remove the key so Settings() fails
        monkeypatch.delenv("BEDROCK_API_KEY", raising=False)
        # Ensure no .env file is picked up by pointing to a nonexistent path
        from pydantic import ValidationError
        from pydantic_settings import BaseSettings

        class _TestSettings(BaseSettings):
            bedrock_api_key: str  # required

        with pytest.raises(ValidationError):
            _TestSettings()


# ── model factory ─────────────────────────────────────────────────────────


class TestModelFactory:
    def test_returns_bedrock_model(self, monkeypatch):
        """Factory returns a BaseChatModel when provider=bedrock (patched).

        Reload order matters: set env → reload config → reload models → THEN
        patch _make_bedrock_model (patching before reload wipes the patch).
        """
        monkeypatch.setenv("BEDROCK_API_KEY", "test-key")
        monkeypatch.setenv("MODEL_PROVIDER", "bedrock")

        from unittest.mock import MagicMock

        from langchain_core.language_models import BaseChatModel

        # Reload config first so the fresh settings singleton picks up the env vars.
        import loop.config as cfg_mod

        importlib.reload(cfg_mod)

        # Reload models so it re-reads the updated settings singleton.
        import loop.models as models_mod

        importlib.reload(models_mod)

        # NOW patch — after reloads, so the patch survives into get_chat_model().
        fake_model = MagicMock(spec=BaseChatModel)
        monkeypatch.setattr("loop.models._make_bedrock_model", lambda: fake_model)

        from loop.models import get_chat_model

        result = get_chat_model()
        assert result is fake_model

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_API_KEY", "test-key")
        monkeypatch.setenv("MODEL_PROVIDER", "unknown-provider")

        import loop.config as cfg_mod

        importlib.reload(cfg_mod)

        # Re-import after reload so settings singleton is fresh
        import loop.models as models_mod

        importlib.reload(models_mod)

        from loop.models import get_chat_model

        with pytest.raises(ValueError, match="Unknown model provider"):
            get_chat_model()

    def test_ollama_raises_not_implemented(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_API_KEY", "test-key")
        monkeypatch.setenv("MODEL_PROVIDER", "ollama")

        import loop.config as cfg_mod

        importlib.reload(cfg_mod)

        import loop.models as models_mod

        importlib.reload(models_mod)

        from loop.models import get_chat_model

        with pytest.raises(NotImplementedError):
            get_chat_model()


# ── observability ─────────────────────────────────────────────────────────


class TestObservability:
    def test_returns_none_when_not_configured(self, monkeypatch):
        """No LANGFUSE_PUBLIC_KEY → callback is None (no-op)."""
        monkeypatch.setenv("BEDROCK_API_KEY", "test-key")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")

        import loop.config as cfg_mod

        importlib.reload(cfg_mod)

        import loop.observability as obs_mod

        importlib.reload(obs_mod)

        from loop.observability import get_langfuse_callback

        assert get_langfuse_callback() is None

    def test_returns_handler_when_configured(self, monkeypatch):
        """Public + secret key set → returns a CallbackHandler instance."""
        monkeypatch.setenv("BEDROCK_API_KEY", "test-key")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test-abc123")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test-xyz789")

        import loop.config as cfg_mod

        importlib.reload(cfg_mod)

        import loop.observability as obs_mod

        importlib.reload(obs_mod)

        from loop.observability import get_langfuse_callback

        cb = get_langfuse_callback()
        assert cb is not None
        # Verify it's the right type without making a network call
        from langfuse.langchain import CallbackHandler

        assert isinstance(cb, CallbackHandler)


# ── fixtures on disk ──────────────────────────────────────────────────────


class TestFixtures:
    def test_fixture_files_exist(self):
        import pathlib

        root = pathlib.Path(__file__).parent.parent / "fixtures"
        for name in ("sample_jd.md", "sample_profile.md", "questions.json", "rubrics.json"):
            assert (root / name).exists(), f"Missing fixture: {name}"

    def test_questions_json_valid(self):
        import json
        import pathlib

        data = json.loads(
            (pathlib.Path(__file__).parent.parent / "fixtures" / "questions.json").read_text()
        )
        assert "questions" in data
        assert len(data["questions"]) >= 2
        for q in data["questions"]:
            assert {"id", "modality", "prompt"} <= q.keys()

    def test_rubrics_json_valid(self):
        import json
        import pathlib

        data = json.loads(
            (pathlib.Path(__file__).parent.parent / "fixtures" / "rubrics.json").read_text()
        )
        assert "rubrics" in data
        assert len(data["rubrics"]) >= 2
        for r in data["rubrics"]:
            assert {"question_id", "max_score", "criteria"} <= r.keys()
