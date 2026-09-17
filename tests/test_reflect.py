"""
Phase 16b tests — reflect() node: reflection/consolidation.

All offline: model calls are stubbed (RunnableLambda), the store is a plain
or fake-embedded InMemoryStore, never a live model or Bedrock call.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.runnables import RunnableLambda
from langgraph.store.memory import InMemoryStore

from loop.memory import procedural_namespace, put_episode, semantic_namespace
from loop.nodes.reflect import reflect
from loop.schemas import ReflectionInsights

_STUB_INSIGHTS = ReflectionInsights(
    semantic_insights=["struggles with concurrency edge cases"],
    procedural_insights=["give more time on system design"],
)


def _fake_model(return_value):
    fake = MagicMock()
    fake.with_structured_output.return_value = RunnableLambda(lambda _: return_value)
    return fake


def _fake_config(store, user_id="u1"):
    """Monkeypatch langgraph.config.get_store/get_config as reflect.py imports them."""
    import loop.nodes.reflect as reflect_mod

    return reflect_mod


class TestReflectFlag:
    def test_reflect_is_noop_when_disabled(self, monkeypatch):
        monkeypatch.setattr("loop.nodes.reflect.settings.reflection_enabled", False)
        # Even with a real store and episodes present, no-op means: never
        # touches get_store/get_config at all, so this must not raise even
        # outside a graph context.
        assert reflect({}) == {}

    def test_reflect_noop_outside_graph_context(self, monkeypatch):
        """No get_config/get_store context -> RuntimeError is swallowed."""
        monkeypatch.setattr("loop.nodes.reflect.settings.reflection_enabled", True)
        assert reflect({}) == {}

    def test_reflect_noop_when_no_store_wired(self, monkeypatch):
        monkeypatch.setattr("loop.nodes.reflect.settings.reflection_enabled", True)
        monkeypatch.setattr("langgraph.config.get_store", lambda: None)
        monkeypatch.setattr(
            "langgraph.config.get_config", lambda: {"configurable": {"user_id": "u1"}}
        )
        assert reflect({}) == {}

    def test_reflect_noop_when_no_episodes(self, monkeypatch):
        monkeypatch.setattr("loop.nodes.reflect.settings.reflection_enabled", True)
        store = InMemoryStore()
        monkeypatch.setattr("langgraph.config.get_store", lambda: store)
        monkeypatch.setattr(
            "langgraph.config.get_config", lambda: {"configurable": {"user_id": "u1"}}
        )
        monkeypatch.setattr(
            "loop.nodes.reflect.get_chat_model", lambda: _fake_model(_STUB_INSIGHTS)
        )
        assert reflect({}) == {}


class TestReflectConsolidation:
    def test_three_episodes_consolidate_into_insights(self, monkeypatch):
        """A stubbed reflection turns N episodes into semantic + procedural
        insight entries in the store (PLAN.md's '3 episodes -> 1 insight' bar)."""
        store = InMemoryStore()
        put_episode(store, "u1", 1, ["sliding-window"])
        put_episode(store, "u1", 2, ["concurrency"])
        put_episode(store, "u1", 3, ["concurrency", "edge-cases"])
        # session_count must reflect 3 completed sessions for get_recent_episodes
        # to find them -- write it via the semantic weak_areas record.
        store.put(semantic_namespace("u1"), "weak_areas", {"weak_areas": [], "session_count": 3})

        monkeypatch.setattr("loop.nodes.reflect.settings.reflection_enabled", True)
        monkeypatch.setattr("langgraph.config.get_store", lambda: store)
        monkeypatch.setattr(
            "langgraph.config.get_config", lambda: {"configurable": {"user_id": "u1"}}
        )
        monkeypatch.setattr(
            "loop.nodes.reflect.get_chat_model", lambda: _fake_model(_STUB_INSIGHTS)
        )

        result = reflect({})
        assert result == {}

        procedural = store.get(procedural_namespace("u1"), "coaching_notes")
        assert procedural is not None
        assert procedural.value["notes"] == _STUB_INSIGHTS.procedural_insights

        # One semantic insight item was written, topic-keyed off its own text.
        semantic_items = [
            store.get(semantic_namespace("u1"), key)
            for key in ("struggles-with-concurrency-edge-cases",)
        ]
        assert any(item is not None for item in semantic_items)

    def test_reflect_bounds_episode_window(self, monkeypatch):
        """reflect() only reads the last _MAX_EPISODES sessions, not the
        whole history -- bounded, per PLAN.md."""
        import loop.nodes.reflect as reflect_mod

        store = InMemoryStore()
        for n in range(1, 21):
            put_episode(store, "u1", n, [f"topic-{n}"])
        store.put(semantic_namespace("u1"), "weak_areas", {"weak_areas": [], "session_count": 20})

        captured = {}

        def _capture_model():
            fake = MagicMock()

            def intercept(prompt_value):
                captured["prompt"] = str(prompt_value)
                return _STUB_INSIGHTS

            fake.with_structured_output.return_value = RunnableLambda(intercept)
            return fake

        monkeypatch.setattr(reflect_mod.settings, "reflection_enabled", True)
        monkeypatch.setattr("langgraph.config.get_store", lambda: store)
        monkeypatch.setattr(
            "langgraph.config.get_config", lambda: {"configurable": {"user_id": "u1"}}
        )
        monkeypatch.setattr(reflect_mod, "get_chat_model", _capture_model)

        reflect_mod.reflect({})

        assert "topic-20" in captured["prompt"]
        assert "topic-1 " not in captured["prompt"] and "topic-1\n" not in captured["prompt"]
