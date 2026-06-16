"""
Phase 2 planner tests — offline, model is stubbed.

Tests verify:
- planner node returns a dict with a "plan" key.
- The plan dict matches the PrepPlan schema (all required fields present).
- Weak areas from state flow into the prompt (not swallowed).
- PrepPlan Pydantic schema validates correctly.
"""

import importlib

import pytest

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_fake_plan():
    """Return a PrepPlan-shaped dict the stubbed model will return."""
    return {
        "role_summary": "Senior backend role focused on distributed payments systems.",
        "total_sessions": 4,
        "sessions": [
            {
                "session_number": 1,
                "modality": "coding",
                "topics": ["sliding-window", "two-pointers"],
                "focus": "Build speed on medium array/string problems.",
            },
            {
                "session_number": 2,
                "modality": "system_design",
                "topics": ["payments", "idempotency"],
                "focus": "Practice structured walk-through for payment systems.",
            },
            {
                "session_number": 3,
                "modality": "behavioral",
                "topics": ["conflict", "ownership"],
                "focus": "Tighten STAR structure and conciseness.",
            },
            {
                "session_number": 4,
                "modality": "system_design",
                "topics": ["distributed-cache", "rate-limiting"],
                "focus": "Cover distributed correctness and failure modes.",
            },
        ],
        "key_gaps": ["Kafka", "distributed systems theory", "STAR storytelling"],
        "rationale": (
            "The JD emphasises async event flows and distributed consistency. "
            "The candidate has strong Spring/AWS basics but gaps in Kafka and "
            "distributed theory. Behavioral rounds need STAR practice."
        ),
    }


def _stub_chain(monkeypatch, fake_plan_dict: dict):
    """Patch the planner so it returns a PrepPlan without calling Bedrock.

    Why RunnableLambda instead of MagicMock:
    The planner builds `chain = _PROMPT | structured_model`.  LangChain's pipe
    operator (|) calls __or__ on the prompt, which creates a RunnableSequence.
    When a bare MagicMock is passed as a step, LangChain wraps it as a callable
    (RunnableLambda), so .invoke() on the mock is never called — the mock itself
    is called as a function instead.  Using RunnableLambda makes the fake step a
    proper Runnable that the sequence can .invoke() correctly.
    """
    from unittest.mock import MagicMock

    from langchain_core.runnables import RunnableLambda

    from loop.schemas import PrepPlan

    fake_plan = PrepPlan(**fake_plan_dict)

    # RunnableLambda wraps a plain function as a proper LangChain Runnable.
    # When _PROMPT | fake_runnable is invoked, fake_runnable.invoke() is called.
    fake_runnable = RunnableLambda(lambda _: fake_plan)

    fake_model = MagicMock()
    fake_model.with_structured_output.return_value = fake_runnable

    monkeypatch.setattr("loop.nodes.planner.get_chat_model", lambda: fake_model)
    return fake_plan


# ── schema validation ─────────────────────────────────────────────────────────


class TestPrepPlanSchema:
    def test_valid_plan_parses(self):
        from loop.schemas import PrepPlan

        plan = PrepPlan(**_make_fake_plan())
        assert plan.total_sessions == 4
        assert len(plan.sessions) == 4
        assert plan.sessions[0].modality == "coding"

    def test_invalid_modality_raises(self):
        from pydantic import ValidationError

        from loop.schemas import Session

        with pytest.raises(ValidationError):
            Session(
                session_number=1,
                modality="invalid_modality",  # not in Literal
                topics=["foo"],
                focus="bar",
            )

    def test_missing_required_field_raises(self):
        from pydantic import ValidationError

        from loop.schemas import PrepPlan

        data = _make_fake_plan()
        del data["role_summary"]
        with pytest.raises(ValidationError):
            PrepPlan(**data)


# ── planner node ──────────────────────────────────────────────────────────────


class TestPlannerNode:
    def test_returns_plan_key(self, monkeypatch):
        """planner() must return a dict containing a 'plan' key."""
        _stub_chain(monkeypatch, _make_fake_plan())

        from loop.nodes.planner import planner
        from loop.state import initial_state

        state = initial_state()
        state["jd"] = "Senior backend engineer role."
        state["profile"] = "Java developer, 7 years exp."

        result = planner(state)
        assert "plan" in result

    def test_plan_has_required_fields(self, monkeypatch):
        """Returned plan dict contains all PrepPlan fields."""
        _stub_chain(monkeypatch, _make_fake_plan())

        from loop.nodes.planner import planner
        from loop.state import initial_state

        state = initial_state()
        state["jd"] = "Senior backend engineer."
        state["profile"] = "Java dev."

        result = planner(state)
        plan = result["plan"]

        for field in ("role_summary", "total_sessions", "sessions", "key_gaps", "rationale"):
            assert field in plan, f"Missing field: {field}"

    def test_sessions_list_non_empty(self, monkeypatch):
        _stub_chain(monkeypatch, _make_fake_plan())

        from loop.nodes.planner import planner
        from loop.state import initial_state

        state = initial_state()
        state["jd"] = "x"
        state["profile"] = "y"

        result = planner(state)
        assert len(result["plan"]["sessions"]) > 0

    def test_weak_areas_passed_to_chain(self, monkeypatch):
        """weak_areas from state must be forwarded to the chain invoke call.

        We capture the input via a closure on RunnableLambda instead of
        checking MagicMock.call_args — because the pipe operator wraps steps
        as Runnables and the captured input is the prompt-rendered messages,
        not the raw dict.  So we intercept at the prompt level instead.
        """
        from unittest.mock import MagicMock

        from langchain_core.runnables import RunnableLambda

        from loop.schemas import PrepPlan

        fake_plan = PrepPlan(**_make_fake_plan())

        captured = {}

        def _capture_and_return(messages):
            # At this point messages is the rendered ChatPromptValue.
            # We stash it so we can inspect it after the call.
            captured["messages"] = messages
            return fake_plan

        fake_model = MagicMock()
        fake_model.with_structured_output.return_value = RunnableLambda(_capture_and_return)
        monkeypatch.setattr("loop.nodes.planner.get_chat_model", lambda: fake_model)

        from loop.nodes.planner import planner
        from loop.state import initial_state

        state = initial_state()
        state["jd"] = "x"
        state["profile"] = "y"
        state["weak_areas"] = ["Kafka", "system design structure"]

        planner(state)

        # The rendered prompt messages should contain our weak areas text.
        assert "messages" in captured
        rendered_text = str(captured["messages"])
        assert "Kafka" in rendered_text


# ── full graph integration (offline) ─────────────────────────────────────────


class TestGraphWithPlanner:
    def test_graph_writes_plan_to_state(self, monkeypatch):
        """End-to-end: graph invocation writes plan into state['plan']."""
        _stub_chain(monkeypatch, _make_fake_plan())

        # Re-import graph after patching so the compiled graph uses the stub.
        import loop.graph as graph_mod

        importlib.reload(graph_mod)

        # After reload, all node functions are real again — stub them in the
        # reloaded module's namespace so the full graph can run without interrupts/answers.
        from langgraph.types import Command as _Command

        monkeypatch.setattr(
            graph_mod,
            "plan_approval",
            lambda state: _Command(goto="session_router", update={"plan_approved": True}),
        )
        monkeypatch.setattr(graph_mod, "grader", lambda state: {"grades": []})
        monkeypatch.setattr(graph_mod, "coach", lambda state: {"weak_areas": []})
        monkeypatch.setattr(
            graph_mod,
            "readiness",
            lambda state: {"readiness_verdict": {"verdict": "ready"}, "verdict_approved": True},
        )

        from loop.state import initial_state

        result = graph_mod.compile_graph().invoke(initial_state())
        assert result["plan"] is not None
        assert result["plan"]["total_sessions"] == 4
