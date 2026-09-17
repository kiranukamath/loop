"""
Phase 9a tests — search seam, tool wrappers, schema, state.

All offline: the DuckDuckGo (ddgs) provider is a *server* activity, exactly
like live Bedrock calls — never exercised over the network in tests/.
Tests stub loop.research.search.web_search (or the underlying DDGS client)
so nothing here touches the network.
"""

from __future__ import annotations

# ── Search seam ────────────────────────────────────────────────────────────


class TestWebSearch:
    def test_web_search_returns_documented_shape(self, monkeypatch):
        """web_search must return a list of {title, url, snippet} dicts."""

        class _FakeDDGS:
            def text(self, query, max_results):
                return [
                    {
                        "title": "Stripe engineering blog",
                        "href": "https://stripe.com/blog",
                        "body": "...",
                    },
                    {
                        "title": "Stripe interview guide",
                        "href": "https://example.com",
                        "body": "...",
                    },
                ]

        monkeypatch.setattr("ddgs.DDGS", lambda: _FakeDDGS())

        from loop.research.search import web_search

        results = web_search("Stripe engineering culture", k=2)
        assert len(results) == 2
        for r in results:
            assert set(r.keys()) == {"title", "url", "snippet"}
        assert results[0]["title"] == "Stripe engineering blog"
        assert results[0]["url"] == "https://stripe.com/blog"

    def test_web_search_respects_k(self, monkeypatch):
        """web_search should request exactly k results from the provider."""
        captured = {}

        class _FakeDDGS:
            def text(self, query, max_results):
                captured["max_results"] = max_results
                return []

        monkeypatch.setattr("ddgs.DDGS", lambda: _FakeDDGS())

        from loop.research.search import web_search

        web_search("anything", k=3)
        assert captured["max_results"] == 3

    def test_web_search_empty_results(self, monkeypatch):
        """An empty provider response returns an empty list, not an error."""

        class _FakeDDGS:
            def text(self, query, max_results):
                return []

        monkeypatch.setattr("ddgs.DDGS", lambda: _FakeDDGS())

        from loop.research.search import web_search

        assert web_search("obscure query with no results") == []

    def test_tavily_provider_returns_documented_shape(self, monkeypatch):
        """Phase 18c: a Tavily key routes to the real (stubbed) Tavily client."""
        monkeypatch.setattr("loop.research.search.settings.tavily_api_key", "fake-key")

        class _FakeTavilySearch:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def invoke(self, input_):
                return {
                    "query": input_["query"],
                    "results": [
                        {
                            "title": "Stripe engineering blog",
                            "url": "https://stripe.com/blog",
                            "content": "We build APIs.",
                            "score": 0.9,
                        }
                    ],
                }

        monkeypatch.setattr("langchain_tavily.TavilySearch", _FakeTavilySearch)

        from loop.research.search import web_search

        results = web_search("Stripe engineering culture", k=2)
        assert len(results) == 1
        assert results[0] == {
            "title": "Stripe engineering blog",
            "url": "https://stripe.com/blog",
            "snippet": "We build APIs.",
        }

    def test_tavily_no_results_returns_empty_list(self, monkeypatch):
        """TavilySearch raises ToolException when the API finds nothing --
        that must degrade to [], same as ddgs's empty-list behaviour."""
        monkeypatch.setattr("loop.research.search.settings.tavily_api_key", "fake-key")

        from langchain_core.tools import ToolException

        class _FakeTavilySearch:
            def __init__(self, **kwargs):
                pass

            def invoke(self, input_):
                raise ToolException("No results found")

        monkeypatch.setattr("langchain_tavily.TavilySearch", _FakeTavilySearch)

        from loop.research.search import web_search

        assert web_search("obscure query with no results") == []

    def test_duckduckgo_is_default_when_no_tavily_key(self, monkeypatch):
        """With no Tavily key configured, the DuckDuckGo path is used."""
        monkeypatch.setattr("loop.research.search.settings.tavily_api_key", "")

        class _FakeDDGS:
            def text(self, query, max_results):
                return [{"title": "x", "href": "y", "body": "z"}]

        monkeypatch.setattr("ddgs.DDGS", lambda: _FakeDDGS())

        from loop.research.search import web_search

        results = web_search("anything")
        assert len(results) == 1


# ── @tool wrapper ─────────────────────────────────────────────────────────


class TestSearchWebTool:
    def test_search_web_is_a_tool(self):
        """search_web must be a proper LangChain tool (callable via .invoke)."""
        from langchain_core.tools import BaseTool

        from loop.research.tools import search_web

        assert isinstance(search_web, BaseTool)
        assert search_web.name == "search_web"

    def test_search_web_formats_results_as_readable_text(self, monkeypatch):
        """The tool's output is a plain string the model can read directly."""

        def _fake_web_search(query, k=5):
            return [
                {
                    "title": "Stripe Engineering",
                    "url": "https://stripe.com/eng",
                    "snippet": "We build APIs.",
                }
            ]

        monkeypatch.setattr("loop.research.tools.web_search", _fake_web_search)

        from loop.research.tools import search_web

        output = search_web.invoke({"query": "Stripe engineering"})
        assert isinstance(output, str)
        assert "Stripe Engineering" in output
        assert "https://stripe.com/eng" in output
        assert "We build APIs." in output

    def test_search_web_handles_no_results(self, monkeypatch):
        monkeypatch.setattr("loop.research.tools.web_search", lambda query, k=5: [])

        from loop.research.tools import search_web

        output = search_web.invoke({"query": "nonexistent company xyz123"})
        assert "No results found" in output


# ── Schema ────────────────────────────────────────────────────────────────


class TestCompanyResearchSchema:
    def test_company_research_schema_valid(self):
        from loop.schemas import CompanyResearch

        cr = CompanyResearch(
            company="Stripe",
            interview_format="4 rounds: coding, system design, behavioral, bar-raiser",
            focus_areas=["distributed systems", "API design"],
            tech_stack=["Ruby", "Go", "AWS"],
            recent_news=["Launched new payments product"],
            sources=["https://stripe.com/blog"],
        )
        assert cr.company == "Stripe"
        assert len(cr.focus_areas) == 2

    def test_company_research_model_dump_is_json_serialisable(self):
        import json

        from loop.schemas import CompanyResearch

        cr = CompanyResearch(
            company="Stripe",
            interview_format="unknown",
            focus_areas=[],
            tech_stack=[],
            recent_news=[],
            sources=[],
        )
        # Should not raise — proves the dict is safe to store in LoopState.
        json.dumps(cr.model_dump())


# ── State ─────────────────────────────────────────────────────────────────


class TestStateCompanyFields:
    def test_initial_state_has_company_fields(self):
        from loop.state import initial_state

        state = initial_state()
        assert state["company"] is None
        assert state["company_research"] is None


# ── Routing (Phase 9b) ───────────────────────────────────────────────────────


class TestRouteAfterIntake:
    def test_routes_to_research_when_company_set(self):
        from loop.graph import _route_after_intake

        assert _route_after_intake({"company": "Stripe"}) == "research"

    def test_routes_to_planner_when_no_company(self):
        from loop.graph import _route_after_intake

        assert _route_after_intake({"company": None}) == "planner"
        assert _route_after_intake({}) == "planner"

    def test_routes_to_planner_for_empty_string_company(self):
        """An empty string is falsy — treated the same as no company."""
        from loop.graph import _route_after_intake

        assert _route_after_intake({"company": ""}) == "planner"


class TestGraphHasResearchNode:
    def test_research_node_registered(self):
        from loop.graph import build_graph

        g = build_graph()
        assert "research" in g.nodes

    def test_intake_does_not_set_company_by_default(self):
        """Confirms the offline-safety design: intake() never sets company,
        so every pre-Phase-9 flow keeps skipping straight to the planner."""
        from loop.graph import intake

        result = intake({})
        assert "company" not in result


# ── Research node (Phase 9b) ─────────────────────────────────────────────────


class TestResearchNode:
    def _make_fake_agent(self, structured_response, captured: dict):
        """Build a fake object matching create_agent()'s return shape closely
        enough for the research() node: just needs .ainvoke(input, config).

        Phase 12b switched research() from agent.invoke() to
        asyncio.run(agent.ainvoke(...)) — see loop/nodes/research.py's
        module docstring for why (MCP-loaded tools are async-only)."""

        class _FakeAgent:
            async def ainvoke(self, input_, config=None):
                captured["input"] = input_
                captured["config"] = config
                return {"structured_response": structured_response}

        return _FakeAgent()

    def test_research_populates_company_research(self, monkeypatch):
        from loop.schemas import CompanyResearch

        canned = CompanyResearch(
            company="Stripe",
            interview_format="4 rounds: coding, system design, behavioral, bar-raiser",
            focus_areas=["distributed systems", "API design"],
            tech_stack=["Ruby", "Go", "AWS"],
            recent_news=["Launched new payments product"],
            sources=["https://stripe.com/blog"],
        )
        captured = {}
        monkeypatch.setattr(
            "loop.nodes.research.create_agent",
            lambda **kwargs: self._make_fake_agent(canned, captured),
        )

        from loop.nodes.research import research

        result = research({"company": "Stripe"})
        assert result["company_research"]["company"] == "Stripe"
        assert result["company_research"]["focus_areas"] == [
            "distributed systems",
            "API design",
        ]

    def test_research_passes_company_in_the_message(self, monkeypatch):
        from loop.schemas import CompanyResearch

        canned = CompanyResearch(
            company="Acme",
            interview_format="unknown",
            focus_areas=[],
            tech_stack=[],
            recent_news=[],
            sources=[],
        )
        captured = {}
        monkeypatch.setattr(
            "loop.nodes.research.create_agent",
            lambda **kwargs: self._make_fake_agent(canned, captured),
        )

        from loop.nodes.research import research

        research({"company": "Acme"})
        messages = captured["input"]["messages"]
        assert "Acme" in messages[0].content

    def test_research_enforces_iteration_bound(self, monkeypatch):
        """recursion_limit passed to agent.invoke must match settings.research_max_iterations."""
        from loop.config import settings
        from loop.schemas import CompanyResearch

        canned = CompanyResearch(
            company="Stripe",
            interview_format="unknown",
            focus_areas=[],
            tech_stack=[],
            recent_news=[],
            sources=[],
        )
        captured = {}
        monkeypatch.setattr(
            "loop.nodes.research.create_agent",
            lambda **kwargs: self._make_fake_agent(canned, captured),
        )

        from loop.nodes.research import research

        research({"company": "Stripe"})
        assert captured["config"]["recursion_limit"] == settings.research_max_iterations

    def test_research_uses_search_web_tool(self, monkeypatch):
        """create_agent must be given the search_web tool."""
        from loop.research.tools import search_web
        from loop.schemas import CompanyResearch

        canned = CompanyResearch(
            company="Stripe",
            interview_format="unknown",
            focus_areas=[],
            tech_stack=[],
            recent_news=[],
            sources=[],
        )
        captured_kwargs = {}

        def _fake_create_agent(**kwargs):
            captured_kwargs.update(kwargs)
            return self._make_fake_agent(canned, {})

        monkeypatch.setattr("loop.nodes.research.create_agent", _fake_create_agent)

        from loop.nodes.research import research

        research({"company": "Stripe"})
        assert search_web in captured_kwargs["tools"]


# ── Planner grounding in company research (Phase 9b) ─────────────────────────


class TestPlannerCompanyResearch:
    def test_format_company_research_with_data(self):
        from loop.nodes.planner import _format_company_research

        cr = {
            "company": "Stripe",
            "interview_format": "4 rounds",
            "focus_areas": ["distributed systems"],
            "tech_stack": ["Ruby", "Go"],
            "recent_news": ["Launched product X"],
            "sources": ["https://stripe.com"],
        }
        text = _format_company_research(cr)
        assert "Stripe" in text
        assert "distributed systems" in text
        assert "Ruby" in text

    def test_format_company_research_when_none(self):
        from loop.nodes.planner import _format_company_research

        text = _format_company_research(None)
        assert "no company research" in text.lower()

    def test_planner_prompt_includes_company_research(self, monkeypatch):
        """The formatted company research must reach the model prompt."""
        from unittest.mock import MagicMock

        from langchain_core.runnables import RunnableLambda

        from loop.schemas import PrepPlan, Session

        stub_plan = PrepPlan(
            role_summary="stub",
            total_sessions=1,
            sessions=[Session(session_number=1, modality="coding", topics=["x"], focus="y")],
            key_gaps=[],
            rationale="stub",
        )

        captured = {}

        def _capturing_planner():
            fake = MagicMock()

            def _capture(prompt_value):
                captured["prompt_text"] = prompt_value.to_string()
                return stub_plan

            fake.with_structured_output.return_value = RunnableLambda(_capture)
            return fake

        monkeypatch.setattr("loop.nodes.planner.get_chat_model", _capturing_planner)
        monkeypatch.setattr("loop.nodes.planner._get_stored_weak_areas", lambda: [])

        from loop.nodes.planner import planner

        state = {
            "jd": "some JD",
            "profile": "some profile",
            "weak_areas": None,
            "company_research": {
                "company": "Stripe",
                "interview_format": "4 rounds",
                "focus_areas": ["distributed systems"],
                "tech_stack": [],
                "recent_news": [],
                "sources": [],
            },
        }
        planner(state)
        assert "Stripe" in captured["prompt_text"]
        assert "distributed systems" in captured["prompt_text"]

    def test_planner_prompt_shows_placeholder_when_no_research(self, monkeypatch):
        from unittest.mock import MagicMock

        from langchain_core.runnables import RunnableLambda

        from loop.schemas import PrepPlan, Session

        stub_plan = PrepPlan(
            role_summary="stub",
            total_sessions=1,
            sessions=[Session(session_number=1, modality="coding", topics=["x"], focus="y")],
            key_gaps=[],
            rationale="stub",
        )

        captured = {}

        def _capturing_planner():
            fake = MagicMock()

            def _capture(prompt_value):
                captured["prompt_text"] = prompt_value.to_string()
                return stub_plan

            fake.with_structured_output.return_value = RunnableLambda(_capture)
            return fake

        monkeypatch.setattr("loop.nodes.planner.get_chat_model", _capturing_planner)
        monkeypatch.setattr("loop.nodes.planner._get_stored_weak_areas", lambda: [])

        from loop.nodes.planner import planner

        state = {
            "jd": "some JD",
            "profile": "some profile",
            "weak_areas": None,
            "company_research": None,
        }
        planner(state)
        assert "no company research" in captured["prompt_text"].lower()
