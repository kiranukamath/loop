"""
Phase 12 tests — MCP & interoperability.

12a — Loop's tools exposed as an MCP server. MCP-over-stdio is a *local
subprocess*, not a network call, so it fits the laptop test gate. These
tests exercise the FastMCP server in-process, asserting:
  1. every tool is registered with the expected name/description/input schema
  2. calling a tool through the MCP layer returns the same data as calling
     the underlying loop.tools function directly (no logic duplication).

12b — Loop consuming an *external* MCP server inside the Phase 9 research
agent. tests/fixtures/mock_mcp_server.py is launched as a REAL subprocess by
the REAL MultiServerMCPClient — a genuine MCP-over-stdio round trip, still
offline (no network involved). Asserts:
  1. load_mcp_tools() returns the fixture server's tool as a BaseTool
  2. invoking it round-trips the canned result through actual MCP-over-stdio
  3. empty config (the default) -> [] and the research node's tool list is
     unchanged (byte-for-byte the Phase 9 flow)
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from loop import tools
from loop.mcp_server import mcp


def _call(tool_name: str, arguments: dict):
    """Call an MCP tool and unwrap the raw Python return value.

    FastMCP.call_tool returns (content_blocks, {"result": <python value>});
    for our purposes we only care about the structured result.
    """
    _, structured = asyncio.run(mcp.call_tool(tool_name, arguments))
    return structured["result"]


class TestToolRegistration:
    def test_all_tools_registered(self):
        registered = asyncio.run(mcp.list_tools())
        names = {t.name for t in registered}
        assert names == {
            "list_questions",
            "search_questions",
            "get_rubric",
            "get_reference_answer",
        }

    def test_tools_have_descriptions_and_schemas(self):
        registered = asyncio.run(mcp.list_tools())
        for t in registered:
            assert t.description, f"{t.name} is missing a description"
            assert t.inputSchema["type"] == "object"

    def test_list_questions_schema_requires_modality(self):
        registered = {t.name: t for t in asyncio.run(mcp.list_tools())}
        schema = registered["list_questions"].inputSchema
        assert schema["required"] == ["modality"]


class TestToolWrappersMatchUnderlyingFunctions:
    def test_list_questions_matches_tools_py(self):
        via_mcp = _call("list_questions", {"modality": "coding"})
        direct = tools.get_questions_by_modality("coding")
        assert via_mcp == direct
        assert len(via_mcp) > 0

    def test_get_rubric_matches_tools_py(self):
        question_id = tools.get_questions_by_modality("coding")[0]["id"]
        via_mcp = _call("get_rubric", {"question_id": question_id})
        direct = tools.get_rubric(question_id)
        assert via_mcp == direct
        assert via_mcp is not None

    def test_get_reference_answer_matches_tools_py(self):
        question_id = tools.get_questions_by_modality("coding")[0]["id"]
        via_mcp = _call("get_reference_answer", {"question_id": question_id})
        direct = tools.get_reference_answer(question_id)
        assert via_mcp == direct

    def test_get_rubric_unknown_id_returns_none(self):
        assert _call("get_rubric", {"question_id": "nope"}) is None

    def test_search_questions_matches_tools_py(self):
        """The autouse stub_embeddings fixture (conftest.py) already keeps
        this offline — every test gets DeterministicFakeEmbedding for free."""
        via_mcp = _call("search_questions", {"query": "arrays and strings", "k": 2})
        direct = tools.search_questions("arrays and strings", k=2)
        assert via_mcp == direct
        assert len(via_mcp) <= 2


# ── 12b: Loop as an MCP client (research agent) ────────────────────────────


class TestLoadMcpToolsDefaultOff:
    def test_empty_config_returns_empty_list(self):
        """settings.mcp_server_configs defaults to {} -- no servers configured,
        no subprocess spawned, no MultiServerMCPClient constructed."""
        from loop.research.mcp_client import load_mcp_tools

        assert load_mcp_tools() == []


class TestLoadMcpToolsRealStdioRoundTrip:
    _CONFIG = {
        "mock": {
            "command": sys.executable,
            "args": ["-m", "tests.fixtures.mock_mcp_server"],
            "transport": "stdio",
        }
    }

    def test_loads_tool_from_fixture_server(self, monkeypatch):
        from loop.config import settings
        from loop.research.mcp_client import load_mcp_tools

        monkeypatch.setattr(settings, "mcp_server_configs", self._CONFIG)
        loaded = load_mcp_tools()
        assert len(loaded) == 1
        assert loaded[0].name == "echo_fact"

    def test_invoking_loaded_tool_roundtrips_canned_result(self, monkeypatch):
        from loop.config import settings
        from loop.research.mcp_client import load_mcp_tools

        monkeypatch.setattr(settings, "mcp_server_configs", self._CONFIG)
        loaded = load_mcp_tools()

        # MCP-loaded tools only implement an async run path (see
        # loop/nodes/research.py's sync/async boundary note) -- .ainvoke(),
        # not .invoke().
        result = asyncio.run(loaded[0].ainvoke({"topic": "MCP"}))
        text = result[0]["text"] if isinstance(result, list) else result
        assert "Canned fact about MCP" in text


class TestResearchNodeUnchangedWithNoMcpConfig:
    def test_tool_list_is_exactly_search_web(self, monkeypatch):
        """With mcp_server_configs empty (the default), the research node's
        tool list must be byte-for-byte the Phase 9 flow: [search_web]."""
        from loop.research.tools import search_web
        from loop.schemas import CompanyResearch

        canned = CompanyResearch(
            company="Acme",
            interview_format="unknown",
            focus_areas=[],
            tech_stack=[],
            recent_news=[],
            sources=[],
        )
        captured_kwargs = {}

        class _FakeAgent:
            async def ainvoke(self, input_, config=None):
                return {"structured_response": canned}

        def _fake_create_agent(**kwargs):
            captured_kwargs.update(kwargs)
            return _FakeAgent()

        monkeypatch.setattr("loop.nodes.research.create_agent", _fake_create_agent)

        from loop.nodes.research import research

        research({"company": "Acme"})
        assert captured_kwargs["tools"] == [search_web]


if __name__ == "__main__":
    pytest.main([__file__])
