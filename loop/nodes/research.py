"""
Research node — Phase 9b: Loop's first dynamic (ReAct) sub-agent.

Every other node in this graph is a FIXED workflow step: it always does the
same thing (call the model once, maybe call interrupt()).  This node is
different — it hands control flow to the model.  create_agent() builds a
small ReAct graph (reason -> call a tool -> observe the result -> repeat)
and we embed that whole sub-graph as a single node in the parent graph.

Only invoked when state["company"] is set — see loop/graph.py's
_route_after_intake, which skips straight to the planner otherwise.

Bounded agency: research_max_iterations (loop/config.py) is passed as
LangGraph's recursion_limit, so a runaway tool-calling loop cannot exceed
that many super-steps.

API note: PLAN.md originally specified langgraph.prebuilt.create_react_agent.
That function is deprecated in the installed langgraph-prebuilt==1.1.0 in
favor of langchain.agents.create_agent (already present via langchain==1.3.9).
Verified via inspect.signature()/help(): response_format=<PydanticModel>
populates result["structured_response"] with a validated instance — the
same mechanism the deprecated function offered.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from loop.config import settings
from loop.models import get_chat_model
from loop.observability import get_langfuse_callback
from loop.research.tools import search_web
from loop.schemas import CompanyResearch

_SYSTEM_PROMPT = """You are a research assistant helping prepare a candidate for a \
technical interview at a specific company.

Use the search_web tool to find real, current information about the company:
its interview process/format, technical and behavioral focus areas, tech stack,
and recent relevant news. Call search_web as many times as you need, but be
efficient — a handful of well-chosen queries beats many redundant ones.

Once you have enough information, produce a structured research brief. If you
cannot find something, say so plainly rather than guessing."""


def research(state: dict) -> dict:
    """Run the ReAct research sub-agent for state["company"].

    Returns {"company_research": CompanyResearch.model_dump()}.
    Only called when state["company"] is truthy — see graph.py routing.
    """
    company = state["company"]

    agent = create_agent(
        model=get_chat_model(),
        tools=[search_web],
        system_prompt=_SYSTEM_PROMPT,
        response_format=CompanyResearch,
    )

    cb = get_langfuse_callback()
    config = {
        "callbacks": [cb] if cb else [],
        # Bounds the ReAct loop: the model can reason/act/observe at most this
        # many super-steps before LangGraph raises GraphRecursionError.
        "recursion_limit": settings.research_max_iterations,
    }

    result = agent.invoke(
        {"messages": [HumanMessage(content=f"Research the company: {company}")]},
        config=config,
    )

    structured: CompanyResearch = result["structured_response"]
    return {"company_research": structured.model_dump()}
