# 10 — MCP roles and primitives

## Three roles: host, client, server

MCP defines three participants in every interaction:

- **Host** — the application the human is actually using: Claude Desktop, an
  IDE, or in our case, Loop's own ReAct research node (Phase 9). The host is
  what decides *when* to reach for a tool.
- **Client** — the piece of code *inside* the host that speaks the MCP wire
  protocol to a specific server. One host can run several clients, one per
  server it's connected to. You (as an app author) rarely hand-write a raw
  client — you use an SDK (`langchain-mcp-adapters`' `MultiServerMCPClient`
  in Loop's case).
- **Server** — the process that actually exposes tools/resources/prompts.
  It doesn't know or care who its host is; it just answers JSON-RPC requests
  like `list_tools` and `call_tool`.

**Loop plays both roles, on purpose, one per sub-phase:**

- **12a — Loop is a server.** `loop/mcp_server.py` exposes our question-bank
  and rubric lookups. Claude Desktop (or any other MCP host) can plug into it
  and use Loop's tools without knowing Python or importing `loop.tools`.
- **12b — Loop is a client (host).** The Phase 9 ReAct research node becomes
  an MCP *host*: it runs a client that connects to *external* MCP servers
  (anyone's) and adds their tools to its own tool list alongside `search_web`.

This is the same client/server duality you already know from a database
connection: your Spring app is a JDBC *client* connecting to a Postgres
*server* — and nothing stops that same JVM from also exposing its own JDBC-
compatible interface to someone else. Client and server are roles a given
process can hold simultaneously, for different connections.

## Three primitives: tools, resources, prompts

MCP servers can expose three kinds of things. Loop's Phase 12 only uses the
first, but knowing all three is worth having:

- **Tools** — functions the *model* decides to call, with arguments it
  chooses, to take an action or fetch dynamic data. This is what Loop
  exposes: `list_questions`, `search_questions`, `get_rubric`,
  `get_reference_answer`. Model-invoked, non-deterministic *when* they're
  called (the model decides).
- **Resources** — readable data addressed by a URI, more like a GET endpoint
  than a function call. The *application* (not necessarily the model)
  decides when to fetch a resource and stuff it into context. Loop's rubrics
  could plausibly be resources instead of tools (`rubric://cod-001`) — we
  chose tools because the interviewer/grader needs to look them up by an ID
  it computes at runtime, which is exactly what a tool call is for.
- **Prompts** — reusable prompt templates a server can hand to a host, so the
  *user* (not the model) can invoke a canned interaction ("summarize this",
  "review this PR"). Loop's interview-question prompts (`loop/nodes/
  interviewers.py`) are a candidate for this if we ever wanted a human to
  trigger "ask me a coding question" directly from Claude Desktop's UI,
  bypassing our own graph entirely — out of scope for Phase 12, but worth
  knowing the primitive exists.

| Primitive | Who decides to use it | Loop's Phase 12 usage |
|---|---|---|
| **Tool** | The model | ✅ all four wrappers in `loop/mcp_server.py` |
| **Resource** | The host application | Not used — mentioned for completeness |
| **Prompt** | The human user | Not used — mentioned for completeness |

## `FastMCP`: the decorator-based server SDK

The official `mcp` Python package ships `mcp.server.fastmcp.FastMCP` — a
class that feels deliberately like FastAPI:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("loop")

@mcp.tool()
def get_rubric(question_id: str) -> dict | None:
    """Get the grading rubric for a question, by question id."""
    return tools.get_rubric(question_id)
```

Two things `FastMCP` derives *automatically* from the plain Python function,
the same way FastAPI derives an OpenAPI schema from a route handler:

- **The input JSON schema** comes from the function's **type hints** —
  `question_id: str` becomes `{"type": "object", "properties": {"question_id":
  {"type": "string"}}, "required": ["question_id"]}`. No schema to hand-write.
- **The tool description** comes from the **docstring** — the first line a
  client sees when it calls `list_tools()`, and the text the *model on the
  other end* reads to decide whether this tool is relevant. A vague docstring
  ("Gets stuff.") means the model on the client side won't know when to call
  it — the docstring isn't documentation for humans here, it's the model's
  only signal.

Verified against the installed `mcp==1.28.1`: `FastMCP.tool()` returns a
decorator; `list_tools()` and `call_tool()` are both `async` — see
[11 — transports and adapters](11-mcp-transports-and-adapters.md) for what
that means for Loop's (currently synchronous) graph nodes.

Next: [11 — MCP transports and the LangChain adapter](11-mcp-transports-and-adapters.md).
