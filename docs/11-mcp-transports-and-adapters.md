# 11 — Transports, the LangChain bridge, and bounded agency

> **Status:** Phase 12 is fully done — both 12a (Loop as an MCP server) and
> 12b (Loop consuming external MCP servers in the Phase 9 research agent) are
> built and tested. Every code snippet below reflects what's actually in the
> repo, including the sync/async boundary finding, which turned out to be a
> real bug caught by empirical testing rather than a design choice either
> option would have satisfied — see that section.

## Transports: stdio vs. streamable HTTP/SSE

MCP separates *what* a server exposes (tools/resources/prompts) from *how*
bytes move between client and server. Two transports matter here:

- **stdio** — the client **spawns the server as a local subprocess** and
  talks to it over the subprocess's stdin/stdout. No network, no port, no
  auth handshake. This is exactly how Claude Desktop runs a local MCP server
  you register in its config: it launches your command, pipes JSON-RPC
  messages over the pipe, and kills the process when you quit.
- **streamable HTTP / SSE** — the server is a long-running network service;
  the client connects over HTTP. Needed for a *remote* server (someone
  else's SaaS exposing MCP), but brings in everything a network call brings:
  auth, retries, latency, availability.

**Loop uses stdio everywhere in Phase 12** — both for the server we publish
(12a: `uv run python -m loop.mcp_server`, launched as a subprocess by
whatever client registers it) and the servers we consume (12b: a test
fixture server, spawned the same way). This is a deliberate, CLAUDE.md-driven
choice: stdio is a **local subprocess, not a network call**, so it fits the
laptop offline test gate — the same reasoning that put streamable-HTTP in the
"seam exists, not implemented" bucket, alongside the Tavily search provider
and the Ollama model swap.

```python
# 12a — running Loop's server (this exists now)
if __name__ == "__main__":
    mcp.run(transport="stdio")
```

## The bridge: `langchain-mcp-adapters`

Phase 9 built a ReAct research agent out of LangChain `BaseTool` objects:

```python
create_agent(model, tools=[search_web])
```

An MCP server, though, doesn't hand you `BaseTool`s — it hands you MCP's own
wire-format tool descriptions. **`langchain-mcp-adapters`** is the glue: its
`MultiServerMCPClient` connects to one or more MCP servers and converts each
tool it finds into a `langchain_core.tools.BaseTool`. Once converted, they're
indistinguishable from `search_web` to `create_agent` — this is the same
decoupling lesson as `get_chat_model()`/`get_embeddings()`: the *consumer*
(the ReAct agent) never needs to know or care that some of its tools came
from an external process instead of local Python.

```
MCP server(s)  ──MultiServerMCPClient.get_tools()──▶  list[BaseTool]  ──▶  create_agent(model, tools=[search_web, *mcp_tools])
```

## The sync/async boundary — a real bug, not just a design choice

`MultiServerMCPClient.get_tools()` is a **coroutine** — MCP communication is
inherently async (it's request/response over a pipe, potentially to several
servers concurrently). Loop's Phase 9 `research()` node, however, was a plain
**sync** function, because the rest of the graph (`intake`, `planner`,
`grader`, ...) is sync and `graph.stream()` is called synchronously from
`api.py`. The PLAN framed this as a design choice between two valid options.
Empirical testing (CLAUDE.md rule #7 — verify, don't assume) showed only
**one** of them actually works, and the other one *silently looks fine until
an MCP tool is actually loaded*:

**Attempt 1 — keep `research()` sync, keep calling `agent.invoke()`.**
Tools loaded via `langchain-mcp-adapters` implement **only** an async run
method (`StructuredTool._arun`, no `_run`). The moment the model calls one of
those tools, LangGraph's `ToolNode` tries its sync execution path first and
raises immediately:

```
NotImplementedError: StructuredTool does not support sync invocation.
```

This only surfaces once an MCP server is actually configured and the model
picks its tool — every Phase 9 test (which only ever used the sync-native
`search_web`) would keep passing while this was silently broken.

**Attempt 2 — make `research()` `async def`.** Confirmed the opposite
failure: building a minimal `StateGraph` with one async node and calling the
*parent* graph's sync `.invoke()`/`.stream()` (exactly what `api.py` does)
raises immediately, before the node ever runs:

```
TypeError: No synchronous function provided to "a".
Either initialize with a synchronous function or invoke via the async API
(ainvoke, astream, etc.)
```

LangGraph does not auto-bridge a sync graph invocation to an async node —
you'd have to convert `api.py`'s entire call chain to async, far outside
Phase 12's scope.

**The actual fix:** keep `research()` a plain sync function (so the parent
graph is untouched), but swap its *internal* agent call from
`agent.invoke()` to `asyncio.run(agent.ainvoke(...))`. Verified this is safe
for the existing Phase 9 behavior too — `agent.ainvoke()` runs a pure-sync
tool list (just `search_web`) correctly, because `BaseTool` provides a
default async wrapper for sync-only tools. One call-site change, zero
observable behavior change when no MCP servers are configured:

```python
# loop/nodes/research.py
result = asyncio.run(
    agent.ainvoke(
        {"messages": [HumanMessage(content=f"Research the company: {company}")]},
        config=config,
    )
)
```

The only ripple effect: `tests/test_research.py`'s `_FakeAgent` test double
had to grow an `async def ainvoke(...)` method instead of `def invoke(...)`
— everything else about those tests (what's asserted) is unchanged.

## Bounded agency, reprised

Phase 9 already established the safety rule for a ReAct agent: it must be
**iteration-bounded**, so a model that keeps calling tools in a loop can't
run forever. Consuming *external* MCP servers in 12b re-raises the same
concern from a new angle:

- **The tool allow-list.** `search_web` is a function Loop's own author
  wrote and audited. A tool loaded from someone else's MCP server is code
  Loop didn't write and (for a remote server) doesn't control — it could
  change behavior after you've configured it. `loop/config.py`'s planned
  `mcp_server_configs` field is an explicit **allow-list**: which servers,
  by name, the research agent is permitted to load tools from. **Empty by
  default** — the same "opt-in, feature-off" pattern as `fallback_model_id`
  (Phase 10) and `company` (Phase 9b): with no config, the research agent's
  tool list is byte-for-byte what Phase 9 already tested.
- **Untrusted tool output is a prompt-injection surface.** Whatever text an
  external tool returns gets fed back into the model's context, just like a
  web search result. This ties directly to Phase 10's guardrails
  (`loop/guardrails.py::detect_injection`) — external MCP tool results are
  exactly the kind of untrusted input that gate exists for.
- **The recursion bound still applies** — an external tool doesn't get a
  bigger iteration budget than a local one.

None of this is exotic: it's the same "don't trust a caller you didn't
write" instinct a backend engineer already applies to a third-party library
or an upstream service — MCP just moves that trust boundary to tool calls
instead of HTTP responses.

## Going deeper

This doc stays at the concept level. Four companion docs go one level
lower, each verified against installed source rather than assumed:

- [12 — The MCP wire protocol](12-mcp-wire-protocol.md) — the actual
  JSON-RPC bytes, the `initialize` handshake, and why stdout must never
  contain anything but protocol messages.
- [13 — `FastMCP` internals](13-fastmcp-internals.md) — exactly how
  `@mcp.tool()` turns a type-hinted function into a JSON Schema and a
  validated call path, read from `mcp`'s own source.
- [14 — Async event loop deep dive](14-async-event-loop-deep-dive.md) —
  the full, line-by-line trace of both stack traces behind the sync/async
  finding above, and why the fix is provably safe rather than "it happened
  to work."
- [15 — MCP security & bounded agency](15-mcp-security-and-bounded-agency.md)
  — tool poisoning, confused deputy, indirect prompt injection via tool
  output, and the per-call session cost `MultiServerMCPClient` carries.
