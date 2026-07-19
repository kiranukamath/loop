# Phase 12 — MCP & interoperability

**Capability taught:** interop via the Model Context Protocol — the same
seam philosophy that's run through every phase (`get_chat_model()`,
`get_embeddings()`, `web_search()`), applied to *tool logic itself* instead
of a provider.

**Status:** 12a and 12b both done & tested. Phase 12 is fully done.

See [docs/09–11](../docs/README.md#phase-12--mcp--interoperability) for the
underlying concepts (what MCP is, roles/primitives, transports/safety) — this
doc is the "what we actually built" walkthrough, in the same spirit as
[phase-10](phase-10-production-hardening.md) and
[phase-11](phase-11-session-history.md).

## Why this phase exists

Every phase so far hid a *provider* behind a factory function so the caller
never cares which concrete implementation answers the call. Phase 12 takes
that one level up: it hides Loop's *tool logic* behind an *industry-standard
protocol* instead of bespoke Python imports. The same `loop/tools.py`
functions Phase 3 wrote and Phase 8 extended become callable by Claude
Desktop, an IDE, or anyone else's agent — with zero duplicated logic.

## Verifying the API before writing any code (CLAUDE.md rule #7)

Neither `mcp` nor `langchain-mcp-adapters` were installed before this phase,
and MCP's SDK moves fast, so nothing here was written from memory. Before
touching `loop/mcp_server.py`:

```bash
uv add mcp langchain-mcp-adapters
# resolved: mcp==1.28.1, langchain-mcp-adapters==0.3.0
```

Then the actual installed shapes were inspected directly, in a throwaway
Python session, rather than trusted from documentation that may lag the
installed version:

```python
from mcp.server.fastmcp import FastMCP
import inspect
inspect.signature(FastMCP.__init__)   # confirms FastMCP(name: str | None = None, ...)
inspect.signature(FastMCP.tool)       # confirms .tool() -> decorator, no required args
inspect.signature(FastMCP.run)        # confirms .run(transport: Literal["stdio","sse","streamable-http"] = "stdio")
```

Two things worth calling out that only showed up by actually running the
code, not by reading the PLAN's expectation:

- **`list_tools()` and `call_tool()` are both `async`**, even though
  registering a tool with `@mcp.tool()` is plain sync code. This matters for
  writing tests (see below) and previews the 12b sync/async boundary
  question for the research node.
- **`call_tool(name, args)` returns a tuple**: `(content_blocks,
  {"result": <actual python value>})`. The first element is MCP's
  wire-format text content (what a real client sees); the second is a
  convenience the Python SDK gives you so you don't have to re-parse JSON
  just to assert on the return value in a test.

## `loop/mcp_server.py` — the server, one tool per existing function

```python
from mcp.server.fastmcp import FastMCP
from loop import tools

mcp = FastMCP("loop")

@mcp.tool()
def get_rubric(question_id: str) -> dict | None:
    """Get the grading rubric for a question, by question id."""
    return tools.get_rubric(question_id)
```

Four tools, each one line of delegation to a function that already existed
and was already tested (Phase 3's `get_questions_by_modality`/`get_rubric`/
`get_reference_answer`, Phase 8's `search_questions`):

| MCP tool | Delegates to | Added in |
|---|---|---|
| `list_questions(modality)` | `tools.get_questions_by_modality` | Phase 3 |
| `search_questions(query, modality, k)` | `tools.search_questions` (→ `loop.retrieval`) | Phase 8 |
| `get_rubric(question_id)` | `tools.get_rubric` | Phase 3 |
| `get_reference_answer(question_id)` | `tools.get_reference_answer` | Phase 8c |

Type hints become the tool's input JSON schema automatically; the docstring
becomes the description a client-side model reads to decide whether the
tool is relevant. Nothing here re-implements fixture-reading, JSON parsing,
or the retrieval index — the entire file is a translation layer.

The stdio entrypoint is the last three lines:

```python
if __name__ == "__main__":
    mcp.run(transport="stdio")
```

`uv run python -m loop.mcp_server` runs this as a long-lived process that
reads JSON-RPC requests from stdin and writes responses to stdout — exactly
what Claude Desktop (or the MCP Inspector) launches when you register it.

## Registering Loop's server in a real MCP client (manual/server activity)

This is a *server activity*, not part of the laptop test gate — the same
category as a live Bedrock call. To try it against a real client:

**Claude Desktop** — add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "loop": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/loop", "python", "-m", "loop.mcp_server"]
    }
  }
}
```

Restart Claude Desktop; Loop's four tools should appear under the 🔌 tools
menu, callable from a normal conversation.

**MCP Inspector** (a browser-based debugging tool, no client app needed):

```bash
npx @modelcontextprotocol/inspector uv run python -m loop.mcp_server
```

This opens a local web UI that lists the tools, lets you call them with
arbitrary arguments, and shows the raw JSON-RPC traffic — the fastest way to
sanity-check a server without wiring up a full client.

## Testing strategy — why this stays inside the laptop gate

MCP-over-stdio is a **local subprocess talking over stdin/stdout**, not a
network call — so unlike Tavily search or a live Bedrock call, it's fully
testable offline. `tests/test_mcp.py` doesn't even need to spawn a
subprocess for 12a: it calls the `FastMCP` instance's own async methods
**in-process**, which already exercises the exact same code path a real
client would hit (`list_tools()` → registration; `call_tool()` → the tool
function + MCP's serialization wrapper):

```python
def _call(tool_name: str, arguments: dict):
    _, structured = asyncio.run(mcp.call_tool(tool_name, arguments))
    return structured["result"]
```

Two things are asserted per tool:

1. **Registration** — `list_tools()` returns exactly the four expected
   names, each with a non-empty description and an object-shaped input
   schema (`test_all_tools_registered`, `test_tools_have_descriptions_and_
   schemas`, `test_list_questions_schema_requires_modality`).
2. **No logic duplication** — calling a tool *through* MCP returns
   identical data to calling the underlying `loop.tools` function directly
   (`TestToolWrappersMatchUnderlyingFunctions`). If someone ever edits
   `mcp_server.py` to reimplement logic instead of delegating, these tests
   catch the drift immediately.

`search_questions` needed no extra stubbing — the existing autouse
`stub_embeddings` fixture in `tests/conftest.py` (Phase 8) already swaps in
`DeterministicFakeEmbedding` for *every* test in the suite, so retrieval
through the MCP layer is offline for free.

A real **stdio subprocess round-trip** (spawn the server, talk to it over an
actual pipe, using the real `MultiServerMCPClient`) is deliberately deferred
to 12b, where it's needed anyway to prove Loop can *consume* an external
server — see the mock fixture server planned there.

## Result

8 new tests, all offline; 250 total passing; `ruff check` clean. Manually
verified `uv run python -m loop.mcp_server` starts cleanly under stdio (no
import/runtime errors) and exits gracefully on stdin EOF.

## 12b — Loop consumes external MCP servers (research agent)

Where 12a made Loop's own tools reachable by *any* MCP client, 12b makes the
Phase 9 ReAct research agent an MCP *client* — able to load and call tools
from *someone else's* MCP server, merged into its existing `[search_web]`
tool list. Same host process, two different roles for two different
connections — see [docs/10](../docs/10-mcp-roles-and-primitives.md).

### `loop/research/mcp_client.py` — the loader

```python
def load_mcp_tools() -> list[BaseTool]:
    if not settings.mcp_server_configs:
        return []
    from langchain_mcp_adapters.client import MultiServerMCPClient
    client = MultiServerMCPClient(settings.mcp_server_configs)
    return asyncio.run(client.get_tools())
```

`settings.mcp_server_configs` (`loop/config.py`) is a `dict[str, dict]`
allow-list, **empty by default** — the same opt-in pattern as Phase 10's
`fallback_model_id` and Phase 9b's `company`. With no servers configured,
`load_mcp_tools()` returns `[]` before ever importing
`langchain_mcp_adapters` or spawning a subprocess.

`loop/nodes/research.py` wires it in as one line:

```python
tools=[search_web, *load_mcp_tools()],
```

### The sync/async boundary — the one real bug this phase found

The PLAN framed the sync/async crossing as a design choice between two
options. Building both and running them proved only one actually works —
see [docs/11's sync/async section](../docs/11-mcp-transports-and-adapters.md#the-syncasync-boundary--a-real-bug-not-just-a-design-choice)
for the full empirical trace. Short version:

- Keeping `research()` sync and calling `agent.invoke()` (the old Phase 9
  code) **breaks the instant an MCP tool is actually loaded**, because
  `langchain-mcp-adapters` tools only implement an async run method, and
  LangGraph's `ToolNode` doesn't fall back to async when the sync path
  fails: `NotImplementedError: StructuredTool does not support sync
  invocation.` This wouldn't have shown up in any existing test — only in
  a live run with a real MCP server configured.
- Making `research()` itself `async def` breaks the *other* direction: the
  parent graph is invoked synchronously everywhere else in Loop
  (`graph.stream()` in `api.py`), and LangGraph refuses to run an async
  node from a sync graph invocation: `TypeError: No synchronous function
  provided to "research". ... invoke via the async API.`
- **The fix:** `research()` stays sync, but its one internal call becomes
  `asyncio.run(agent.ainvoke(...))` instead of `agent.invoke(...)`. Verified
  this doesn't change Phase 9's existing behavior — `agent.ainvoke()` runs a
  pure-sync tool list (`search_web` alone) exactly as before.

This is a good illustration of why CLAUDE.md rule #7 ("verify library APIs,
don't trust memorized behavior") matters even for architectural decisions,
not just import paths — the PLAN's own text presented this as two equally
valid choices, and only one of them actually worked once tested against the
real installed `langgraph`/`langchain-mcp-adapters` versions.

### Testing strategy — a real fixture server, spawned for real

`tests/fixtures/mock_mcp_server.py` is a second, tiny `FastMCP` server (one
tool, `echo_fact`) that stands in for "someone else's MCP server." Unlike
12a's in-process tests, 12b's tests genuinely spawn it as a subprocess via
the real `MultiServerMCPClient` — proving the whole stdio wire protocol
round-trips, not just the Python-level plumbing:

```python
_CONFIG = {"mock": {"command": sys.executable,
                     "args": ["-m", "tests.fixtures.mock_mcp_server"],
                     "transport": "stdio"}}

def test_loads_tool_from_fixture_server(self, monkeypatch):
    monkeypatch.setattr(settings, "mcp_server_configs", self._CONFIG)
    loaded = load_mcp_tools()
    assert loaded[0].name == "echo_fact"
```

This still satisfies the laptop offline gate — stdio spawns a **local
subprocess**, never touches the network. `tests/test_mcp.py` (12 tests total
now) also asserts the empty-config path returns `[]`, and that with no MCP
servers configured, the research node's tool list is `== [search_web]` —
byte-for-byte the Phase 9 flow, no regression.

## Result (both sub-steps)

12 tests in `tests/test_mcp.py` (4 registration/wrapper tests + 1 empty-config
test + 2 real-subprocess round-trip tests + 1 tool-list-unchanged test, plus
3 covered under `TestToolWrappersMatchUnderlyingFunctions`), all offline.
254 tests total across the whole suite, `ruff check` clean. Phase 12 —
MCP & interoperability — is done.
