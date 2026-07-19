# 9 — What MCP is, and why it exists

## The problem it solves

Every LLM tool-calling framework invented its own way to describe "here are
the functions you can call." LangChain has `@tool`. OpenAI's function-calling
has its own JSON schema convention. LlamaIndex has another. If you write a
tool as a LangChain `@tool`, only LangChain-based code can use it. Wire it up
to Claude Desktop, Cursor, or someone else's agent runtime, and you're
rewriting the same function against their glue code.

**MCP (Model Context Protocol)**, published by Anthropic in late 2024, is an
attempt to end that duplication: **one open protocol, over JSON-RPC, that any
LLM host can speak to any tool provider.** Write the server once; every
MCP-aware client — Claude Desktop, an IDE, your own agent — can discover and
call its tools without custom glue per client.

## The analogy: JDBC (or LSP)

You already know this pattern from two places in the Java/backend world:

- **JDBC/ODBC** — before it existed, every application had to write custom
  code per database vendor. JDBC standardized *one* driver interface
  (`Connection`, `Statement`, `ResultSet`); any JDBC-compliant app can talk to
  any JDBC-compliant database, and switching from Postgres to MySQL is a
  driver swap, not a rewrite.
- **LSP (Language Server Protocol)** — before it existed, every editor
  (VS Code, IntelliJ, Vim) had to write its own integration for every
  language's autocomplete/go-to-definition/diagnostics. LSP standardized the
  *protocol* between editor and language server. One Python language server
  now works in every LSP-compliant editor.

MCP does the same thing for LLM tools and context: **one protocol, many
hosts, many servers, zero N×M glue code.**

```
Without MCP:              With MCP:

Claude Desktop ─┐         Claude Desktop ─┐
Cursor ─────────┼─ N×M    Cursor ─────────┼── MCP ── Loop's server
Your agent ─────┘  glue   Your agent ─────┘         (and any other MCP server)
     ↕
 Loop's tools (bespoke integration per host)
```

## Why this matters for Loop specifically

Loop already has a "seam philosophy" running through every phase: the model
is hidden behind `get_chat_model()`, embeddings behind `get_embeddings()`,
web search behind `web_search()`. Each seam means *the caller doesn't care
which concrete implementation is behind it.*

MCP is the same lesson taken one level up: instead of hiding a *provider*
behind a factory function, we hide our *tool logic* behind a *protocol*.
`loop/tools.py::get_rubric()` doesn't change at all — Phase 12 puts a thin
MCP wrapper (`loop/mcp_server.py`) in front of it, so the exact same,
already-tested function becomes callable by any MCP client on the planet,
with zero duplicated logic.

## MCP vs. a LangChain `@tool`

| | LangChain `@tool` | MCP tool |
|---|---|---|
| Who can call it | Only LangChain-based code (agents built with LangChain/LangGraph) | Any MCP client — Claude Desktop, Cursor, a LangGraph agent (via an adapter), a raw Python MCP client |
| Transport | In-process Python function call | JSON-RPC over stdio or HTTP — can run in a different process, container, or machine |
| Discovery | You import the Python module and read the code | The client calls `list_tools()` at runtime and gets name + description + JSON schema — no source access needed |
| Loop's usage | Still used internally for tools only *our* graph calls (Phase 3, 9) | Used for tools we want to expose *externally* (12a) and external tools we want to *consume* (12b) |

Loop keeps both. `loop/tools.py` functions remain plain Python, called
directly by our own graph nodes. `loop/mcp_server.py` wraps the *same*
functions in MCP so *other* clients can reach them too — the wrapper adds
nothing, duplicates nothing, and can be deleted without breaking Loop itself.

Next: [10 — MCP roles and primitives](10-mcp-roles-and-primitives.md).
