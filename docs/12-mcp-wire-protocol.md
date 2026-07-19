# 12 — The MCP wire protocol: JSON-RPC, the handshake, and stdio framing

Docs 9–11 covered MCP conceptually — the protocol's purpose, roles, transports.
This doc goes one level lower: **what actually crosses the wire** when
`loop/mcp_server.py` talks to a client, byte for byte. Understanding this
layer matters because it explains two very concrete, very easy-to-hit
gotchas (stdout purity, and why every call is a two-step negotiation) that
no amount of reading `FastMCP`'s Python API would tell you.

## MCP is JSON-RPC 2.0, nothing more exotic

Every message on the wire is a JSON-RPC 2.0 object. If you've ever worked
with a JSON-RPC or a simple bidirectional message bus, this will look
completely familiar — no custom binary framing, no special encoding.

**Requests** (expect a response, correlated by `id`):

```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
```

**Responses:**

```json
{"jsonrpc": "2.0", "id": 1, "result": {"tools": [...]}}
```

**Notifications** (fire-and-forget, no `id`, no response expected) — used for
things like "my tool list changed, re-fetch it":

```json
{"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}
```

The methods Loop's server actually implements (via `FastMCP`, automatically,
from the `@mcp.tool()` decorators): `tools/list` and `tools/call`. A server
that also exposed resources/prompts (docs/10) would additionally answer
`resources/list`/`resources/read` and `prompts/list`/`prompts/get`.

## The `mcp.call_tool()` you called in tests — one level up

`tests/test_mcp.py`'s `_call()` helper calls `mcp.call_tool(name, args)`
directly on the `FastMCP` **object** — that's calling Python code in-process,
skipping JSON-RPC entirely (no client, no subprocess, no serialization).
That's exactly why 12a's tests could assert on Python dicts directly. The
**12b** tests are different in kind: `MultiServerMCPClient` genuinely spawns
`tests/fixtures/mock_mcp_server.py` as a subprocess and exchanges real
JSON-RPC messages with it over its stdin/stdout — the wire protocol
described in this doc is actually exercised there, not simulated.

## The handshake: `initialize` before anything else

A client can't just fire off `tools/list` the instant the subprocess starts.
MCP requires a capability-negotiation handshake first — the same idea as a
TLS handshake or an HTTP/2 SETTINGS frame: both sides agree on what they
support *before* doing real work.

1. **Client → Server: `initialize` request** — carries the client's
   `protocolVersion` and which capabilities it supports (does it support
   sampling? roots? elicitation?).
2. **Server → Client: `initialize` response** — the server's own
   `protocolVersion` + capabilities (does it have tools? resources? prompts?
   does it support `list_changed` notifications for each?) + a `serverInfo`
   block (name, version).
3. **Client → Server: `initialized` notification** — "handshake
   acknowledged, you may now send other requests."

Only after step 3 will a well-behaved server accept `tools/list`. You can
see this exact sequence in `langchain-mcp-adapters`' own code
(`MultiServerMCPClient.session()`, `loop/../.venv/.../langchain_mcp_adapters/
client.py`): it opens the transport, then explicitly calls
`await session.initialize()` before ever calling `get_tools()` — the
adapter library is doing this handshake *for you*, which is exactly why
`load_mcp_tools()` (Phase 12b) never has to think about it.

## stdio framing: why `stdout` must contain *only* protocol messages

This is the gotcha that will bite you the first time you add a `print()`
statement for debugging. Over the stdio transport, the two processes agree
on the simplest possible framing: **each JSON-RPC message is one line of
JSON, newline-delimited, written to stdout (server → client) or stdin
(client → server).** There is no length prefix, no envelope, no
out-of-band channel — the parser just reads a line, and expects it to
`json.loads()` cleanly into a JSON-RPC object.

**Consequence:** if your server's code (or any library it imports) writes
*anything* else to stdout — a stray `print()`, a library's startup banner,
a warning printed via `print()` instead of `logging` — the client's line
parser will try to parse that line as JSON-RPC and either crash or silently
desync the whole session. This is exactly why `FastMCP` (and the `mcp`
SDK generally) route **all logging to stderr**, never stdout, and why
`loop/mcp_server.py` doesn't add any `print()` calls of its own — stdout is
a dedicated, single-purpose pipe for protocol traffic only. If you ever need
to debug this server by hand, always print to stderr (`print(..., file=sys.stderr)`)
or use Python's `logging` module (which defaults to stderr).

The Java-world analogy: this is the same discipline as never writing debug
output to a socket you're using for a binary protocol — you'd reach for a
side-channel logger, not `System.out.println`, inside a Netty channel
handler. stdio MCP is the same constraint, just visible because stdout
*looks* like a safe place to print until you remember a subprocess pipe
is being read as a structured stream, not a terminal.

## Session lifecycle for a stdio server (what actually happens on each test run)

Putting the whole thing together — this is literally what happens every
time `tests/test_mcp.py::TestLoadMcpToolsRealStdioRoundTrip` runs:

```
1. MultiServerMCPClient spawns:  python -m tests.fixtures.mock_mcp_server
                                  (a new OS process, stdin/stdout piped)
2. Client → Server (stdin):      {"jsonrpc":"2.0","id":1,"method":"initialize",...}
3. Server → Client (stdout):     {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":...,"capabilities":{"tools":{}},...}}
4. Client → Server (stdin):      {"jsonrpc":"2.0","method":"notifications/initialized"}
5. Client → Server (stdin):      {"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
6. Server → Client (stdout):     {"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"echo_fact",...}]}}
7. Client → Server (stdin):      {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"echo_fact","arguments":{"topic":"MCP"}}}
8. Server → Client (stdout):     {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"Canned fact about MCP: ..."}]}}
9. Client closes stdin -> server sees EOF -> process exits.
```

Every one of those eight round-trip lines is genuinely happening — this
isn't a mock at the JSON-RPC layer, only the *tool's business logic* is
canned (`echo_fact` always returns the same string). That's precisely why
this test satisfies the "MCP-over-stdio is a local subprocess, not a
network call" reasoning from CLAUDE.md: the entire protocol is exercised,
for real, with zero network involvement — the pipe is process-to-process
on the same machine.

## Why `Connecting to a server for each tool call` matters (forward pointer)

One detail that becomes very relevant for a research-agent-shaped consumer:
`MultiServerMCPClient.get_tools()`'s own docstring says *"a new session will
be created for each tool call."* That means every `tools/call` on an
MCP-loaded tool re-runs steps 1–4 above (spawn a fresh subprocess,
handshake again) before it even gets to step 7. [Doc 14](14-async-event-loop-deep-dive.md)
covers the async mechanics this implies; the performance and design
tradeoff is worth its own note in [doc 15](15-mcp-security-and-bounded-agency.md#the-per-call-session-cost-a-production-tradeoff-not-just-a-safety-one).
