# 13 — `FastMCP` internals: how `@mcp.tool()` actually builds a tool

Doc 10 said "`FastMCP` derives the input schema from type hints and the
description from the docstring — the same way FastAPI derives an OpenAPI
schema." This doc verifies *exactly* how, by reading the installed
`mcp==1.28.1` source (`mcp/server/fastmcp/tools/base.py`,
`mcp/server/fastmcp/utilities/func_metadata.py`) rather than assuming.

## `Tool.from_function()` — the registration path

Every `@mcp.tool()` decorator call eventually runs
`Tool.from_function(fn, ...)`. Its four load-bearing lines
(`mcp/server/fastmcp/tools/base.py:66-77`):

```python
func_doc = description or fn.__doc__ or ""
is_async = _is_async_callable(fn)
func_arg_metadata = func_metadata(fn, ...)
parameters = func_arg_metadata.arg_model.model_json_schema(by_alias=True)
```

Four concrete, verified facts fall out of this:

1. **The description is the *whole* docstring, verbatim, not the first
   line.** `func_doc = description or fn.__doc__ or ""` — there's no
   truncation, no "first sentence only" logic. If `get_rubric`'s docstring
   in `loop/mcp_server.py` had three paragraphs, all three would be sent to
   every client on every `tools/list` call. Keep tool docstrings tight —
   they're bandwidth on every listing, and (more importantly) they're the
   *only* signal the calling model gets to decide whether your tool is
   relevant to what it's trying to do.
2. **There's no docstring-parsing for per-parameter descriptions.**
   `func_metadata.py` builds each parameter's JSON Schema entry purely from
   the function's **type hints and defaults** — it does not parse
   Google/NumPy/Sphinx-style `Args:` sections out of the docstring the way
   some other tools do. If you want a specific parameter to carry its own
   description in the schema, you'd need `Annotated[str, Field(description=
   "...")]` on that parameter — plain type hints (what `loop/mcp_server.py`
   uses throughout) give the client a name and a type, but no per-field
   prose.
3. **The input schema comes from a *dynamically constructed Pydantic
   model*, not manual JSON.** `func_metadata(fn)` builds an
   `arg_model` — an actual `pydantic.BaseModel` subclass created on the fly,
   one field per parameter, matching each parameter's type hint and
   default. `arg_model.model_json_schema(by_alias=True)` is what turns that
   into the JSON Schema a client sees. This is the same mechanism FastAPI
   uses to build OpenAPI schemas from route handler signatures — a
   `Callable`'s signature becomes a Pydantic model becomes a JSON Schema,
   automatically, with no schema-writing by hand. It's why `loop/mcp_server.py`
   never manually writes a schema: `def get_rubric(question_id: str) -> dict
   | None` is *sufficient* input for the whole pipeline.
4. **Sync vs. async is detected once, at registration time, and cached.**
   `is_async = _is_async_callable(fn)` runs exactly once when you call
   `@mcp.tool()`, stored on the `Tool` object as `self.is_async`. This is a
   different mechanism from — and unrelated to — the LangChain/LangGraph
   sync/async story in [doc 14](14-async-event-loop-deep-dive.md): FastMCP's
   own tool-calling path (`Tool.run()`, below) happily calls *either* a sync
   or an async Python function you decorated, because MCP server-side tool
   execution is always awaited from inside the server's own (always-async)
   event loop. The sync/async pain in Phase 12b is specific to
   *LangGraph's* `ToolNode`, not to FastMCP.

## `Tool.run()` — what actually executes when a client calls your tool

```python
async def run(self, arguments, context=None, convert_result=False) -> Any:
    result = await self.fn_metadata.call_fn_with_arg_validation(
        self.fn, self.is_async, arguments, ...
    )
    if convert_result:
        result = self.fn_metadata.convert_result(result)
    return result
```

`call_fn_with_arg_validation` does two things worth naming explicitly:

- **Validates `arguments` against the same Pydantic `arg_model`** used to
  build the schema — so a client that sends `{"question_id": 42}` (an int,
  not a string) for `get_rubric(question_id: str)` gets a clean validation
  error back over MCP, not a Python `TypeError` deep inside `loop/tools.py`.
  This is free correctness Loop didn't have to write — the exact same
  "deserialize into a typed object instead of parsing strings" idea from
  Phase 2's `with_structured_output()`, applied to *inbound* tool arguments
  instead of *outbound* model responses.
- **Awaits `self.fn` regardless of whether it's sync or async** — this is
  where `is_async` (fact #4 above) is actually used: async functions are
  awaited directly, sync functions are called and their plain return value
  is used as-is. Every function in `loop/mcp_server.py`
  (`list_questions`, `search_questions`, `get_rubric`,
  `get_reference_answer`) is a plain `def`, not `async def` — entirely
  fine, because `Tool.run()` handles both.

## Why `call_tool()` returned a *tuple* — what you actually saw in the tests

`tests/test_mcp.py`'s `_call()` helper does:

```python
_, structured = asyncio.run(mcp.call_tool(tool_name, arguments))
return structured["result"]
```

Verified directly (docs/README shows the exact throwaway-script output used
to confirm this before writing any code): `FastMCP.call_tool()` returns
`(content_blocks, {"result": <python value>})`. The two halves serve two
different audiences:

- **`content_blocks`** — a `list[TextContent | ImageContent | ...]`, the
  actual over-the-wire MCP response shape a *real* client (Claude Desktop,
  the MCP Inspector) would receive and render. For a `dict`/`list` return
  value, FastMCP serializes it to a JSON string wrapped in one
  `TextContent(type="text", text="...")` block — this is what a model on
  the client side actually reads.
- **`{"result": ...}`** — a Python-SDK-only convenience: the *original,
  unserialized* Python return value, so code calling `FastMCP` directly (as
  our tests do) doesn't have to `json.loads()` the text block just to
  assert on it. This half never crosses the wire in a real client/server
  pair — it's purely an ergonomic shortcut for in-process callers.

`tests/test_mcp.py`'s `TestToolWrappersMatchUnderlyingFunctions` tests rely
on exactly this second element to assert `via_mcp == direct` (the MCP
wrapper's output equals `loop.tools`'s output) without any JSON round-trip
noise getting in the way of the comparison.

## Tying back to the "no logic duplication" goal

None of this machinery — schema generation, argument validation, async
detection, content-block serialization — required a single line of code in
`loop/mcp_server.py` beyond four `@mcp.tool()`-decorated one-liners. That's
the entire point of choosing `FastMCP` over hand-rolling MCP's raw
`Server` class: the "decorator does the boilerplate" trade Phase 0's
`pydantic-settings` already taught you (`Settings` fields → typed config,
no manual `os.environ.get()` parsing) shows up again here, one layer up
the stack.
