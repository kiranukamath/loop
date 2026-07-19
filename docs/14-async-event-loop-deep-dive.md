# 14 — Event loops, and why the sync/async bug happened (full trace)

`doc/phase-12-mcp-interop.md` and `docs/11` summarize the sync/async bug
Phase 12b hit. This doc is the deep dive: what an event loop actually *is*,
traced line-by-line through the two real stack traces that were produced
while building this phase, with the exact source lines responsible — not a
restatement, a full technical account with the Java-world equivalents
called out explicitly.

## What an event loop actually is (for a thread-per-request mental model)

You've spent your career mostly in a **thread-per-request** world: Spring
Boot hands each HTTP request its own OS thread (or a small pool), and
blocking I/O (a JDBC call, an HTTP client call) just... blocks that thread.
The OS scheduler handles concurrency by context-switching between threads.

Python's `asyncio` is a **single-thread, cooperative-multitasking** model
instead — much closer to **Netty's `EventLoopGroup`** or Node.js's event
loop than to a thread pool. One thread runs an **event loop**: a `while
True` loop that holds a queue of pending callbacks/coroutines and, each
iteration, resumes whichever one is ready to make progress (its I/O
finished, its timer fired, etc.). A coroutine that hits `await
some_io()` doesn't block the thread — it *suspends*, handing control back
to the loop, which runs something else, and resumes the coroutine later
when that I/O completes.

**The rule that matters here:** there is at most **one event loop running
per thread at a time.** `asyncio.run(coro)` is the entry point that (a)
creates a brand-new event loop, (b) runs `coro` to completion on it, (c)
closes the loop. If you call `asyncio.run()` from a thread that's already
inside a running event loop, you get `RuntimeError: asyncio.run() cannot
be called from a running event loop` — the Python-world equivalent of
trying to start a second `EventLoopGroup` inside a Netty channel handler
that's already running on one.

This single rule is the entire reason both of the following errors happen,
and the entire reason the fix works.

## Trace 1 — why `agent.invoke()` broke with an MCP tool loaded

Reproduced directly (not from memory) while building 12b: `agent =
create_agent(model, tools=[search_web, *load_mcp_tools()])`, then
`agent.invoke(...)` (sync), with the model choosing to call an MCP-loaded
tool. The full, unmodified traceback:

```
File ".../langgraph/prebuilt/tool_node.py", line 822, in _func
    outputs = list(executor.map(self._run_one, tool_calls, input_types, tool_runtimes))
File ".../langgraph/prebuilt/tool_node.py", line 1006, in _execute_tool_sync
    content = _handle_tool_error(e, flag=self._handle_tool_errors)
File ".../langgraph/prebuilt/tool_node.py", line 958, in _execute_tool_sync
    response = tool.invoke(call_args, config)
File ".../langchain_core/tools/base.py", line 640, in invoke
    return self.run(tool_input, **kwargs)
File ".../langchain_core/tools/base.py", line 968, in run
    response = context.run(self._run, *tool_args, **tool_kwargs)
File ".../langchain_core/tools/structured.py", line 99, in _run
    raise NotImplementedError(msg)
NotImplementedError: StructuredTool does not support sync invocation.
```

Reading it top-down, with the actual installed source at each frame:

1. `ToolNode._func` (`tool_node.py:822`) is the entry point LangGraph uses
   when the **parent agent graph is invoked synchronously** — it maps
   `self._run_one` over the pending tool calls via a `ThreadPoolExecutor`
   (`executor.map`, not `asyncio.gather` — this is LangGraph's own
   sync-path parallelism, one OS thread per tool call).
2. `_run_one` → `_execute_tool_sync` (`tool_node.py:958`) calls
   `tool.invoke(call_args, config)` — the **synchronous** `BaseTool` API.
   Critically, there is **no fallback to `.ainvoke()`** anywhere in this
   path — if the sync call fails because the tool has no sync
   implementation, that's it, the exception propagates.
3. `BaseTool.invoke()` (`langchain_core/tools/base.py:640`) calls
   `self.run(...)`, which eventually calls the tool's `_run()` method — the
   actual "do the work" hook every `BaseTool` subclass is supposed to
   implement.
4. **The MCP-loaded tool is a `StructuredTool` built with only a
   `coroutine=` set, no `func=`** (verified: `langchain-mcp-adapters`'
   `convert_mcp_tool_to_langchain_tool` wraps the async `call_tool`
   coroutine from doc 13 as the tool's `coroutine`, never as `func`). Look
   at `StructuredTool._run()` itself
   (`langchain_core/tools/structured.py:74-98`):

   ```python
   def _run(self, *args, config, run_manager=None, **kwargs):
       if self.func:
           ...
           return self.func(*args, **kwargs)
       msg = "StructuredTool does not support sync invocation."
       raise NotImplementedError(msg)
   ```

   `self.func` is `None` for an MCP-loaded tool (only `self.coroutine` is
   set), so this falls straight through to the `raise` — the exact
   exception message we saw.

**The key insight:** this isn't really about event loops directly — it's
that **an MCP-loaded tool's `_run()` genuinely doesn't exist**, by
construction, because the underlying operation (talk to a subprocess over
JSON-RPC) can only sensibly be expressed as a coroutine. `ToolNode`'s sync
path has no way to run a coroutine without an event loop to run it on, so
rather than silently spinning one up, LangChain just refuses.

## Trace 2 — why making `research()` `async def` also broke

The natural instinct once trace 1 is understood: "fine, make the *node*
async, so its internals can `await` things directly." Verified this
independently with a minimal reproduction — a two-line `StateGraph` with one
`async def` node, invoked via the graph's plain sync `.invoke()` (exactly
what `api.py`'s `stream_session` route does to `_graph.stream(...)`):

```
File ".../langgraph/_internal/_runnable.py", line 356, in invoke
    raise TypeError(
TypeError: No synchronous function provided to "a".
Either initialize with a synchronous function or invoke via the async API (ainvoke, astream, etc.)
```

The relevant source, `RunnableCallable.invoke` (`_runnable.py:353-361`):

```python
def invoke(self, input, config=None, **kwargs):
    if self.func is None:
        raise TypeError(
            f'No synchronous function provided to "{self.name}".'
            "\nEither initialize with a synchronous function or invoke"
            " via the async API (ainvoke, astream, etc.)"
        )
```

Every node you register with `StateGraph.add_node(name, fn)` gets wrapped in
a `RunnableCallable`. If `fn` is a plain function, LangGraph populates
`self.func` (the sync slot); if `fn` is a coroutine function, it populates
`self.afunc` (the async slot) **instead of** `self.func`. The graph's own
sync `.invoke()`/`.stream()` entry points call each node's `RunnableCallable
.invoke()` directly — and for a node whose `self.func` is `None` (an
async-only node), that raises immediately, **before the node body ever
runs**. There is no fallback to `.ainvoke()` here either, for the same
reason as trace 1: nothing decides to spin up an event loop on your behalf
just because it would make things work.

This confirms the two options the PLAN framed as "either is valid" are
actually **mutually exclusive with each other's failure mode**: sync agent
call + async-only tool fails one way; async node + sync graph invocation
fails the other way. There was no version of "pick either" that actually
worked — only the third option below did.

## The fix, and why it's actually safe (not just "it happened to work")

```python
# loop/nodes/research.py — research() stays a plain `def`, sync signature
result = asyncio.run(
    agent.ainvoke({"messages": [...]}, config=config)
)
```

Two things had to be true for this to be correct, both verified rather than
assumed:

1. **`agent.ainvoke()` must work identically to `agent.invoke()` when *no*
   MCP tools are loaded** (the default, and every existing Phase 9 test's
   scenario) — verified directly: a `create_agent(model, tools=[search_web])`
   agent (a pure-sync tool, `search_web`, no `coroutine=` set) runs
   correctly through `.ainvoke()`, because `BaseTool` provides a **default
   async wrapper for sync-only tools** — it runs the sync `_run()` in a
   thread-pool executor under the hood and awaits that. So switching the
   call from `.invoke()` to `asyncio.run(...ainvoke...)` changes *nothing*
   observable for Phase 9's existing tool list; it only *adds* the ability
   to also run genuinely async-only (MCP) tools correctly.
2. **`asyncio.run()` must not be called from a thread that already has an
   event loop running**, or it raises the "cannot be called from a running
   event loop" error described at the top of this doc. Traced this all the
   way up `research()`'s actual call stack in the real app: `research()` is
   a plain LangGraph node, invoked by `_graph.stream(...)` (sync) inside
   `stream_session()` in `loop/api.py` — and that route handler is a plain
   `def`, not `async def`. `loop/api.py`'s own top-of-file comment already
   states the relevant fact: **"FastAPI runs sync routes in a threadpool
   automatically"** — meaning `stream_session` executes on a plain
   worker thread from Starlette's threadpool, a thread that has **no event
   loop of its own** (uvicorn's single event loop lives on the main thread,
   handling the *dispatch* to that threadpool, not the request body
   itself). `asyncio.run()` inside `research()`, called from that worker
   thread, is therefore creating the **first and only** event loop on that
   thread — completely safe, by construction of how FastAPI/Starlette
   already dispatches sync routes, a design decision Phase 7 made and
   documented well before Phase 12 ever needed to rely on it.

Point 2 is worth sitting with: Phase 12b didn't have to change anything
about `api.py`, precisely *because* Phase 7 already made the "sync routes
run off the main event loop's thread" choice. A different Phase 7 design
(e.g., an `async def` route calling `_graph.stream()` directly on the main
event loop thread) would have made this exact `asyncio.run()` call in
`research()` raise the "already running" error — the fix would then have had
to be different (likely: making the whole call chain async, or using
`asyncio.run_coroutine_threadsafe` against a loop running in a background
thread). This is a good illustration of why "verify the actual call path,
not just the function you're editing" matters — the correctness of a
three-line change in `research.py` depended on an architectural decision
made five phases earlier.

## The test-double ripple effect

One more small, honest consequence: `tests/test_research.py`'s `_FakeAgent`
test double had a `def invoke(self, input_, config=None)` method. Once
`research()` calls `agent.ainvoke()`, that fake needed an `async def
ainvoke(...)` instead — otherwise `asyncio.run(fake_agent.ainvoke(...))`
would fail because there'd be no `ainvoke` attribute at all. This is a
one-line change per test double, not a logic change — every assertion in
`test_research.py` (what's captured in `input`/`config`, what
`company_research` ends up containing) is byte-for-byte unchanged. It's
mentioned here only because it's a small, easy-to-forget ripple whenever
you change a real dependency's call convention: the test doubles standing
in for that dependency have to track the same interface change.
