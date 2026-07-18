# Phase 10 — Production hardening

**Capability taught:** resilience, safety, cost — the concerns that separate a
weekend demo from something you'd actually run in production.

## Why this phase exists

Every phase before this one made Loop *smarter* (planning, orchestration,
memory, HITL, eval, RAG, tool use). Phase 10 makes it *safe to operate*.
Three backend-engineer instincts, applied to an LLM app instead of a REST API:

1. **What happens when the downstream call fails?** (resilience)
2. **What happens when the input is hostile?** (guardrails)
3. **What happens when the meter is running and nobody's watching it?** (budget)

If you've built a Spring service that calls a flaky third-party API, you've
already solved all three of these problems once — Resilience4j retries,
request validation, and a rate limiter / quota. This phase is the same three
patterns, aimed at model calls instead of HTTP calls.

---

## 10a — Resilience in the model seam

### The concept

LangChain's `Runnable` base class — which `ChatBedrockConverse`, prompts,
parsers, and chains all implement — ships two decorators:

- **`.with_retry(...)`** — re-invoke on failure, with exponential backoff.
  Same idea as `@Retry` in Resilience4j.
- **`.with_fallbacks([...])`** — if the wrapped runnable's retries are
  exhausted, try a different runnable instead. Same idea as a circuit
  breaker's fallback path.

Both are generic — they don't know or care whether the thing they're wrapping
is a chat model, a chain, or a plain function. That genericity is also where
the first real gotcha of this phase showed up.

### The gotcha (and why we deviated from the plan text)

The original `PLAN.md` text says: *"wrap the returned model with
`.with_retry(...)` and `.with_fallbacks(...)`"* — implying you'd do this
inside `get_chat_model()`, once, and every caller benefits.

That doesn't work here. Every node in Loop calls
`model.with_structured_output(Schema)` to get Pydantic-typed output back
instead of a raw `AIMessage`. `.with_structured_output()` is a method
*specific to `BaseChatModel`* — it's not part of the generic `Runnable`
interface. I verified this directly against the installed `langchain-core`:

```python
model = get_chat_model()
retried = model.with_retry()
hasattr(retried, "with_structured_output")   # False
```

`model.with_retry()` returns a `RunnableRetry` — a generic wrapper that does
**not** proxy unknown attributes through to the model it wraps. So if
`get_chat_model()` returned a retry-wrapped model, every node's
`.with_structured_output(Schema)` call would immediately blow up with
`AttributeError`.

**The fix:** wrap resilience around the *finished chain*
(`prompt | model.with_structured_output(Schema)`), not the bare model.
`.with_retry()`/`.with_fallbacks()` work fine there because at that point
you're not calling any chat-model-specific method anymore — you're just
composing runnables, and retry/fallback are runnable-level concerns.

```python
# loop/models.py
def with_resilience(chain: Runnable, fallback_chain: Runnable | None = None) -> Runnable:
    if fallback_chain is not None:
        chain = chain.with_fallbacks([fallback_chain])
    return chain.with_retry(stop_after_attempt=settings.retry_max_attempts)
```

Each node (`planner.py`, `grader.py`, `coach.py`, `readiness.py`) builds its
chain exactly as before, then wraps it:

```python
chain = _PROMPT | structured_model

fallback_chain = None
if settings.fallback_model_id:
    fallback_model = get_chat_model(settings.fallback_model_id)
    fallback_chain = _PROMPT | fallback_model.with_structured_output(PrepPlan)
chain = with_resilience(chain, fallback_chain)
```

This is a good example of CLAUDE.md's rule #7 in action — *"verify library
APIs against installed versions; do NOT trust memorized APIs."* The plan text
was a reasonable first guess, but the installed `langchain-core==1.4.7`
behavior said otherwise, so the implementation deviated and the deviation is
documented in the code (and here) rather than silently done differently.

### `get_chat_model(model_id=None)`

To build a fallback chain you need a *different model* — so
`get_chat_model()` grew an optional override parameter:

```python
def get_chat_model(model_id: str | None = None) -> BaseChatModel:
    if settings.model_provider == "bedrock":
        return _make_bedrock_model(model_id or settings.bedrock_model_id)
    ...
```

Calling it with no arguments is 100% backward compatible (every existing test
that stubs `get_chat_model` with a zero-arg lambda still works, since the
fallback branch is only exercised when `settings.fallback_model_id` is set —
which defaults to `""`).

### New config

```python
retry_max_attempts: int = 3       # includes the first attempt
fallback_model_id: str = ""       # empty = no fallback configured
```

### Tests — [`tests/test_models.py`](../tests/test_models.py)

All offline. The trick: build a fake chain out of `RunnableLambda` wrapping a
small counter object that raises N times then succeeds (or always raises),
and assert on the call count after `with_resilience(chain).invoke({})`. No
real model, no network — retry/fallback logic is pure Runnable composition,
so it's fully testable without Bedrock.

---

## 10b — Guardrails

### The concept

The JD, the candidate's profile, and every interview answer are **untrusted
input** — the same category as a request body hitting a REST endpoint. Two
independent concerns, both handled with plain deterministic regex (no model
call, no network, instant):

- **PII redaction** — mask emails, phone numbers, and long digit runs
  (SSNs, card numbers) before text is stored in state or sent to a model.
- **Prompt-injection detection** — flag (not silently drop) text that looks
  like it's trying to hijack the system prompt: *"ignore previous
  instructions"*, *"you are now a different assistant"*, *"reveal your
  system prompt"*, etc.

### Why flag instead of block

Blocking outright risks false positives killing a legitimate answer (a
candidate might reasonably type "the interviewer said to ignore the previous
requirement and focus on X" — not an attack, just unlucky phrasing). Flagging
means the human stays in the loop: the interview continues, but the flag
surfaces in the UI so a human can see what got caught and judge for
themselves. This is consistent with Loop's whole HITL philosophy — the model
proposes, a human disposes.

### `loop/guardrails.py`

Two pure functions:

```python
def redact_pii(text: str) -> str: ...
def detect_injection(text: str) -> bool: ...
```

Order matters inside `redact_pii`: email first (has its own `@` delimiter),
then phone-shaped runs (digits with separators, 10+ characters so it doesn't
catch short numbers like "8/10"), then any remaining bare 9+ digit run (a
9-digit SSN with no dashes wouldn't match the phone pattern's minimum
length, so it needs its own pass).

### Wiring

- **`intake()`** in `graph.py` redacts the JD and profile text, and appends
  to `state["flagged_inputs"]` if either trips `detect_injection()`. Today
  these come from static fixtures (so nothing ever actually triggers), but
  this is the seam where a real JD upload would land — the safety wiring is
  already in place for when v2 replaces the fixture read with a real upload.
- **`_ask_question()`** in `interviewers.py` redacts the candidate's answer
  and flags it *before* it's stored in `state["answers"]` — so a flagged or
  PII-containing answer never reaches the grader's prompt un-redacted.

### New state field

```python
flagged_inputs: Annotated[Optional[list[dict]], _append_list]
```

Uses the same `_append_list` reducer as `answers`/`grades` — each node
returns only the *new* flags it found, and LangGraph appends them to
whatever accumulated across the session so far, instead of overwriting.

### Surfacing to the UI

`api.py`'s `_safe_payload()` now forwards `flagged_inputs` in the SSE stream
when a node returns any; `index.html`'s `showFlagged()` renders a `⚠ Input
flagged for review` entry in the activity log.

### Tests — [`tests/test_guardrails.py`](../tests/test_guardrails.py)

Pure unit tests on `redact_pii`/`detect_injection` directly — no graph, no
model, just regex behavior against known-PII and known-injection strings,
plus a "clean text passes through unchanged" case for each.

---

## 10c — Cost & token budget

### The concept

Every Bedrock Converse response carries a `usage_metadata` block on the
returned `AIMessage`: `{"input_tokens": N, "output_tokens": N, "total_tokens": N}`.
Accumulate that across a session, price it against a per-model rate table,
and enforce a ceiling — the same shape as a rate limiter guarding an
expensive downstream call, just counting tokens instead of requests.

### The design choice: a callback, not a code change in every node

The naive approach would be to have every node read its own model call's
usage and manually accumulate it into state — but that means touching
`planner.py`, `grader.py`, `coach.py`, `readiness.py`, *and* remembering to do
it correctly every time a new node is added later.

Instead, `loop/budget.py` implements a **LangChain callback handler**:

```python
class BudgetCallbackHandler(BaseCallbackHandler):
    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs) -> None:
        for generation_list in response.generations:
            for generation in generation_list:
                usage = getattr(generation.message, "usage_metadata", None)
                if usage:
                    self.budget.add(self.model_id, usage["input_tokens"], usage["output_tokens"])
```

`on_llm_end` fires automatically after **every** model call in the graph,
regardless of which node made it — because LangChain (and LangGraph on top
of it) propagate a top-level `config["callbacks"]` list down into every
nested runnable invocation. This is exactly how `get_langfuse_callback()`
already worked in earlier phases; Phase 10c reuses the same mechanism for a
different purpose.

**Where it's attached:** not inside any node — at the single top-level
`_graph.stream(...)` call in `api.py`. One line of wiring, zero changes to
any node file:

```python
budget = _budgets.setdefault(thread_id, SessionBudget())
config = {
    "configurable": {"thread_id": thread_id, "user_id": user_id},
    "callbacks": [BudgetCallbackHandler(budget)],
}
```

### `SessionBudget` vs. `BudgetCallbackHandler`

Split deliberately into two classes:

- **`SessionBudget`** — plain accumulator (`tokens_used`, `cost_usd`,
  `.add(...)`, raises `BudgetExceeded` on breach). Zero LangChain
  dependencies — trivially unit-testable.
- **`BudgetCallbackHandler`** — thin LangChain glue that unpacks an
  `LLMResult` and calls `SessionBudget.add(...)`.

This split means the arithmetic (accumulation, pricing, ceiling logic) is
tested completely independent of any LangChain machinery, and the callback
wiring is tested separately by constructing a real `LLMResult`/`AIMessage`
by hand and calling `on_llm_end()` directly — no graph run needed for either.

### Stopping gracefully

`SessionBudget.add()` raises `BudgetExceeded` when the ceiling is crossed.
Since it fires from inside `on_llm_end` — synchronously, during whatever
node's `chain.invoke()` triggered the model call — the exception propagates
up through that node, through `graph.stream()`, and into `api.py`'s SSE
generator, where it's caught and turned into a clean event instead of a raw
500:

```python
except BudgetExceeded as exc:
    yield _sse({"type": "error", "reason": "budget_exceeded", "message": str(exc), ...})
    return
```

The browser sees `{"type": "error", ...}` and shows a readable message
instead of a dropped connection.

### Known simplification

The callback prices every call against a single `model_id` (the primary
model from `settings`), because a callback has no reliable way to know which
model actually served a given call. This only matters in the rare case where
a Phase 10a fallback model fires — the cost would be priced as if the
primary model served it. Good enough for a v1 cost *estimate*; not
billing-accurate. Documented in the module docstring rather than silently
assumed.

### Tests — [`tests/test_budget.py`](../tests/test_budget.py)

Three layers, matching the two-class split above: `price_usage()` math,
`SessionBudget` accumulation/ceiling behavior, and `BudgetCallbackHandler`
driven by hand-built `LLMResult`/`ChatGeneration`/`AIMessage` objects — the
exact shapes `langchain_core` uses internally, confirmed by constructing one
in a throwaway Python shell before writing the test.

---

## What to take away

- **Runnable composition order matters.** `.with_retry()` before
  `.with_structured_output()` silently breaks; after, it works. When a
  library API doesn't behave the way the plan assumed, verify against the
  installed version rather than pushing forward on a guess — and write down
  *why* it changed, not just that it did.
- **A callback is the right tool when a cross-cutting concern (cost, tracing)
  needs to see every model call without every node knowing about it.** This
  is the same shape as a Spring `@Around` advice / servlet filter — one seam,
  applied once, invisible to the business logic it wraps.
- **Flag, don't always block, on untrusted input** when a human is already
  in the loop downstream — it preserves the "model proposes, human disposes"
  contract instead of silently overriding it.
