"""
Langfuse observability — no-op when unconfigured.

The Langfuse CallbackHandler is a LangChain lifecycle hook: LangChain calls
it automatically on every on_llm_start / on_llm_end / on_llm_error event,
recording each call as a trace in Langfuse.  Analogy: a servlet filter or
Spring AOP interceptor that fires around every model call.

Usage in a node:
    from loop.observability import get_langfuse_callback
    cb = get_langfuse_callback()
    callbacks = [cb] if cb else []
    model.invoke(messages, config={"callbacks": callbacks})

When LANGFUSE_PUBLIC_KEY is not set, get_langfuse_callback() returns None
and the callbacks list is empty — tracing is completely skipped.
"""

from __future__ import annotations

from langchain_core.callbacks import BaseCallbackHandler

from loop.config import settings


def get_langfuse_callback() -> BaseCallbackHandler | None:
    """Return a Langfuse callback handler, or None if not configured.

    Returns None when LANGFUSE_PUBLIC_KEY is absent or empty, so callers
    can pass `callbacks=[cb] if cb else []` without any conditional logic.
    """
    if not settings.langfuse_public_key:
        return None

    # Import only when Langfuse is configured — avoids any startup error if the
    # langfuse package were missing (it's always present here, but good habit).
    from langfuse.langchain import CallbackHandler

    # Langfuse v4 CallbackHandler accepts public_key; it reads LANGFUSE_SECRET_KEY
    # and LANGFUSE_HOST from the environment automatically if needed.
    return CallbackHandler(public_key=settings.langfuse_public_key)
