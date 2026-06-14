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

    Returns None when public or secret key is absent, so callers can use
    `callbacks=[cb] if cb else []` without any conditional logic.

    Requires all three: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST.
    Tracing is silently skipped when any of the first two are missing.
    """
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None

    from langfuse.langchain import CallbackHandler

    # v4 CallbackHandler only accepts public_key in its constructor.
    # It reads LANGFUSE_SECRET_KEY and LANGFUSE_HOST from os.environ automatically.
    # load_dotenv() in config.py already put the .env values into os.environ,
    # so the handler picks them up without any extra wiring.
    return CallbackHandler(public_key=settings.langfuse_public_key)
