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

import os

from langchain_core.callbacks import BaseCallbackHandler

from loop.config import settings


def get_langfuse_callback() -> BaseCallbackHandler | None:
    """Return a Langfuse callback handler, or None if not configured.

    Returns None when public or secret key is absent, so callers can use
    `callbacks=[cb] if cb else []` without any conditional logic.

    Why no-args CallbackHandler():
    Langfuse's SDK calls get_client() internally, which only creates a real
    (enabled) client when LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY are present
    in os.environ.  Passing public_key= to the constructor bypasses that path
    and returns a disabled client that drops all traces.
    load_dotenv() in config.py already populated os.environ from .env, so
    CallbackHandler() with no args finds the credentials and creates a live client.
    """
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None

    from langfuse.langchain import CallbackHandler

    # No args — SDK reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
    # from os.environ (set by load_dotenv() at startup).
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    if settings.langfuse_host:
        os.environ["LANGFUSE_HOST"] = settings.langfuse_host
    return CallbackHandler()
