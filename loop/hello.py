"""
Phase 0 smoke-test: one real Bedrock call via the model factory.

Run with:  uv run python -m loop.hello

This is NOT a pytest test — it makes a live network call to Bedrock.
Run it manually to verify the factory + creds work end-to-end.
If LANGFUSE_PUBLIC_KEY is set the call will also appear as a trace.
"""

from loop.models import get_chat_model
from loop.observability import get_langfuse_callback


def main() -> None:
    print("Loop — Phase 0 hello-world")
    print("Building model via factory...")

    model = get_chat_model()

    cb = get_langfuse_callback()
    callbacks = [cb] if cb else []
    if cb:
        print("Langfuse callback active — call will be traced.")
    else:
        print("Langfuse not configured — tracing skipped.")

    print("Calling Bedrock...")
    response = model.invoke(
        "Say exactly: 'Loop is alive.' and nothing else.",
        config={"callbacks": callbacks},
    )

    print(f"\nModel response: {response.content}")
    print("\nPhase 0 complete — factory + Bedrock wired correctly.")


if __name__ == "__main__":
    main()
