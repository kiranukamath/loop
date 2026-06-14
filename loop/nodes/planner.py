"""
Planner node — Phase 2.

Reads jd + profile (+ weak_areas in Phase 4) from state, calls the model
with structured output, and writes a PrepPlan back into state["plan"].

This is the first LLM node in the graph.  Every concept introduced here
(ChatPromptTemplate, with_structured_output, the pipe operator) is reused
by every subsequent node.
"""

from langchain_core.prompts import ChatPromptTemplate

from loop.models import get_chat_model
from loop.observability import get_langfuse_callback
from loop.schemas import PrepPlan

# ── Prompt ────────────────────────────────────────────────────────────────────
# Defined at module level — constructed once, reused on every invocation.
# Placeholders: {jd}, {profile}, {weak_areas}.

_SYSTEM = """You are an expert technical-interview coach.
Given a job description and a candidate profile, produce a structured prep plan.
Be specific: name real topics (e.g. "sliding window", "outbox pattern", "STAR format").
Tailor the plan to the gap between the JD requirements and the candidate's current skills."""

_HUMAN = """## Job Description
{jd}

## Candidate Profile
{profile}

## Known Weak Areas (from previous sessions — empty on first session)
{weak_areas}

Produce a PrepPlan for this candidate."""

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM),
        ("human", _HUMAN),
    ]
)


# ── Node ──────────────────────────────────────────────────────────────────────


def planner(state: dict) -> dict:
    """Plan a prep curriculum from JD + profile and write it to state["plan"].

    Node contract: receive full state, return only the keys changed.

    The chain is:  prompt | model.with_structured_output(PrepPlan)
    - prompt.invoke(vars) → ChatPromptValue (the filled-in messages)
    - model.with_structured_output(PrepPlan) wraps the model so it returns a
      PrepPlan instance instead of a raw AIMessage string.
    - The | pipe chains them: output of prompt feeds into the model call.
    """
    model = get_chat_model()

    # with_structured_output(PrepPlan) tells the model to fill in the PrepPlan
    # schema.  LangChain converts the Pydantic model to a tool definition,
    # calls the model, and deserialises the result back to PrepPlan.
    # Analogy: Jackson @JsonDeserialize — you get a typed object, not a string.
    structured_model = model.with_structured_output(PrepPlan)

    # Build the chain: prompt template → structured model.
    # The | operator is LangChain's pipe — same as Unix pipes.
    chain = _PROMPT | structured_model

    # Collect observability callback (no-op if Langfuse not configured).
    cb = get_langfuse_callback()
    config = {"callbacks": [cb]} if cb else {}

    # Format weak_areas as a readable string (empty list → "none").
    weak_areas = state.get("weak_areas") or []
    weak_areas_str = ", ".join(weak_areas) if weak_areas else "none"

    plan: PrepPlan = chain.invoke(
        {
            "jd": state["jd"],
            "profile": state["profile"],
            "weak_areas": weak_areas_str,
        },
        config=config,
    )

    # Return only the changed key — LangGraph merges this into the full state.
    return {"plan": plan.model_dump()}
