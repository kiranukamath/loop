"""
Pydantic schemas for structured output across all phases.

These are the "DTOs" of Loop — each schema is both the shape the model must
return AND the shape stored in state / passed between nodes.

All schemas use Pydantic v2.  Field descriptions are important: LangChain
passes them to the model as the tool description, so the model knows what
each field means and how to fill it.
"""

from typing import Literal

from pydantic import BaseModel, Field

# ── Phase 2: Planning ─────────────────────────────────────────────────────────


class Session(BaseModel):
    """One mock-interview session within the prep plan."""

    session_number: int = Field(description="1-based session index")
    modality: Literal["coding", "system_design", "behavioral"] = Field(
        description="Interview type for this session"
    )
    topics: list[str] = Field(
        description="Specific topics to cover, e.g. ['sliding-window', 'two-pointers']"
    )
    focus: str = Field(
        description="One sentence on what to emphasise given the candidate's weak areas"
    )


class PrepPlan(BaseModel):
    """Full prep curriculum produced by the planner node.

    The planner reads the JD + candidate profile and returns this object.
    Later nodes (Phase 3+) iterate over sessions to run mock interviews.
    """

    role_summary: str = Field(
        description="One sentence summarising the target role and its core skill demands"
    )
    total_sessions: int = Field(
        description="Recommended number of mock-interview sessions (typically 4–8)"
    )
    sessions: list[Session] = Field(
        description="Ordered list of sessions — one entry per planned mock interview"
    )
    key_gaps: list[str] = Field(
        description=(
            "Candidate's most important skill gaps relative to the JD, "
            "e.g. ['Kafka', 'distributed systems theory', 'STAR storytelling']"
        )
    )
    rationale: str = Field(
        description=(
            "2–3 sentence explanation of why this plan is structured this way "
            "given the JD requirements and the candidate's profile"
        )
    )


# ── Phase 3: Interview loop ───────────────────────────────────────────────────


class Answer(BaseModel):
    """A candidate's answer to one interview question.

    Stored as model_dump() dicts in state["answers"].
    Phase 5 will populate these via human interrupt; Phase 3 pre-injects them.
    """

    question_id: str = Field(description="ID of the question being answered")
    text: str = Field(description="The candidate's full answer text")


class Grade(BaseModel):
    """Grader output for one answer."""

    question_id: str = Field(description="ID of the question being graded")
    score: int = Field(description="Score out of 10")
    criterion_scores: dict[str, int] = Field(
        description="Per-criterion scores, keys match rubric criterion names"
    )
    strengths: list[str] = Field(description="What the candidate did well")
    improvements: list[str] = Field(description="Specific things to improve")
    overall_feedback: str = Field(description="2–3 sentence summary feedback")


class Feedback(BaseModel):
    """Coach output — actionable feedback after a graded session."""

    summary: str = Field(description="Overall session performance in 1–2 sentences")
    action_items: list[str] = Field(
        description="Concrete things to study or practise before the next session"
    )
    weak_areas_update: list[str] = Field(
        description="Topics the candidate should focus on in future sessions"
    )
