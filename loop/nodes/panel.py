"""
Panel grading — Phase 13a: parallel fan-out/fan-in ("a panel of graders").

Where loop/nodes/grader.py grades one answer with ONE model call, this module
grades the SAME answer with K independent model calls running concurrently —
one per persona in settings.grader_personas — then reduces their partial
scores into a single Grade with the identical shape grader.py produces.
coach and every downstream node never know which path ran.

Three functions, three distinct roles in the map-reduce shape:

  grade_dispatch(state) -> list[Send]
      The "map" step. A conditional-edge path function (registered via
      add_conditional_edges, not add_node) that returns one Send per persona.
      Send(node, arg) is NOT a graph-state update — `arg` becomes panel_grader's
      ENTIRE input, replacing state entirely for that invocation (verified: a
      Send-invoked node receives exactly `arg`, not LoopState).  So grade_dispatch
      must pack everything panel_grader needs into `arg` — persona label, the
      question/answer/rubric context, and (for the 13c debate round) peer scores.

  panel_grader(payload) -> dict
      One persona's scoring pass. Reads `payload`, NEVER `state` — this
      function runs once per Send, concurrently, with no visibility into the
      other personas' invocations or the rest of the graph.  Writes one
      partial into the transient state["panel_grades"] scratch channel (the
      _reset_or_append reducer in loop/state.py appends it).

  grade_aggregator(state) -> Command
      The "reduce" step, reached by a static edge (panel_grader -> grade_aggregator)
      so it runs after EVERY Send has landed — LangGraph's superstep barrier
      guarantees this node fires exactly once per round, not once per Send
      (verified: 3 Sends into one aggregator -> aggregator runs 1x).
      With settings.panel_debate off (default), it reduces round 0 straight
      to a final Grade and hands off to "coach" via Command(goto=...).  With
      it on, round 0 instead triggers ONE more fan-out via Command(goto=[Send,
      ...]) (each persona sees the others' round-0 scores and revises) before
      reducing the LATEST round only — see the debate-round section below.
      ALWAYS returns Command (never a plain dict) for the same reason
      plan_approval does: a static outgoing edge would compete with it.

Phase 13c (stretch): panel_debate extends this into a tiny multi-agent
debate — personas see peer scores once and can revise, which is a real
(if small) eval-quality lever: a persona that scored an answer highly in
isolation sometimes revises after seeing a peer flag something it missed.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command, Send

from loop.config import settings
from loop.models import get_chat_model, with_resilience
from loop.observability import get_langfuse_callback
from loop.schemas import Grade, PersonaGrade
from loop.tools import get_question_by_id, get_reference_answer, get_rubric

_NO_REFERENCE_TEXT = (
    "(no reference answer available for this question — grade against the rubric alone)"
)

# One-sentence steer per persona — each still sees the FULL rubric and answer,
# but is asked to weight its scoring toward this lens.  Personas are labels
# for a grading emphasis, not a 1:1 mapping to specific rubric criterion
# names (rubric criteria vary per question — e.g. cod-002's criteria are
# "correctness"/"synchronization_strategy"/"edge_cases"/"testability").
_PERSONA_FOCUS = {
    "correctness": (
        "Weight your scoring toward whether the solution is actually correct, "
        "including edge cases — this is your primary lens."
    ),
    "communication": (
        "Weight your scoring toward how clearly the candidate explained their "
        "reasoning, structured the answer, and narrated tradeoffs."
    ),
    "depth": (
        "Weight your scoring toward the depth of technical understanding shown: "
        "complexity analysis, alternatives considered, and how well choices were justified."
    ),
}
_DEFAULT_FOCUS = "Grade the answer against the rubric as a generalist reviewer."

_SYSTEM = """\
You are one of several expert interviewers independently grading the same \
candidate answer, each from a different angle. Your angle: {persona_focus}

You still score against the FULL rubric — your angle changes what you weight, \
not which criteria you're allowed to score. Grade honestly and constructively. \
Each criterion score must be between 0 and its weight value (inclusive).
"""

_HUMAN = """\
Question: {question_prompt}

Reference answer (for grounding — the candidate did not see this):
{reference_answer}

Candidate's answer:
{answer_text}

Rubric (max score: {max_score}):
{rubric_criteria}
{peer_feedback_block}
Grade this answer from your assigned angle."""

_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])


def _build_context(question_id: str, answer_text: str) -> dict:
    """Look up everything panel_grader needs to score one answer.

    Shared by grade_dispatch (round 0) and grade_aggregator (round 1's
    debate fan-out) so the question/rubric/reference lookup isn't duplicated.
    """
    question = get_question_by_id(question_id)
    rubric = get_rubric(question_id)
    if rubric is None:
        raise ValueError(f"No rubric found for question_id={question_id!r}")

    criteria_text = "\n".join(
        f"- {c['name']} (weight {c['weight']}): {c['description']}" for c in rubric["criteria"]
    )
    reference_answer = get_reference_answer(question_id) or _NO_REFERENCE_TEXT

    return {
        "question_prompt": question["prompt"] if question else "",
        "reference_answer": reference_answer,
        "answer_text": answer_text,
        "max_score": rubric["max_score"],
        "rubric_criteria": criteria_text,
    }


# ── Map step ──────────────────────────────────────────────────────────────────


def grade_dispatch(state: dict) -> list[Send]:
    """Fan out one Send("panel_grader", payload) per persona (round 0).

    Registered as a conditional-edge path function (graph.py), NOT a node —
    add_conditional_edges accepts a path function that returns a list of
    Send objects directly (no path_map needed; each Send names its own target).
    """
    question_id = state["current_question_id"]
    answers = state.get("answers") or []
    answer = next((a for a in answers if a["question_id"] == question_id), None)
    if answer is None:
        raise ValueError(
            f"No answer in state for question_id={question_id!r}. "
            "Pre-populate state['answers'] before invoking the graph."
        )

    context = _build_context(question_id, answer["text"])

    return [
        Send(
            "panel_grader",
            {
                "persona": persona,
                "question_id": question_id,
                "round": 0,
                "peer_scores": None,
                **context,
            },
        )
        for persona in settings.grader_personas
    ]


# ── Per-persona scoring (runs once per Send, concurrently) ───────────────────


def panel_grader(payload: dict) -> dict:
    """Score one answer from one persona's lens. Reads `payload`, not `state`."""
    persona = payload["persona"]
    focus = _PERSONA_FOCUS.get(persona, _DEFAULT_FOCUS)

    peer_scores = payload.get("peer_scores")
    peer_feedback_block = ""
    if peer_scores:
        lines = "\n".join(
            f"- {p['persona']}: scored {sum(p['criterion_scores'].values())}, notes: {p['notes']}"
            for p in peer_scores
        )
        peer_feedback_block = (
            f"\nOther graders' round-1 scores (revise your own view if warranted, "
            f"but don't just conform):\n{lines}\n"
        )

    model = get_chat_model()
    structured_model = model.with_structured_output(PersonaGrade)
    chain = _PROMPT | structured_model

    fallback_chain = None
    if settings.fallback_model_id:
        fallback_model = get_chat_model(settings.fallback_model_id)
        fallback_chain = _PROMPT | fallback_model.with_structured_output(PersonaGrade)
    chain = with_resilience(chain, fallback_chain)

    cb = get_langfuse_callback()
    config = {"callbacks": [cb]} if cb else {}

    result: PersonaGrade = chain.invoke(
        {
            "persona_focus": focus,
            "question_prompt": payload["question_prompt"],
            "reference_answer": payload["reference_answer"],
            "answer_text": payload["answer_text"],
            "max_score": payload["max_score"],
            "rubric_criteria": payload["rubric_criteria"],
            "peer_feedback_block": peer_feedback_block,
        },
        config=config,
    )

    # persona/question_id/round come from OUR payload, not the model's output —
    # a model asked to echo an ID back is a needless place for it to drift.
    partial = {
        "persona": persona,
        "question_id": payload["question_id"],
        "round": payload["round"],
        **result.model_dump(),
    }
    return {"panel_grades": [partial]}


# ── Reduce step (+ optional 13c debate fan-out) ──────────────────────────────


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _reduce_to_grade(question_id: str, partials: list[dict]) -> Grade:
    """Combine K persona partials (same round) into one Grade.

    Per-criterion scores are averaged across personas (not summed — each
    persona scores the FULL rubric, so summing would blow past max_score).
    Strengths/improvements are unioned (order-preserving, deduped);
    overall_feedback concatenates each persona's note under its own label so
    the coach (and a human reading the trace) can see where a piece of
    feedback came from.
    """
    criterion_names = {name for p in partials for name in p["criterion_scores"]}
    criterion_scores = {
        name: round(sum(p["criterion_scores"].get(name, 0) for p in partials) / len(partials))
        for name in criterion_names
    }
    score = sum(criterion_scores.values())

    strengths = _dedup([s for p in partials for s in p["strengths"]])
    improvements = _dedup([s for p in partials for s in p["improvements"]])
    overall_feedback = " ".join(f"[{p['persona']}] {p['notes']}" for p in partials)

    return Grade(
        question_id=question_id,
        score=score,
        criterion_scores=criterion_scores,
        strengths=strengths,
        improvements=improvements,
        overall_feedback=overall_feedback,
    )


def grade_aggregator(state: dict) -> Command:
    """Fan-in: reduce this question's panel partials into a final Grade.

    Runs once per superstep (the panel_grader -> grade_aggregator barrier),
    regardless of how many personas fired. With settings.panel_debate off,
    round 0 is final. With it on, seeing only round-0 partials means a
    debate round is due: fan out ONE more Send per persona (a node CAN
    return Command(goto=[Send(...), ...]) to fan out, not just a
    conditional-edge path function — verified empirically), each carrying
    the OTHER personas' round-0 scores as peer_scores. The next time this
    node runs, partials include round 1, so it reduces those and finalizes.

    ALWAYS returns Command, never a plain dict — same reason plan_approval
    (graph.py) does: this node sometimes routes to more Sends and sometimes
    to "coach", so graph.py registers NO static outgoing edge for it. A
    static edge would compete with the Command and run both paths at once
    (verified against the debate-round case: a static "grade_aggregator ->
    coach" edge alongside a Command(goto=[Send...]) return would fire coach
    AND the round-1 Sends in the same superstep).
    """
    question_id = state["current_question_id"]
    all_partials = [p for p in (state.get("panel_grades") or []) if p["question_id"] == question_id]
    if not all_partials:
        raise ValueError(f"No panel grades found for question_id={question_id!r}")

    latest_round = max(p["round"] for p in all_partials)
    current_round = [p for p in all_partials if p["round"] == latest_round]

    if settings.panel_debate and latest_round == 0:
        answers = state.get("answers") or []
        answer = next((a for a in answers if a["question_id"] == question_id), None)
        context = _build_context(question_id, answer["text"] if answer else "")

        sends = [
            Send(
                "panel_grader",
                {
                    "persona": p["persona"],
                    "question_id": question_id,
                    "round": 1,
                    "peer_scores": [
                        other for other in current_round if other["persona"] != p["persona"]
                    ],
                    **context,
                },
            )
            for p in current_round
        ]
        return Command(goto=sends)

    grade = _reduce_to_grade(question_id, current_round)
    # Reset the scratch channel: {"panel_grades": None} -> the _reset_or_append
    # reducer treats None as "clear", not "append nothing" (see loop/state.py).
    return Command(goto="coach", update={"grades": [grade.model_dump()], "panel_grades": None})
