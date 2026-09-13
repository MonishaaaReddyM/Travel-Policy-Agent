"""
agent.py
--------
Day 3 deliverable, Parts 4-5: The agent that decides whether a question
needs policy retrieval (RAG), a tool call, or both, then combines the
results into one final answer.

    User Question
        -> Agent (decides what's needed)
        -> RAG Search  and/or  Tool(s)
        -> Final Answer

Decision logic (deliberately rule-based and transparent, not an LLM guess):
    - Employee ID + a trip amount mentioned  -> eligibility tool + trip
      validation tool + RAG (for the policy text backing the limit used)
    - Employee ID only (no amount)           -> eligibility tool only
    - No employee ID                          -> RAG only

This mirrors the three scenarios in the Day 3 spec exactly, and keeps the
routing decision testable and reproducible - not dependent on how an LLM
happens to interpret a given phrasing on a given day.

Run directly for a quick manual test:
    python src/agent.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag import generate_answer, RagAnswer
from tools import (
    check_employee_eligibility,
    validate_trip,
    EligibilityResult,
    TripValidationResult,
)

# ---------------------------------------------------------------------------
# Part 4: Entity extraction (turns free text into tool arguments)
# ---------------------------------------------------------------------------

EMPLOYEE_ID_PATTERN = re.compile(r"\bEMP\d+\b", re.IGNORECASE)
TIME_PATTERN = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
INR_PATTERN = re.compile(r"(?:₹|INR)\s?([\d,]+(?:\.\d+)?)", re.IGNORECASE)
USD_PATTERN = re.compile(r"(?:\$|USD)\s?([\d,]+(?:\.\d+)?)", re.IGNORECASE)
BARE_NUMBER_PATTERN = re.compile(r"\b([\d,]{3,}(?:\.\d+)?)\b")  # fallback: a 3+ digit number

TRIP_TYPE_KEYWORDS = {
    "airport": "airport",
    "office": "office",
    "client": "client_meeting",
    "meeting": "client_meeting",
}


@dataclass
class ExtractedEntities:
    employee_id: str | None = None
    amount: float | None = None
    currency: str | None = None
    time: str | None = None
    trip_type: str = "other"


def extract_entities(question: str) -> ExtractedEntities:
    """
    Pulls structured fields out of a free-text question so tools can be
    called with real arguments, e.g.:
        "Can EMP001 take a ₹2,500 airport trip at 22:30?"
        -> employee_id="EMP001", amount=2500, currency="INR",
           trip_type="airport", time="22:30"
    """
    entities = ExtractedEntities()

    emp_match = EMPLOYEE_ID_PATTERN.search(question)
    if emp_match:
        entities.employee_id = emp_match.group(0).upper()

    time_match = TIME_PATTERN.search(question)
    if time_match:
        entities.time = time_match.group(0)

    inr_match = INR_PATTERN.search(question)
    usd_match = USD_PATTERN.search(question)
    if inr_match:
        entities.amount = float(inr_match.group(1).replace(",", ""))
        entities.currency = "INR"
    elif usd_match:
        entities.amount = float(usd_match.group(1).replace(",", ""))
        entities.currency = "USD"
    else:
        bare_match = BARE_NUMBER_PATTERN.search(question)
        if bare_match:
            entities.amount = float(bare_match.group(1).replace(",", ""))
            # currency left as None - validate_trip() will infer it from
            # the employee's country if not given

    lowered = question.lower()
    for keyword, trip_type in TRIP_TYPE_KEYWORDS.items():
        if keyword in lowered:
            entities.trip_type = trip_type
            break

    return entities


# ---------------------------------------------------------------------------
# Part 5: Agent decision + orchestration
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    question: str
    route: str  # "rag_only" | "tool_only" | "tool_and_rag"
    final_answer: str
    eligibility: EligibilityResult | None = None
    trip_validation: TripValidationResult | None = None
    rag_answer: RagAnswer | None = None
    sources: list[str] = field(default_factory=list)


def decide_route(entities: ExtractedEntities) -> str:
    """
    The core agent decision, made explicit and testable:

        Employee ID + amount  -> "tool_and_rag"  (eligibility + trip validation + policy context)
        Employee ID only      -> "tool_only"      (eligibility check)
        Neither                -> "rag_only"       (pure policy question)
    """
    if entities.employee_id and entities.amount is not None:
        return "tool_and_rag"
    elif entities.employee_id:
        return "tool_only"
    else:
        return "rag_only"


def run_agent(question: str) -> AgentResponse:
    """
    The full agent workflow described in the Day 3 spec:

        User Question -> Agent -> RAG Search and/or Tool(s) -> Final Answer
    """
    entities = extract_entities(question)
    route = decide_route(entities)

    if route == "rag_only":
        rag_result = generate_answer(question)
        return AgentResponse(
            question=question,
            route=route,
            final_answer=rag_result.answer,
            rag_answer=rag_result,
            sources=rag_result.sources,
        )

    if route == "tool_only":
        eligibility = check_employee_eligibility(entities.employee_id)
        return AgentResponse(
            question=question,
            route=route,
            final_answer=eligibility.message,
            eligibility=eligibility,
        )

    # route == "tool_and_rag"
    eligibility = check_employee_eligibility(entities.employee_id)

    if not eligibility.found:
        # Can't validate a trip for an employee we have no record of -
        # stop here rather than guessing a country/limit.
        return AgentResponse(
            question=question,
            route=route,
            final_answer=eligibility.message,
            eligibility=eligibility,
        )

    validation = validate_trip(
        employee_id=entities.employee_id,
        trip_type=entities.trip_type,
        amount=entities.amount,
        time=entities.time,
        currency=entities.currency,
    )

    # Pull the relevant policy text too, so the final answer can cite the
    # source document backing the limit that was applied - not just a
    # bare number computed in code.
    rag_result = generate_answer(question)

    final_answer = _compose_tool_and_rag_answer(validation, rag_result)

    return AgentResponse(
        question=question,
        route=route,
        final_answer=final_answer,
        eligibility=eligibility,
        trip_validation=validation,
        rag_answer=rag_result,
        sources=rag_result.sources,
    )


def _compose_tool_and_rag_answer(validation: TripValidationResult, rag_result: RagAnswer) -> str:
    """
    Combines the deterministic tool result (trip validation - the number
    that must be exactly right) with the RAG answer (the policy language
    backing it) into one final answer. The tool result drives the verdict;
    the RAG text adds the "why", grounded in the actual policy wording.
    """
    lines = [f"Status: {validation.status}", f"Reason: {validation.reason}"]

    if validation.reimbursement:
        r = validation.reimbursement
        lines.append(
            f"Reimbursable amount: {validation.currency} {r.reimbursable_amount:,.0f}"
            + (f" (Amount requiring review: {validation.currency} {r.amount_requiring_review:,.0f})"
               if r.amount_requiring_review > 0 else "")
        )

    if rag_result.sources:
        lines.append(f"Policy source: {', '.join(rag_result.sources)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Example scenarios (matching the Day 3 spec exactly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scenarios = [
        ("Scenario 1 - Policy Question", "What is the travel limit in India?"),
        ("Scenario 2 - Eligibility Question", "Is EMP001 eligible for business travel?"),
        ("Scenario 3 - Complex Question", "Can EMP001 take a ₹2,500 airport trip?"),
    ]

    for label, question in scenarios:
        print(f"\n=== {label} ===")
        print(f"Q: {question}")
        result = run_agent(question)
        print(f"Route chosen: {result.route}")
        print(f"A:\n{result.final_answer}")