"""
memory.py
---------
Day 4 deliverable, Part 1: Conversational memory so the assistant can
understand follow-up questions.

    User: Can I take an airport trip?
    Assistant: Yes, airport trips are permitted for approved business travel.
    User: What if it costs ₹2,500?
                 ^ "it" only makes sense with the previous turn in mind

Approach: rather than trying to resolve pronouns with regex, each new
question is passed through an LLM condensation step (reusing the same
LLM configured in rag.py) that rewrites it into a standalone question
using the recent conversation history. The rest of the pipeline (agent,
RAG, tools) then only ever sees standalone questions - it doesn't need to
know memory exists.

Run directly for a quick manual test:
    python src/memory.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from prompts import CONDENSE_FOLLOWUP_PROMPT

MAX_HISTORY_TURNS = 6  # how many recent exchanges to keep in the prompt context


@dataclass
class Turn:
    question: str
    answer: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


class ConversationMemory:
    """
    In-memory conversation history for a single session. Session-scoped
    (not global) so the Day 4 Flask app can keep one ConversationMemory per
    browser session without conversations bleeding into each other.
    """

    def __init__(self, max_turns: int = MAX_HISTORY_TURNS):
        self.turns: list[Turn] = []
        self.max_turns = max_turns

    def add_turn(self, question: str, answer: str) -> None:
        self.turns.append(Turn(question=question, answer=answer))
        # Keep only the most recent max_turns exchanges so the prompt
        # context doesn't grow unbounded over a long conversation.
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def clear(self) -> None:
        self.turns = []

    def is_empty(self) -> bool:
        return len(self.turns) == 0

    def format_for_prompt(self) -> str:
        """Renders recent history as plain text for the condensation prompt."""
        if self.is_empty():
            return "(no previous conversation)"
        lines = []
        for turn in self.turns:
            lines.append(f"User: {turn.question}")
            lines.append(f"Assistant: {turn.answer}")
        return "\n".join(lines)

    def as_list(self) -> list[dict]:
        """Renders history as a list of dicts, e.g. for the Flask UI or an API response."""
        return [
            {"question": t.question, "answer": t.answer, "timestamp": t.timestamp}
            for t in self.turns
        ]


def resolve_followup_question(question: str, memory: ConversationMemory) -> str:
    """
    Rewrites `question` into a standalone question using `memory`, via the
    LLM. If there's no history yet, the question is returned unchanged
    (skips an unnecessary LLM call for the first turn of a conversation).

    Example:
        History: "User: Can I take an airport trip?
                   Assistant: Yes, airport trips are permitted..."
        question: "What if it costs ₹2,500?"
        -> "What if my airport trip costs ₹2,500?"
    """
    if memory.is_empty():
        return question

    # Imported lazily to avoid a circular import (rag.py doesn't import
    # memory.py, but importing rag at module load time would still force
    # embeddings.py / the vector store to load just to use ConversationMemory
    # on its own, e.g. in a unit test).
    from rag import _call_llm

    prompt = CONDENSE_FOLLOWUP_PROMPT.format(
        history=memory.format_for_prompt(),
        question=question,
    )
    try:
        rewritten = _call_llm(prompt).strip()
        return rewritten if rewritten else question
    except Exception:
        # If condensation fails for any reason (LLM hiccup, etc.), fall back
        # to the original question rather than blocking the conversation.
        return question


if __name__ == "__main__":
    memory = ConversationMemory()

    print("=== Turn 1 (no history yet) ===")
    q1 = "Can I take an airport trip?"
    resolved1 = resolve_followup_question(q1, memory)
    print(f"Q: {q1}\nResolved: {resolved1} (unchanged, as expected)")
    memory.add_turn(q1, "Yes, airport trips are permitted for approved business travel.")

    print("\n=== Turn 2 (follow-up, needs resolving) ===")
    q2 = "What if it costs ₹2,500?"
    resolved2 = resolve_followup_question(q2, memory)
    print(f"Q: {q2}\nResolved: {resolved2}")
    memory.add_turn(q2, "Then it would exceed the standard limit and need approval.")

    print("\n=== Full history ===")
    for turn in memory.as_list():
        print(turn)