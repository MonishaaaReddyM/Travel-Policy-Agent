"""
rag.py
------
Day 2 deliverable, Parts 3-6: Semantic search over the policy vector store,
the full RAG pipeline (retrieve -> augment prompt -> generate), source
attribution, and hallucination-resistant behavior.

    User Question
        -> Query Embedding
        -> Vector Search (FAISS)
        -> Top-K Policy Chunks
        -> Prompt + Retrieved Context
        -> LLM
        -> Answer + Source

Supports two LLM providers via env var LLM_PROVIDER: "openai" or
"anthropic". Set the corresponding API key as an environment variable
(OPENAI_API_KEY or ANTHROPIC_API_KEY), e.g. in a local .env file (see
.env.example).

Run directly for a quick manual test (including a hallucination check):
    python src/rag.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads OPENAI_API_KEY / ANTHROPIC_API_KEY / LLM_PROVIDER from a local .env file, if present
except ImportError:
    pass  # python-dotenv not installed - fall back to whatever is already in the environment

from embeddings import get_embedding_model, load_vector_store
from prompts import POLICY_QA_PROMPT_PTCF

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()  # "openai" | "anthropic" | "gemini"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

DEFAULT_TOP_K = 3


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    chunk_id: str
    source: str
    policy_type: str
    country: str
    text: str
    score: float


@dataclass
class RagAnswer:
    answer: str
    sources: list[str]
    retrieved_chunks: list[RetrievedChunk]


# ---------------------------------------------------------------------------
# Part 3: Semantic Search
# ---------------------------------------------------------------------------

_index = None
_chunks = None


def _get_store():
    """Lazily loads the FAISS index + chunk metadata (built by embeddings.py)."""
    global _index, _chunks
    if _index is None or _chunks is None:
        _index, _chunks = load_vector_store()
    return _index, _chunks


def search_policy(query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
    """
    Embeds the query and retrieves the top_k most similar policy chunks
    from the FAISS vector store.

    Example:
        search_policy("Can I use Uber for an airport trip late at night?")
        -> chunks from airport_policy.txt, travel_policy_india.txt, etc.
    """
    index, chunks = _get_store()
    model = get_embedding_model()

    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        results.append(
            RetrievedChunk(
                chunk_id=chunk["chunk_id"],
                source=chunk["source"],
                policy_type=chunk["policy_type"],
                country=chunk["country"],
                text=chunk["text"],
                score=float(score),
            )
        )
    return results


def explain_retrieval(query: str, results: list[RetrievedChunk]) -> str:
    """Human-readable explanation of why each chunk was retrieved (by score)."""
    lines = [f"Query: {query}", "Retrieved (ranked by cosine similarity):"]
    for i, r in enumerate(results, 1):
        lines.append(f"  {i}. {r.source} (chunk {r.chunk_id}, score={r.score:.3f}, policy_type={r.policy_type})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Part 4 & 5: RAG Pipeline + Source Attribution
# ---------------------------------------------------------------------------

def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Builds the context block injected into the prompt, each chunk labeled with its source."""
    blocks = []
    for c in chunks:
        blocks.append(f"[source: {c.source}]\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def _call_llm(prompt: str) -> str:
    """Dispatches to the configured LLM provider."""
    if LLM_PROVIDER == "anthropic":
        return _call_anthropic(prompt)
    elif LLM_PROVIDER == "openai":
        return _call_openai(prompt)
    elif LLM_PROVIDER == "gemini":
        return _call_gemini(prompt)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER!r} (use 'openai', 'anthropic', or 'gemini')")

def _call_openai(prompt: str) -> str:
    from openai import OpenAI  # imported lazily so the module loads without the package installed

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def _call_anthropic(prompt: str) -> str:
    import anthropic  # imported lazily so the module loads without the package installed

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your environment or .env file.")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()

def _call_gemini(prompt: str) -> str:
    # Uses the current "google-genai" SDK (the older "google-generativeai"
    # package is deprecated). Install with: pip install google-genai
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your environment or .env file.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text.strip()


def _parse_answer_and_source(llm_output: str, retrieved: list[RetrievedChunk]) -> tuple[str, list[str]]:
    """
    Parses the "Answer: ... / Source: ..." structure requested by the prompt
    template. Falls back gracefully if the model didn't follow the format
    exactly, so a formatting slip never crashes the pipeline.
    """
    answer_line = ""
    source_line = ""

    for line in llm_output.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("answer:"):
            answer_line = stripped[len("answer:"):].strip()
        elif stripped.lower().startswith("source:"):
            source_line = stripped[len("source:"):].strip()

    if not answer_line:
        # Model didn't follow the format - use the raw output as the answer
        # rather than silently dropping it.
        answer_line = llm_output.strip()

    if source_line and source_line.lower() != "none":
        sources = [s.strip() for s in source_line.split(",") if s.strip()]
    else:
        sources = []

    return answer_line, sources


def generate_answer(question: str, top_k: int = DEFAULT_TOP_K) -> RagAnswer:
    """
    Full RAG pipeline: retrieve top_k chunks, build the grounded prompt,
    call the LLM, and return a structured answer with source attribution.

    Hallucination prevention relies on:
    1. The prompt explicitly instructing the model to only use the
       provided context and to say so plainly when the context doesn't
       answer the question (see prompts.POLICY_QA_PROMPT_PTCF).
    2. Source attribution being parsed out separately, so a "Source: None"
       response is a clear, checkable signal rather than free text.
    """
    retrieved = search_policy(question, top_k=top_k)
    context = _format_context(retrieved)

    prompt = POLICY_QA_PROMPT_PTCF.format(context=context, question=question)
    llm_output = _call_llm(prompt)

    answer, sources = _parse_answer_and_source(llm_output, retrieved)
    return RagAnswer(answer=answer, sources=sources, retrieved_chunks=retrieved)


# ---------------------------------------------------------------------------
# Part 6: Hallucination Testing
# ---------------------------------------------------------------------------

HALLUCINATION_TEST_QUESTIONS = [
    "Does the company reimburse hotel expenses?",
    "What is the maximum flight ticket price?",
    "Can I book a rental car through this program?",
]


def run_hallucination_tests() -> None:
    print("\n=== Hallucination Tests ===")
    for q in HALLUCINATION_TEST_QUESTIONS:
        result = generate_answer(q)
        print(f"\nQ: {q}")
        print(f"A: {result.answer}")
        print(f"Sources: {result.sources or 'None'}")


if __name__ == "__main__":
    sample_question = "Can I use Uber for an airport trip late at night?"

    print("=== Semantic Search Demo ===")
    retrieved = search_policy(sample_question, top_k=3)
    print(explain_retrieval(sample_question, retrieved))

    print("\n=== RAG Pipeline Demo ===")
    result = generate_answer(sample_question)
    print(f"Q: {sample_question}")
    print(f"A: {result.answer}")
    print(f"Sources: {result.sources}")

    run_hallucination_tests()
