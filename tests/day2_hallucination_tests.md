# Day 2 — Hallucination Test Cases

These questions are deliberately outside the scope of the 7 policy
documents. The assistant is expected to say it cannot find the
information, rather than inventing a plausible-sounding policy.

| # | Question | Expected behavior |
|---|---|---|
| 1 | Does the company reimburse hotel expenses? | `Answer: I could not find information about this in the available policy documents.` `Source: None` |
| 2 | What is the maximum flight ticket price? | Same "not found" response — no flight-price policy exists in the dataset. |
| 3 | Can I book a rental car through this program? | Same "not found" response — no rental-car policy exists in the dataset. |

## How this is enforced

1. **Prompt-level constraint** (`src/prompts.py` → `POLICY_QA_PROMPT_PTCF`):
   the model is explicitly instructed to answer *only* from the retrieved
   context and to say so plainly when the context doesn't cover the
   question, using a fixed "Answer: ... / Source: ..." format.
2. **Structured parsing** (`src/rag.py` → `_parse_answer_and_source`):
   `Source: None` is parsed into an empty `sources` list, so downstream
   code (the Day 3 agent, the Day 4 Flask UI) can check
   `if not result.sources` to detect a "not found" response
   programmatically instead of string-matching the answer text.
3. **Retrieval is not the gate — the LLM's grounding is.** `search_policy`
   will always return its `top_k` nearest chunks even for an unrelated
   question (FAISS doesn't have a "no match" state). It's the prompt
   instruction + the LLM's judgment of whether that retrieved context
   actually answers the question that prevents a fabricated answer. This
   is why the prompt wording matters more here than the retrieval step.

Run `python src/rag.py` (with an API key configured) to execute these
automatically via `run_hallucination_tests()`.
