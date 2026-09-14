# Prompt Evaluation & Improvement

Two meaningful prompt improvements, evaluated using the 4 Cs: Context to
Clarity to Constraints to Customization.

---

## Improvement 1 - Policy Q&A Prompt (POLICY_QA_PROMPT_PTCF)

### Initial Prompt

Answer the employee's question using the policy context below.

Context: {context}
Question: {question}

### Problem Identified

Tested against a deliberately out-of-scope question: "Does the company
reimburse hotel expenses?"

The retrieved context (from search_policy) always returns the top-K
nearest chunks even when none of them actually answer the question -
FAISS has no "no good match" state, it just returns the closest ones.
With the initial prompt, the model treated whatever chunks were retrieved
as license to construct a plausible-sounding answer anyway, e.g. blending
unrelated expense-policy language into a fabricated "yes, hotel costs are
reimbursed up to X" answer. This is exactly the hallucination behavior
the Day 2 spec explicitly tests for.

Root cause: the prompt had Context (the retrieved chunks) but no
Constraints - nothing telling the model what to do when the context
doesn't actually cover the question - and no Customization (output
format), so there was no structured way to detect a "didn't know" answer
programmatically downstream.

### Improved Prompt

This is the current POLICY_QA_PROMPT_PTCF in src/prompts.py. It adds a
PERSONA section, a TASK section instructing the model to answer ONLY from
context and say so explicitly when the context doesn't cover the
question, and a FORMAT section requiring the exact structure:

Answer: (direct answer in 1-3 sentences)
Source: (the source filename(s) the answer came from)

with a specific fallback when nothing applies:

Answer: I could not find information about this in the available policy documents.
Source: None

### Output Comparison

Question: "What is the standard travel limit in India?"
- Initial prompt: correct answer, no source
- Improved prompt: correct answer plus Source: travel_policy_india.txt

Question: "Does the company reimburse hotel expenses?"
- Initial prompt: fabricated a plausible-sounding reimbursement policy
- Improved prompt: "Answer: I could not find information about this in
  the available policy documents." / "Source: None"

### Final Prompt

The improved version is the one shipped in src/prompts.py and used
throughout src/rag.py. The Constraints section (explicit "say so"
instruction) fixed the hallucination; the Customization (the
Answer: / Source: structure) is what let _parse_answer_and_source() in
rag.py programmatically detect a "not found" response - this is also
what the Day 3 agent and Day 4 Flask app rely on downstream to know
whether a policy source exists.

---

## Improvement 2 - Follow-up Question Condensation (CONDENSE_FOLLOWUP_PROMPT)

### Initial Prompt

Given this conversation history: {history}
And this question: {question}
Rewrite the question so it makes sense on its own.

### Problem Identified

Tested with a two-turn conversation: Turn 1 asked "Can I take an airport
trip?" and got an answer confirming airport trips are permitted. Turn 2
asked "What if it costs ₹2,500?"

The initial prompt had Context (history) and roughly the right Task, but
lacked Clarity about what "rewrite" actually meant in edge cases, and no
Constraints on output format. In testing, the model sometimes:

- Answered the question instead of just rewriting it (conflating condense
  with respond)
- Added an explanation before the rewritten question, which then got
  passed into the agent as part of the "question," creating awkward
  context for the RAG step
- Left first-turn (non-follow-up) questions unnecessarily rephrased,
  wasting a full LLM call for something that needed no rewriting

### Improved Prompt

This is the current CONDENSE_FOLLOWUP_PROMPT in src/prompts.py. It adds
an explicit instruction: "Do not answer the question - only rewrite it,"
a rule that an already-standalone question should be returned unchanged,
and a strict output constraint: "Respond with ONLY the rewritten
question, no explanation, no quotes."

### Output Comparison

Follow-up "What if it costs ₹2,500?" after airport trip context:
- Initial prompt: sometimes included explanation text alongside the rewrite
- Improved prompt: "What if my airport trip costs ₹2,500?" - clean, no extra text

First-turn question with no history:
- Initial prompt: occasionally rephrased unnecessarily
- Improved prompt: handled entirely in code (memory.py skips the LLM
  call when history is empty) rather than relying on the prompt to
  recognize "no history"

### Final Prompt

The improved version is what ships in src/prompts.py. Two fixes worked
together: the explicit Constraint ("Do not answer... only rewrite... no
explanation") stopped the model from blending an answer into the
rewrite, and moving the "skip if no history" logic out of the prompt and
into resolve_followup_question() in src/memory.py (a code-level
Constraint rather than a prompt-level one) made that behavior 100%
reliable instead of depending on the model noticing "(no previous
conversation)" in the context and doing the right thing on its own.