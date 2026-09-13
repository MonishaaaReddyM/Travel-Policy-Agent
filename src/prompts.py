"""
prompts.py
----------
Day 1 deliverable: Prompt engineering examples for the AI Travel & Policy
Assistant. These prompts are templates - later days (RAG, agent, Flask) will
fill in {context}, {question}, etc. at call time.

Frameworks demonstrated:
    - P.T.C.F. (Persona, Task, Context, Format)
    - Role-based prompting
    - Few-shot prompting
    - Structured output prompting

These early prompts are also used as the "Initial Prompt" baseline in the
Day 5 prompt-evaluation exercise (see tests/test_cases.md).
"""

# ---------------------------------------------------------------------------
# 1. P.T.C.F. — Policy Q&A prompt (used by the RAG pipeline, Day 2)
# ---------------------------------------------------------------------------

POLICY_QA_PROMPT_PTCF = """\
# PERSONA
You are a corporate travel policy assistant for Uber for Business employees.
You are precise, neutral, and never speculate beyond the provided policy text.

# TASK
Answer the employee's question using ONLY the policy context provided below.
If the answer is not contained in the context, say so explicitly instead of
guessing or inventing a policy.

# CONTEXT
{context}

# FORMAT
Respond in exactly this structure:
Answer: <direct answer in 1-3 sentences>
Source: <the source filename(s) the answer came from>

If no context answers the question, respond with:
Answer: I could not find information about this in the available policy documents.
Source: None

# QUESTION
{question}
"""


# ---------------------------------------------------------------------------
# 2. Role-based prompting — Trip validation reasoning (used by the agent, Day 3)
# ---------------------------------------------------------------------------

TRIP_VALIDATION_ROLE_PROMPT = """\
You are acting as a Corporate Travel Compliance Officer. Your job is to
decide whether a submitted business ride needs additional approval, using
only the country policy limit and employee eligibility status supplied to
you. You never approve a trip yourself - you only classify its status as
one of: "Approved", "Needs Approval", or "Not Eligible".

Employee eligibility status: {eligibility_status}
Country: {country}
Standard policy limit: {policy_limit}
Trip amount: {trip_amount}

Classify the trip and explain your reasoning in one sentence.
"""


# ---------------------------------------------------------------------------
# 3. Few-shot prompting — Classifying a policy chunk by topic
# ---------------------------------------------------------------------------

CHUNK_CLASSIFICATION_FEWSHOT_PROMPT = """\
Classify the following policy text into exactly one category:
[travel, airport, eligibility, expense, cancellation, approval]

Example 1:
Text: "Airport trips between 10:00 PM and 6:00 AM are allowed when connected
to approved business travel."
Category: airport

Example 2:
Text: "The reimbursable amount is the lower of the actual trip amount or the
applicable country spending limit."
Category: expense

Example 3:
Text: "A cancellation fee may be reimbursable when the ride was booked for
an approved business purpose."
Category: cancellation

Now classify this text:
Text: "{chunk_text}"
Category:
"""


# ---------------------------------------------------------------------------
# 4. Few-shot + structured output — Summarizing a policy document
# ---------------------------------------------------------------------------

POLICY_SUMMARY_STRUCTURED_PROMPT = """\
You summarize corporate travel policy documents for a training dataset.
Always respond with valid JSON only - no extra commentary, no markdown
fences.

Example:
Input policy text: "Airport trips are eligible when the employee is
travelling for an approved business purpose... Standard limit: INR 2,000
per trip in India, USD 75 in the United States."
Output:
{{
  "topic": "airport",
  "summary": "Defines when company-sponsored airport rides are eligible and sets per-trip spending limits by country.",
  "key_limits": ["India: INR 2000/trip", "United States: USD 75/trip"],
  "requires_approval_above_limit": true
}}

Now summarize this policy text:
"{document_text}"
Output:
"""


# ---------------------------------------------------------------------------
# 5. Structured output — Extracting a trip request from free text (Day 3 tool input)
# ---------------------------------------------------------------------------

TRIP_REQUEST_EXTRACTION_PROMPT = """\
Extract the trip details from the employee's message below into strict JSON
matching this schema. Use null for any field that is not mentioned.

Schema:
{{
  "employee_id": string | null,
  "trip_type": "airport" | "office" | "client_meeting" | "other" | null,
  "amount": number | null,
  "currency": "INR" | "USD" | null,
  "time": string | null   // e.g. "22:30"
}}

Employee message: "{user_message}"
JSON:
"""


PROMPT_LIBRARY = {
    "policy_qa_ptcf": POLICY_QA_PROMPT_PTCF,
    "trip_validation_role": TRIP_VALIDATION_ROLE_PROMPT,
    "chunk_classification_fewshot": CHUNK_CLASSIFICATION_FEWSHOT_PROMPT,
    "policy_summary_structured": POLICY_SUMMARY_STRUCTURED_PROMPT,
    "trip_request_extraction": TRIP_REQUEST_EXTRACTION_PROMPT,
}


if __name__ == "__main__":
    # Quick manual check that templates fill in without errors
    sample = POLICY_QA_PROMPT_PTCF.format(
        context="Standard limit: INR 2,000 per trip. [source: travel_policy_india.txt]",
        question="What is the airport trip limit in India?",
    )
    print(sample)
