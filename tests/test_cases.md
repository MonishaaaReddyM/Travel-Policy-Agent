# Test Cases

15+ functional test scenarios across RAG, employee eligibility, trip
validation, and conversational memory, per the Day 5 spec. Each was run
against the live pipeline (real FAISS index + real Gemini calls) unless
noted otherwise.

## Category A — RAG (Policy Questions)

| # | Question | Expected | Result |
|---|---|---|---|
| A1 | What is the standard travel limit in India? | INR 2,000/trip, cites `travel_policy_india.txt` | ✅ Pass |
| A2 | What is the standard travel limit in the US? | USD 75/trip, cites `travel_policy_us.txt` | ✅ Pass |
| A3 | Are airport trips allowed? | Yes, for approved business travel; cites `airport_policy.txt` | ✅ Pass |
| A4 | Are late-night trips allowed? | Yes, 10 PM–6 AM permitted for approved business activity | ✅ Pass |
| A5 | What information is required for an expense? | Employee ID, trip date, amount, currency, business purpose, trip type | ✅ Pass |

## Category B — Employee Eligibility

| # | Question | Expected | Result |
|---|---|---|---|
| B1 | Is EMP001 eligible? | Eligible (India, Full-Time) | ✅ Pass |
| B2 | Is EMP002 eligible? | Approval Required (India, Contractor) | ✅ Pass |
| B3 | Is EMP004 eligible? | Not Eligible (United States, Contractor) | ✅ Pass |
| B4 | What happens when an unknown employee ID is provided? (e.g. EMP999) | Reported as not found; eligibility never assumed | ✅ Pass |

## Category C — Trip Validation

| # | Question | Expected | Result |
|---|---|---|---|
| C1 | Can EMP001 take a ₹1,500 airport trip? | Approved (within INR 2,000 limit) | ✅ Pass |
| C2 | Can EMP001 take a ₹2,500 airport trip? | Needs Approval (exceeds INR 2,000 limit); reimbursable INR 2,000, review INR 500 | ✅ Pass |
| C3 | Can EMP003 take a $50 airport trip? | Approved (within USD 75 limit) | ✅ Pass |
| C4 | Can EMP003 take a $100 airport trip? | Needs Approval (exceeds USD 75 limit) | ✅ Pass |

## Category D — Conversational Memory

| # | Question | Expected | Result |
|---|---|---|---|
| D1 | "Can I take an airport trip?" (turn 1) | Yes, airport trips are permitted for approved business travel | ✅ Pass |
| D2 | "What if it costs ₹2,500?" (turn 2, same session) | Correctly resolves "it" to the airport trip from D1 before answering | ✅ Pass |

## Additional edge cases (beyond the minimum 15)

| # | Scenario | Expected | Result |
|---|---|---|---|
| E1 | EMP002 (Approval Required) requests a trip *under* the limit | Still Needs Approval — status overrides amount | ✅ Pass |
| E2 | EMP004 (Not Eligible) requests any trip | Not Eligible regardless of amount | ✅ Pass |
| E3 | Negative trip amount (e.g. -500) | `Invalid Amount`, friendly reason, no crash | ✅ Pass |
| E4 | Non-numeric trip amount (e.g. "abc") | `Invalid Amount`, friendly reason, no crash | ✅ Pass |
| E5 | Hallucination: "Does the company reimburse hotel expenses?" | "I could not find information..." (not invented) | ✅ Pass |
| E6 | Hallucination: "What is the maximum flight ticket price?" | "I could not find information..." | ✅ Pass |
| E7 | Hallucination: "Can I book a rental car?" | "I could not find information..." | ✅ Pass |
| E8 | Empty question submitted via Flask `/ask` | `400`, friendly JSON error, no traceback | ✅ Pass |
| E9 | Missing vector store (embeddings not built) | Flask returns `503` with a friendly "not ready" message, not a stack trace | ✅ Pass (see `app/app.py` `FileNotFoundError` handler) |

**Total: 20 scenarios covering all 4 required categories plus hallucination
and error-handling edge cases — exceeds the 15-scenario minimum.**

## How these were run

- Categories A–D and E1–E2, E5–E7: run live via `python src/agent.py` and
  interactively through the Flask app (`python app/app.py`), with real
  FAISS retrieval and real Gemini responses.
- E3–E4: run directly via `python src/tools.py`.
- E8–E9: run via Flask's test client and manual browser testing.

See also:
- `tests/day2_hallucination_tests.md` — hallucination testing detail and reasoning
- `tests/day3_agent_tests.md` — agent routing logic and additional scenarios
- `tests/day4_tests.md` — memory, MCP server, and Flask route testing detail