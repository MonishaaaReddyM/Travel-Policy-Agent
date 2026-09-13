# Chunking Notes — Day 1

## What was tried

| Chunk size | Overlap | Result |
|---|---|---|
| 300 chars | 50 | Too small — split single numbered sections (e.g. "4. Standard Trip Limit") mid-sentence across 2 chunks. Retrieval later would need to stitch chunks back together to get a full rule. |
| 800 chars | 150 | **Selected.** Each chunk holds 1–2 full numbered sections. Overlap keeps the tail of one section visible at the start of the next chunk, so a rule split across a chunk boundary is still findable. |
| 1500 chars | 200 | Too large for these documents — each file (1,200–1,650 chars) collapsed into essentially 1 chunk, defeating the purpose of chunking and returning the whole document for narrow queries. |

## Final values

- **Chunk size:** 800 characters
- **Chunk overlap:** 150 characters
- **Splitting strategy:** paragraph-aware greedy accumulation (split on blank lines, not a fixed character offset), so numbered policy sections and bullet lists stay intact rather than being cut mid-line. A per-paragraph fallback hard-split only triggers if a single paragraph exceeds the chunk size (doesn't happen in this dataset, but keeps the function robust for future policy documents).

## Why these values

These policy documents are short (1,200–1,650 characters each) and structured as numbered sections. The goal of chunking here isn't to manage huge documents — it's to let semantic search return a *specific rule* (e.g. "Standard Trip Limit") rather than an entire policy file as context. 800 characters is large enough to capture a full section with its heading and 2–3 sentences, small enough that 2–3 chunks per document result, giving the retriever meaningful granularity to choose from.

## How chunk size affects retrieval quality

- **Too small:** rules get fragmented; a chunk might contain a number (e.g. "INR 2,000") without the sentence explaining what it's the limit for, hurting embedding quality and answer grounding.
- **Too large:** chunks approach full-document size, so semantic search stops discriminating between documents — every query returns "the whole airport policy" instead of the one paragraph that answers the question. This also wastes LLM context and increases the chance the model pulls in irrelevant nearby text.
- **Overlap:** without overlap, a rule sentence that falls right at a chunk boundary can lose its lead-in context (e.g. the overlap ensures "3. Restricted Trips" heading isn't orphaned from its bullets if a boundary falls near it).
