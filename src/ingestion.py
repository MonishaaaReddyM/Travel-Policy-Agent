"""
ingestion.py
------------
Day 1 deliverable: Load, clean, tag, and chunk the company policy documents
so they are ready for embedding + retrieval in later days.

Pipeline:
    .txt files  ->  load_documents()  ->  clean_document()
                ->  attach_metadata()  ->  chunk_document()
                ->  chunks.json (used by src/embeddings.py on Day 2)

Run directly:
    python src/ingestion.py
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Chunk size / overlap were chosen after a brief experiment - see
# docs/chunking_notes.md for the reasoning and the values that were tried.
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # characters of overlap between consecutive chunks

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "company_policy"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "chunks.json"

# Static metadata lookup. In a larger system this would live in a config
# file or be inferred from a metadata header in each document; for this
# fictional 7-document policy set, an explicit map keeps things transparent.
DOCUMENT_METADATA = {
    "travel_policy_india.txt": {"policy_type": "travel", "country": "India"},
    "travel_policy_us.txt": {"policy_type": "travel", "country": "United States"},
    "airport_policy.txt": {"policy_type": "airport", "country": "Global"},
    "employee_eligibility.txt": {"policy_type": "eligibility", "country": "Global"},
    "expense_policy.txt": {"policy_type": "expense", "country": "Global"},
    "cancellation_policy.txt": {"policy_type": "cancellation", "country": "Global"},
    "approval_policy.txt": {"policy_type": "approval", "country": "Global"},
}

EXPECTED_FILES = list(DOCUMENT_METADATA.keys())


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PolicyChunk:
    chunk_id: str
    source: str
    policy_type: str
    country: str
    chunk_index: int
    text: str
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Step 1: Load
# ---------------------------------------------------------------------------

def load_documents(data_dir: Path = DATA_DIR) -> dict[str, str]:
    """
    Reads every expected .txt policy file from data_dir.

    Handles:
    - Missing files (skipped with a warning, not a crash)
    - Empty files (kept, flagged as empty)
    - Any file that isn't .txt (ignored)

    Returns: {filename: raw_text}
    """
    documents: dict[str, str] = {}

    if not data_dir.exists():
        raise FileNotFoundError(f"Policy directory not found: {data_dir}")

    for filename in EXPECTED_FILES:
        file_path = data_dir / filename
        if not file_path.exists():
            print(f"[WARN] Missing expected file, skipping: {filename}")
            continue

        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"[WARN] Could not decode {filename} as UTF-8, skipping.")
            continue

        if not text.strip():
            print(f"[WARN] {filename} is empty.")
            documents[filename] = ""
            continue

        documents[filename] = text

    # Also warn about any .txt files present that we don't have metadata for,
    # so new policy docs don't silently fail to get tagged.
    for file_path in data_dir.glob("*.txt"):
        if file_path.name not in DOCUMENT_METADATA:
            print(f"[WARN] Found untracked policy file (no metadata mapping): {file_path.name}")

    return documents


def print_document_summary(documents: dict[str, str]) -> None:
    print(f"\n{'Document':<30}{'Characters':>12}")
    print("-" * 42)
    for name, text in documents.items():
        print(f"{name:<30}{len(text):>12,}")
    print("-" * 42)
    print(f"{'TOTAL':<30}{sum(len(t) for t in documents.values()):>12,}\n")


# ---------------------------------------------------------------------------
# Step 2: Clean
# ---------------------------------------------------------------------------

def clean_document(text: str) -> str:
    """
    Light-touch cleaning that preserves headings and policy structure
    (numbered sections, bullet lists) while removing noise.
    """
    if not text:
        return ""

    # Normalize Windows/Mac line endings to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse runs of 3+ blank lines down to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse repeated spaces/tabs (but keep newlines intact)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Strip trailing whitespace on each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Remove non-printable / control characters except newline
    text = re.sub(r"[^\x20-\x7E\n]", "", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Step 3: Metadata
# ---------------------------------------------------------------------------

def get_metadata(filename: str) -> dict:
    meta = DOCUMENT_METADATA.get(filename)
    if meta is None:
        # Sensible fallback for any future/unmapped document
        return {"policy_type": "unknown", "country": "Global"}
    return {"source": filename, **meta}


# ---------------------------------------------------------------------------
# Step 4: Chunk
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits cleaned text into overlapping chunks, breaking on paragraph/section
    boundaries where possible rather than mid-sentence.

    Strategy: greedily accumulate paragraphs (split on blank lines) until the
    chunk_size budget would be exceeded, then start a new chunk that repeats
    the last `overlap` characters of the previous chunk for context
    continuity. This keeps numbered policy sections mostly intact instead of
    slicing at a fixed character offset.
    """
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        # Current chunk is full - close it out
        if current:
            chunks.append(current)

        # Start the new chunk with overlap from the end of the previous chunk
        overlap_text = current[-overlap:] if current and overlap > 0 else ""

        if len(para) > chunk_size:
            # A single paragraph is bigger than chunk_size on its own
            # (rare in these short policies, but handled for robustness) -
            # hard-split it with overlap.
            start = 0
            while start < len(para):
                end = start + chunk_size
                piece = para[start:end]
                chunks.append((overlap_text + "\n\n" + piece).strip() if overlap_text else piece)
                overlap_text = piece[-overlap:] if overlap > 0 else ""
                start = end - overlap if end - overlap > start else end
            current = ""
        else:
            current = (overlap_text + "\n\n" + para).strip() if overlap_text else para

    if current:
        chunks.append(current)

    return chunks


def chunk_document(filename: str, cleaned_text: str) -> list[PolicyChunk]:
    meta = get_metadata(filename)
    raw_chunks = chunk_text(cleaned_text)

    policy_chunks = []
    for idx, chunk in enumerate(raw_chunks):
        policy_chunks.append(
            PolicyChunk(
                chunk_id=f"{filename.replace('.txt', '')}_{idx:03d}",
                source=filename,
                policy_type=meta["policy_type"],
                country=meta["country"],
                chunk_index=idx,
                text=chunk,
            )
        )
    return policy_chunks


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_all(data_dir: Path = DATA_DIR, save: bool = True) -> list[PolicyChunk]:
    documents = load_documents(data_dir)
    print_document_summary(documents)

    all_chunks: list[PolicyChunk] = []
    for filename, raw_text in documents.items():
        cleaned = clean_document(raw_text)
        doc_chunks = chunk_document(filename, cleaned)
        all_chunks.extend(doc_chunks)
        print(f"{filename}: {len(doc_chunks)} chunk(s) "
              f"(avg {sum(c.char_count for c in doc_chunks) // max(len(doc_chunks), 1)} chars/chunk)")

    print(f"\nTotal chunks across all documents: {len(all_chunks)}")

    if save:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in all_chunks], f, indent=2)
        print(f"Saved chunks to: {OUTPUT_PATH}")

    return all_chunks


if __name__ == "__main__":
    process_all()
