"""
embeddings.py
-------------
Day 2 deliverable, Part 1 & 2: Convert policy chunks into vector embeddings
using Sentence Transformers, and store them (chunk + embedding + metadata)
in a local FAISS vector store.

Pipeline:
    data/chunks.json (from src/ingestion.py)
        -> embed each chunk's text
        -> build a FAISS index (cosine similarity via normalized inner product)
        -> save index + chunk metadata to disk (vector_store/)

Run directly:
    python src/embeddings.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# all-MiniLM-L6-v2: small (~80MB), fast on CPU, strong general-purpose
# sentence embedding quality - a good fit for a short policy corpus like
# this one where we don't need a huge model.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks.json"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
INDEX_PATH = VECTOR_STORE_DIR / "policy.index"
METADATA_PATH = VECTOR_STORE_DIR / "chunk_metadata.pkl"

_model_cache: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Loads (and caches) the Sentence Transformers model."""
    global _model_cache
    if _model_cache is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
        _model_cache = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model_cache


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/ingestion.py` first to generate it."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_chunks(chunks: list[dict], model: SentenceTransformer | None = None) -> np.ndarray:
    """
    Converts each chunk's text into an embedding vector.

    Policy Chunk -> Embedding Model -> Vector
    """
    model = model or get_embedding_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so inner product == cosine similarity
    )
    return embeddings.astype("float32")


def build_vector_store(chunks: list[dict], embeddings: np.ndarray) -> faiss.Index:
    """
    Builds a FAISS index over the embeddings. Using IndexFlatIP (inner
    product) on normalized vectors gives us cosine similarity search,
    which is the standard choice for sentence-embedding retrieval.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def save_vector_store(index: faiss.Index, chunks: list[dict]) -> None:
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "wb") as f:
        # chunks are stored in the same order they were added to the FAISS
        # index, so index position i <-> chunks[i]
        pickle.dump(chunks, f)
    print(f"Saved FAISS index to {INDEX_PATH}")
    print(f"Saved chunk metadata to {METADATA_PATH}")


def load_vector_store() -> tuple[faiss.Index, list[dict]]:
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Vector store not found. Run `python src/embeddings.py` first."
        )
    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def build_and_save() -> None:
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    embeddings = embed_chunks(chunks)
    print(f"Generated embeddings: shape={embeddings.shape}")

    index = build_vector_store(chunks, embeddings)
    print(f"Built FAISS index with {index.ntotal} vectors (dim={index.d})")

    save_vector_store(index, chunks)


if __name__ == "__main__":
    build_and_save()
