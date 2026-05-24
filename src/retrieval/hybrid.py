"""Hybrid retriever — BM25 + dense, min-max normalized and weighted.

Same fusion approach as my RAG project: normalize each scorer's outputs to
[0, 1] over the candidate pool, then take a weighted sum. alpha=0.4 leans
slightly dense (semantic match matters more than exact term match for natural-
language questions over code), but it's a knob.

The whole `HybridRetriever` is a thin coordinator: builds both indexes, holds
the chunks list, and runs scoring. Saving/loading uses a directory because we
have multiple artifacts (chunks json, embeddings npy, bm25 pickle).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.config import RETRIEVAL
from src.retrieval.bm25 import BM25Index
from src.retrieval.chunker import Chunk
from src.retrieval.dense import DenseIndex


def _minmax(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. If all scores are equal returns zeros
    (so the scorer contributes nothing for that query, which is correct)."""
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class HybridRetriever:
    """Wraps BM25 + DenseIndex over the same chunk list.

    Build once with `.build(chunks)`, then call `.search(query, k)` as many
    times as you want. Save/load via a directory of artifacts.
    """

    def __init__(self, chunks: list[Chunk], bm25: BM25Index, dense: DenseIndex):
        if not (len(chunks) == bm25.n_docs == dense.n_docs):
            raise ValueError(
                f"Index size mismatch: chunks={len(chunks)}, bm25={bm25.n_docs}, dense={dense.n_docs}"
            )
        self.chunks = chunks
        self.bm25 = bm25
        self.dense = dense

    @classmethod
    def build(cls, chunks: list[Chunk]) -> HybridRetriever:
        if not chunks:
            raise ValueError("Cannot build a retriever over zero chunks")
        texts = [c.text for c in chunks]
        bm25 = BM25Index.build(texts)
        dense = DenseIndex.build(texts)
        return cls(chunks=chunks, bm25=bm25, dense=dense)

    # ---------- Search ----------

    def search(
        self,
        query: str,
        k: int = RETRIEVAL.default_top_k,
        alpha: float = RETRIEVAL.hybrid_alpha,
    ) -> list[tuple[Chunk, float]]:
        """Return top-k (chunk, fused_score) pairs.

        alpha is the BM25 weight: score = alpha * bm25_norm + (1-alpha) * dense_norm.
        """
        bm25_scores = np.array(self.bm25.score(query), dtype=np.float32)
        dense_scores = self.dense.score(query)

        fused = alpha * _minmax(bm25_scores) + (1 - alpha) * _minmax(dense_scores)
        top_indices = np.argsort(-fused)[:k]
        return [(self.chunks[i], float(fused[i])) for i in top_indices]

    # ---------- Persistence ----------

    def save(self, directory: Path) -> None:
        """Write all artifacts to a directory. Idempotent — overwrites."""
        directory.mkdir(parents=True, exist_ok=True)
        # Chunks as JSON (small, human-inspectable).
        with (directory / "chunks.json").open("w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f)
        # Dense embeddings.
        self.dense.save(directory / "embeddings.npy")
        # BM25 — pickle is fine here, the data is just dicts and floats.
        # We use protocol=4 for cross-version compatibility.
        with (directory / "bm25.pkl").open("wb") as f:
            pickle.dump(self.bm25, f, protocol=4)

    @classmethod
    def load(cls, directory: Path) -> HybridRetriever:
        with (directory / "chunks.json").open(encoding="utf-8") as f:
            chunks = [Chunk(**c) for c in json.load(f)]
        dense = DenseIndex.load(directory / "embeddings.npy")
        with (directory / "bm25.pkl").open("rb") as f:
            bm25: BM25Index = pickle.load(f)  # noqa: S301 — trusted local file
        return cls(chunks=chunks, bm25=bm25, dense=dense)
