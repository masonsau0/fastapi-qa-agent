"""Dense retrieval with sentence-transformers.

Keeps embeddings in a single numpy array; cosine similarity is just a dot
product after L2 normalization. We don't need FAISS at this scale — the
FastAPI repo produces ~5k chunks and brute-force cosine is microseconds.

If you ever index a 100k+ chunk corpus, swap in faiss-cpu and call it a day.
The interface here (encode, search) is FAISS-shaped on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.config import MODEL

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def _load_model() -> SentenceTransformer:
    """Lazy import. Loads once per process — the import alone is a few hundred ms."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL.embedding_model)


@dataclass
class DenseIndex:
    """Holds a normalized embedding matrix and provides cosine search.

    Built once, queried many times. The matrix is float32 — float16 would
    halve memory but we're at single-digit MB for this corpus, not worth it.
    """

    embeddings: np.ndarray  # shape (n_docs, dim), L2-normalized
    n_docs: int

    @classmethod
    def build(cls, docs: list[str], batch_size: int = 64) -> DenseIndex:
        if not docs:
            raise ValueError("Cannot build a dense index over zero documents")
        model = _load_model()
        # show_progress_bar=False because this gets called from CLI scripts and
        # the progress bar fights with the surrounding logger.
        raw = model.encode(
            docs,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize so dot == cosine
        )
        return cls(embeddings=raw.astype(np.float32), n_docs=len(docs))

    def score(self, query: str) -> np.ndarray:
        """Cosine similarity of query against every doc. Returns array of length n_docs."""
        model = _load_model()
        q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        # q is shape (1, dim); embeddings is (n, dim); result is (n,).
        return (self.embeddings @ q.T).flatten()

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.score(query)
        # argsort returns ascending; flip and slice.
        top_indices = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top_indices]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.embeddings)

    @classmethod
    def load(cls, path: Path) -> DenseIndex:
        emb = np.load(path)
        return cls(embeddings=emb, n_docs=emb.shape[0])
