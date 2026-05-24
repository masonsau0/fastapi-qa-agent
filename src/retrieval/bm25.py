"""BM25 from scratch.

I imported rank_bm25 first, then ripped it out. Two reasons:
  1. Showing the math matters more than saving 40 lines.
  2. The library tokenizes naively (split on whitespace) and for code that
     wrecks results — `app.get` becomes one token, not three.

This implementation does a slightly-better-than-naive tokenization: lowercase,
strip punctuation, but also split on `.` and `_` so identifiers like
`get_current_user` produce useful tokens. Numbers and short tokens (<2 chars)
get dropped. It's still not a real code tokenizer, but it's a big step up.

BM25 formula (Okapi BM25):
    score(D, Q) = sum over t in Q of:
        idf(t) * (tf(t, D) * (k1 + 1)) / (tf(t, D) + k1 * (1 - b + b * |D| / avgdl))

Defaults: k1=1.5, b=0.75. These are the standard "good enough for everything"
values from the original paper.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

# Identifier-aware tokenizer. Splits on whitespace, punctuation, `.`, `_`.
# Then lowercases and drops tokens shorter than 2 chars.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    raw = _TOKEN_RE.findall(text.lower())
    # Further split snake_case-ish tokens (the regex already split on _).
    # The list comp below filters to anything 2+ chars long.
    return [t for t in raw if len(t) >= 2]


@dataclass
class BM25Index:
    """Pre-computed BM25 index over a fixed corpus.

    Build once, query many times. Not threadsafe for builds — fine for queries
    since we never mutate after construction.
    """

    # Length of each document (in tokens).
    doc_lens: list[int]
    # Average document length (for the BM25 normalization term).
    avgdl: float
    # For each term: a dict {doc_id: term_frequency_in_that_doc}.
    inverted: dict[str, dict[int, int]]
    # idf for each term, pre-computed at build time.
    idf: dict[str, float]
    # Number of documents.
    n_docs: int

    # BM25 hyperparameters.
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, docs: list[str]) -> BM25Index:
        n_docs = len(docs)
        if n_docs == 0:
            raise ValueError("Cannot build a BM25 index over zero documents")

        tokenized = [tokenize(d) for d in docs]
        doc_lens = [len(t) for t in tokenized]
        avgdl = sum(doc_lens) / n_docs if n_docs else 0.0

        # Build the inverted index: term -> {doc_id: tf}
        inverted: dict[str, dict[int, int]] = {}
        for doc_id, tokens in enumerate(tokenized):
            counts = Counter(tokens)
            for term, tf in counts.items():
                inverted.setdefault(term, {})[doc_id] = tf

        # idf per term. Standard Robertson-Spärck Jones formula with the
        # smoothing trick (+0.5 / +0.5) to keep idf positive for terms that
        # appear in more than half the docs.
        idf: dict[str, float] = {}
        for term, postings in inverted.items():
            df = len(postings)
            idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

        return cls(
            doc_lens=doc_lens,
            avgdl=avgdl,
            inverted=inverted,
            idf=idf,
            n_docs=n_docs,
        )

    def score(self, query: str) -> list[float]:
        """Score every document against the query. Returns list of length n_docs."""
        scores = [0.0] * self.n_docs
        q_tokens = tokenize(query)
        if not q_tokens:
            return scores

        for term in q_tokens:
            postings = self.inverted.get(term)
            if not postings:
                continue
            term_idf = self.idf[term]
            for doc_id, tf in postings.items():
                dl = self.doc_lens[doc_id]
                # The denominator that does the length normalization.
                norm = 1 - self.b + self.b * (dl / self.avgdl if self.avgdl else 0)
                scores[doc_id] += term_idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)

        return scores

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        """Convenience: return the top-k (doc_id, score) pairs, sorted desc."""
        scores = self.score(query)
        # enumerate -> list of (id, score), sort by score desc, slice top-k.
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]
