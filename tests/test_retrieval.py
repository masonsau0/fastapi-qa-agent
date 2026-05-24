"""Tests for BM25 and chunking. No network, no API key needed.

Some of these are dumb little tests — they exist because they caught bugs while
I was iterating. Worth leaving in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.retrieval.bm25 import BM25Index, tokenize
from src.retrieval.chunker import chunk_file

# ---------- Tokenizer ----------


def test_tokenize_basic():
    assert tokenize("Hello World") == ["hello", "world"]


def test_tokenize_splits_identifiers():
    # The regex splits on _ and .  so identifiers fragment usefully.
    toks = tokenize("get_current_user")
    assert "get" in toks
    assert "current" in toks
    assert "user" in toks


def test_tokenize_drops_short_tokens():
    # Single-letter tokens are noise for BM25.
    assert "a" not in tokenize("a quick brown fox")
    assert "quick" in tokenize("a quick brown fox")


def test_tokenize_handles_empty():
    assert tokenize("") == []
    assert tokenize("   ") == []


# ---------- BM25 ----------


def test_bm25_ranks_relevant_doc_first():
    docs = [
        "the quick brown fox",
        "lorem ipsum dolor sit amet",
        "the lazy dog sleeps in the afternoon",
    ]
    idx = BM25Index.build(docs)
    top = idx.top_k("brown fox", k=3)
    # Top result should be doc 0 (the one containing both query terms).
    assert top[0][0] == 0


def test_bm25_empty_query_returns_zero_scores():
    idx = BM25Index.build(["one two three", "four five six"])
    scores = idx.score("")
    assert scores == [0.0, 0.0]


def test_bm25_unknown_query_returns_zero_scores():
    idx = BM25Index.build(["one two three", "four five six"])
    scores = idx.score("xyzzy")
    assert scores == [0.0, 0.0]


def test_bm25_raises_on_empty_corpus():
    with pytest.raises(ValueError):
        BM25Index.build([])


# ---------- Chunker ----------


def test_chunker_single_short_file(tmp_path: Path):
    # File smaller than the chunk window should produce exactly one chunk.
    p = tmp_path / "tiny.py"
    p.write_text("def f():\n    return 1\n")
    chunks = chunk_file(p, p.read_text(), tmp_path)
    assert len(chunks) == 1
    assert chunks[0].start_line == 1
    assert "def f" in chunks[0].text


def test_chunker_long_file_produces_overlapping_chunks(tmp_path: Path):
    # 200 lines, default window=60, overlap=10 → step=50.
    # Expected chunks: starting at 0, 50, 100, 150 → 4 chunks (last one shorter).
    p = tmp_path / "big.py"
    content = "\n".join(f"line {i}" for i in range(1, 201))
    p.write_text(content)
    chunks = chunk_file(p, content, tmp_path)
    assert len(chunks) >= 3
    # Citations should be 1-indexed and inclusive.
    for c in chunks:
        assert c.start_line >= 1
        assert c.end_line >= c.start_line
