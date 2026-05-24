"""Splits source files into overlapping line-windowed chunks.

I went back and forth on this. The fashionable thing is to chunk by AST nodes
(functions, classes). It's smarter but it's also fragile — half the things you
want to retrieve aren't whole functions (a comment block, a regex pattern, an
error message). Line-windowed chunking with overlap is dumber but covers more.

Overlap matters more than you'd think. Without it a definition gets split across
chunks and BM25 misses the query. With ~15% overlap most short queries find a
chunk that contains both the call site and the definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config import RETRIEVAL


@dataclass(frozen=True)
class Chunk:
    """One indexable unit. `text` is what gets searched; the path/lines let us
    cite back to the source."""

    path: str  # relative to the target repo root
    start_line: int  # 1-indexed, inclusive
    end_line: int  # 1-indexed, inclusive
    text: str

    @property
    def citation(self) -> str:
        """Human-readable citation like `fastapi/routing.py:120-179`."""
        return f"{self.path}:{self.start_line}-{self.end_line}"


def chunk_file(path: Path, content: str, repo_root: Path) -> list[Chunk]:
    """Split a single file into chunks. Tiny files become a single chunk."""
    lines = content.splitlines()
    if not lines:
        return []

    rel_path = str(path.relative_to(repo_root))
    chunks: list[Chunk] = []

    window = RETRIEVAL.chunk_lines
    overlap = RETRIEVAL.chunk_overlap_lines
    step = max(1, window - overlap)

    # Walk by `step` so each chunk overlaps the next by `overlap` lines.
    for start_idx in range(0, len(lines), step):
        end_idx = min(start_idx + window, len(lines))
        chunk_text = "\n".join(lines[start_idx:end_idx])
        # Skip whitespace-only chunks (mostly in markdown with big gaps).
        if not chunk_text.strip():
            continue
        chunks.append(
            Chunk(
                path=rel_path,
                start_line=start_idx + 1,
                end_line=end_idx,
                text=chunk_text,
            )
        )
        # If we already covered the whole file, stop.
        if end_idx >= len(lines):
            break

    return chunks
