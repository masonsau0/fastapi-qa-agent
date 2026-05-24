"""Walks the cloned target repo and yields (path, content) for indexable files.

Two passes: `iter_code_files()` for source, `iter_docs_files()` for docs. We
separate them because the agent's two retrieval tools hit different indexes.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from src.config import RETRIEVAL


def _should_skip(path: Path, repo_root: Path) -> bool:
    """True if any directory in the path is on the ignore list."""
    try:
        parts = path.relative_to(repo_root).parts
    except ValueError:
        return True
    return any(p in RETRIEVAL.ignore_dirs for p in parts)


def _iter_files(repo_root: Path, extensions: tuple[str, ...]) -> Iterator[tuple[Path, str]]:
    """Yield (path, content) for every file under repo_root matching extensions.
    Skips ignored directories. Silently skips files that can't be decoded as UTF-8
    — we'd rather miss a file than crash on a stray binary."""
    if not repo_root.exists():
        raise FileNotFoundError(f"Repo not found at {repo_root}. Run scripts/clone_repo.sh first.")
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in extensions:
            continue
        if _should_skip(path, repo_root):
            continue
        try:
            yield path, path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue


def iter_code_files(repo_root: Path) -> Iterator[tuple[Path, str]]:
    return _iter_files(repo_root, RETRIEVAL.code_extensions)


def iter_docs_files(repo_root: Path) -> Iterator[tuple[Path, str]]:
    return _iter_files(repo_root, RETRIEVAL.docs_extensions)
