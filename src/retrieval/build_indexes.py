"""Build the code and docs indexes from the cloned target repo.

Run once after cloning. Re-run if you've edited config (chunk size, etc).
Produces two directories under data/index/: `code/` and `docs/`.

    python -m src.retrieval.build_indexes
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from src.config import INDEX_DIR, TARGET_REPO_DIR
from src.retrieval.chunker import Chunk, chunk_file
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.walker import iter_code_files, iter_docs_files

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def _chunk_all(file_iter, repo_root: Path, label: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    n_files = 0
    for path, content in file_iter:
        chunks.extend(chunk_file(path, content, repo_root))
        n_files += 1
        if n_files % 200 == 0:
            log.info("[%s] processed %d files, %d chunks so far", label, n_files, len(chunks))
    log.info("[%s] done: %d files, %d chunks total", label, n_files, len(chunks))
    return chunks


def build_one(label: str, file_iter, repo_root: Path, out_dir: Path) -> None:
    log.info("== Building %s index ==", label)
    t0 = time.time()
    chunks = _chunk_all(file_iter, repo_root, label)
    if not chunks:
        log.warning("[%s] no chunks produced — skipping", label)
        return
    retriever = HybridRetriever.build(chunks)
    retriever.save(out_dir)
    log.info("[%s] saved to %s in %.1fs", label, out_dir, time.time() - t0)


def main() -> int:
    if not TARGET_REPO_DIR.exists():
        log.error("Target repo not found at %s.", TARGET_REPO_DIR)
        log.error("Run scripts/clone_repo.sh first.")
        return 1

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    build_one("code", iter_code_files(TARGET_REPO_DIR), TARGET_REPO_DIR, INDEX_DIR / "code")
    build_one("docs", iter_docs_files(TARGET_REPO_DIR), TARGET_REPO_DIR, INDEX_DIR / "docs")
    log.info("All indexes built.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
