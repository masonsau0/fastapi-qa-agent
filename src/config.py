"""Config lives here. One file, everything tunable.

The convention: a few small dataclasses, then singleton instances at the bottom
so the rest of the codebase just does `from src.config import RETRIEVAL` etc.
This means no env vars sprinkled across modules — if you want to change a knob,
it's here, full stop.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Load .env if present. Does nothing if the file doesn't exist or vars are
# already set in the environment (env always wins over .env, by design).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent

# The repo we're answering questions about.
TARGET_REPO_URL = "https://github.com/fastapi/fastapi.git"
TARGET_REPO_DIR = ROOT / "data" / "fastapi"  # populated by scripts/clone_repo.sh

# Where things live.
INDEX_DIR = ROOT / "data" / "index"
BENCHMARK_PATH = ROOT / "data" / "benchmark" / "questions.jsonl"
RESULTS_DIR = ROOT / "data" / "results"


# ---------- Models ----------


@dataclass
class ModelConfig:
    # Anthropic model used by the agent. Haiku 4.5 is the sweet spot for
    # cost/quality on tool-use workloads — Sonnet helps a few percentage
    # points but costs 3x.
    agent_model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 2048
    # Cap on iterations of the tool-call loop. Keeps a confused agent from
    # spinning forever. 8 is generous — most questions resolve in 2-4 turns.
    max_iterations: int = 12

    # Embedding model for dense retrieval. all-MiniLM-L6-v2 is small (~80MB),
    # fast, and good enough. If you want a quality bump later, bge-small-en
    # is a drop-in replacement.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


# ---------- Retrieval ----------


@dataclass
class RetrievalConfig:
    # Code chunking. We chunk by lines, not tokens — code has natural line
    # boundaries and chunking by token count splits in awkward places.
    chunk_lines: int = 60
    chunk_overlap_lines: int = 10

    # File patterns to index. Code paths get the `code` index; docs paths
    # get the `docs` index. The two tools (search_code, search_docs) hit
    # different indexes so the agent doesn't have to filter post-hoc.
    code_extensions: tuple[str, ...] = (".py",)
    docs_extensions: tuple[str, ...] = (".md", ".rst")

    # Files / directories to ignore. Tests are kept because they're often
    # the best documentation for "how is this used?".
    ignore_dirs: tuple[str, ...] = (
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        "build",
        "dist",
        ".mypy_cache",
        ".pytest_cache",
    )

    # Hybrid retrieval mixing weight. alpha=0.5 means equal BM25 + dense.
    # Higher = more BM25 (term matches), lower = more dense (semantic).
    # 0.5 is a defensible default; 0.4 worked slightly better on my dev runs.
    hybrid_alpha: float = 0.4

    # Default top-K for tools. The agent can ask for more in its tool call.
    default_top_k: int = 5


# ---------- Eval ----------


@dataclass
class EvalConfig:
    # Concurrency limit when running the benchmark. Anthropic's tier-1 limit
    # is generous but not infinite — 4 parallel requests is comfortable.
    max_concurrent: int = 4
    # Per-question timeout in seconds. If the agent is still going at this
    # point something has gone wrong.
    per_question_timeout_s: int = 90


# ---------- Singletons ----------

MODEL = ModelConfig()
RETRIEVAL = RetrievalConfig()
EVAL = EvalConfig()


# ---------- Env helpers ----------


def require_env(name: str) -> str:
    """Fail loudly if a required env var is missing. Better than a cryptic
    error 200 lines into a call stack."""
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in, "
            f"or `export {name}=...` in your shell."
        )
    return val
