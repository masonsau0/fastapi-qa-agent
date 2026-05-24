"""RAG-only baseline.

The honest baseline for "does the agent help?". Same model, same retrieval,
but no tool loop — just retrieve once over code+docs combined, stuff into
the prompt, generate.

If the agent doesn't beat this by a meaningful margin, the agent isn't earning
its complexity. That's a fine finding to report — the project's point is
measurement, not winning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from anthropic import Anthropic

from src.config import INDEX_DIR, MODEL, RETRIEVAL, require_env
from src.retrieval.hybrid import HybridRetriever

log = logging.getLogger(__name__)


BASELINE_SYSTEM_PROMPT = """\
You are an assistant that answers questions about the FastAPI codebase.

The user message will include retrieved code and documentation excerpts.
Use them to answer the question. Cite the file path and line numbers from the
excerpts (e.g. "see fastapi/routing.py:142-180"). If the excerpts don't
contain enough information, say so rather than guessing.
"""


@dataclass
class BaselineResult:
    answer: str
    retrieved_citations: list[str]


class RAGBaseline:
    """Single-shot RAG: one query, top-k from each index, then generate."""

    def __init__(self, k_each: int = 5):
        self.client = Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))
        self.k_each = k_each
        self.code = HybridRetriever.load(INDEX_DIR / "code")
        self.docs = HybridRetriever.load(INDEX_DIR / "docs")

    def ask(self, question: str) -> BaselineResult:
        code_hits = self.code.search(question, k=self.k_each, alpha=RETRIEVAL.hybrid_alpha)
        docs_hits = self.docs.search(question, k=self.k_each, alpha=RETRIEVAL.hybrid_alpha)

        citations = [c.citation for c, _ in code_hits] + [c.citation for c, _ in docs_hits]

        excerpts = []
        for label, hits in (("CODE", code_hits), ("DOCS", docs_hits)):
            for chunk, _ in hits:
                excerpts.append(f"[{label}] {chunk.citation}\n```\n{chunk.text}\n```")

        prompt = f"Question: {question}\n\nRetrieved excerpts:\n\n" + "\n\n".join(excerpts)

        response = self.client.messages.create(
            model=MODEL.agent_model,
            max_tokens=MODEL.max_tokens,
            system=BASELINE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return BaselineResult(answer=text, retrieved_citations=citations)
