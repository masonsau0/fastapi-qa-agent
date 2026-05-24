"""Run the benchmark and score agent vs baseline.

Scoring uses three signals:
  1. Keyword recall: fraction of `curated_answer_keywords` present in the answer
     (case-insensitive, after light normalization). The same EM-style metric I
     used in the LoRA project — pragmatic, not fancy.
  2. File-citation accuracy: of the files cited in the answer, what fraction are
     in `expected_files`? Catches "right answer, wrong source" cases.
  3. Tool-usage stats: average number of tool calls, distribution of tools used.
     Not a quality metric, but useful color for the README.

Run:
    python -m src.eval.run_eval --benchmark data/benchmark/questions.jsonl --output data/results/eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import string
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.agent.baseline import RAGBaseline
from src.agent.loop import CodebaseAgent
from src.config import BENCHMARK_PATH, EVAL, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ---------- Scoring helpers ----------

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")


def _normalize(text: str) -> str:
    return _PUNCT_RE.sub(" ", text.lower())


def keyword_recall(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    norm = _normalize(answer)
    hits = sum(1 for kw in keywords if _normalize(kw) in norm)
    return hits / len(keywords)


# File-citation regex: `path/like/this.py` or `path/like/this.py:123-456`
_CITATION_RE = re.compile(r"[a-zA-Z0-9_./-]+\.(py|md|rst)(?::\d+(?:-\d+)?)?")


def file_citation_accuracy(answer: str, expected_files: list[str]) -> float:
    """Fraction of cited files that are in the expected set. Returns 0 if
    nothing was cited (which is itself a partial fail — penalizes uncited answers)."""
    cited = set()
    for m in _CITATION_RE.finditer(answer):
        # Strip line number suffix if present.
        path = m.group(0).split(":")[0]
        cited.add(path)
    if not cited:
        return 0.0
    expected = set(expected_files)
    return len(cited & expected) / len(cited)


# ---------- Per-question records ----------


@dataclass
class QuestionResult:
    question: str
    expected_keywords: list[str]
    expected_files: list[str]

    agent_answer: str = ""
    agent_keyword_recall: float = 0.0
    agent_file_accuracy: float = 0.0
    agent_tool_trace: list[str] = field(default_factory=list)
    agent_iterations: int = 0
    agent_error: str = ""

    baseline_answer: str = ""
    baseline_keyword_recall: float = 0.0
    baseline_file_accuracy: float = 0.0
    baseline_error: str = ""


def _run_agent(agent: CodebaseAgent, q: dict[str, Any], r: QuestionResult) -> None:
    try:
        out = agent.ask(q["curated_question"])
        r.agent_answer = out.answer
        r.agent_tool_trace = out.tool_trace
        r.agent_iterations = out.iterations
        r.agent_keyword_recall = keyword_recall(out.answer, q["curated_answer_keywords"])
        r.agent_file_accuracy = file_citation_accuracy(out.answer, q["expected_files"])
    except Exception as e:  # noqa: BLE001
        log.exception("Agent failed on Q: %s", q["curated_question"][:80])
        r.agent_error = f"{type(e).__name__}: {e}"


def _run_baseline(baseline: RAGBaseline, q: dict[str, Any], r: QuestionResult) -> None:
    try:
        out = baseline.ask(q["curated_question"])
        r.baseline_answer = out.answer
        r.baseline_keyword_recall = keyword_recall(out.answer, q["curated_answer_keywords"])
        r.baseline_file_accuracy = file_citation_accuracy(out.answer, q["expected_files"])
    except Exception as e:  # noqa: BLE001
        log.exception("Baseline failed on Q: %s", q["curated_question"][:80])
        r.baseline_error = f"{type(e).__name__}: {e}"


# ---------- Main ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--benchmark", default=str(BENCHMARK_PATH))
    p.add_argument("--output", default=str(RESULTS_DIR / "eval.json"))
    p.add_argument(
        "--limit", type=int, default=None, help="Only run the first N questions (for smoke tests)."
    )
    p.add_argument("--no-baseline", action="store_true", help="Skip RAG baseline.")
    p.add_argument("--no-agent", action="store_true", help="Skip agent.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bench_path = Path(args.benchmark)
    if not bench_path.exists():
        log.error("Benchmark not found at %s. Run the miner + curation first.", bench_path)
        return 1

    questions = []
    with bench_path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            q = json.loads(line)
            if not q.get("curated"):
                continue
            questions.append(q)

    if args.limit:
        questions = questions[: args.limit]
    log.info("Running on %d curated questions", len(questions))

    if not questions:
        log.error("No curated questions in benchmark. Curate some first.")
        return 1

    agent = None if args.no_agent else CodebaseAgent()
    baseline = None if args.no_baseline else RAGBaseline()

    results: list[QuestionResult] = []

    # Run with bounded concurrency. ThreadPoolExecutor is fine here — every
    # blocking call is network I/O, the GIL is not the bottleneck.
    def _run_one(q: dict[str, Any]) -> QuestionResult:
        r = QuestionResult(
            question=q["curated_question"],
            expected_keywords=q["curated_answer_keywords"],
            expected_files=q["expected_files"],
        )
        if agent is not None:
            _run_agent(agent, q, r)
        if baseline is not None:
            _run_baseline(baseline, q, r)
        return r

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=EVAL.max_concurrent) as ex:
        futures = {ex.submit(_run_one, q): i for i, q in enumerate(questions)}
        for i, fut in enumerate(as_completed(futures), start=1):
            results.append(fut.result())
            if i % 5 == 0 or i == len(questions):
                log.info("[%d / %d] done", i, len(questions))
    log.info("Eval complete in %.1fs", time.time() - t0)

    # Aggregate.
    def _avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "n_questions": len(results),
        "agent": {
            "keyword_recall": _avg([r.agent_keyword_recall for r in results if not r.agent_error]),
            "file_accuracy": _avg([r.agent_file_accuracy for r in results if not r.agent_error]),
            "avg_iterations": _avg([r.agent_iterations for r in results if not r.agent_error]),
            "tool_distribution": dict(Counter(t for r in results for t in r.agent_tool_trace)),
            "errors": sum(1 for r in results if r.agent_error),
        },
        "baseline": {
            "keyword_recall": _avg(
                [r.baseline_keyword_recall for r in results if not r.baseline_error]
            ),
            "file_accuracy": _avg(
                [r.baseline_file_accuracy for r in results if not r.baseline_error]
            ),
            "errors": sum(1 for r in results if r.baseline_error),
        },
    }
    summary["delta"] = {
        "keyword_recall": summary["agent"]["keyword_recall"]
        - summary["baseline"]["keyword_recall"],
        "file_accuracy": summary["agent"]["file_accuracy"] - summary["baseline"]["file_accuracy"],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "per_question": [asdict(r) for r in results],
            },
            indent=2,
        )
    )
    log.info("Wrote results to %s", out_path)

    # Print summary table.
    print("\n" + "=" * 60)
    print(f"{'Metric':<25} {'Baseline':>12} {'Agent':>12} {'Δ':>10}")
    print("-" * 60)
    print(
        f"{'Keyword recall':<25} {summary['baseline']['keyword_recall']:>12.3f} {summary['agent']['keyword_recall']:>12.3f} {summary['delta']['keyword_recall']:>+10.3f}"
    )
    print(
        f"{'File citation accuracy':<25} {summary['baseline']['file_accuracy']:>12.3f} {summary['agent']['file_accuracy']:>12.3f} {summary['delta']['file_accuracy']:>+10.3f}"
    )
    print(f"{'Avg agent iterations':<25} {'-':>12} {summary['agent']['avg_iterations']:>12.2f}")
    print("=" * 60)
    print("Tool distribution:", summary["agent"]["tool_distribution"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
