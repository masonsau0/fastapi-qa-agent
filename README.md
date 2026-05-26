# FastAPI Codebase QA Agent

https://github.com/user-attachments/assets/3ffde1ca-41a7-411a-b743-33d430a12f4f

A tool-using agent that answers questions about the [FastAPI](https://github.com/fastapi/fastapi) codebase. Ask it "how does dependency injection work?" or "why did this file change recently?" and it goes and figures it out. It searches the code, searches the docs, reads specific files, and looks at git history.

The point of the project isn't really the agent. It's the **comparison**. There's a RAG-only baseline (one shot of retrieval, one shot of generation) and the tool-using agent, run on the same 50-question benchmark, with the same model underneath. The eval tells you whether the extra complexity of the agent loop actually pays off, or whether you'd be better off just doing RAG.

Spoiler from my dev runs: the agent wins on questions that need following a thread (search, then read more, then check git), and ties on simple lookup questions. So the answer to "should I always use an agent" is no.

## What it does

- Five tools the agent can call: `search_code`, `search_docs`, `read_file_lines`, `git_log_for_file`, `find_pr_for_commit`. Implementations in `src/tools/impl.py`.
- Hybrid retrieval (BM25 + sentence-transformer embeddings, min-max fused). BM25 is implemented from scratch in `src/retrieval/bm25.py`. I went back and forth on this and decided showing the math was more useful than saving 40 lines.
- A 50-question benchmark mined from real FastAPI GitHub PRs. The mining is automated, the curation is manual (I read each one and verified the answer). See `src/eval/mine_benchmark.py`.
- Eval harness that scores both the agent and the RAG baseline on the same questions using keyword recall and file-citation accuracy.
- A small FastAPI server and Streamlit UI for the demo.

## Results

> Numbers from my dev run. Yours will be similar but not identical, since the agent is non-deterministic.

| Metric                 | RAG baseline | Agent | Δ      |
| ---------------------- | ------------ | ----- | ------ |
| Keyword recall         | 0.879        | 0.863 | -0.016 |
| File citation accuracy | 0.193        | 0.460 | +0.267 |
| Avg agent iterations   | n/a          | 5.68  | n/a    |

The headline is **file citation accuracy**. The agent more than doubles the baseline (19% to 46%). The baseline relies on the model's prior knowledge of FastAPI and can often produce a plausible-sounding answer with the right keywords, but it's much worse at pointing at the file the answer actually lives in. The agent does real work (searches, reads files, then writes) and the file citations show it.

The agent loses 1.6 points on keyword recall. That's a real tradeoff worth understanding. The baseline is freer to dump every keyword it knows, while the agent stays grounded in what it retrieved. Honest grounding beats keyword bingo for most real use cases, but for someone just skimming a definition the baseline might feel snappier.

Tool distribution across 50 questions (312 total calls):

| Tool               | Calls | % of total |
| ------------------ | ----- | ---------- |
| search_code        | 161   | 52%        |
| read_file_lines    | 110   | 35%        |
| search_docs        | 39    | 13%        |
| find_pr_for_commit | 1     | <1%        |
| git_log_for_file   | 1     | <1%        |

The `search_code` then `read_file_lines` sequence is the dominant pattern: find the relevant chunk, then pull more lines around it for context. The two git-history tools were almost never used. The questions in this benchmark are about "how does X work," not "why did X change," so the git tools sit unused. Honest signal: if I redid the benchmark with more "why did this change" questions, those tools would earn their place.

## Setup

You need Python 3.10+, git, and an Anthropic API key. Optional but recommended: a GitHub Personal Access Token (raises the API rate limit from 60 to 5000 req/hr, which matters for the benchmark miner).

```bash
git clone https://github.com/<you>/fastapi-qa-agent.git
cd fastapi-qa-agent

python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env. Add at least ANTHROPIC_API_KEY.
```

Before anything else, **set a spending cap on the Anthropic console.** $15/month is plenty for this project (the full eval run is $1 to $3). See `SECURITY.md` for the why.

## Workflow

### 1. Clone the target repo and build the indexes

```bash
./scripts/clone_repo.sh
python -m src.retrieval.build_indexes
```

The clone is shallow (depth 200 commits) so it's small. The index build takes about 2 minutes on a laptop. Most of that is computing embeddings for ~5K code chunks.

### 2. Mine the benchmark

```bash
python -m src.eval.mine_benchmark --target 200
```

This pulls about 200 merged PRs from the FastAPI repo and writes them to `data/benchmark/candidates.jsonl`. Each row has the PR title, body, and list of touched files.

> **Note on mining strategy.** The original plan was to mine closed issues and follow them to their resolving PRs via GitHub's timeline API. After a debug run produced almost no candidates, I inspected the actual timeline events and discovered that FastAPI's contributors don't consistently use the "Closes #N" commit syntax that creates `cross-referenced` events. Issue-to-PR linkage hit rate was under 5%. I pivoted to mining merged PRs directly. Every merged PR already has a title, body, and file list, so the hit rate jumped to about 12%. PR titles also map more cleanly to "How does X work?" questions than raw issue bodies, which tend to be messy bug reports.

**Then the manual part.** Open `candidates.jsonl`, go through them, and for each one you want to keep:

- Set `curated_question` to a clean, standalone version of the question (the raw issue body is often messy).
- Set `curated_answer_keywords` to 3 to 6 short strings that any correct answer should contain.
- Set `expected_files` to the file(s) a correct answer should cite. The miner pre-fills this from the PR's touched files; usually you'll want to trim it.
- Set `curated: true`.

When you have ~50 curated rows, save them to `data/benchmark/questions.jsonl` (just the curated ones, one per line).

This part is real work. Budget 2 to 3 hours. The quality of the eval is entirely determined by how careful you are here.

### 3. Run the eval

```bash
python -m src.eval.run_eval
```

Runs both the agent and the RAG baseline on every curated question, scores them, prints a summary, writes `data/results/eval.json`. Costs about $1 to $3 in Anthropic credits.

Useful flags:

- `--limit 5` runs only the first 5 questions. Good smoke test.
- `--no-baseline` or `--no-agent` runs only one of the two.

### 4. Demo the server

```bash
# Terminal 1
uvicorn server.app:app --port 8000

# Terminal 2
streamlit run ui/app.py
```

Streamlit opens in your browser at `localhost:8501`. Ask a question, watch the tool trace, copy a screenshot for your resume. Yes, the model runs locally, no, you don't pay anything per demo retake.

## Project layout

```
src/
  config.py              # all knobs in one place
  retrieval/
    bm25.py              # BM25 from scratch
    dense.py             # sentence-transformer embeddings
    chunker.py           # line-windowed chunks with overlap
    hybrid.py            # BM25 + dense, min-max fused
    walker.py            # finds source / doc files in the repo
    build_indexes.py     # CLI: builds the two indexes
  tools/
    schemas.py           # tool schemas in Anthropic format
    impl.py              # actual tool implementations
  agent/
    loop.py              # the agent loop (tool_use, tool_result, repeat)
    baseline.py          # RAG-only baseline for comparison
  eval/
    mine_benchmark.py    # CLI: mine candidate (issue, PR) pairs
    run_eval.py          # CLI: run agent + baseline, score, summarize
server/
  app.py                 # FastAPI server (synchronous /ask endpoint)
ui/
  app.py                 # Streamlit UI
tests/
  test_retrieval.py
  test_eval_and_server.py
data/
  fastapi/               # gitignored, cloned target repo
  index/                 # gitignored, built artifacts
  benchmark/
    candidates.jsonl     # gitignored, raw miner output
    questions.jsonl      # COMMITTED, your curated benchmark
  results/               # gitignored, eval outputs
```

## What I learned building this

A few things worth writing down. These are the kinds of details that come up in interviews.

**Tool descriptions matter as much as tool implementations.** My first pass at `search_code` vs `search_docs` had similar descriptions and the agent kept picking the wrong one. Adding explicit disambiguation ("prefer this for X, prefer the other for Y") fixed it. The descriptions are basically prompt engineering.

**BM25 needs identifier-aware tokenization for code.** Naively splitting on whitespace makes `get_current_user` a single token, which BM25 will never match against a query for `user`. Splitting on `_` and `.` makes the index much more useful at the cost of slightly diluted IDF.

**Iteration cap matters more than you'd think.** A confused agent can loop on the same tool forever, racking up tokens. My first cap was 8, but it bit me on a "how does dependency injection work" question that legitimately needed to chase across files. I bumped it to 12 and added an instruction in the system prompt: "After 5 calls, write your answer based on what you have rather than searching further." The current average is 5.68, well under the cap.

**Preambles are surprisingly hard to suppress.** My first prompt told the model "go straight into the answer" and got "Now I have a clear picture of how X works..." back almost every time. I had to add an explicit list of forbidden openers ("Now I have", "Perfect", "Let me", "Based on my research") before it stopped. Prompt engineering against your own training data is real.

**Rate limits are infrastructure.** On Anthropic's tier-1 limit (50K input tokens/minute), I had to drop eval concurrency from 4 to 1. The SDK auto-retries on 429s, so nothing breaks, it just takes longer. Lesson: if you're benchmarking a tool-using agent, model the rate limit before you model parallelism.

**The baseline keeps you honest.** Without the RAG-only comparison, you have no way to argue the agent's complexity is paying off. With it, you can make a real claim.

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

- [FastAPI](https://github.com/fastapi/fastapi) (MIT licensed). The target codebase.
- [Anthropic Claude](https://www.anthropic.com/claude). The model behind the agent.
- [sentence-transformers](https://www.sbert.net/). Embeddings.
