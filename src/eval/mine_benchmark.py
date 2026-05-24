"""Mine candidate (question, ground-truth-PR) pairs from the target repo.

What this does:
  1. Hit the GitHub API for closed issues with linked PRs.
  2. For each, extract the issue title + body as a candidate question, and
     the linked PR(s) + the files they touched as the candidate answer.
  3. Write out to data/benchmark/candidates.jsonl.

You (the human) then review candidates.jsonl, pick ~50, and curate them into
questions.jsonl. That manual step is the whole point — the eval is only as
good as the ground truth, and an LLM signing off on its own ground truth is
circular.

Usage:
    python -m src.eval.mine_benchmark --target 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx

from src.config import BENCHMARK_PATH, TARGET_REPO_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def _parse_repo_url(url: str) -> tuple[str, str]:
    """github.com/owner/repo[.git] -> (owner, repo)."""
    import re

    m = re.match(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if not m:
        raise ValueError(f"Cannot parse GitHub URL: {url}")
    return m.group(1), m.group(2)


def _headers() -> dict[str, str]:
    h = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "fastapi-qa-agent-miner",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    else:
        log.warning("GITHUB_TOKEN not set — rate limited to 60 req/hr. Set it in .env.")
    return h


def _fetch_closed_issues(
    owner: str, repo: str, per_page: int = 100, max_pages: int = 5
) -> list[dict]:
    """Pull recent closed issues. 'Closed' = they got resolved, hopefully by a PR."""
    all_issues = []
    for page in range(1, max_pages + 1):
        url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        params = {
            "state": "closed",
            "per_page": per_page,
            "page": page,
            "sort": "comments",  # well-discussed issues are usually better questions
            "direction": "desc",
        }
        resp = httpx.get(url, headers=_headers(), params=params, timeout=30.0)
        if resp.status_code != 200:
            log.error("GitHub API returned %s: %s", resp.status_code, resp.text[:200])
            break
        page_issues = resp.json()
        # The issues endpoint also returns PRs (legacy API quirk). Filter PRs out.
        real_issues = [i for i in page_issues if "pull_request" not in i]
        all_issues.extend(real_issues)
        log.info(
            "Page %d: fetched %d issues (filtered from %d items)",
            page,
            len(real_issues),
            len(page_issues),
        )
        if len(page_issues) < per_page:
            break
        time.sleep(0.5)  # gentle on the API even with a token
    return all_issues


def _find_linked_prs(issue: dict, owner: str, repo: str) -> list[dict]:
    """Look for "Closed by #1234" or PR links in the issue's timeline.

    Simple version: scan the issue body + comments for #NNN refs, fetch each,
    keep the ones that are PRs. Not perfect (misses cross-references) but
    catches the common case.
    """
    import re

    text = issue.get("body") or ""
    # Look for #NNN patterns. We'll trust that the FIRST one referenced in the
    # body is the most likely "this is the fix" PR.
    refs = re.findall(r"#(\d{2,6})", text)
    if not refs:
        return []

    prs = []
    for ref in refs[:3]:  # cap at 3 to keep API usage bounded
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{ref}"
        resp = httpx.get(url, headers=_headers(), timeout=15.0)
        if resp.status_code == 200:
            prs.append(resp.json())
        time.sleep(0.3)
    return prs


def _pr_files(pr: dict, owner: str, repo: str) -> list[str]:
    """Files touched by the PR. Returns paths only (no diff)."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr['number']}/files"
    resp = httpx.get(url, headers=_headers(), timeout=15.0)
    if resp.status_code != 200:
        return []
    return [f["filename"] for f in resp.json()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--target", type=int, default=100, help="Max candidates to write.")
    p.add_argument("--output", type=str, default=str(BENCHMARK_PATH.parent / "candidates.jsonl"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    owner, repo = _parse_repo_url(TARGET_REPO_URL)
    log.info("Mining candidate questions from %s/%s", owner, repo)

    issues = _fetch_closed_issues(owner, repo, max_pages=5)
    log.info("Got %d closed issues to inspect", len(issues))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0

    with out_path.open("w", encoding="utf-8") as f:
        for issue in issues:
            if n_written >= args.target:
                break
            # Need a body to use as the question.
            if not (issue.get("body") or "").strip():
                continue
            prs = _find_linked_prs(issue, owner, repo)
            if not prs:
                continue
            # Take the first merged PR as ground truth.
            merged = [p for p in prs if p.get("merged_at")]
            if not merged:
                continue
            pr = merged[0]
            files = _pr_files(pr, owner, repo)

            candidate = {
                "issue_number": issue["number"],
                "issue_url": issue["html_url"],
                "title": issue["title"],
                "body": issue["body"][:1500],  # truncate huge bodies
                "pr_number": pr["number"],
                "pr_url": pr["html_url"],
                "pr_title": pr["title"],
                "pr_files": files,
                # The fields below are what YOU fill in during curation.
                "curated_question": "",
                "curated_answer_keywords": [],
                "expected_files": files,  # default = PR files; you'll trim this
                "curated": False,
            }
            f.write(json.dumps(candidate) + "\n")
            n_written += 1
            log.info(
                "[%d/%d] issue #%d -> PR #%d (%d files)",
                n_written,
                args.target,
                issue["number"],
                pr["number"],
                len(files),
            )

    log.info("Wrote %d candidates to %s", n_written, out_path)
    log.info(
        "Next: open %s, set `curated_question`, `curated_answer_keywords`, "
        "and `expected_files` for ~50 high-quality ones, set `curated`=true, "
        "then save curated rows to %s.",
        out_path,
        BENCHMARK_PATH,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
