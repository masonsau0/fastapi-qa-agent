"""Mine candidate benchmark items from merged PRs in the target repo.

Original plan was to start from closed issues and find their resolving PRs,
but FastAPI's contributors don't reliably use "Closes #N" syntax, so the
issue→PR linkage was hit-rate ~5%. Going PR-first instead gives ~95% — every
merged PR already has a title, body, and file list.

What you (the human) do after this runs:
  Open data/benchmark/candidates.jsonl, read through entries, and for each
  one you keep:
    - rewrite `curated_question` as a natural standalone question
    - fill in `curated_answer_keywords` (3-6 short phrases that any correct
      answer must contain)
    - trim `expected_files` to just the file(s) that contain THE answer
      (the miner pre-fills it from all touched files, often you'll narrow)
    - set "curated": true

  When you have ~50 curated rows, save them to questions.jsonl.

Usage:
    python -m src.eval.mine_benchmark --target 200
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import httpx

from src.config import BENCHMARK_PATH, TARGET_REPO_URL, require_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def _parse_repo_url(url: str) -> tuple[str, str]:
    m = re.match(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if not m:
        raise ValueError(f"Cannot parse GitHub URL: {url}")
    return m.group(1), m.group(2)


def _headers() -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "User-Agent": "fastapi-qa-agent-miner",
        "X-GitHub-Api-Version": "2022-11-28",
        "Authorization": f"Bearer {require_env('GITHUB_TOKEN')}",
    }


def _fetch_merged_prs(owner: str, repo: str, max_pages: int = 30) -> list[dict]:
    """Pull recently-merged PRs. The /pulls endpoint with state=closed returns
    closed PRs (some merged, some not), sorted by update time."""
    all_prs: list[dict] = []
    for page in range(1, max_pages + 1):
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        params = {
            "state": "closed",
            "per_page": 100,
            "page": page,
            "sort": "updated",
            "direction": "desc",
        }
        resp = httpx.get(url, headers=_headers(), params=params, timeout=30.0)
        if resp.status_code != 200:
            log.error("GitHub API returned %s: %s", resp.status_code, resp.text[:200])
            break
        page_prs = resp.json()
        # Keep only PRs that were actually merged (not just closed-unmerged).
        merged = [p for p in page_prs if p.get("merged_at")]
        all_prs.extend(merged)
        log.info("Page %d: %d merged PRs (of %d closed)", page, len(merged), len(page_prs))
        if len(page_prs) < 100:
            break
        time.sleep(0.3)
    return all_prs


def _pr_files(pr_number: int, owner: str, repo: str) -> list[str]:
    """List file paths touched by a PR."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
    resp = httpx.get(url, headers=_headers(), timeout=15.0, params={"per_page": 100})
    if resp.status_code != 200:
        return []
    return [f["filename"] for f in resp.json()]


def _is_useful_pr(pr: dict, files: list[str]) -> bool:
    """Heuristic filter. We want PRs that are likely to map to a real question."""
    title = (pr.get("title") or "").lower()
    body = (pr.get("body") or "").strip()

    # Skip empty bodies (nothing for a curator to read).
    if not body:
        return False
    # Skip dependabot / ci / docs-only / release / version-bump PRs — these
    # don't make for good "how does X work" questions.
    skip_prefixes = (
        "bump ",
        "release ",
        "update changelog",
        "[pre-commit",
        "ci:",
        "build:",
        "chore:",
        "refactor:",
        "test:",
    )
    if any(title.startswith(p) for p in skip_prefixes):
        return False
    if "dependabot" in (pr.get("user", {}).get("login") or "").lower():
        return False
    # Skip PRs that only touch docs/translations (these are about the project,
    # not about the code).
    if files and all(f.startswith(("docs/", ".github/")) for f in files):
        return False
    # Skip PRs with no Python files touched (mostly translation PRs).
    return any(f.endswith(".py") for f in files)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--target", type=int, default=200, help="How many candidates to write.")
    p.add_argument("--output", default=str(BENCHMARK_PATH.parent / "candidates.jsonl"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    owner, repo = _parse_repo_url(TARGET_REPO_URL)
    log.info("Mining merged PRs from %s/%s", owner, repo)

    prs = _fetch_merged_prs(owner, repo, max_pages=30)
    log.info("Got %d merged PRs to inspect", len(prs))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0

    with out_path.open("w", encoding="utf-8") as f:
        for pr in prs:
            if n_written >= args.target:
                break

            files = _pr_files(pr["number"], owner, repo)
            time.sleep(0.2)  # be polite to the API

            if not _is_useful_pr(pr, files):
                continue

            candidate = {
                "pr_number": pr["number"],
                "pr_url": pr["html_url"],
                "pr_title": pr["title"],
                "pr_body": (pr.get("body") or "")[:2000],
                "pr_files": files,
                "merge_commit_sha": pr.get("merge_commit_sha"),
                # Fields YOU fill in during curation:
                "curated_question": "",
                "curated_answer_keywords": [],
                "expected_files": [f for f in files if f.endswith(".py")],
                "curated": False,
            }
            f.write(json.dumps(candidate) + "\n")
            n_written += 1
            log.info(
                "[%d/%d] PR #%d: %s (%d files)",
                n_written,
                args.target,
                pr["number"],
                pr["title"][:60],
                len(files),
            )

    log.info("Wrote %d candidates to %s", n_written, out_path)
    log.info(
        "Next: open %s. For each one you keep, set `curated_question`, "
        "`curated_answer_keywords`, trim `expected_files`, set curated=true. "
        "Save the curated rows to %s.",
        out_path,
        BENCHMARK_PATH,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
