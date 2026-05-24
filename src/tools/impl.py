"""Tool implementations.

Each function corresponds to one schema in `schemas.py`. They return strings
or string-serializable structures because that's what gets fed back to the
agent as tool_result content.

Defensive coding throughout: an exception in a tool kills the whole agent
turn, so we catch and return a useful error string instead.

Security note: read_file_lines, git_log_for_file, and find_pr_for_commit all
take user-controlled input. The path-taking ones explicitly resolve against
the repo root and reject paths that escape it. find_pr_for_commit hits the
GitHub API — we validate the SHA against a regex so we can't be tricked into
hitting a weird URL.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from typing import Any

import httpx

from src.config import INDEX_DIR, TARGET_REPO_DIR
from src.retrieval.hybrid import HybridRetriever

log = logging.getLogger(__name__)


# ---------- Shared state (lazily loaded) ----------

_retrievers: dict[str, HybridRetriever] = {}


def _get_retriever(kind: str) -> HybridRetriever:
    """Load (and cache) one of the two retrievers. 'code' or 'docs'."""
    if kind not in _retrievers:
        path = INDEX_DIR / kind
        if not path.exists():
            raise RuntimeError(
                f"Index '{kind}' not found at {path}. Run `python -m src.retrieval.build_indexes` first."
            )
        _retrievers[kind] = HybridRetriever.load(path)
    return _retrievers[kind]


# ---------- Tool: search_code ----------


def search_code(query: str, k: int = 5) -> str:
    return _do_search("code", query, k)


# ---------- Tool: search_docs ----------


def search_docs(query: str, k: int = 5) -> str:
    return _do_search("docs", query, k)


def _do_search(kind: str, query: str, k: int) -> str:
    k = max(1, min(int(k), 10))
    try:
        results = _get_retriever(kind).search(query, k=k)
    except Exception as e:  # noqa: BLE001
        log.exception("Retrieval failed")
        return f"Error: retrieval failed ({type(e).__name__}: {e})"

    if not results:
        return "No results."

    parts = []
    for i, (chunk, score) in enumerate(results, start=1):
        parts.append(
            f"--- Result {i} (score={score:.3f}) ---\n"
            f"File: {chunk.citation}\n"
            f"```\n{chunk.text}\n```"
        )
    return "\n\n".join(parts)


# ---------- Tool: read_file_lines ----------

# Lines we'll read in one go. The schema also caps end_line - start_line at 200,
# but enforce here too in case the schema validation is bypassed.
_MAX_READ_LINES = 200


def read_file_lines(path: str, start_line: int, end_line: int) -> str:
    # Resolve and validate the path stays inside the repo.
    repo_root = TARGET_REPO_DIR.resolve()
    try:
        target = (repo_root / path).resolve()
        target.relative_to(repo_root)  # raises if outside
    except (ValueError, OSError):
        return f"Error: invalid path {path!r} (must be inside the target repo)."

    if not target.is_file():
        return f"Error: {path} is not a file in the repo."

    start_line = max(1, int(start_line))
    end_line = max(start_line, int(end_line))
    end_line = min(end_line, start_line + _MAX_READ_LINES)

    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return f"Error: {path} is not UTF-8 text."

    # Convert to 0-indexed slice.
    selected = lines[start_line - 1 : end_line]
    if not selected:
        return f"No lines in range {start_line}-{end_line} (file has {len(lines)} lines)."

    numbered = [f"{i:>5}: {line}" for i, line in enumerate(selected, start=start_line)]
    return f"File: {path} (lines {start_line}-{start_line + len(selected) - 1})\n" + "\n".join(
        numbered
    )


# ---------- Tool: git_log_for_file ----------

_MAX_LOG_COMMITS = 30


def git_log_for_file(path: str, limit: int = 10) -> str:
    repo_root = TARGET_REPO_DIR.resolve()
    try:
        target = (repo_root / path).resolve()
        target.relative_to(repo_root)
    except (ValueError, OSError):
        return f"Error: invalid path {path!r}."

    limit = max(1, min(int(limit), _MAX_LOG_COMMITS))

    # `--` separates rev/path arguments from any ambiguous-looking path.
    # `--no-pager` keeps `less` from spawning. Format: hash | author | date | subject.
    fmt = "%h | %an | %ad | %s"
    cmd = [
        "git",
        "-C",
        str(repo_root),
        "--no-pager",
        "log",
        f"-n{limit}",
        "--date=short",
        f"--pretty=format:{fmt}",
        "--",
        path,
    ]
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "Error: git log timed out."
    if out.returncode != 0:
        return f"Error: git log failed: {out.stderr.strip()}"
    if not out.stdout.strip():
        return f"No commits found for {path}."
    return out.stdout.strip()


# ---------- Tool: find_pr_for_commit ----------

# GitHub-flavored SHA: hex, 7 to 40 chars.
_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")


def find_pr_for_commit(commit_hash: str) -> str:
    if not _SHA_RE.match(commit_hash or ""):
        return "Error: invalid commit hash."

    # Extract owner/repo from the configured target URL, defensively.
    # We assume the URL looks like https://github.com/<owner>/<repo>.git.
    from src.config import TARGET_REPO_URL

    match = re.match(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", TARGET_REPO_URL)
    if not match:
        return f"Error: cannot parse owner/repo from {TARGET_REPO_URL}."
    owner, repo = match.group(1), match.group(2)

    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_hash}/pulls"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "fastapi-qa-agent",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = httpx.get(url, headers=headers, timeout=10.0)
    except httpx.HTTPError as e:
        return f"Error: GitHub API request failed: {e}"

    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        return "Error: GitHub API rate limit hit. Set GITHUB_TOKEN in .env to raise the limit."
    if resp.status_code == 404:
        return f"No PR found for commit {commit_hash}."
    if resp.status_code >= 400:
        return f"Error: GitHub API returned {resp.status_code}."

    prs: list[dict[str, Any]] = resp.json()
    if not prs:
        return f"No PR found for commit {commit_hash}."

    pr = prs[0]
    body = (pr.get("body") or "").strip()
    if len(body) > 500:
        body = body[:500] + "..."
    return f"PR #{pr['number']}: {pr['title']}\nURL: {pr['html_url']}\nBody excerpt:\n{body}"


# ---------- Dispatcher ----------

TOOL_FUNCS = {
    "search_code": search_code,
    "search_docs": search_docs,
    "read_file_lines": read_file_lines,
    "git_log_for_file": git_log_for_file,
    "find_pr_for_commit": find_pr_for_commit,
}


def dispatch(name: str, args: dict[str, Any]) -> str:
    """Call a tool by name with kwargs. Returns the tool's string output."""
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return f"Error: unknown tool {name!r}."
    try:
        return fn(**args)
    except TypeError as e:
        return f"Error: bad arguments to {name}: {e}"
    except Exception as e:  # noqa: BLE001
        log.exception("Tool %s raised", name)
        return f"Error: {name} failed ({type(e).__name__}: {e})"
