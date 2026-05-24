"""Tool schemas in the Anthropic native tool-use format.

Each tool: name, description (the agent reads this to decide when to call),
and input_schema (JSON schema for the args). Descriptions matter a lot — they
ARE the prompt for tool selection.

Lessons I learned while iterating on these:
  - Be explicit about when to use a tool, not just what it does.
  - If two tools could plausibly answer a query, the agent will pick wrong unless
    the descriptions explicitly disambiguate (search_code vs search_docs).
  - Mention the return format briefly so the agent knows how to use the output.
"""

TOOL_SCHEMAS = [
    {
        "name": "search_code",
        "description": (
            "Search the project's source code (Python files) by query. Use this "
            "when the question is about how something is implemented, where a "
            "function lives, what a class does, or to find specific patterns in "
            "the code. Returns the top-k matching code chunks with file paths "
            "and line numbers. Prefer this over search_docs for 'how does X "
            "work internally' questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language or code-flavored query. Function names, error messages, and identifiers work well.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return. Default 5, max 10.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_docs",
        "description": (
            "Search the project's documentation (Markdown and reST files). "
            "Use this for 'how do I use X', conceptual explanations, tutorials, "
            "and configuration references. Prefer this over search_code when the "
            "question is about API usage from the perspective of a library "
            "consumer rather than internal implementation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_file_lines",
        "description": (
            "Read a specific line range from a file in the repo. Use this as a "
            "follow-up after search_code or search_docs returns a promising "
            "result but you need more context around it (the surrounding lines, "
            "the rest of a function, etc). Returns the requested lines verbatim "
            "with line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Repo-relative path, e.g. 'fastapi/routing.py'",
                },
                "start_line": {"type": "integer", "description": "1-indexed, inclusive."},
                "end_line": {
                    "type": "integer",
                    "description": "1-indexed, inclusive. Capped at start_line + 200.",
                },
            },
            "required": ["path", "start_line", "end_line"],
        },
    },
    {
        "name": "git_log_for_file",
        "description": (
            "Get the recent commit history for a specific file. Use this when "
            "the question is about why a piece of code is the way it is, or "
            "when something changed. Returns the last N commits touching the "
            "file (hash, author, date, subject)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repo-relative path."},
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Max commits to return. Default 10, max 30.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "find_pr_for_commit",
        "description": (
            "Given a commit hash, find the GitHub Pull Request that introduced "
            "it. Useful as a follow-up to git_log_for_file when you want the "
            "discussion or rationale behind a change. Returns the PR number, "
            "title, body excerpt, and URL — or null if no PR found."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "commit_hash": {
                    "type": "string",
                    "description": "Full or short Git SHA.",
                },
            },
            "required": ["commit_hash"],
        },
    },
]
