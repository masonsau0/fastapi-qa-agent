#!/usr/bin/env bash
# Clone the target repo into data/fastapi/. Idempotent (skips if already there).

set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="data/fastapi"

if [ -d "$TARGET/.git" ]; then
    echo "Already cloned at $TARGET. To refresh, delete the directory and rerun."
    exit 0
fi

# Shallow clone — we only need the current state + recent history.
# depth=200 gives us enough history for git_log_for_file to be useful without
# pulling a 50MB+ history.
git clone --depth 200 https://github.com/fastapi/fastapi.git "$TARGET"
echo "Cloned to $TARGET"
