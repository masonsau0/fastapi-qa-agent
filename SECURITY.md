# Security notes

This project uses an Anthropic API key. If that key leaks to a public repo,
bots find it in minutes. Read this before pushing.

## Before pushing the repo public

1. **Set a monthly cap** on the Anthropic console.
   Settings → Billing → Usage limits. $15/month is plenty for this project —
   the full eval run costs $1–3.

2. **Use a project-scoped key.** Name it something like
   `fastapi-qa-agent-dev`. Easy to revoke if anything goes wrong, and you
   won't take down anything else.

3. **Verify `.env` is ignored.** Run `git check-ignore .env`. It should print
   `.env`. If it doesn't, stop and figure out why before you commit.

4. **Install the pre-commit hook.** `pip install pre-commit && pre-commit install`.
   Runs gitleaks on every commit and refuses anything that looks like a
   credential.

5. **Run a manual scan once before the first push.** `pre-commit run --all-files`.

## What's gitignored

`.env`, `*.key`, `*.pem`, the cloned target repo (`data/fastapi/`), all built
indexes (`data/index/`), and all results (`data/results/`). See `.gitignore`.

## What's committed

`.env.example` (template with no values), the curated benchmark
(`data/benchmark/questions.jsonl`) once you've curated it, and all source code.

## Deploying the server

The FastAPI server in `server/app.py` is for local use. It has no auth and no
rate limiting. If you put it on the internet, a single bot with a script can
drain your Anthropic credits via the `/ask` endpoint.

If you must deploy it publicly:
- Add an API key header check (~10 lines).
- Add rate limiting with `slowapi` (~10 lines).
- Use a separate scoped Anthropic key with a low cap (so the worst case is
  bounded).
- Better: don't deploy it. Record a Loom demo instead.

## If a key leaks

1. Revoke it on the Anthropic console (one click).
2. Generate a new one.
3. Update your local `.env`.
4. If the leak was in your last commit, you can usually just delete and
   re-init the repo.
5. Check the Anthropic usage dashboard for anything unfamiliar.
