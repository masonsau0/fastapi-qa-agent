"""FastAPI server exposing the agent over HTTP.

Endpoints:
    GET  /health
    POST /ask           — synchronous, returns answer + trace

I deliberately did not add streaming. The agent loop calls multiple tools
internally and streaming tokens through that machinery is more code than it's
worth for a portfolio demo. If you want streaming, the right way is to
emit per-step events (tool_call, tool_result, answer_chunk) over SSE —
that's a worthwhile follow-up.

Security caveats (same as the LoRA project — read SECURITY.md):
  - No auth, no rate limiting. Bind to localhost only.
  - The agent uses your ANTHROPIC_API_KEY. Do not expose this server to the
    public internet without auth + rate limiting + a separate scoped key.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


class _State:
    agent = None


state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Lazy import so the module is importable without the heavy deps installed.
    use_mock = os.environ.get("USE_MOCK", "false").lower() == "true"
    if use_mock:
        log.info("USE_MOCK=true — agent will return canned responses.")

        class _Mock:
            def ask(self, q):
                from src.agent.loop import AgentResult, ToolCall

                return AgentResult(
                    answer=f"[MOCK] Q: {q}",
                    tool_calls=[ToolCall(name="search_code", args={"query": q}, result="...")],
                    iterations=1,
                    stop_reason="end_turn",
                )

        state.agent = _Mock()
    else:
        from src.agent.loop import CodebaseAgent

        state.agent = CodebaseAgent()
        log.info("Agent loaded.")
    yield
    state.agent = None


app = FastAPI(
    title="FastAPI codebase QA agent",
    version="0.1.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    tool_trace: list[str]
    iterations: int


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    if state.agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")
    # The agent is sync (Anthropic SDK call is blocking). For low concurrency
    # this is fine; under load we'd want to wrap in run_in_threadpool.
    result = state.agent.ask(req.question)
    return AskResponse(
        answer=result.answer,
        tool_trace=result.tool_trace,
        iterations=result.iterations,
    )
