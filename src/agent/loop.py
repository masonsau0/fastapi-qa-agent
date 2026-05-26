"""The agent loop.

Standard Anthropic tool-use pattern:
  1. Send user question + tool schemas.
  2. Model returns either a text response OR one or more tool_use blocks.
  3. If tool_use: execute each tool, send back tool_result blocks, loop.
  4. If pure text and stop_reason is end_turn: done.
  5. Cap iterations so a confused agent can't loop forever.

We capture the full trace (which tools were called, with what args, what they
returned) because the evaluation depends on it. The trace is also what makes
the agent debuggable — when an answer is wrong, you can read what it tried.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from anthropic import Anthropic

from src.config import MODEL, require_env
from src.tools.impl import dispatch
from src.tools.schemas import TOOL_SCHEMAS

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are a careful assistant that answers questions about the FastAPI codebase.

You have tools that let you search code, search docs, read specific file ranges,
and inspect git history. Use them — don't guess from memory. The codebase
changes over time; only your tools see the current state.

Guidelines:
  - For "how is X implemented" questions, lead with search_code.
  - For "how do I use X" questions, lead with search_docs.
  - If a search result looks promising but is truncated, follow up with
    read_file_lines for more context.
  - Cite the file path and line numbers from your tool results in your final
    answer (e.g. "see fastapi/routing.py:142-180"). Don't fabricate citations.
  - Keep answers concise. A paragraph or two plus citations is usually right.
  - If you genuinely cannot find an answer after 2-3 search attempts, say so
    rather than inventing one.
  - You typically need only 2-4 tool calls. After 5 calls, write your answer
    based on what you have rather than searching further.
  - Your final answer must start directly with the substantive content (a
    heading, a sentence stating the answer, or a code example). NEVER begin
    with transitional phrases like "Now I have", "Perfect", "Let me", "Based on
    my research", or "Here is". These are forbidden and will be considered an
    error.
    CRITICAL: Your final answer MUST begin with a Markdown heading (## or #) or
    a direct sentence stating the answer. Words that MUST NEVER appear in the
    first 10 words of your answer: "Now", "Perfect", "Excellent", "Let me",
    "Based on", "Here is", "I have", "I now", "Great", "I'll".
"""


@dataclass
class ToolCall:
    """One tool call made by the agent during a question. Captured for eval."""

    name: str
    args: dict[str, Any]
    result: str  # truncated for logging; full result was fed to the model


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    iterations: int = 0
    stop_reason: str = ""

    # Convenience for eval: just the tool names, in order.
    @property
    def tool_trace(self) -> list[str]:
        return [tc.name for tc in self.tool_calls]


class CodebaseAgent:
    """One agent instance, configured for one Anthropic client + model.

    Stateless across calls to `.ask()` — you can use the same instance to
    run a whole benchmark in parallel.
    """

    def __init__(
        self,
        model: str = MODEL.agent_model,
        max_iterations: int = MODEL.max_iterations,
        max_tokens: int = MODEL.max_tokens,
    ):
        self.client = Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens

    def ask(self, question: str) -> AgentResult:
        """Run the agent on one question. Returns the final answer + trace."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": question},
        ]
        result = AgentResult(answer="")

        for iteration in range(self.max_iterations):
            result.iterations = iteration + 1

            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
            result.stop_reason = response.stop_reason or ""

            # Collect any text blocks (the model may interleave text and tool_use).
            text_parts = [b.text for b in response.content if b.type == "text"]
            tool_uses = [b for b in response.content if b.type == "tool_use"]

            # If no tool calls and the model said it's done, we're done.
            if not tool_uses and response.stop_reason == "end_turn":
                result.answer = "\n".join(text_parts).strip()
                return result

            # If the model wants to call tools, run them and feed results back.
            if tool_uses:
                # Add the assistant turn (with the tool_use blocks) to history.
                messages.append({"role": "assistant", "content": response.content})

                # Execute each tool and build the user-turn reply.
                tool_result_blocks = []
                for tu in tool_uses:
                    tool_output = dispatch(tu.name, dict(tu.input))
                    # Truncate tool output that gets fed to the model — very long
                    # results blow context for no benefit. 8K chars is generous.
                    truncated = tool_output
                    if len(truncated) > 8000:
                        truncated = truncated[:8000] + "\n... [truncated]"
                    tool_result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": truncated,
                        }
                    )
                    # Capture in trace — store a short preview, not the full text.
                    preview = tool_output[:300]
                    result.tool_calls.append(
                        ToolCall(name=tu.name, args=dict(tu.input), result=preview)
                    )

                messages.append({"role": "user", "content": tool_result_blocks})
                continue

            # If we got here, model stopped for another reason (max_tokens, etc).
            # Take whatever text was generated and bail out.
            result.answer = "\n".join(text_parts).strip() or "[no answer]"
            return result

        # Hit the iteration cap.
        result.answer = (
            f"[agent hit iteration cap of {self.max_iterations} without producing a final answer]"
        )
        log.warning("Agent exceeded max_iterations on question: %s", question[:80])
        return result
