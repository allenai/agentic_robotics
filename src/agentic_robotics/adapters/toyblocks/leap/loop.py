"""The experimentation-phase loop: the repo's loop plus the paper's status line, context wipes,
dollar ledger, and per-turn tool listing.

This class overrides the hook methods that `AgenticLoop.enter_loop` calls, so that:

- the tool list is fetched every turn, bypassing the client cache, so a docstring update or an
  allowlist change takes effect on the next turn;
- the per-turn status message reports tool calls left, context used against the wipe threshold,
  and dollars spent against the cap, with one-time warnings at 50, 90 and 95 %;
- once the last prompt reached the threshold, the agent gets one turn with the file tools only,
  then the history is replaced by the initial messages plus a note listing the workspace files;
  when the agent asked via `clear_context` at 90 % or more, the history is replaced after that
  turn with no extra turn (the rules as the developer's plan file states them);
- when the ledger passes its cap, the agent gets one turn with the file tools only, then the
  final-answer turn;
- the phase has two ends: the dollar cap, and `max_idle_turns` consecutive turns without a tool
  call (three by default), which forfeit the remaining budget. A tool-less turn before that is
  re-prompted with a note that warns of the exit (`confirm_episode_end`), because the base loop
  would otherwise exit on the first one and an agent could end the phase with most of its budget
  unspent; the idle exit exists because an agent that has declared itself done otherwise keeps
  answering re-prompts at a few cents each until the cap.

"Context used" is the previous model call's prompt size in tokens, which includes the system
prompt and the tool definitions; that is what the provider counts, so it is what the threshold
is measured against.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mcp.server.mcpserver.exceptions import ToolError

from agentic_robotics.adapters.toyblocks.leap import messages
from agentic_robotics.adapters.toyblocks.leap.context_clearing_agent import (
    MAX_CONTEXT_SIZE_TOKENS,
    MIN_CLEARABLE_CONTEXT_PERCENT,
    ContextClearingAgent,
)
from agentic_robotics.adapters.toyblocks.leap.declarations import register_declared_tool
from agentic_robotics.core.agentic_loop import AgenticLoop, ToolCall
from agentic_robotics.core.foundation_models import (
    History,
    OpenAIModelClient,
    Transcript,
    create_openai_message,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from mcp import Client as MCPClient
    from mcp.server.mcpserver import MCPServer
    from mcp_types import Tool as MCPTool
    from openai.types.responses.response_input_item_param import FunctionCallOutput

    from agentic_robotics.adapters.toyblocks.leap.composed_server import ComposedServer
    from agentic_robotics.adapters.toyblocks.leap.cost import CostLedger
    from agentic_robotics.adapters.toyblocks.leap.workspace import Workspace
    from agentic_robotics.core.foundation_models.usage_tracking import TokenUsageSnapshot

logger = logging.getLogger("agentic_robotics.examples.leap.loop")

KEEP_ALIVE_INTERVAL_S = 240.0
"""Seconds a tool call may run before the previous model request is re-sent to keep its cache.

Anthropic's prompt cache expires five minutes after its last use. The harness pilot of
2026-10-05 lost $1.00 of $5.34 to four experiments that ran six to ten minutes, each followed by
a full context rewrite. A resend of the previous request with a minimal output cap reads the
cached prefix, which refreshes it, for about the cache-read price (`cache_ttl_probe.py`). Four
minutes leaves a margin for a slow request. The resend is the model client's `keep_alive`; a
client without one is left alone.
"""


@dataclass(frozen=True)
class ContextPolicy:
    """The context-wipe rules of the developer's plan file, identical across conditions."""

    limit_tokens: int = MAX_CONTEXT_SIZE_TOKENS
    warn_fractions: tuple[float, ...] = (0.5, 0.9, 0.95)
    self_wipe_fraction: float = MIN_CLEARABLE_CONTEXT_PERCENT / 100


@dataclass
class ContextControl:
    """Shared by the loop, which sets `fraction_used`, and clear_context, which sets the request."""

    self_wipe_fraction: float
    fraction_used: float = 0.0
    wipe_requested: bool = False


def register_context_tool(server: MCPServer, control: ContextControl) -> str:
    """Register clear_context on `server` as `ContextClearingAgent` declares it.

    :return: The tool's name, for allowlists
    """

    async def clear_context() -> None:
        """Flag the wipe for the end of this turn; refused below the threshold (see the ABC)."""
        percent = round(100 * control.fraction_used)
        if control.fraction_used < control.self_wipe_fraction:
            raise ToolError(
                messages.clear_context_refused(percent, round(100 * control.self_wipe_fraction))
            )
        control.wipe_requested = True

    return register_declared_tool(server, ContextClearingAgent, "clear_context", clear_context)


class LeapAgenticLoop(AgenticLoop):
    """The base loop under the experimentation phase's budget, context and tool rules.

    Hooks on the base loop: the status line is `status_text`,
    the dollar cap and the context wipes are `after_turn`, the keep-alive wraps
    `process_tool_calls`, and a tool-less turn is re-prompted or, after `max_idle_turns` of them
    in a row, ends the phase, in `confirm_episode_end`. What the hooks cannot reach is the base's
    own call counter, so the one to three calls of a file-only turn at the context limit are no
    longer charged against `max_tool_calls`; the dollar cap and the idle exit are the intended
    stops and the counter a safety ceiling, so this does not change when a phase ends.
    """

    def __init__(
        self,
        composed: ComposedServer,
        mcp_client: MCPClient,
        model_client: OpenAIModelClient,
        transcript: Transcript,
        *,
        ledger: CostLedger,
        workspace: Workspace,
        file_tool_names: list[str],
        control: ContextControl,
        context_policy: ContextPolicy | None = None,
        metadata_callback: Callable[[], dict[str, Any]] | None = None,
        max_idle_turns: int = 3,
    ) -> None:
        """Initialize the loop.

        :param composed: The server behind `mcp_client`, for the final-turn allowlist
        :param mcp_client: Client connected to `composed.server`
        :param model_client: The model, whose usage history feeds the ledger
        :param transcript: Transcript of the conversation recorded to file
        :param ledger: Dollar ledger this loop spends against
        :param workspace: The agent's workspace, listed in the wipe note
        :param file_tool_names: The tools left available on the final turn after the cap
        :param control: The context control shared with clear_context
        :param context_policy: Threshold and warning fractions (defaults to the decided ones)
        :param metadata_callback: Optional callback to query domain-specific tool call metadata
        :param max_idle_turns: Consecutive tool-less turns after which the phase ends with the
            remaining budget forfeited; a very large value re-prompts until the cap instead
        """
        super().__init__(mcp_client, model_client, transcript, metadata_callback)
        self._composed = composed
        self._ledger = ledger
        self._workspace = workspace
        self._file_tool_names = list(file_tool_names)
        self._policy = context_policy or ContextPolicy()
        self.control = control
        self._max_idle_turns = max_idle_turns

        self._usage_seen = 0
        self._initial_history: History = []
        self._task = ""
        self._prompt_tokens = 0
        self._warned: set[float] = set()
        self._done = False
        self.wipe_count = 0
        self.tool_calls_made = 0
        self.reprompts = 0
        """Turns without a tool call that were re-prompted rather than ending the phase."""
        self.idle_turns = 0
        """Consecutive tool-less turns so far; reset by any turn that ran a tool."""
        self.ended_idle = False
        """Whether the phase ended on `max_idle_turns` rather than on the dollar cap."""
        self.keep_alives = 0
        self.keep_alive_cost_usd = 0.0

    async def list_mcp_tools(self) -> list[MCPTool]:
        """List the tools as they are now, not as the client cached them."""
        result = await self._mcp_client.list_tools(cache_mode="bypass")
        return result.tools

    def _record_usage(self) -> TokenUsageSnapshot | None:
        """Add any new usage snapshots to the ledger; return the latest one."""
        snapshots = self._model_client.usage_history
        new = snapshots[self._usage_seen :]
        self._usage_seen = len(snapshots)
        for snapshot in new:
            self._ledger.add(snapshot.cost_usd)  # Raises on an unpriced call
        return new[-1] if new else None

    def _status_text(self, tool_calls_left: int, prompt_tokens: int, warned: set[float]) -> str:
        """The per-turn status message, with any warning threshold crossed for the first time."""
        limit = self._policy.limit_tokens
        lines = [messages.status_line(tool_calls_left, prompt_tokens, limit, self._ledger.status())]
        fraction = prompt_tokens / limit if limit else 0.0
        for threshold in self._policy.warn_fractions:
            if fraction >= threshold and threshold not in warned:
                warned.add(threshold)
                lines.append(
                    messages.context_warning(
                        threshold,
                        limit,
                        may_clear_early=threshold >= self._policy.self_wipe_fraction,
                    )
                )
        return "\n".join(lines)

    def status_text(self, tool_calls_left: int) -> str:
        """The base hook: the status line against the previous prompt's size."""
        return self._status_text(tool_calls_left, self._prompt_tokens, self._warned)

    def should_continue(self) -> bool:
        """The loop runs until `after_turn` has given the post-cap file turn."""
        return not self._done

    async def take_turn(self, task: str, status_text: str, tools: list[MCPTool]) -> History:
        """The base turn (which records the call's usage on the status message), then the cost
        goes to the ledger and the prompt size becomes what the context rules measure.

        "Context used" is the previous model call's prompt size in tokens, which includes the
        system prompt and the tool definitions; that is what the provider counts, so it is what
        the threshold is measured against.
        """
        response = await super().take_turn(task, status_text, tools)
        usage = self._record_usage()
        if usage is not None:
            self._prompt_tokens = usage.input_tokens
        return response

    async def process_tool_calls(self, tool_calls: list[ToolCall]) -> list[FunctionCallOutput]:
        """Run the calls, re-sending the previous model request every `KEEP_ALIVE_INTERVAL_S`
        they are still running so its cache entry does not expire.

        The keep-alive's cost goes to the ledger like any model call; its output is discarded. A
        failed keep-alive is logged and the tool calls run on, since the worst case is the
        rewrite the keep-alive was meant to avoid.
        """
        keep_alive = getattr(self._model_client, "keep_alive", None)
        work = asyncio.ensure_future(super().process_tool_calls(tool_calls))
        if keep_alive is None:
            return await work
        try:
            while True:
                done, _ = await asyncio.wait({work}, timeout=KEEP_ALIVE_INTERVAL_S)
                if done:
                    return work.result()
                try:
                    await keep_alive()
                except Exception:
                    logger.exception("Cache keep-alive failed; the tool calls continue.")
                    continue
                usage = self._record_usage()
                cost = (usage.cost_usd or 0.0) if usage else 0.0
                self.keep_alives += 1
                self.keep_alive_cost_usd += cost
                logger.info(
                    "Cache keep-alive (%d so far): read %s tokens for $%.4f while tool calls run.",
                    self.keep_alives,
                    f"{usage.tokens_read:,}" if usage else "?",
                    cost,
                )
        finally:
            if not work.done():
                work.cancel()

    async def run_tool_calls(self, calls: list[ToolCall]) -> list[FunctionCallOutput]:
        """The base run, counted."""
        outputs = await super().run_tool_calls(calls)
        self.tool_calls_made += len(outputs)
        if outputs:
            self.idle_turns = 0
        return outputs

    def _wipe(self, reason: str) -> None:
        """Replace the history with the initial messages and a note listing the workspace."""
        self.wipe_count += 1
        first, *rest = self._initial_history
        self._transcript.start_new_episode(first)
        self._transcript.extend(rest)
        note = messages.context_cleared_note(reason, self._workspace.listing())
        self._transcript.append(create_openai_message(note, role="system"))
        self.control.wipe_requested = False
        self.control.fraction_used = 0.0
        self._prompt_tokens = 0
        self._warned.clear()
        logger.info(f"Context wiped ({self.wipe_count} so far): {reason}.")

    async def confirm_episode_end(self) -> bool:
        """A tool-less turn is re-prompted, until `max_idle_turns` in a row end the phase.

        While budget remains, the first `max_idle_turns - 1` consecutive turns without a tool call
        each get a system note stating the budget and the coming exit, and the next turn's status
        line follows as usual; the turn that reaches `max_idle_turns` ends the phase through the
        base loop's final-answer turn, with no file-tools turn first and the remaining budget
        forfeited. Any turn that runs a tool resets the count. Over the cap, `after_turn` handles
        the end instead. Identical in the naive and harness conditions, which both run this loop.
        """
        if self._ledger.exceeded:
            return False  # `after_turn` gives the file-tools turn, then the final answer
        self.idle_turns += 1
        if self.idle_turns >= self._max_idle_turns:
            self.ended_idle = True
            logger.info(
                f"Phase ended on {self.idle_turns} consecutive tool-less turns "
                f"(max {self._max_idle_turns}); budget spent: {self._ledger.status()}."
            )
            return True
        self.reprompts += 1
        note = messages.no_tool_call_note(self._ledger.status(), self._max_idle_turns)
        self._transcript.append(create_openai_message(note, role="system"))
        logger.info(f"Tool-less turn re-prompted ({self.reprompts} so far); budget remains.")
        return False

    async def after_turn(self) -> None:
        """The context-wipe rules, after every turn: over the cap, one file-tools turn and the
        phase ends; at the context limit, one file-tools turn and a wipe; a wipe the agent asked
        for at 90 % or more happens now with no extra turn."""
        if self._ledger.exceeded:
            await self._final_file_turn(self._task)
            self._done = True
            return
        limit = self._policy.limit_tokens
        self.control.fraction_used = self._prompt_tokens / limit if limit else 0.0
        if self._prompt_tokens >= limit:
            await self._limit_file_turn(self._task)
            self._wipe(messages.context_limit_reason(limit))
        elif self.control.wipe_requested:
            self._wipe(messages.CLEAR_CONTEXT_REASON)

    async def _file_only_turn(self, task: str, note: str) -> int:
        """One turn with the file tools alone, restoring the allowlist afterwards.

        :return: How many tool calls the turn made
        """
        previous = self._composed.allowlist
        self._composed.set_allowlist(self._file_tool_names)
        try:
            response = await self.take_turn(task, note, await self.list_mcp_tools())
            return len(await self.run_tool_calls(self._filter_tool_calls(response)))
        finally:
            self._composed.set_allowlist(previous)

    async def _final_file_turn(self, task: str) -> None:
        """The one turn the agent gets after the cap: file tools only."""
        await self._file_only_turn(task, messages.final_file_turn_note(self._ledger.status()))

    async def _limit_file_turn(self, task: str) -> int:
        """The one file-tools turn the agent gets at the context limit, before the forced clear."""
        note = messages.context_limit_file_turn_note(
            self._policy.limit_tokens, self._workspace.listing()
        )
        calls = await self._file_only_turn(task, note)
        logger.info(f"Context limit reached; file-only turn made {calls} tool call(s).")
        return calls

    def get_final_update(self, task: str) -> str:
        """The final-answer request, from `messages.py` rather than the core loop."""
        return messages.final_update(task)

    async def enter_loop(
        self,
        task: str,
        history: History,
        max_tool_calls: int,
        opening_note: str | None = None,
    ) -> Transcript:
        """Run the experimentation phase until the budget, the call ceiling, or the agent stops.

        :param task: Fixed task instructions given to the agent each turn
        :param history: Initial conversation history, restored after every wipe
        :param max_tool_calls: Ceiling on tool calls; the dollar cap and the idle exit are the
            intended stops
        :param opening_note: A system message after the initial history on the first context
            only (a resumed phase's note); wipes restore `history` without it
        :return: Full transcript of the finished conversation(s)
        """
        self._task = task
        self._initial_history = list(history)
        self._done = False
        opening = (
            [*history, create_openai_message(opening_note, role="system")]
            if opening_note
            else history
        )
        if self._ledger.exceeded:
            # A resumed phase already over its cap: the file-tools turn, then the final answer.
            self._transcript.extend(opening)
            await self._final_file_turn(task)
            self._done = True
            await self.take_final_turn(task, await self.list_mcp_tools())
            return self._transcript
        return await super().enter_loop(task, opening, max_tool_calls)
