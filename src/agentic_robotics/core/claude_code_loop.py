"""An agentic loop hosted by Claude Code through the Claude Agent SDK, as an alternative to
`AgenticLoop`, which hosts the model↔tool loop itself over the OpenAI Responses API.

Here the loop runs inside the Claude Code CLI that the SDK bundles: one `query` per episode, the
task as the whole system prompt, every built-in tool removed, and the MCP tools reached over
streamable HTTP. What this module adds around that loop is what the repo's own loop records and
enforces, through the SDK's hooks and message stream:

- `PreToolUse` asks `deny_reason`, so a subclass can refuse a call (the evaluation cutoff);
- `PostToolUse` calls `after_tool_call` (the per-call state snapshot), lets `edit_tool_output`
  replace what the model sees of a result (the batched-image rule), and appends `status_text`
  to the result as additional context (the per-turn status line, delivered per call here);
- `UserPromptSubmit` appends `status_text` to the prompt, the first status line;
- every assistant message, tool call and tool result is written to the repo's `Transcript` in
  its OpenAI-Responses item shapes, so the readers of `transcript.jsonl` (`tool_call_counts`,
  `tool_call_seconds`, `final_assistant_message`) work unchanged;
- per-step token usage becomes `TokenUsageSnapshot`s, deduplicated by message id as the SDK's
  cost-tracking guide says (parallel tool calls repeat one message's usage), with `cost_usd`
  left None: the SDK prices only the whole run, on the `ResultMessage` kept in `result`, and
  that price is a client-side estimate.

Tool calls are tracked per assistant message (`ToolCallRecord.message_id`, `batch_of`), which
is what lets a subclass apply a rule at a turn boundary rather than per call: the stream
delivers the assistant message before the CLI runs its tools, so by the time a hook fires the
record exists; the hook waits briefly for it rather than assuming the order.

The subprocess inherits this process's environment. Before it starts, `ANTHROPIC_API_KEY`,
`ANTHROPIC_AUTH_TOKEN` and the OpenRouter key are removed from it, since an API key outranks the
OAuth token the developer's seat uses, and the credential is never read here beyond checking
that one is present. Bare mode (`CLAUDE_CODE_SIMPLE`) is never set: it ignores the OAuth token.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from agentic_robotics.core.foundation_models import Transcript, create_openai_message
from agentic_robotics.core.foundation_models.usage_tracking import TokenUsageSnapshot

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from claude_agent_sdk._internal.transport import Transport
    from claude_agent_sdk.types import EffortLevel, HookContext

logger = logging.getLogger(__name__)

CREDENTIAL_VARIABLES = ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY")
"""One of these must be set for the CLI to authenticate; only presence is checked."""

VARIABLES_HIDDEN_FROM_CLI = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENROUTER_API_KEY")
"""Removed from the subprocess environment so the OAuth token is the credential that is used."""

BUILTIN_TOOLS = (
    "Agent",
    "AskUserQuestion",
    "Bash",
    "BashOutput",
    "Edit",
    "ExitPlanMode",
    "Glob",
    "Grep",
    "KillShell",
    "ListMcpResourcesTool",
    "MultiEdit",
    "NotebookEdit",
    "NotebookRead",
    "Read",
    "ReadMcpResourceTool",
    "Skill",
    "SlashCommand",
    "Task",
    "TaskCreate",
    "TaskGet",
    "TaskList",
    "TaskOutput",
    "TaskStop",
    "TaskUpdate",
    "TodoWrite",
    "ToolSearch",
    "WebFetch",
    "WebSearch",
    "Write",
)
"""Claude Code's built-in tools, denied by name on top of `tools=[]`, which removes them all.
Which tools remained is reported by the init message and logged at the start of a run."""

CLI_ENVIRONMENT = {
    "DISABLE_COMPACT": "1",
    "ENABLE_TOOL_SEARCH": "false",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
}
"""No compaction (the context is never rewritten, as in the repo's loop), every MCP tool loaded
up front rather than deferred behind tool search, and no auto-update or telemetry traffic."""

HOOK_WAIT_S = 10.0
"""How long a hook waits for the assistant message that issued its tool call to be consumed."""

ToolCallOutcome = Literal["Tool Success", "Tool Failure", "Denied"]


@dataclass(frozen=True)
class ClaudeCodeConfig:
    """How the Claude Code CLI is launched for one episode."""

    model: str
    """The full model id, as the Messages API names it (`claude-opus-5-5`)."""

    mcp_server_key: str
    """The key the MCP server is registered under; tools are named `mcp__<key>__<tool>`. The
    adapter chooses it, so this module names no domain."""

    effort: EffortLevel | None = None
    """The CLI's `--effort`; None leaves the model's own default."""

    max_turns: int | None = None
    """Ceiling on tool-use round trips, the CLI's own stop; None for no ceiling."""

    cwd: Path | None = None
    """Working directory of the CLI process; None inherits this process's."""


@dataclass
class ToolCallRecord:
    """One tool call the model issued, from its assistant message to its result."""

    tool_use_id: str
    name: str
    """The tool's own name, the `mcp__<key>__` prefix removed."""
    raw_name: str
    arguments: dict[str, Any]
    message_id: str | None
    index: int
    """Position among the tool calls of the same assistant message."""
    started_at: str | None = None
    duration_s: float | None = None
    denied: str | None = None
    pre: dict[str, Any] | None = None
    post: dict[str, Any] | None = None
    is_error: bool | None = None
    written: bool = False


@dataclass
class _Step:
    """Bookkeeping for one assistant message id (one model step)."""

    calls: list[ToolCallRecord] = field(default_factory=list)
    usage: TokenUsageSnapshot | None = None
    """The step's snapshot, taken on its first frame, written with its first transcript row."""
    usage_written: bool = False


class ClaudeCodeLoop:
    """One episode hosted by Claude Code, recorded to the repo's transcript.

    Subclasses override the hook methods (`status_text`, `deny_reason`, `after_tool_call`,
    `edit_tool_output`, `on_assistant_message`), none of which does anything here.
    """

    def __init__(
        self,
        transcript: Transcript,
        config: ClaudeCodeConfig,
        mcp_url: str,
        tool_names: list[str],
        metadata_callback: Callable[[], dict[str, Any]] | None = None,
        transport: Transport | None = None,
    ) -> None:
        """Initialize the loop; nothing is launched until `run`.

        :param transcript: Transcript of the conversation recorded to file
        :param config: How the CLI is launched
        :param mcp_url: Streamable-HTTP URL of the MCP server the agent uses
        :param tool_names: The tools served there, allowlisted by name
        :param metadata_callback: Optional callback to query domain-specific tool call metadata,
            called before and after every tool call as the repo's loop does
        :param transport: An SDK `Transport` to use instead of the bundled CLI (tests)
        """
        self._transcript = transcript
        self._config = config
        self._mcp_url = mcp_url
        self._tool_names = list(tool_names)
        self._metadata_callback = metadata_callback
        self._transport = transport

        self._records: dict[str, ToolCallRecord] = {}
        self._steps: dict[str, _Step] = {}
        self._registered: dict[str, asyncio.Event] = {}
        self._seen_message_ids: set[str] = set()
        self._last_event_at: str = datetime.now().astimezone().isoformat()
        self._last_event_ns: int = time.monotonic_ns()
        self._task = ""

        self.usage_history: list[TokenUsageSnapshot] = []
        self.result: ResultMessage | None = None
        self.init_data: dict[str, Any] | None = None
        """The CLI's init message: model, tools and MCP server status, for checking the setup."""
        self.denials = 0
        self.ordering_fallbacks = 0
        """Hooks that fired before their assistant message was consumed (expected never)."""

    # Hook methods for subclasses

    def status_text(self) -> str | None:
        """Text appended to the prompt and to every tool result; None for none."""
        return None

    def deny_reason(self, call: ToolCallRecord) -> str | None:
        """The text the model gets instead of running `call`, or None to run it."""
        return None

    async def after_tool_call(self, call: ToolCallRecord, response: Any) -> None:  # noqa: ANN401
        """Called after a tool ran, with the CLI's `tool_response`, before the model sees it."""

    def edit_tool_output(self, call: ToolCallRecord, response: Any) -> Any | None:  # noqa: ANN401
        """A replacement for what the model sees of `call`'s result, or None to leave it."""
        return None

    def on_assistant_message(self, message_id: str, calls: list[ToolCallRecord]) -> None:
        """Called once per model step, when its assistant message is consumed."""

    # Tracking

    def tool_name(self, raw_name: str) -> str:
        """Strip the MCP prefix the CLI gives a served tool."""
        return raw_name.removeprefix(f"mcp__{self._config.mcp_server_key}__")

    def batch_of(self, call: ToolCallRecord) -> list[ToolCallRecord]:
        """The tool calls of `call`'s assistant message, in order (just `call` if unknown)."""
        if call.message_id is None:
            return [call]
        return list(self._steps[call.message_id].calls)

    def _event(self, tool_use_id: str) -> asyncio.Event:
        if tool_use_id not in self._registered:
            self._registered[tool_use_id] = asyncio.Event()
        return self._registered[tool_use_id]

    async def _record_for_hook(self, hook_input: dict[str, Any]) -> ToolCallRecord:
        """The record for a hook's tool call, waiting for its assistant message if needed."""
        tool_use_id = hook_input["tool_use_id"]
        try:
            await asyncio.wait_for(self._event(tool_use_id).wait(), timeout=HOOK_WAIT_S)
        except TimeoutError:
            self.ordering_fallbacks += 1
            logger.warning(
                f"Hook for {tool_use_id} fired without its assistant message; tracked alone."
            )
            record = ToolCallRecord(
                tool_use_id=tool_use_id,
                name=self.tool_name(hook_input["tool_name"]),
                raw_name=hook_input["tool_name"],
                arguments=dict(hook_input.get("tool_input") or {}),
                message_id=None,
                index=0,
            )
            self._records[tool_use_id] = record
        return self._records[tool_use_id]

    # Hooks wired to the SDK

    async def _pre_tool_use(
        self, hook_input: dict[str, Any], tool_use_id: str | None, context: HookContext
    ) -> dict[str, Any]:
        record = await self._record_for_hook(hook_input)
        reason = self.deny_reason(record)
        if reason is not None:
            self.denials += 1
            record.denied = reason
            logger.info(f"Denied {record.name} ({record.tool_use_id}).")
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        record.started_at = datetime.now().astimezone().isoformat()
        if self._metadata_callback is not None:
            record.pre = self._metadata_callback()
        return {}

    async def _post_tool_use(
        self, hook_input: dict[str, Any], tool_use_id: str | None, context: HookContext
    ) -> dict[str, Any]:
        record = await self._record_for_hook(hook_input)
        duration_ms = hook_input.get("duration_ms")
        if duration_ms is not None:
            record.duration_s = round(duration_ms / 1000, ndigits=3)
        if self._metadata_callback is not None:
            record.post = self._metadata_callback()
        response = hook_input.get("tool_response")
        await self.after_tool_call(record, response)
        output: dict[str, Any] = {"hookEventName": "PostToolUse"}
        edited = self.edit_tool_output(record, response)
        if edited is not None:
            output["updatedToolOutput"] = edited
        status = self.status_text()
        if status is not None:
            output["additionalContext"] = status
            self._transcript.append(
                create_openai_message(status, role="system"),
                metadata={"channel": "PostToolUse.additionalContext", "call_id": tool_use_id},
            )
        return {"hookSpecificOutput": output} if len(output) > 1 else {}

    async def _user_prompt_submit(
        self, hook_input: dict[str, Any], tool_use_id: str | None, context: HookContext
    ) -> dict[str, Any]:
        status = self.status_text()
        if status is None:
            return {}
        self._transcript.append(
            create_openai_message(status, role="system"),
            metadata={"channel": "UserPromptSubmit.additionalContext"},
        )
        return {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": status,
            }
        }

    # Message stream

    def _usage_snapshot(self, message: AssistantMessage) -> TokenUsageSnapshot | None:
        """One snapshot per message id; the first message of a step carries it."""
        if message.message_id is None or message.message_id in self._seen_message_ids:
            return None
        self._seen_message_ids.add(message.message_id)
        usage = message.usage or {}
        read = int(usage.get("cache_read_input_tokens") or 0)
        written = int(usage.get("cache_creation_input_tokens") or 0)
        uncached = int(usage.get("input_tokens") or 0)
        now_ns = time.monotonic_ns()
        snapshot = TokenUsageSnapshot(
            input_tokens=uncached + read + written,
            tokens_read=read,
            tokens_written=written,
            cost_usd=None,
            started_at=self._last_event_at,
            duration_s=round((now_ns - self._last_event_ns) / 1e9, ndigits=3),
        )
        self.usage_history.append(snapshot)
        return snapshot

    def _consume_assistant(self, message: AssistantMessage) -> None:
        """Register the step's tool calls, then write the message to the transcript."""
        step_id = message.message_id
        step = self._steps.setdefault(step_id, _Step()) if step_id is not None else _Step()
        new_calls: list[ToolCallRecord] = []
        text_parts: list[str] = []
        for block in message.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                record = ToolCallRecord(
                    tool_use_id=block.id,
                    name=self.tool_name(block.name),
                    raw_name=block.name,
                    arguments=dict(block.input),
                    message_id=step_id,
                    index=len(step.calls),
                )
                step.calls.append(record)
                new_calls.append(record)
                self._records[block.id] = record
            elif isinstance(block, ThinkingBlock):
                continue
        snapshot = self._usage_snapshot(message)
        if snapshot is not None:
            step.usage = snapshot
            # First message of this step: the subclass sees the step before any of its calls.
            self.on_assistant_message(step_id or "", list(step.calls))
        # The CLI streams one message as several frames (a thinking block, then text, then tool
        # calls); only the first frame takes the snapshot, and a thinking-only frame writes no
        # row, so the usage goes with the step's first row whichever frame carries it.
        usage_metadata = (
            {"usage": asdict(step.usage)}
            if step.usage is not None and not step.usage_written
            else {}
        )
        metadata: dict[str, Any] = {
            "message_id": step_id,
            "stop_reason": message.stop_reason,
            **usage_metadata,
            **({"error": message.error} if message.error else {}),
        }
        if text_parts:
            self._transcript.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "\n".join(text_parts)}],
                },
                metadata=metadata,
            )
            step.usage_written = step.usage_written or bool(usage_metadata)
            metadata = {"message_id": step_id}
        for record in new_calls:
            self._transcript.append(
                {
                    "type": "function_call",
                    "call_id": record.tool_use_id,
                    "name": record.name,
                    "arguments": json.dumps(record.arguments),
                },
                metadata={**metadata, "raw_name": record.raw_name},
            )
            step.usage_written = step.usage_written or "usage" in metadata
            metadata = {"message_id": step_id}
        for record in new_calls:
            self._event(record.tool_use_id).set()

    def _consume_user(self, message: UserMessage) -> None:
        """Write each tool result as the model saw it."""
        if isinstance(message.content, str):
            return
        for block in message.content:
            if not isinstance(block, ToolResultBlock):
                continue
            record = self._records.get(block.tool_use_id)
            if record is None or record.written:
                continue
            record.written = True
            record.is_error = block.is_error
            outcome: ToolCallOutcome = (
                "Denied"
                if record.denied is not None
                else "Tool Failure"
                if block.is_error
                else "Tool Success"
            )
            metadata: dict[str, Any] = {"outcome": outcome}
            if record.started_at is not None:
                metadata["started_at"] = record.started_at
            if record.duration_s is not None:
                metadata["duration_s"] = record.duration_s
            if record.pre is not None:
                metadata["pre"] = record.pre
            if record.post is not None:
                metadata["post"] = record.post
            self._transcript.append(
                {
                    "type": "function_call_output",
                    "call_id": block.tool_use_id,
                    "output": tool_result_to_openai_output(block.content),
                },
                metadata=metadata,
            )

    def _consume(self, message: Any) -> None:  # noqa: ANN401
        """Bookkeeping for one stream message; synchronous so hooks see a consistent state."""
        if isinstance(message, AssistantMessage):
            if message.parent_tool_use_id is None:
                self._consume_assistant(message)
        elif isinstance(message, UserMessage):
            self._consume_user(message)
        elif isinstance(message, SystemMessage):
            if message.subtype == "init":
                self.init_data = dict(message.data)
                logger.info(
                    "Claude Code init: model %s, tools %s, MCP servers %s.",
                    message.data.get("model"),
                    message.data.get("tools"),
                    message.data.get("mcp_servers"),
                )
        elif isinstance(message, ResultMessage):
            self.result = message
        self._last_event_at = datetime.now().astimezone().isoformat()
        self._last_event_ns = time.monotonic_ns()

    # Launch

    def _options(self) -> ClaudeAgentOptions:
        key = self._config.mcp_server_key
        return ClaudeAgentOptions(
            system_prompt=self._task,
            model=self._config.model,
            effort=self._config.effort,
            max_turns=self._config.max_turns,
            mcp_servers={key: {"type": "http", "url": self._mcp_url}},
            strict_mcp_config=True,
            tools=[],
            disallowed_tools=list(BUILTIN_TOOLS),
            allowed_tools=[f"mcp__{key}__{name}" for name in self._tool_names],
            permission_mode="dontAsk",
            setting_sources=[],
            env=dict(CLI_ENVIRONMENT),
            cwd=self._config.cwd,
            hooks={
                "PreToolUse": [HookMatcher(matcher=None, hooks=[self._pre_tool_use])],
                "PostToolUse": [HookMatcher(matcher=None, hooks=[self._post_tool_use])],
                "UserPromptSubmit": [HookMatcher(matcher=None, hooks=[self._user_prompt_submit])],
            },
        )

    def _prepare_environment(self) -> None:
        """Check a credential is present and hide the keys that would outrank the OAuth token."""
        if not any(os.environ.get(name) for name in CREDENTIAL_VARIABLES):
            raise RuntimeError(
                f"No Claude Code credential: set one of {', '.join(CREDENTIAL_VARIABLES)}."
            )
        for name in VARIABLES_HIDDEN_FROM_CLI:
            if os.environ.pop(name, None) is not None:
                logger.info(f"{name} removed from the CLI's environment.")

    async def run(self, task: str, prompt: str) -> ResultMessage:
        """Run one episode: `task` as the system prompt, `prompt` as the first user message.

        :return: The SDK's result message, also kept in `result`
        :raises RuntimeError: If no credential is set, or the run ended without a result
        """
        self._task = task
        if self._transport is None:
            self._prepare_environment()
        self._transcript.append(create_openai_message(prompt, role="user"))
        self._last_event_at = datetime.now().astimezone().isoformat()
        self._last_event_ns = time.monotonic_ns()
        async with ClaudeSDKClient(self._options(), transport=self._transport) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                self._consume(message)
        if self.result is None:
            raise RuntimeError("The Claude Code run ended without a result message.")
        if self.result.is_error:
            logger.warning(
                f"Claude Code result {self.result.subtype!r}: {self.result.errors or ''}"
            )
        return self.result


def tool_result_to_openai_output(content: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    """Convert a tool result's content (Messages API blocks, or a string) to Responses blocks.

    Images become `input_image` data URLs, which `Transcript` spills to its image directory.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    output: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            output.append({"type": "input_text", "text": str(block)})
        elif block.get("type") == "text":
            output.append({"type": "input_text", "text": block.get("text", "")})
        elif block.get("type") == "image":
            source = block.get("source") or {}
            if source.get("type") == "base64":
                output.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{source.get('media_type', 'image/png')};base64,"
                        f"{source.get('data', '')}",
                    }
                )
            else:
                output.append({"type": "input_image", "image_url": str(source.get("url", ""))})
        else:
            output.append({"type": "input_text", "text": json.dumps(block)})
    return output


def strip_images(response: Any, note: str) -> Any | None:  # noqa: ANN401
    """`response` with its image blocks replaced by one text block `note`, or None if it has
    no image. Handles a list of content blocks, and a dict carrying one under `content`."""
    if isinstance(response, dict) and isinstance(response.get("content"), list):
        inner = strip_images(response["content"], note)
        return None if inner is None else {**response, "content": inner}
    if not isinstance(response, list):
        return None
    if not any(isinstance(b, dict) and b.get("type") == "image" for b in response):
        return None
    kept = [b for b in response if not (isinstance(b, dict) and b.get("type") == "image")]
    return [*kept, {"type": "text", "text": note}]
