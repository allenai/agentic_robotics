"""Human interjection over `core`'s agentic loop, written as the hooks that loop provides.

The loop is `AgenticLoop.enter_loop`, unmodified. This subclass adds no loop of its own: it fills
in the three hooks the base leaves open, wraps tool execution to report it to a page, and does
nothing else. A human who never types changes nothing about how an episode behaves.

Where the human gets in:

- **`before_turn`** — hold while paused, apply a queued episode reset, then deliver everything
  typed since the last turn, each as a plain `user` item. Pause holds between turns and never
  interrupts one: a skill is atomic and a model call is billed the moment it is issued.
- **`process_tool_calls`** — a message arriving, or a stop, drops the rest of the batch. Calls
  that already ran cannot be taken back, so the ones that did not run are answered with a
  `function_call_output` saying they were skipped. A pause does not: it holds the next turn and
  lets the queued calls finish, which is what makes pausing to look free. That is the tool
  channel answering the model's own call, not a message inserted into the conversation, and it
  leaves a line in `transcript.jsonl` recording that an interjection stopped something.
- **`request_more_tool_calls`** — a spent budget asks the human for more instead of ending the
  episode. Granting zero ends it.
- **`confirm_episode_end`** — the model calling no tool is a proposal to finish, which is declined
  while someone is mid-sentence.

A model call in flight is never cancelled, and a response is never discarded. Interjecting while
the model is generating lands one seam later: the response is kept, and every call in it comes
back skipped, so the record says what happened rather than hiding it.

**Resetting an episode resets the scene and the conversation together**, because a history
describing blocks that have since moved is worse than no history. The page's feed is cleared
wherever the model's window is -- a reset, and the start of every episode after the first -- so a
human watching reads what the model is holding rather than a record the model has lost.
`transcript.jsonl` keeps every item ever appended, so the discarded attempt survives on disk. A
reset also invalidates the cache prefix, so the turn after one pays a full write.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from openai.types.responses import ResponseInputTextContentParam
from openai.types.responses.response_input_item_param import FunctionCallOutput

from agentic_robotics.core.agentic_loop import AgenticLoop, ToolCall
from agentic_robotics.core.foundation_models import History, Transcript, create_openai_message

if TYPE_CHECKING:
    from mcp import Client as MCPClient

    from agentic_robotics.adapters.toyblocks.web_ui.interjection import InterjectionChannel
    from agentic_robotics.core.foundation_models import OpenAIModelClient

logger = logging.getLogger("agentic_robotics.examples.web_ui.interjecting_loop")

Event = dict[str, Any]
"""One UI event: a JSON-serializable dict with a "type" key. `server.py` fans these out."""

EmitEvent = Callable[[Event], None]
"""Called with each event as it happens. Must not block; the server buffers per client."""

ResetScene = Callable[[], Awaitable[None]]
"""Puts the simulated scene back to a fresh episode. Awaited between turns, never during one."""

SKIPPED_CALL_TEXT = (
    "Skipped: the user interjected before this tool ran. Read their message before continuing; "
    "the plan you were following may no longer apply."
)


class EpisodeAborted(Exception):
    """Raised out of the loop, between turns, when the human loads another task mid-episode."""


class InterjectingAgenticLoop(AgenticLoop):
    """The base agentic loop, with a channel a watching human interjects through."""

    def __init__(
        self,
        mcp_client: MCPClient,
        model_client: OpenAIModelClient,
        transcript: Transcript,
        *,
        channel: InterjectionChannel,
        emit: EmitEvent,
        reset_scene: ResetScene | None = None,
        metadata_callback: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the base loop's three clients, plus a channel and an event sink.

        :param mcp_client: Client used to access MCP tools
        :param model_client: Client used to access the model
        :param transcript: Transcript of the conversation recorded to file
        :param channel: Channel the watching human interjects through
        :param emit: Called with each UI event as it happens
        :param reset_scene: Optional coroutine function putting the scene back to a fresh episode;
            without one, the page's reset control reports that this session has no scene to reset
        :param metadata_callback: Passed to the base loop, which samples it around each tool call
        """
        super().__init__(mcp_client, model_client, transcript, metadata_callback=metadata_callback)
        self._channel = channel
        self._emit = emit
        self._reset_scene = reset_scene

        self._opening_history: History = []
        self._emitted_upto = 0
        self._calls_left = 0
        self._calls_granted = 0

    @property
    def can_reset(self) -> bool:
        """Whether this session has a scene the human can reset."""
        return self._reset_scene is not None

    async def enter_loop(
        self,
        task: str,
        history: History,
        max_tool_calls: int,
        *,
        new_episode: bool = False,
    ) -> Transcript:
        """Set up the page's view of the episode, run the base loop, then report it finished.

        :param task: Task instructions given to the agent each loop
        :param history: Conversation history providing context to the model
        :param max_tool_calls: Budget of permitted MCP tool calls for the whole episode
        :param new_episode: Whether this transcript already holds an episode, in which case the
            opening goes in through `start_new_episode` so the model's window begins at it and a
            marker records the boundary. The base loop appends the history it is given, so it is
            handed nothing when this has already done it.
        :return: Full transcript of the finished conversation(s)
        """
        self._opening_history = list(history)
        self._calls_left = max_tool_calls
        self._calls_granted = max_tool_calls

        starting_new = new_episode and bool(history)
        if starting_new:
            self._emit({"type": "conversation_cleared"})

        self._emit_opening(history)
        self._emit_budget()

        if starting_new:
            opening, *rest = history
            self._transcript.start_new_episode(opening)
            self._transcript.extend(rest)
            history = []

        try:
            return await super().enter_loop(task, history, max_tool_calls)
        finally:
            self._drain_transcript_events()  # The base's final-answer turn appends after the loop
            self._emit({"type": "episode_done"})

    # -- The hooks -----------------------------------------------------------------------------

    async def before_turn(self) -> None:
        """Hold while paused, apply a queued reset, and deliver anything the human typed.

        Reset is applied here, between turns, for the reason a skill is atomic: it moves the
        world, and doing that while a motion is in flight has no defined meaning.
        """
        # A task switch ends the episode here, before any pause hold or model call. Checked
        # again after the hold, since the switch is what releases a paused loop.
        self._raise_if_switching()
        if self._channel.paused:
            self._emit({"type": "status", "state": "paused"})
            await self._channel.wait_until_resumed()
            self._emit({"type": "resumed"})
        self._raise_if_switching()

        if self._channel.take_reset_request():
            await self._reset_episode()

        for message in self._channel.take():
            self._transcript.append(create_openai_message(message, role="user"))
            self._emit({"type": "user_message", "text": message})

        self._emit({"type": "status", "state": "thinking"})

    async def request_more_tool_calls(self) -> int:
        """Ask the human to extend a spent budget, and wait for the answer.

        :return: Additional tool calls granted; zero to end the episode
        """
        self._emit({"type": "budget_exhausted"})
        granted = await self._next_grant_or_switch()
        self._emit({"type": "budget_granted", "count": granted})

        if granted > 0:
            self._calls_left += granted
            self._calls_granted += granted
            self._emit_budget()

        return granted

    def _raise_if_switching(self) -> None:
        """End the episode if the human has asked for another task."""
        if self._channel.task_request_pending:
            raise EpisodeAborted("the human loaded another task")

    async def _next_grant_or_switch(self) -> int:
        """Wait for a budget grant, unless a task switch arrives first, which ends the episode.

        `next_grant` alone would hold the loop until someone answered the budget prompt, and a
        human who picks another task instead has answered it.
        """
        grant = asyncio.ensure_future(self._channel.next_grant())
        try:
            while True:
                activity = asyncio.ensure_future(self._channel.wait_for_activity())
                done, _ = await asyncio.wait({grant, activity}, return_when=asyncio.FIRST_COMPLETED)
                if grant in done:
                    activity.cancel()
                    return grant.result()
                self._raise_if_switching()
        finally:
            if not grant.done():
                grant.cancel()

    async def confirm_episode_end(self) -> bool:
        """Decline the model's proposal to finish while the human is mid-sentence.

        :return: Whether the episode should end now
        """
        return not self._channel.has_pending()

    # -- Tool calls ----------------------------------------------------------------------------

    async def process_tool_calls(self, tool_calls: list[ToolCall]) -> list[FunctionCallOutput]:
        """Run a response's tool calls, stopping the batch if the human interjects.

        Every call still receives an output, which the Responses API requires. This is also where
        the model's own text and reasoning reach the page, because it is the one method the base
        loop calls once per turn with the response already in the transcript.

        :param tool_calls: The model response's tool calls, as the base loop filters them
        :return: One output per tool call
        """
        self._drain_transcript_events()

        outputs: list[FunctionCallOutput] = []
        for call in tool_calls:
            stopped = self._channel.stop_requested
            switching = self._channel.task_request_pending
            if self._channel.has_pending() or stopped or switching:
                outputs.append(self._skipped_output(call))
                # A switch, a stop and an interjection drop a call for different reasons, and
                # someone watching a demo needs to see which. Checked in that order: when more
                # than one is true, the earlier is the stronger claim about why the arm is
                # where it is.
                self._emit(
                    {
                        "type": "tool_skipped",
                        "name": call["name"],
                        "reason": "task_switch"
                        if switching
                        else "stop"
                        if stopped
                        else "interjection",
                    }
                )
            else:
                outputs.append(await self.process_tool_call(call))

        # Mirrors the base loop's own budget arithmetic (`tool_call_cost`, one per call), which
        # is not reachable from a hook. Charging a skipped call is the base's arithmetic, not a
        # choice made here; the page would read differently if the base charged only what it ran.
        self._calls_left -= len(outputs)
        self._emit_budget()

        return outputs

    async def process_tool_call(self, item: ToolCall) -> FunctionCallOutput:
        """Run one tool call through the base implementation, reporting it to the page.

        :param item: Function tool call to execute
        :return: The call's output, from the base implementation
        """
        self._emit({"type": "status", "state": "tool_running", "detail": item["name"]})
        result = await super().process_tool_call(item)
        self._emit_tool_result(item, result)

        return result

    @staticmethod
    def _skipped_output(call: ToolCall) -> FunctionCallOutput:
        """Build the output answering a call the human's interjection stopped."""
        return FunctionCallOutput(
            call_id=call["call_id"],
            output=[ResponseInputTextContentParam(text=SKIPPED_CALL_TEXT, type="input_text")],
            type="function_call_output",
        )

    async def _reset_episode(self) -> None:
        """Put the scene back to a fresh state and start a new episode in the transcript.

        The model's window goes back to the opening prompt and the page's feed follows it, so the
        human is reading the conversation the model has. Nothing is discarded by either:
        `Transcript` moves the index its current episode starts at and writes a marker where the
        boundary fell, so the attempt being abandoned stays on disk.
        """
        if self._reset_scene is None:
            self._emit({"type": "error", "text": "This session has no scene to reset."})
            return

        await self._reset_scene()

        opening, *rest = self._opening_history
        self._transcript.start_new_episode(opening)
        self._transcript.extend(rest)

        logger.info("Episode reset; the model's window is back to the opening prompt.")
        self._emit({"type": "conversation_cleared"})
        self._emit({"type": "episode_reset"})

    # -- Events --------------------------------------------------------------------------------

    def _emit_budget(self) -> None:
        """Report the tool-call budget to the page."""
        self._emit(
            {"type": "budget", "remaining": self._calls_left, "granted": self._calls_granted}
        )

    def _emit_opening(self, history: History) -> None:
        """Emit the opening conversation items, so a connecting page shows the task."""
        for item in history:
            if item.get("role") != "user":
                continue

            content = item.get("content")
            if isinstance(content, str):
                text = content
            else:
                text = "\n".join(
                    block.get("text", "")
                    for block in content or []
                    if isinstance(block, dict) and block.get("type") == "input_text"
                )

            if text:
                self._emit({"type": "user_message", "text": text})

    def _drain_transcript_events(self) -> None:
        """Emit UI events for every transcript item appended since this was last called.

        Reading the transcript rather than a response means the base loop's final-answer turn is
        reported too, which no hook is called during. User and system items are skipped here
        because they are emitted where they originate.
        """
        items = (
            self._transcript._items
        )  # Private: `Transcript` exposes only the current episode, not the whole history
        new_items = items[self._emitted_upto :]
        self._emitted_upto = len(items)

        for item in new_items:
            item_type = item.get("type")

            if item_type == "message" and item.get("role") == "assistant":
                texts = [
                    block.get("text", "")
                    for block in item.get("content", [])
                    if isinstance(block, dict) and "text" in block
                ]
                if any(texts):
                    self._emit({"type": "assistant_text", "text": "\n".join(texts)})

            elif item_type == "reasoning":
                blocks = list(item.get("summary") or []) + list(item.get("content") or [])
                texts = [b.get("text", "") for b in blocks if isinstance(b, dict)]
                if any(texts):
                    self._emit({"type": "reasoning", "text": "\n".join(t for t in texts if t)})

            elif item_type == "function_call":
                self._emit(
                    {
                        "type": "tool_call",
                        "name": item["name"],
                        "call_id": item["call_id"],
                        "arguments": item.get("arguments", ""),
                    }
                )

    def _emit_tool_result(self, call: ToolCall, result: FunctionCallOutput) -> None:
        """Emit one tool call's result, including the images the model was shown."""
        blocks = result.get("output") or []
        if isinstance(blocks, str):
            blocks = [ResponseInputTextContentParam(text=blocks, type="input_text")]

        self._emit(
            {
                "type": "tool_result",
                "name": call["name"],
                "call_id": call["call_id"],
                "text": "\n".join(
                    text
                    for block in blocks
                    if block.get("type") == "input_text" and (text := block.get("text", ""))
                ),
                "images": [
                    str(block.get("image_url", ""))
                    for block in blocks
                    if block.get("type") == "input_image"
                ],
            }
        )
