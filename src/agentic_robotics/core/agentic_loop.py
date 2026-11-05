"""Construct a basic agentic loop using an MCP client and the OpenAI Responses API."""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from typing import Any, Literal, TypeGuard

from mcp import Client as MCPClient
from mcp_types import Tool as MCPTool
from openai.types.responses import (
    ResponseFunctionCallOutputItemListParam,
    ResponseInputItemParam,
    ResponseInputTextContentParam,
)
from openai.types.responses import ResponseFunctionToolCallParam as ToolCall
from openai.types.responses.response_input_item_param import FunctionCallOutput

from agentic_robotics.core.foundation_models import (
    History,
    OpenAIModelClient,
    Transcript,
    create_openai_message,
)
from agentic_robotics.core.mcp_conversion import mcp_result_to_openai_output

logger = logging.getLogger(__name__)

ToolCallOutcome = Literal["Invalid JSON", "Tool Failure", "Tool Success", "Tool Raised Exception"]


def is_tool_call(item: ResponseInputItemParam) -> TypeGuard[ToolCall]:
    """Determine whether the given conversation item is a function tool call."""
    return item.get("type") == "function_call"


class AgenticLoop:
    """A basic agentic loop based on MCP tool calls chosen using the OpenAI Responses API."""

    def __init__(
        self,
        mcp_client: MCPClient,
        model_client: OpenAIModelClient,
        transcript: Transcript,
        metadata_callback: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the clients used by the agentic loop.

        :param mcp_client: Client used to access MCP tools and resources
        :param model_client: Client used to access the LLM/VLM model
        :param transcript: Transcript of the conversation recorded to file
        :param metadata_callback: Optional callback to query domain-specific tool call metadata
        """
        self._mcp_client = mcp_client
        self._model_client = model_client
        self._transcript = transcript

        self._call_metadata: dict[str, dict[str, Any]] = {}
        """Metadata associated with each MCP tool call; populated in process_tool_call."""

        self._metadata_callback = metadata_callback
        """Optional callback used to query domain-specific metadata per tool call."""

    def get_final_update(self, task: str) -> str:
        """Build an update message to prompt the agent after it exits the tool-call loop."""
        return (
            f"Recall your current task:\n\n{task}\n\n"
            "You have no further tool calls available. Respond with your final answer."
        )

    def status_text(self, tool_calls_left: int) -> str:
        """Build system-level status text to be passed to the agent."""
        return f"You have {tool_calls_left} tool calls remaining."

    async def list_mcp_tools(self) -> list[MCPTool]:
        """List all tools available to the MCP client."""
        list_tools_result = await self._mcp_client.list_tools()
        return list_tools_result.tools

    def _filter_tool_calls(self, messages: History) -> list[ToolCall]:
        """Filter out the MCP tool calls from the messages of an OpenAI API response."""
        return [m for m in messages if is_tool_call(m)]

    async def process_tool_call(self, item: ToolCall) -> FunctionCallOutput:
        """Process a single MCP tool call.

        :param item: Function tool call, represented as a message from the OpenAI Responses API
        :return: Resulting function output, even in cases where the tool call raises
        """
        call_id = item["call_id"]
        output: ResponseFunctionCallOutputItemListParam = []
        started_at = datetime.now().astimezone().isoformat()
        start_time_ns = time.monotonic_ns()

        callback_metadata = {"pre": self._metadata_callback()} if self._metadata_callback else {}

        try:
            parsed_args = json.loads(item["arguments"])
        except json.JSONDecodeError:
            logger.warning(f"Malformed arguments for tool call {call_id}: {item['arguments']}")
            output = [
                ResponseInputTextContentParam(
                    text="The tool call's arguments were not valid JSON.", type="input_text"
                )
            ]
            call_outcome: ToolCallOutcome = "Invalid JSON"

        else:
            try:
                result = await self._mcp_client.call_tool(name=item["name"], arguments=parsed_args)
                call_outcome = "Tool Failure" if result.is_error else "Tool Success"
                output = mcp_result_to_openai_output(result)

            except Exception as exc:
                logger.exception(f"Tool call {call_id} ({item['name']}) raised.")
                output = [
                    ResponseInputTextContentParam(
                        text=f"The tool call raised an exception ({type(exc).__name__}): {exc}.",
                        type="input_text",
                    )
                ]
                call_outcome = "Tool Raised Exception"

        duration_s = round((time.monotonic_ns() - start_time_ns) / 1e9, ndigits=3)
        if self._metadata_callback:
            callback_metadata["post"] = self._metadata_callback()

        self._call_metadata[call_id] = {
            "started_at": started_at,
            "duration_s": duration_s,
            "outcome": call_outcome,
            **callback_metadata,
        }

        return FunctionCallOutput(call_id=call_id, output=output, type="function_call_output")

    async def process_tool_calls(self, tool_calls: list[ToolCall]) -> list[FunctionCallOutput]:
        """Process MCP tool calls from the messages of an OpenAI API response."""
        outputs: list[FunctionCallOutput] = []
        for call in tool_calls:
            output = await self.process_tool_call(call)
            outputs.append(output)
        return outputs

    async def run_tool_calls(self, calls: list[ToolCall]) -> list[FunctionCallOutput]:
        """Run MCP tool calls and log their outputs with any associated metadata.

        :param calls: MCP tool calls from an OpenAI API model response
        :return: Outputs from the tool calls (empty if no calls were made)
        """
        tool_outputs = await self.process_tool_calls(calls)
        if not tool_outputs:
            return []

        for tool_output in tool_outputs:
            metadata = self._call_metadata.pop(tool_output["call_id"], None)
            self._transcript.append(tool_output, metadata=metadata)

        return tool_outputs

    def tool_call_cost(self, call: ToolCall) -> int:
        """Compute the tool call cost for the given tool call.

        Subclasses can override this method to change how the tool budget behaves.
        """
        return 1

    async def take_turn(self, task: str, status_text: str, tools: list[MCPTool]) -> History:
        """Prompt the model to take one conversation turn, and record its response.

        :param task: Fixed task instructions given to the model each loop
        :param status_text: System message for this turn (e.g., remaining tool call budget)
        :param tools: MCP tools offered to the model this turn
        :return: Model's response, as the conversation items appended to the transcript
        """
        new_usage_idx = len(self._model_client.usage_history)

        response_messages = await self._model_client.prompt(
            instructions=task,
            history=[
                *self._transcript.current_episode,
                create_openai_message(
                    text=status_text, role="system", caching=self._model_client.config.caching
                ),  # Keep cache-breakpoint-annotated messages out of the transcript
            ],
            tools=tools,
        )

        new_usage = self._model_client.usage_history[new_usage_idx:]
        metadata = {"usage": asdict(new_usage[-1])} if new_usage else None

        self._transcript.append(
            create_openai_message(text=status_text, role="system"), metadata=metadata
        )
        self._transcript.extend(response_messages)

        return response_messages

    async def take_final_turn(self, task: str, tools: list[MCPTool]) -> History | None:
        """Prompt the model to take its final conversation turn, and record its response.

        :param task: Fixed task instructions given to the model each loop
        :param tools: MCP tools offered to the model this turn
        :return: Model's response, as the conversation items appended to the transcript (or None)
        """
        last_item = self._transcript.current_episode[-1]
        if last_item.get("type") == "message" and last_item.get("role") == "assistant":
            return None

        final_text = self.get_final_update(task=task)

        # Provide the list of available tools so that the model cache isn't wiped
        return await self.take_turn(task=task, status_text=final_text, tools=tools)

    async def enter_loop(self, task: str, history: History, max_tool_calls: int) -> Transcript:
        """Enter an agentic loop initialized using the given conversation history.

        :param task: Fixed task instructions given to the agent each loop
        :param history: Conversation history providing context to the model
        :param max_tool_calls: Maximum budget of permitted MCP tool calls
        :return: Full transcript of the finished conversation(s)
        """
        self._transcript.extend(history)
        tool_calls_left = max_tool_calls

        while self.should_continue():
            if tool_calls_left <= 0:
                extra_calls = await self.request_more_tool_calls()
                if extra_calls <= 0:
                    break
                tool_calls_left += extra_calls

            await self.before_turn()

            status_text = self.status_text(tool_calls_left=tool_calls_left)
            available_tools = await self.list_mcp_tools()  # Changing the tools clears the cache

            response_messages = await self.take_turn(
                task=task, status_text=status_text, tools=available_tools
            )  # Solicit the model to select one or more MCP tool calls

            tool_calls = self._filter_tool_calls(response_messages)
            for call in tool_calls:
                tool_calls_left -= self.tool_call_cost(call)

            tool_outputs = await self.run_tool_calls(tool_calls)
            if not tool_outputs and (await self.confirm_episode_end()):
                break  # Exit if the model didn't call any MCP tools

            await self.after_turn()

        # Allow the model to respond to the final state of the conversation (if needed)
        available_tools = await self.list_mcp_tools()
        await self.take_final_turn(task=task, tools=available_tools)

        return self._transcript

    def should_continue(self) -> bool:
        """Return whether or not the agent loop should continue.

        Subclasses can override this method to add their own stopping conditions.

        :return: True, if the agentic loop should continue, else False.
        """
        return True

    async def before_turn(self) -> None:
        """Perform operations before the agent's conversation turn.

        This is a hook method for subclasses; the base loop does nothing.
        """
        pass

    async def after_turn(self) -> None:
        """Perform operations after the agent's conversation turn.

        This is a hook method for subclasses; the base loop does nothing.
        """
        pass

    async def request_more_tool_calls(self) -> int:
        """Request more MCP tool calls once the budget is spent.

        This is a hook method for subclasses; the base loop does not grant additional calls.

        :return: Number of additional MCP tool calls given to the agent
        """
        return 0

    async def confirm_episode_end(self) -> bool:
        """Check whether the current episode should terminate.

        This is a hook method for subclasses; the base loop simply exits.
        """
        return True
