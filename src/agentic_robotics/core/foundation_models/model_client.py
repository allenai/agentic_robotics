"""Define an interface for programmatically calling models (i.e., LLMs/VLMs)."""

from typing import Protocol

from mcp_types import Tool as MCPTool
from openai.types.responses import ResponseInputParam

History = ResponseInputParam
"""List of messages representing the history of a conversation."""


class ModelClient(Protocol):
    """An interface for programmatically calling LLMs and VLMs."""

    async def prompt(
        self, instructions: str, history: History, tools: list[MCPTool] | None = None
    ) -> History:
        """Prompt the model using the given conversation history and available MCP tools.

        :param instructions: System-level instructions to the model
        :param history: Conversation history upon which the model response is conditioned
        :param tools: Optional list of MCP tools available to the model (defaults to None)
        :return: Model response, represented as a list of additional conversation messages
        """
        ...
