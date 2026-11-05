"""Define an abstract base class for an agent with an `update_docstring` MCP tool."""

from abc import ABC, abstractmethod


class UpdateDocstringAgent(ABC):
    """An agent with an `update_docstring` MCP tool."""

    @abstractmethod
    async def update_tool_docstring(self, tool_name: str, description: str) -> str:
        """Replace the docstring of one of your tools with new text.

        The docstring will be updated on your next conversation turn, and every update
        is saved as a numbered file under 'logs/docstrings/'.

        :param tool_name: Name of the MCP tool described by the given description
        :param description: New description for the tool (overwrites the existing description)
        :return: Result message describing the outcome of the tool
        """
        ...
