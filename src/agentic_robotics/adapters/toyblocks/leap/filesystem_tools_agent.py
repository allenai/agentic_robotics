"""Define an abstract base class for an agent that can interact with the filesystem."""

from abc import ABC, abstractmethod
from typing import Annotated

from mcp.server.mcpserver import Image as MCPImage
from pydantic import Field

DEFAULT_READ_LINES = 200
"""Default number of lines read from a file by the `read_file` tool."""

MAX_READ_LINES = 500
"""Maximum number of lines that can be read by a single call to the `read_file` tool."""


class FilesystemAgent(ABC):
    """An agent that can interact with the local filesystem."""

    @abstractmethod
    async def read_text_file(
        self,
        path: str,
        start_line: int = 1,
        max_lines: Annotated[int, Field(ge=1, le=MAX_READ_LINES)] = DEFAULT_READ_LINES,
    ) -> str:
        """Read a window of lines from the specified text file.

        Paths are relative to your workspace:
            'notes/' contains files you wrote
            'logs/' contains files the system wrote for you (e.g., tool description snapshots)

        :param path: Relative path to the file to be read
        :param start_line: First line of the window read from the file (1-based indexing)
        :param max_lines: Maximum number of lines read from the file
        :return: Requested lines of text, numbered, and the file's total line count
        """
        ...

    @abstractmethod
    async def read_image_file(self, path: str) -> MCPImage:
        """Read an image from the specified file.

        :param path: Relative path to the file to be read
        :return: Image read from the file
        """
        ...

    @abstractmethod
    async def write_text_file(self, path: str, content: str) -> str:
        """Create or overwrite a file under 'notes/' with the given text.

        To change part of an existing file, use `edit_text_file` instead.

        :param path: Filepath to be created or overwritten (must begin with 'notes/')
        :param content: Content of the new file, or replaces the whole file if it exists
        :return: Result message describing the outcome of the tool
        """
        ...

    @abstractmethod
    async def edit_text_file(self, path: str, old_string: str, new_string: str) -> str:
        """Replace one exact occurrence of `old_string` with `new_string` in the specified file.

        :param path: Filepath to be edited (must begin with 'notes/')
        :param old_string: Text that must appear exactly once in the file, so include enough
            surrounding text to make it unique. Matching is exact.
        :param new_string: Text used to replace `old_string`
        :return: Result message describing the outcome of the tool
        """
        ...

    @abstractmethod
    async def list_files(self) -> str:
        """List every file in your workspace, with its size in bytes.

        'notes/' holds files you wrote; 'logs/' holds files the system wrote for you to read.
        """
        ...
