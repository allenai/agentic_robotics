"""The per-run workspace the file tools act on: notes the agent owns and logs the system writes.

Immutability is a path policy rather than a separate tool: `notes/` accepts writes and edits,
`logs/` (docstring snapshots, subagent logs) is read-only, and nothing outside the two is
reachable. The read is windowed because a tool result is capped in size and a log will not fit in
one; the edit takes an exact unique string because that form pays output tokens only for the
delta. These are the read, write, edit semantics coding agents are trained on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import Image as MCPImage
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from agentic_robotics.adapters.toyblocks.leap.declarations import register_declared_tool
from agentic_robotics.adapters.toyblocks.leap.filesystem_tools_agent import (
    DEFAULT_READ_LINES,
    MAX_READ_LINES,
    FilesystemAgent,
)

NOTES_DIR = "notes"
LOGS_DIR = "logs"

FILE_TOOL_NAMES = (
    "read_text_file",
    "read_image_file",
    "write_text_file",
    "edit_text_file",
    "list_files",
)
"""The five tools `FilesystemAgent` declares, in registration order."""

IMAGE_SUFFIXES = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg"}
"""Files `read_image_file` accepts, by suffix, with the format the SDK's Image takes."""

BINARY_SUFFIXES = {*IMAGE_SUFFIXES, ".mp4", ".pkl"}


class Workspace:
    """Two subtrees under one root, addressed by paths relative to the root."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.notes_dir = root / NOTES_DIR
        self.logs_dir = root / LOGS_DIR
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def resolve(self, path: str, *, writable: bool = False) -> Path:
        """Map a model-supplied relative path to a file under the workspace.

        :param path: Path relative to the workspace root, starting with `notes/` or `logs/`
        :param writable: Whether the caller intends to write, which only `notes/` permits
        :return: The absolute path
        :raises ToolError: If the path escapes the workspace, names a directory, or is read-only
        """
        text = path.strip()
        relative = Path(text)
        if not text or relative.is_absolute() or ".." in relative.parts:
            raise ToolError(f"Path must be relative to the workspace, with no '..': {path!r}.")
        top = relative.parts[0]
        if top not in (NOTES_DIR, LOGS_DIR):
            raise ToolError(f"Path must start with '{NOTES_DIR}/' or '{LOGS_DIR}/': {path!r}.")
        if writable and top != NOTES_DIR:
            raise ToolError(
                f"Only files under '{NOTES_DIR}/' can be written; '{LOGS_DIR}/' is read-only."
            )
        if len(relative.parts) < 2:
            raise ToolError(f"Path names a directory, not a file: {path!r}.")
        return self.root / relative

    def relative(self, path: Path) -> str:
        """The workspace-relative form of an absolute path, as the agent should write it."""
        return path.relative_to(self.root).as_posix()

    def listing(self) -> str:
        """Every file in the workspace, one per line with its size."""
        lines = []
        for file in sorted(p for p in self.root.rglob("*") if p.is_file()):
            if file.suffix.lower() in BINARY_SUFFIXES:
                detail = f"{file.stat().st_size} bytes"
            else:
                with file.open(encoding="utf-8", errors="replace") as handle:
                    detail = f"{sum(1 for _ in handle)} lines"
            lines.append(f"{self.relative(file)}  ({detail})")
        return "\n".join(lines) if lines else "(the workspace is empty)"


def register_file_tools(server: MCPServer, workspace: Workspace) -> list[str]:
    """Register the five file tools on `server` as `FilesystemAgent` declares them.

    Docstrings here are developer notes; the model reads the ABC's. Text and images are two tools:
    a subagent's recorded observations under `logs/` are read with `read_image_file`, everything
    else with `read_text_file`.

    :param server: The MCP server the tools are registered on
    :param workspace: The workspace they act on
    :return: Names of the registered tools, for allowlists
    """

    async def read_text_file(
        path: str,
        start_line: int = 1,
        max_lines: Annotated[int, Field(ge=1, le=MAX_READ_LINES)] = DEFAULT_READ_LINES,
    ) -> str:
        """Numbered lines of a text file."""
        file = workspace.resolve(path)
        if not file.is_file():
            raise ToolError(f"No file at {path!r}. Call list_files to see what exists.")
        if file.suffix.lower() in IMAGE_SUFFIXES:
            raise ToolError(f"{path} is an image; read it with read_image_file.")
        if file.suffix.lower() in BINARY_SUFFIXES:
            raise ToolError(f"{path} is a binary file and cannot be read as text.")
        if start_line < 1:
            raise ToolError("start_line must be at least 1.")

        lines = file.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)
        if total == 0:
            return f"{path} is empty (0 lines)."
        if start_line > total:
            raise ToolError(f"start_line {start_line} is past the end of {path} ({total} lines).")

        end = min(start_line + max_lines - 1, total)
        body = "\n".join(f"{n:>5}\t{lines[n - 1]}" for n in range(start_line, end + 1))
        return f"{path} (lines {start_line}-{end} of {total})\n{body}"

    async def read_image_file(path: str) -> MCPImage:
        """The image file's bytes as an MCP image."""
        file = workspace.resolve(path)
        if not file.is_file():
            raise ToolError(f"No file at {path!r}. Call list_files to see what exists.")
        suffix = file.suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise ToolError(
                f"{path} is not an image ({', '.join(sorted(IMAGE_SUFFIXES))}); read it with "
                "read_text_file."
            )
        return MCPImage(data=file.read_bytes(), format=IMAGE_SUFFIXES[suffix])

    async def write_text_file(path: str, content: str) -> str:
        """Create or replace a file under notes/."""
        file = workspace.resolve(path, writable=True)
        existed = file.exists()
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content, encoding="utf-8")
        verb = "Overwrote" if existed else "Created"
        return f"{verb} {path} ({len(content.splitlines())} lines)."

    async def edit_text_file(path: str, old_string: str, new_string: str) -> str:
        """Replace the one exact occurrence of old_string; whitespace counts."""
        file = workspace.resolve(path, writable=True)
        if not file.is_file():
            raise ToolError(f"No file at {path!r}. Use write_text_file to create it.")
        if not old_string:
            raise ToolError("old_string must not be empty.")

        text = file.read_text(encoding="utf-8")
        count = text.count(old_string)
        if count == 0:
            raise ToolError(
                f"old_string was not found in {path}. Matching is exact; read the file to check."
            )
        if count > 1:
            raise ToolError(
                f"old_string occurs {count} times in {path}; include more surrounding text so it "
                "matches exactly once."
            )

        updated = text.replace(old_string, new_string, 1)
        file.write_text(updated, encoding="utf-8")
        return f"Edited {path}: replaced 1 occurrence ({len(updated.splitlines())} lines now)."

    async def list_files() -> str:
        """Every workspace file with its size."""
        return workspace.listing()

    implementations = {
        "read_text_file": read_text_file,
        "read_image_file": read_image_file,
        "write_text_file": write_text_file,
        "edit_text_file": edit_text_file,
        "list_files": list_files,
    }
    return [
        register_declared_tool(server, FilesystemAgent, name, implementations[name])
        for name in FILE_TOOL_NAMES
    ]
