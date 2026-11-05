"""One MCP server per agent, composed from the simulator's skill tools and locally registered ones.

The pinned MCP SDK (2.0.0) cannot mount one server inside another, and a client connects to exactly
one server. The agentic loop holds one client, so this is a lowlevel `Server` whose two handlers,
list and call, route to client sessions behind it: one to the simulator (an in-process `MCPServer`
for the top-level agent, an HTTP URL for a pool simulator), and one to `local`, the `MCPServer`
that the file, docstring, context and spawn tools are registered on.

It is also where the learned model lives. `update_description` overrides what the listing reports
for a tool, and every override writes a numbered snapshot of the whole description set under the
snapshot directory, which is what E1 scores. An allowlist selects which tools an agent sees, which
is the only thing that differs between conditions and between a parent and its subagents.

Tool results are capped in size here, so that no single result can cross the context-wipe
threshold in one turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any

from mcp import Client as MCPClient
from mcp.server.lowlevel import Server
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import CallToolResult, ListToolsResult, TextContent
from mcp_types import Tool as MCPTool

from agentic_robotics.adapters.toyblocks.leap.declarations import register_declared_tool
from agentic_robotics.adapters.toyblocks.leap.update_docstring_agent import UpdateDocstringAgent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable, Mapping
    from pathlib import Path

logger = logging.getLogger("agentic_robotics.examples.leap.composed_server")

MAX_TOOL_RESULT_CHARS = 20_000
"""Text beyond this in one tool result is cut. About 5K tokens against a 150K wipe threshold."""

SimulatorTarget = MCPServer | str
"""An in-process skill server, or the URL of one served over streamable HTTP."""


class ComposedServer:
    """A lowlevel MCP server routing tool listing and calls to the client sessions behind it."""

    def __init__(
        self,
        name: str,
        simulator: SimulatorTarget,
        snapshot_dir: Path | None = None,
        max_result_chars: int = MAX_TOOL_RESULT_CHARS,
        initial_descriptions: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize the composed server; nothing is connected until it is entered.

        :param name: Server name, reported to the client
        :param simulator: The skill server, in-process or by URL
        :param snapshot_dir: Where description snapshots are written; None disables them. If it
            already holds snapshots (a resumed run's copied workspace), numbering continues
        :param max_result_chars: Cap on the text of any one tool result
        :param initial_descriptions: Descriptions to override on entry, before the initial
            snapshot, for a served tool named in it: a resumed run's final docstring set, or an
            evaluation run's learned one. Names not served are ignored
        """
        self.name = name
        self.local = MCPServer(f"{name}-local")
        """Register tools that run in this process here, before the server is entered."""

        self._simulator = simulator
        self._snapshot_dir = snapshot_dir
        self._max_result_chars = max_result_chars
        self._snapshot_count = 0

        self._clients: list[MCPClient] = []
        self._routes: dict[str, MCPClient] = {}
        self._tools: dict[str, MCPTool] = {}
        self._overrides: dict[str, str] = dict(initial_descriptions or {})
        self._refused: dict[str, str] = {}
        self._allowed: set[str] | None = None
        self._exit_stack: AsyncExitStack | None = None

        self.server: Server[Any] = Server(
            name, on_list_tools=self._on_list_tools, on_call_tool=self._on_call_tool
        )
        """What the agent's client connects to."""

    async def __aenter__(self) -> ComposedServer:
        """Connect to the simulator and the local server, then list what they serve."""
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            self._clients = [
                await stack.enter_async_context(MCPClient(self._simulator)),
                await stack.enter_async_context(MCPClient(self.local)),
            ]
            await self.refresh()
            unknown = [name for name in self._refused if name not in self._tools]
            if unknown:
                raise RuntimeError(f"Tools to refuse are not served: {unknown}.")
            self._overrides = {k: v for k, v in self._overrides.items() if k in self._tools}
        except BaseException:
            await stack.aclose()
            raise
        self._exit_stack = stack

        if self._snapshot_dir is not None:
            existing = (
                [int(p.stem) for p in self._snapshot_dir.glob("[0-9]*.json")]
                if self._snapshot_dir.is_dir()
                else []
            )
            self._snapshot_count = max(existing, default=-1) + 1
            self.write_snapshot(reason="initial")
        return self

    async def __aexit__(self, *exc_info: Any) -> None:  # the exit-stack protocol
        """Disconnect from every source."""
        if self._exit_stack is not None:
            await self._exit_stack.__aexit__(*exc_info)
            self._exit_stack = None

    async def refresh(self) -> None:
        """Re-list every source's tools, bypassing the client cache."""
        routes: dict[str, MCPClient] = {}
        tools: dict[str, MCPTool] = {}
        for client in self._clients:
            listing = await client.list_tools(cache_mode="bypass")
            for tool in listing.tools:
                if tool.name in routes:
                    raise RuntimeError(f"Tool name {tool.name!r} is served by two sources.")
                routes[tool.name] = client
                tools[tool.name] = tool
        self._routes = routes
        self._tools = tools

    @property
    def tool_names(self) -> list[str]:
        """Every tool name across the sources, before the allowlist."""
        return list(self._tools)

    @property
    def skill_tool_names(self) -> list[str]:
        """The tools the simulator serves."""
        simulator = self._clients[0]
        return [name for name, client in self._routes.items() if client is simulator]

    @property
    def visible_tool_names(self) -> list[str]:
        """The tools the agent currently sees."""
        return [name for name in self._tools if self._visible(name)]

    @property
    def allowlist(self) -> list[str] | None:
        """The current allowlist, or None when every served tool is visible."""
        return sorted(self._allowed) if self._allowed is not None else None

    def set_allowlist(self, names: Iterable[str] | None) -> None:
        """Restrict the listing to the named tools; None lifts the restriction.

        :raises ValueError: If a name matches no served tool
        """
        if names is None:
            self._allowed = None
            return
        chosen = list(names)
        unknown = [name for name in chosen if name not in self._tools]
        if unknown:
            raise ValueError(f"Unknown tool names {unknown}; served tools are {self.tool_names}.")
        self._allowed = set(chosen)

    def refuse_calls(self, names: Iterable[str], refusal_text: Callable[[str], str]) -> None:
        """Keep the named tools listed, with their descriptions, but answer every call to one
        with `refusal_text(name)` as an error result. Call before entering; the names are checked
        against the served tools then. This is how the Harness agent's motor skills work.
        """
        for name in names:
            self._refused[name] = refusal_text(name)

    def visible_tools(self) -> list[MCPTool]:
        """The tools as the agent sees them: the allowlist applied, overrides in place."""
        return [
            tool.model_copy(update={"description": self.description_of(name)})
            for name, tool in self._tools.items()
            if self._visible(name)
        ]

    def description_of(self, name: str) -> str:
        """The description the agent sees for a tool: the override if any, else the source's."""
        return self._overrides.get(name, self._tools[name].description or "")

    def descriptions(self) -> dict[str, str]:
        """Every served tool's current description."""
        return {name: self.description_of(name) for name in self._tools}

    def update_description(self, name: str, description: str) -> Path | None:
        """Override a tool's description and snapshot the whole set.

        :param name: A tool the agent currently sees
        :param description: Its new description, replacing the old one entirely
        :return: The snapshot written, or None when snapshots are disabled
        :raises KeyError: If the agent sees no tool by that name
        """
        if name not in self._tools or not self._visible(name):
            raise KeyError(name)
        self._overrides[name] = description
        if self._snapshot_dir is None:
            return None
        return self.write_snapshot(reason=f"update:{name}")

    def write_snapshot(self, reason: str) -> Path:
        """Write the current description set as the next numbered JSON file."""
        assert self._snapshot_dir is not None
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = self._snapshot_dir / f"{self._snapshot_count:04d}.json"
        payload = {
            "version": self._snapshot_count,
            "written_at": datetime.now().astimezone().isoformat(),
            "reason": reason,
            "tools": self.descriptions(),
        }
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        self._snapshot_count += 1
        return path

    def _visible(self, name: str) -> bool:
        return self._allowed is None or name in self._allowed

    async def _on_list_tools(self, ctx: Any, params: Any) -> ListToolsResult:  # noqa: ANN401
        return ListToolsResult(tools=self.visible_tools())

    async def _on_call_tool(self, ctx: Any, params: Any) -> CallToolResult:  # noqa: ANN401
        name = params.name
        client = self._routes.get(name)
        if client is None or not self._visible(name):
            available = ", ".join(self.visible_tool_names)
            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text=f"Unknown tool {name!r}. Available tools: {available}."
                    )
                ],
                is_error=True,
            )
        if name in self._refused:
            logger.warning(f"Refused a call to {name!r}: {params.arguments}")
            return CallToolResult(
                content=[TextContent(type="text", text=self._refused[name])], is_error=True
            )
        result = await client.call_tool(name, params.arguments or {})
        return self._cap(result)

    def _cap(self, result: CallToolResult) -> CallToolResult:
        """Truncate any text block longer than the cap, saying how much was cut."""
        content = []
        for block in result.content:
            if isinstance(block, TextContent) and len(block.text) > self._max_result_chars:
                kept = block.text[: self._max_result_chars]
                cut = len(block.text) - len(kept)
                block = block.model_copy(
                    update={"text": f"{kept}\n[... result truncated: {cut} more characters]"}
                )
            content.append(block)
        return result.model_copy(update={"content": content})


def free_port(host: str = "127.0.0.1") -> int:
    """A port the kernel reports free right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


@asynccontextmanager
async def serve_over_http(
    composed: ComposedServer, host: str = "127.0.0.1", port: int | None = None, path: str = "/mcp"
) -> AsyncIterator[str]:
    """Serve an entered composed server over streamable HTTP on this event loop, for an agent
    that runs in another process (the Claude Code CLI); yields the URL and stops on exit.

    :param composed: An entered `ComposedServer`
    :param port: Port to bind; a free one when None
    """
    import uvicorn

    bound = free_port(host) if port is None else port
    app = composed.server.streamable_http_app(streamable_http_path=path, host=host)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=bound, log_level="warning"))
    serving = asyncio.ensure_future(server.serve())
    while not server.started:
        if serving.done():
            serving.result()  # Re-raise whatever stopped it
            raise RuntimeError("The MCP HTTP server stopped before it started.")
        await asyncio.sleep(0.05)
    url = f"http://{host}:{bound}{path}"
    logger.info(f"Composed server {composed.name!r} serving at {url}")
    try:
        yield url
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(serving, timeout=10.0)
        except (TimeoutError, asyncio.CancelledError, Exception):
            logger.exception("The MCP HTTP server did not stop cleanly.")


def register_docstring_tool(server: MCPServer, composed: ComposedServer) -> str:
    """Register update_tool_docstring on `server` as `UpdateDocstringAgent` declares it, acting
    on `composed`'s descriptions.

    :return: The tool's name, for allowlists
    """

    async def update_tool_docstring(tool_name: str, description: str) -> str:
        """Override the description and snapshot; the return names the saved file (see the ABC)."""
        if not description.strip():
            raise ToolError(
                "The description must not be empty; a tool with no description is unusable."
            )
        try:
            path = composed.update_description(tool_name, description)
        except KeyError:
            raise ToolError(
                f"No tool named {tool_name!r}. Your tools are: "
                f"{', '.join(composed.visible_tool_names)}."
            ) from None
        saved = f" Saved as logs/docstrings/{path.name}." if path is not None else ""
        return f"Updated the description of {tool_name!r}; it takes effect next turn.{saved}"

    return register_declared_tool(
        server, UpdateDocstringAgent, "update_tool_docstring", update_tool_docstring
    )
