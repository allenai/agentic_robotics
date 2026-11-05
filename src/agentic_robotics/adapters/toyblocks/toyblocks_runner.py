"""Define an MCP client for the ToyBlocks server and begin an agentic loop.

Note: Isaac must be stepped from the thread that created it, and its toolkit (Kit)
pumps an asyncio loop each time the simulator advances. Therefore, no other async
loop can run on that thread. To get around this, we run the MCP loop on a second
thread and hand its asynchronous work to the simulator (i.e., main) thread.

To run this overall demo, use the commands:

    1. Launch the policy server (`agentic-robotics serve`)

    2. Run the runner itself:

        cd <ToyBlocks checkout>
        python -m agentic_robotics.adapters.toyblocks.toyblocks_runner

"""

from __future__ import annotations

import asyncio
import functools
import logging
import queue
from collections.abc import Callable, Coroutine
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, TypeVar

from mcp import Client as MCPClient
from openai.types.responses import EasyInputMessageParam

from agentic_robotics.adapters.toyblocks.config import SkillBackend, ToyBlocksConfig
from agentic_robotics.core.agentic_loop import AgenticLoop
from agentic_robotics.core.foundation_models import OpenAIModelClient, Transcript

if TYPE_CHECKING:
    import threading

    from agentic_robotics.adapters.toyblocks.toyblocks_mcp_server import ToyBlocksMCPServer

logger = logging.getLogger(__name__)


Job = tuple[Coroutine[Any, Any, Any], Future]

T = TypeVar("T")


class SimulatorDispatcher:
    """Execute the body of MCP tool calls on the thread that created the simulator.

    Isaac's toolkit (Kit) creates an asyncio loop on the thread that boots Isaac,
    and pumps that loop each simulator step. But if there's already a loop on its
    thread, Kit pumps it re-entrantly, stranding both loops' tasks. Thus, we
    isolate the simulator to the main thread and provide it work using this queue.
    """

    def __init__(self) -> None:
        """Initialize the dispatcher with an empty queue of work."""
        self._jobs = queue.Queue[Job]()

    def submit(self, coroutine: Coroutine[Any, Any, Any]) -> Future:
        """Submit a coroutine to be executed on the simulator's thread.

        :param coroutine: Coroutine to be executed
        :return: Future that will hold the coroutine's result once the simulator runs it
        """
        future = Future()
        self._jobs.put((coroutine, future))
        return future

    @staticmethod
    def drive(coroutine: Coroutine[Any, Any, T]) -> T:
        """Run a coroutine that never awaits, on the simulator's thread.

        MCP tool bodies are `async def` for the framework's sake, but are synchronous
        inside (i.e., they just step the simulator and return). By driving them
        manually, we allow the simulator's thread to execute them while the MCP
        client awaits the result from another thread.

        :param coroutine: Coroutine to be executed to completion
        :return: Value returned by the coroutine
        :raises RuntimeError: If the coroutine awaits, because its work can't actually be run here
        """
        try:
            coroutine.send(None)  # Should StopIteration unless the coroutine contains `await`
        except StopIteration as returned:
            return returned.value

        coroutine.close()
        raise RuntimeError(f"{coroutine.__qualname__} awaited; its work cannot run on this thread.")

    def execute(self, finished: threading.Event) -> None:
        """Execute submitted work until the session ends.

        :param finished: Event set once no further work will be submitted
        """
        while not finished.is_set():
            try:
                coroutine, future = self._jobs.get(timeout=0.1)  # Wait at most 0.1 sec for work
            except queue.Empty:
                continue

            if future.cancelled():
                coroutine.close()
                logger.warning("Skipping a cancelled job.")
                continue

            try:
                result = SimulatorDispatcher.drive(coroutine)
            except Exception as exc:
                if future.cancelled():
                    logger.warning("A cancelled job raised an error; dropping the error.")
                else:
                    future.set_exception(exc)  # Propagate error to re-raise on the awaiting side
                continue

            if future.cancelled():
                logger.warning("A job finished after it was cancelled; dropping its result.")
            else:
                future.set_result(result)


def _boot_mcp_server(
    dispatcher: SimulatorDispatcher, config: ToyBlocksConfig
) -> ToyBlocksMCPServer:
    """Boot Isaac on the calling thread and build an MCP server whose tools run on that thread.

    These imports are written here so that Isaac doesn't boot within a separate async context.

    :param dispatcher: Dispatcher owning the thread this function is called on
    :param config: Run-specific configuration parameters
    :return: A `ToyBlocksMCPServer` that dispatches every tool call to that thread
    """
    match config.skill_backend:
        case SkillBackend.MANIFLOW:
            from agentic_robotics.adapters.toyblocks.maniflow_mcp_server import (
                ManiFlowMCPServer as SkillServer,
            )
        case SkillBackend.ORACLE:
            from agentic_robotics.adapters.toyblocks.oracle_mcp_server import (
                OracleMCPServer as SkillServer,
            )

    class SimThreadMCPServer(SkillServer):
        """A ToyBlocks MCP server that runs every registered skill on the simulator's thread."""

        def __init__(self, dispatcher: SimulatorDispatcher, *args: Any, **kwargs: Any) -> None:
            """Initialize the simulator-thread MCP server."""
            self._dispatcher = dispatcher  # Set before super.__init__() registers the MCP tools
            super().__init__(*args, **kwargs)

        def _register_tool(self, title: str, skill: Callable) -> None:
            """Register an MCP tool whose body will be executed on the simulator's thread."""
            dispatcher = self._dispatcher

            @functools.wraps(skill)
            async def dispatched(*args: Any, **kwargs: Any) -> object:
                return await asyncio.wrap_future(dispatcher.submit(skill(*args, **kwargs)))

            super()._register_tool(title, dispatched)

    return SimThreadMCPServer(dispatcher, config)


def _agentic_thread(server: ToyBlocksMCPServer, finished: threading.Event) -> None:
    """Run the agentic loop on this thread, which is not the simulator's.

    We run the loop manually because Isaac's toolkit (Kit) replaces asyncio.run
    with a version tied to its loop on the main thread.

    :param server: MCP server providing tools to the agent
    :param finished: Event set once the agent loop is over, used to release the simulator's thread
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run_agentic_loop(server))
    finally:
        loop.close()
        finished.set()


async def _run_agentic_loop(server: ToyBlocksMCPServer) -> None:
    """Connect a client to the running MCP server and enter an agentic loop."""
    model_client = OpenAIModelClient(config=server.config.model_config)
    transcript = Transcript(out_dir=server.config.output_dir)

    async with MCPClient(server.mcp) as mcp_client:
        capabilities = mcp_client.server_capabilities.model_dump(exclude_none=True)
        logger.info(f"ToyBlocks MCP server capabilities: {capabilities}")

        agent = AgenticLoop(
            mcp_client,
            model_client,
            transcript,
            metadata_callback=lambda: {"frame_count": server.frame_count},
        )
        prompt = server.config.task_config.prompt
        initial_message = EasyInputMessageParam(content=prompt, role="user")

        try:
            await agent.enter_loop(
                task=prompt,
                history=[initial_message],
                max_tool_calls=server.config.task_config.max_tool_calls,
            )
        except Exception:
            logger.exception("Agentic loop failed; ending episode after cleanup.")
