"""One pre-warmed simulator for subagent trials, run as `agentic-robotics pool`:

    agentic-robotics pool --task ai2_easy --pool-port 8101

`agentic-robotics` owns the boot; this module supplies the thread that serves the skill server over
streamable HTTP at `http://HOST:PORT/mcp` until the process is stopped. `SimulatorDispatcher` itself
survives a client leaving mid-skill. Start one per simulator the pool should hold, each on its own
port and pinned to the same variant slug as the `agentic-robotics leap` run, then hand the URLs to
it as `--pool-url`. A boot costs 60-90 s, which is why these are warmed before the parent agent
starts rather than per trial. The policy server serves every pool simulator at once.

Each process records its own `video.mp4` and manifest in its own run directory, continuous
across the trials it serves. Trials set the scene they need through `set_sim_state`, so nothing
here resets between them. Isaac ignores SIGTERM; stop a pool process with Ctrl-C in its tmux
session or SIGKILL by PID.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import threading

    from agentic_robotics.adapters.toyblocks.toyblocks_mcp_server import ToyBlocksMCPServer

logger = logging.getLogger("agentic_robotics.examples.leap.pool_server")

MCP_PATH = "/mcp"


def serve_thread(
    server: ToyBlocksMCPServer, finished: threading.Event, host: str, port: int
) -> None:
    """Serve the skill server over HTTP on this thread, which is not the simulator's."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def serve() -> None:
        task = asyncio.ensure_future(
            server.mcp.run_streamable_http_async(
                host=host, port=port, streamable_http_path=MCP_PATH
            )
        )
        logger.info(f"Pool simulator serving at http://{host}:{port}{MCP_PATH}")
        while not finished.is_set():
            if task.done():
                task.result()  # Re-raise whatever stopped the server
                return
            await asyncio.sleep(0.5)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    try:
        loop.run_until_complete(serve())
    except Exception:
        logger.exception("The pool simulator's HTTP server stopped.")
    finally:
        loop.close()
        finished.set()
