"""Web interface for watching an autonomous episode and interjecting into it.

What it serves:

- `GET /` -- `index.html` (the entire front end, one file beside this one). A large iframe
  embedding the viser viewer, beside a column showing the loop's own events: the opening prompt,
  assistant text, reasoning summaries, tool calls, and tool results including the post-state
  images the model saw. Controls are Pause/Resume, a message box, and Reset episode.
- `GET /ws` -- a websocket. The browser sends `user_message`, `pause`, `resume`, `reset_episode`
  and `grant_tool_calls`; the server pushes every loop event to every connected socket, and
  replays the whole event log to a socket that connects late, so a refresh keeps the record.

**The episode is autonomous** by default. It starts when the process starts, runs on the prompt
the runner configured, and spends the one per-episode budget from its `TaskConfig`. A human
who never opens the page changes nothing. Everything the page can do reaches the loop through
`InterjectionChannel`; see `interjecting_loop.py` for where the loop asks.

With ``task_from_page=True`` nothing starts until someone types: the first message becomes the
task and the opening user message, `standing_instructions` are appended to the task the model is
instructed with, and when that episode ends the page waits for the next task. While waiting, a
reset request is applied at once since no turn is in flight. `agentic-robotics menu` uses this.

Bound to 127.0.0.1, not 0.0.0.0: there is no authentication here, and anything that can reach the
port can move the arm and read the transcript. Port-forwarding over SSH reaches localhost, so the
loopback bind costs nothing and the wildcard bind would put an unauthenticated robot control
surface on every interface the machine has.

Port 8083, because 8000 is the policy server, 8080 Viser, 8081 the manual driver, and 8082 the run
portal. The viser iframe points at http://localhost:8080 by default, so forward both ports the
same way.

Two ways to run it:

1. Against the real adapter: `agentic-robotics page` (one episode) or `agentic-robotics menu`
   (tasks loaded from the page), which boot Isaac through `runner_bootstrap` and hand this the
   skill module's tool surface.

2. Standalone mock -- no Isaac, no GPU, three fake skills -- to exercise the page and the
   interjection points. It still calls the real model, so it needs `OPENROUTER_API_KEY` and costs
   per episode:

       .venv/bin/python -m agentic_robotics.adapters.toyblocks.web_ui.server --mock

   `agentic-robotics mock` serves the same mock.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from aiohttp import WSMsgType, web
from mcp import Client as MCPClient
from mcp.server.mcpserver import Image as MCPImage  # noqa: TC002 (used for MCP tool registration)

from agentic_robotics.adapters.toyblocks.web_ui.interjecting_loop import (
    EpisodeAborted,
    Event,
    InterjectingAgenticLoop,
)
from agentic_robotics.adapters.toyblocks.web_ui.interjection import InterjectionChannel
from agentic_robotics.core import configure_logging
from agentic_robotics.core.foundation_models import (
    OpenAIModelClient,
    Transcript,
    create_openai_message,
)

if TYPE_CHECKING:
    import threading
    from collections.abc import Callable

    from agentic_robotics.adapters.toyblocks.cli import OracleSceneLoader
    from agentic_robotics.adapters.toyblocks.toyblocks_mcp_server import ToyBlocksMCPServer

logger = logging.getLogger("agentic_robotics.examples.web_ui.server")

INDEX_PATH = Path(__file__).with_name("index.html")

DEFAULT_PORT = 8083
DEFAULT_VISER_URL = "http://localhost:8080"
DEFAULT_GRANT = 25
"""Tool calls added when the human extends an exhausted budget without naming a count."""


class WebInterjectionSession:
    """Fans loop events out to every connected browser tab, and their input into the channel."""

    def __init__(
        self,
        channel: InterjectionChannel,
        validate_task: Callable[[str], str | None] | None = None,
    ) -> None:
        """Initialize an empty session around the channel the loop reads.

        :param channel: Channel the loop asks between turns
        :param validate_task: Given a menu name, why loading it would be refused, or None. Asked
            before a load request reaches the channel, since a request during an episode ends
            that episode and a refused load should not cost one.
        """
        self.channel = channel
        self._validate_task = validate_task
        self._event_log: list[Event] = []
        self._client_queues: set[asyncio.Queue[Event]] = set()

    def emit(self, event: Event) -> None:
        """Log an event and fan it out to every connected client."""
        self._event_log.append(event)
        for client_queue in self._client_queues:
            client_queue.put_nowait(event)

    def attach_client(self) -> asyncio.Queue[Event]:
        """Register a new client, replaying the event log so it catches up."""
        client_queue: asyncio.Queue[Event] = asyncio.Queue()
        for event in self._event_log:
            client_queue.put_nowait(event)
        self._client_queues.add(client_queue)

        return client_queue

    def detach_client(self, client_queue: asyncio.Queue[Event]) -> None:
        """Unregister a client."""
        self._client_queues.discard(client_queue)

    def handle_browser_message(self, data: dict) -> None:
        """Apply one message from a browser tab to the channel.

        Echoing a submitted message immediately would show it above the tool calls that are still
        running when it was typed, which is the wrong story: it is delivered before the *next*
        turn. The loop emits it when it lands, so the column reads in the order the model saw.

        :param data: The decoded websocket message
        """
        match data.get("type"):
            case "user_message" if data.get("text", "").strip():
                text = data["text"].strip()
                self.channel.submit(text)
                self.emit({"type": "message_queued", "text": text})

            case "stop":
                self.channel.request_stop()
                self.emit({"type": "status", "state": "stopping"})

            case "pause":
                self.channel.pause()
                self.emit({"type": "status", "state": "pause_requested"})

            case "resume":
                self.channel.resume()

            case "reset_episode":
                self.channel.request_reset()
                self.emit({"type": "status", "state": "reset_queued"})

            case "load_task" if isinstance(data.get("name"), str):
                name = data["name"]
                reason = self._validate_task(name) if self._validate_task is not None else None
                if reason is not None:
                    self.emit({"type": "error", "text": f"Cannot load {name}: {reason}."})
                    return
                self.channel.request_task(name)
                # A paused or stopped loop is released so it reaches the turn boundary where
                # the switch ends the episode; the arm stays halted, since the motion layer
                # reads the pending switch beside the stop flag.
                self.channel.resume()
                self.emit({"type": "status", "state": "task_load_queued", "detail": name})

            case "grant_tool_calls":
                count = data.get("count")
                self.channel.grant_tool_calls(
                    int(count) if isinstance(count, (int, float)) else DEFAULT_GRANT
                )


def _lan_ip() -> str | None:
    """Best-effort LAN address of this machine (a UDP connect sends no traffic)."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return None


async def _index_handler(request: web.Request) -> web.Response:
    """Serve the front end, substituting the viser URL the iframe should embed."""
    html = INDEX_PATH.read_text(encoding="utf-8")
    html = html.replace("{{VISER_URL}}", request.app["viser_url"])

    return web.Response(text=html, content_type="text/html")


async def _websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """Pump session events out to one browser tab and its input back in."""
    session: WebInterjectionSession = request.app["session"]

    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    client_queue = session.attach_client()

    async def send_events() -> None:
        while True:
            await ws.send_json(await client_queue.get())

    sender = asyncio.create_task(send_events())
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                session.handle_browser_message(json.loads(msg.data))
            except json.JSONDecodeError:
                continue
    finally:
        sender.cancel()
        session.detach_client(client_queue)

    return ws


def build_app(session: WebInterjectionSession, viser_url: str) -> web.Application:
    """Build the aiohttp application serving the page and the websocket."""
    app = web.Application()
    app["session"] = session
    app["viser_url"] = viser_url
    app.add_routes([web.get("/", _index_handler), web.get("/ws", _websocket_handler)])

    return app


async def _run_page_driven_episodes(
    session: WebInterjectionSession,
    channel: InterjectionChannel,
    *,
    standing_instructions: str,
    mcp_server: ToyBlocksMCPServer | SimpleNamespace | None = None,
    task_loader: OracleSceneLoader | None = None,
) -> None:
    """Wait for a human to type the task, run that episode, then wait for the next one.

    With a `task_loader`, no scene exists until the page loads one: the loader's `tasks` are sent
    to the page, `load(name)` builds a scene and returns the server for it, and `viser_url()` says
    where that scene's viewer listens. Without one, `mcp_server` is the only scene. A load asked
    for during an episode ends that episode (`EpisodeAborted`) and is applied straight after.

    Each episode starts the model on an empty conversation, and nothing is thrown away: one
    `Transcript` is kept per scene, and every episode after the first opens a new one in it, so
    `transcript.jsonl` under a scene's output directory holds them all with markers between.
    """
    current = mcp_server
    transcripts: dict[Any, Transcript] = {}
    episodes_run: dict[Any, int] = {}
    """Episodes started per scene. It indexes the scene's RNG, so a layout is a function of which
    episode it is rather than of how many draws the previous one made, and it tells the loop
    whether a transcript needs a new episode opened in it or is being written for the first time."""

    if task_loader is not None:
        session.emit({"type": "task_list", "tasks": task_loader.tasks})

    def wire_stop(server: ToyBlocksMCPServer | SimpleNamespace) -> None:
        """Let the page's stop control reach the motion layer of whichever scene is loaded.

        Re-done per scene rather than once, because a scene swap builds a new server over a new
        arm and the previous wire points at the disposed one.
        """
        set_should_stop = getattr(server, "set_should_stop", None)
        if set_should_stop is not None:
            # A pending task switch halts the arm the way a stop does: the episode is ending.
            set_should_stop(lambda: channel.stop_requested or channel.task_request_pending)

    if current is not None:
        wire_stop(current)

    async def apply_task_request() -> None:
        """Load the task the page asked for, if any, and tell the page how it went."""
        nonlocal current
        requested = channel.take_task_request()
        if requested is None or task_loader is None:
            return
        session.emit({"type": "status", "state": "loading_task", "detail": requested})
        try:
            current = await task_loader.load(requested)
        except Exception as exc:
            logger.exception(f"Could not load task {requested!r}.")
            session.emit({"type": "error", "text": f"Could not load {requested}: {exc}"})
        else:
            wire_stop(current)
            session.emit(
                {
                    "type": "task_loaded",
                    "name": requested,
                    "description": task_loader.describe(requested),
                    "viser_url": task_loader.viser_url(),
                }
            )
        session.emit({"type": "status", "state": "waiting" if current is not None else "no_task"})

    while True:
        # First, so a switch that ended the previous episode is applied before anything typed
        # during it is read as the next task.
        await apply_task_request()
        session.emit({"type": "status", "state": "waiting" if current is not None else "no_task"})

        while not channel.has_pending():
            await apply_task_request()

            if channel.take_reset_request():
                reset_scene = getattr(current, "reset_scene", None)
                if reset_scene is None:
                    session.emit({"type": "error", "text": "There is no scene to reset."})
                else:
                    await reset_scene(episodes_run.get(current.config.output_dir, 0))
                    session.emit({"type": "episode_reset"})

            await channel.wait_for_activity()

        texts = channel.take()
        if current is None:
            session.emit({"type": "error", "text": "Load a task first; nothing is on the table."})
            continue

        task_text = "\n\n".join(texts)
        task = f"{task_text}\n\n{standing_instructions}" if standing_instructions else task_text
        config = current.config
        model_client = OpenAIModelClient(config=config.model_config)
        transcript = transcripts.setdefault(
            config.output_dir, Transcript(out_dir=config.output_dir)
        )

        episode_index = episodes_run.get(config.output_dir, 0)
        episodes_run[config.output_dir] = episode_index + 1

        logger.info(
            f"Starting episode {episode_index} on the task typed on the page: {task_text!r}"
        )
        async with MCPClient(current.mcp) as mcp_client:
            agent = InterjectingAgenticLoop(
                mcp_client,
                model_client,
                transcript,
                channel=channel,
                emit=session.emit,
                reset_scene=getattr(current, "reset_scene", None),
                metadata_callback=lambda s=current: {
                    "frame_count": getattr(s, "frame_count", None)
                },
            )
            try:
                await agent.enter_loop(
                    task=task,
                    history=[create_openai_message(task_text, role="user")],
                    max_tool_calls=config.task_config.max_tool_calls,
                    new_episode=episode_index > 0,
                )
            except EpisodeAborted as exc:
                logger.info(f"Episode {episode_index} ended early: {exc}.")
                session.emit({"type": "episode_aborted", "reason": str(exc)})


async def run_web_session(
    mcp_server: ToyBlocksMCPServer | SimpleNamespace | None,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    viser_url: str = DEFAULT_VISER_URL,
    *,
    task_from_page: bool = False,
    standing_instructions: str = "",
    task_loader: OracleSceneLoader | None = None,
) -> None:
    """Serve the page, run one autonomous episode over the server's tools, then keep serving.

    The page outlives the episode on purpose: when the loop ends, the human still wants to read
    what happened and look at the final scene.

    :param mcp_server: An object with `.mcp` (the `MCPServer`) and `.config` (carrying
        `model_config`, `output_dir`, `prompt` and `max_tool_calls`) -- a `ToyBlocksMCPServer`,
        or the mock below. None when a `task_loader` supplies scenes instead.
    :param host: Interface to bind; loopback by default, deliberately
    :param port: Port to serve the page on
    :param viser_url: URL the page's iframe embeds until a loaded task names its own
    :param task_from_page: Wait for the human to type the task instead of starting on
        the task's own prompt, and run another episode each time they type one after it ends
    :param standing_instructions: Appended to every page-typed task before it is handed to the
        model: coordinate conventions, exit rule, the note that a human may speak
    :param task_loader: Builds scenes on demand from the page's task menu; implies
        `task_from_page`. Needs `tasks`, `describe(name)`, `load(name)` and `viser_url()`.
    """
    channel = InterjectionChannel()
    session = WebInterjectionSession(channel, validate_task=getattr(task_loader, "rejection", None))

    runner = web.AppRunner(build_app(session, viser_url))
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info(f"Chat UI at http://localhost:{port}/ (viser iframe -> {viser_url})")
    lan_ip = _lan_ip()
    if lan_ip and host == "0.0.0.0":
        # The page rewrites "localhost" in the viser URL to the host the browser dialed, so any
        # machine that can reach this one needs no port forwarding.
        logger.info(f"From other machines: http://{lan_ip}:{port}/")

    try:
        if task_loader is not None or task_from_page:
            await _run_page_driven_episodes(
                session,
                channel,
                standing_instructions=standing_instructions,
                mcp_server=mcp_server,
                task_loader=task_loader,
            )
        else:
            config = mcp_server.config
            model_client = OpenAIModelClient(config=config.model_config)
            transcript = Transcript(out_dir=config.output_dir)
            async with MCPClient(mcp_server.mcp) as mcp_client:
                agent = InterjectingAgenticLoop(
                    mcp_client,
                    model_client,
                    transcript,
                    channel=channel,
                    emit=session.emit,
                    reset_scene=getattr(mcp_server, "reset_scene", None),
                    metadata_callback=lambda s=mcp_server: {
                        "frame_count": getattr(s, "frame_count", None)
                    },
                )
                await agent.enter_loop(
                    task=config.task_config.prompt,
                    history=[create_openai_message(config.task_config.prompt, role="user")],
                    max_tool_calls=config.task_config.max_tool_calls,
                )
    except Exception as exc:
        logger.exception("Agentic loop failed; the page stays up to show what happened.")
        session.emit({"type": "error", "text": f"{type(exc).__name__}: {exc}"})

    logger.info("Episode over. Serving until killed (Ctrl+C).")
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


def agentic_web_thread(
    server: ToyBlocksMCPServer | None,
    finished: threading.Event,
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    viser_url: str = DEFAULT_VISER_URL,
    task_from_page: bool = False,
    standing_instructions: str = "",
    task_loader: OracleSceneLoader | None = None,
) -> None:
    """Drop-in for `toyblocks_runner._agentic_thread`, serving the page around the episode.

    Same signature and the same manual event loop, for the same reason: Kit replaces
    `asyncio.run` on the main thread. `finished` is set on the way out so the simulator's
    dispatcher thread is released however this ends.

    :param server: MCP server providing tools to the agent
    :param finished: Event set once the session is over, releasing the simulator's thread
    :param host: Interface to bind
    :param port: Port to serve the page on
    :param viser_url: URL the page's iframe embeds
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            run_web_session(
                server,
                host=host,
                port=port,
                viser_url=viser_url,
                task_from_page=task_from_page,
                standing_instructions=standing_instructions,
                task_loader=task_loader,
            )
        )
    except Exception:
        logger.exception("Web interjection session failed.")
    finally:
        loop.close()
        finished.set()


# ---------------------------------------------------------------------------------------------
# Standalone mock: the page and the interjection points, with no Isaac behind the tools
# ---------------------------------------------------------------------------------------------


def build_mock_server() -> SimpleNamespace:
    """Build a stand-in for a `ToyBlocksMCPServer`: same `.mcp` and `.config`, no Isaac.

    Tools return a synthetic frame with a marker that moves on every call, so the post-state
    images visibly change. `wait_seconds` awaits -- a real dispatched skill never may (see
    `SimulatorDispatcher.drive`) -- and exists to make a skill slow enough to type during.

    Output lands in a temp dir rather than under `output_root_dir`: a mock episode is not a run of
    anything, and putting one in the runs tree would leave a manifest-less directory there.
    """
    import numpy as np
    from mcp.server import MCPServer

    from agentic_robotics.core.foundation_models.openai_model_client import (
        OPENAI_API_MODEL_CONFIGS,
    )
    from agentic_robotics.core.images import array_to_mcp_image

    state = {"calls": 0}
    mcp = MCPServer("MockToyBlocks")

    def next_frame() -> MCPImage:
        state["calls"] += 1
        height, width = 360, 640
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
        image[:, :, 2] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
        offset = 20 + 70 * (state["calls"] % 8)
        image[20:60, offset : offset + 40] = (255, 255, 255)

        return array_to_mcp_image(image, image_format="PNG")

    @mcp.tool(title="Get Image Observation")
    async def get_current_image() -> MCPImage:
        """Return the current image observation of the mock scene."""
        return next_frame()

    @mcp.tool(title="Pick")
    async def pick(x: int, y: int) -> MCPImage:
        """Pretend to pick at a point normalized into [0,1000); returns the post-state image."""
        return next_frame()

    @mcp.tool(title="Wait")
    async def wait_seconds(seconds: float) -> MCPImage:
        """Do nothing for the given number of seconds (capped at 30); a stand-in for a slow skill."""  # noqa: E501
        await asyncio.sleep(min(seconds, 30.0))
        return next_frame()

    async def reset_scene() -> None:
        state["calls"] = 0

    output_dir = Path(tempfile.mkdtemp(prefix="webui-mock-"))
    config = SimpleNamespace(
        model_config=OPENAI_API_MODEL_CONFIGS["sonnet-5.5"],
        output_dir=output_dir,
        # Shaped like a `TaskConfig`, because that is where the prompt and the budget live.
        # A stub rather than the real class: importing the adapter's `config` through its package
        # boots Isaac, and running without one is the whole point of the mock.
        task_config=SimpleNamespace(
            name="Mock Scene",
            task_slug="mock",
            max_tool_calls=15,
            prompt=(
                "You are driving a mock scene that only pretends to act. Take an image, pick "
                "somewhere, wait a few seconds, and describe what changed between the images."
            ),
        ),
    )
    logger.info(f"Mock session transcript will land in {output_dir}")

    return SimpleNamespace(mcp=mcp, config=config, reset_scene=reset_scene)


def main() -> None:
    """Run the mock session. The real one is started from `agentic-robotics page` or
    `agentic-robotics menu`."""
    parser = argparse.ArgumentParser(description="Web interjection UI over the agentic loop.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Serve the page over three fake skills (no Isaac). Calls the real model: costs money.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--viser-url", default=DEFAULT_VISER_URL)
    args = parser.parse_args()

    if not args.mock:
        raise SystemExit(
            "Standalone mode only serves the mock; pass --mock to acknowledge it calls the real "
            "model. To run against Isaac, use `agentic-robotics page` or `agentic-robotics menu`."
        )

    configure_logging()
    asyncio.run(run_web_session(build_mock_server(), port=args.port, viser_url=args.viser_url))


if __name__ == "__main__":
    main()
