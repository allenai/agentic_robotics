"""Subagent episodes: one blocking batch tool that runs each episode on a pool simulator under
the parent's budget and returns what each one reported.

The tool is `RecursiveAgent.spawn_subagents` from the ABCs, which fixes its name, docstring and
the `Episode` schema: a prompt, a tool allowlist (the skills, the perception and state tools and
the file tools; never docstring updates, context clearing or spawning), a tool-call ceiling of at
most `MAX_EPISODE_TOOL_CALLS`, and a starting scene, either a `SimState` or a workspace path to a
file holding one as JSON. Every episode runs on the parent's model. An
episode ends when it stops calling tools, when it reaches its ceiling, or when the shared ledger
passes its cap, in which case it returns a budget-exceeded result with no final message.
Subagents inherit the parent's current tool descriptions and share its workspace; the parent gets
back an `EpisodeResult` per episode. The parent's prompt is the first user message; the system
instructions are the developer's preamble under the ABC package's `prompts/` directory.

Simulators come from `SimulatorPool`: pre-warmed `pool_server.py` processes serving one scene
each over streamable HTTP. An episode checks one out for its duration, so more episodes than
simulators queue. An episode's transcript lands under `workspace/logs/subagents/`, with `log.md`
rendered once from it by `render_log` and its images beside it, all readable by the parent with
`read_file`.

An unpriced model call inside an episode raises `UnpricedModelCallError` out of the episode and
out of the tool, which the MCP layer reports to the parent as a tool error rather than a crash.
The parent's own next model call goes through the same endpoint, so the ledger raises there and
ends the phase within one turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from mcp import Client as MCPClient
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import ValidationError

from agentic_robotics.adapters.toyblocks.leap import messages
from agentic_robotics.adapters.toyblocks.leap.composed_server import ComposedServer
from agentic_robotics.adapters.toyblocks.leap.cost import CostLedger, UnpricedModelCallError
from agentic_robotics.adapters.toyblocks.leap.declarations import register_declared_tool
from agentic_robotics.adapters.toyblocks.leap.prompts_dir import PROMPTS_DIR
from agentic_robotics.adapters.toyblocks.leap.recursive_agent import (
    Episode,
    EpisodeResult,
    RecursiveAgent,
)
from agentic_robotics.adapters.toyblocks.leap.workspace import (
    FILE_TOOL_NAMES,
    Workspace,
    register_file_tools,
)
from agentic_robotics.core.agentic_loop import AgenticLoop, ToolCall
from agentic_robotics.core.foundation_models import (
    History,
    OpenAIModelClient,
    Transcript,
    create_openai_message,
)
from agentic_robotics.core.states import SimState

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from mcp.server.mcpserver import MCPServer
    from mcp_types import Tool as MCPTool
    from openai.types.responses.response_input_item_param import FunctionCallOutput

    from agentic_robotics.core.foundation_models.openai_model_client import OpenAIModelConfig

logger = logging.getLogger("agentic_robotics.examples.leap.subagents")

SUBAGENT_LOGS_DIR = "subagents"
"""Under the workspace's logs directory: one directory per batch, one per episode inside it."""

DEFAULT_PREAMBLE_FILE = PROMPTS_DIR / "subagent_preamble.md"
"""The developer's system prompt for every subagent episode, their file and their spelling."""

SET_STATE_TOOL = "set_sim_state"


class SimulatorPool:
    """Pre-warmed simulators, each serving one scene's tools at a URL, checked out one at a time."""

    def __init__(self, urls: list[str]) -> None:
        if not urls:
            raise ValueError("A simulator pool needs at least one URL.")
        self.urls = list(urls)
        self.tool_names: list[str] = []
        self._free: asyncio.Queue[str] | None = None

    async def check(self) -> list[str]:
        """Connect to every simulator and list its tools; all must serve the same set.

        :return: The tool names every simulator serves
        :raises RuntimeError: If a simulator is unreachable or serves a different set
        """
        listings: dict[str, list[str]] = {}
        for url in self.urls:
            try:
                async with MCPClient(url) as client:
                    result = await client.list_tools(cache_mode="bypass")
            except Exception as exc:
                raise RuntimeError(f"Pool simulator at {url} is not serving: {exc}") from exc
            listings[url] = sorted(tool.name for tool in result.tools)
        first = listings[self.urls[0]]
        for url, names in listings.items():
            if names != first:
                raise RuntimeError(
                    f"Pool simulators disagree on their tools: {self.urls[0]} serves {first}, "
                    f"{url} serves {names}."
                )
        self.tool_names = first
        return first

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[str]:
        """Check a simulator out until the block ends; waits when all are in use."""
        if self._free is None:
            self._free = asyncio.Queue()
            for url in self.urls:
                self._free.put_nowait(url)
        url = await self._free.get()
        try:
            yield url
        finally:
            self._free.put_nowait(url)


class SubagentLoop(AgenticLoop):
    """The base loop with a per-turn call count, the shared ledger, and a budget stop.

    Hooks only: the loop itself, the turn and the tool-call recording are the base's. A
    tool-less turn ends the episode, as the base does.
    """

    def __init__(
        self,
        mcp_client: MCPClient,
        model_client: OpenAIModelClient,
        transcript: Transcript,
        *,
        ledger: CostLedger,
    ) -> None:
        super().__init__(mcp_client, model_client, transcript)
        self._ledger = ledger
        self._usage_seen = 0
        self.tool_calls_made = 0
        self.cost_usd = 0.0
        self.budget_exceeded = False

    def _record_usage(self) -> None:
        """Add any new usage snapshots to the ledger and this episode's own total."""
        snapshots = self._model_client.usage_history
        new = snapshots[self._usage_seen :]
        self._usage_seen = len(snapshots)
        for snapshot in new:
            self._ledger.add(snapshot.cost_usd)  # Raises on an unpriced call
            self.cost_usd += snapshot.cost_usd or 0.0

    def status_text(self, tool_calls_left: int) -> str:
        """The subagent's per-turn status line, from `messages.py`."""
        return messages.subagent_status_line(tool_calls_left)

    def should_continue(self) -> bool:
        """Stop, and say so, once the shared ledger is over its cap."""
        if self._ledger.exceeded:
            self.budget_exceeded = True
            return False
        return True

    async def take_turn(self, task: str, status_text: str, tools: list[MCPTool]) -> History:
        """The base turn, then the call's cost goes to the ledger and this episode's total."""
        response = await super().take_turn(task, status_text, tools)
        self._record_usage()
        return response

    async def run_tool_calls(self, calls: list[ToolCall]) -> list[FunctionCallOutput]:
        """The base run, counted."""
        outputs = await super().run_tool_calls(calls)
        self.tool_calls_made += len(outputs)
        return outputs

    async def take_final_turn(self, task: str, tools: list[MCPTool]) -> History | None:
        """No final answer once the ledger is over its cap; otherwise the base's."""
        if self._ledger.exceeded:
            self.budget_exceeded = True
            return None
        return await super().take_final_turn(task, tools)

    def get_final_update(self, task: str) -> str:
        """The final-answer request, from `messages.py` rather than the core loop."""
        return messages.final_update(task)


def _text_of(content: object) -> str:
    """The text in a message's content, which is a string or a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(block.get("text", "")) for block in content if isinstance(block, dict))
    return ""


def final_message_of(items: History) -> str:
    """The text of the last assistant message, or empty when there is none."""
    for item in reversed(items):
        if item.get("type") == "message" and item.get("role") == "assistant":
            return _text_of(item.get("content"))
    return ""


def render_log(transcript_path: Path, image_dir: str = "images") -> str:
    """Render an episode's `transcript.jsonl` as Markdown: prompts, agent text, calls and results.

    Images are named by file under `image_dir`, which the spawner passes as the workspace-relative
    path of the episode's `images/` directory so the parent can hand it straight to `read_file`.
    Reasoning items are left out: the API returns them summarised at best, and the model's own
    text carries the account.
    """
    lines = ["# Episode log", ""]
    for raw in transcript_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(raw)
        # Messages the loop writes itself carry a role and no type; the API's carry both.
        kind = item.get("type") or ("message" if "role" in item else None)
        if kind == "message":
            role = item.get("role")
            heading = {"assistant": "Agent", "user": "Instructions"}.get(role, "System")
            lines += [f"## {heading}", _text_of(item.get("content")), ""]
        elif kind == "function_call":
            lines += [
                f"## Tool call: {item.get('name')}",
                "```json",
                str(item.get("arguments", "")),
                "```",
                "",
            ]
        elif kind == "function_call_output":
            metadata = item.get("metadata") or {}
            outcome = metadata.get("outcome", "?")
            duration = metadata.get("duration_s", "?")
            lines.append(f"## Tool result ({outcome}, {duration} s)")
            output = item.get("output")
            if isinstance(output, str):
                lines.append(output)
            else:
                for block in output or []:
                    if block.get("type") == "input_image":
                        lines.append(f"[image: {image_dir}/{block.get('image_url')}]")
                    else:
                        lines.append(str(block.get("text", "")))
            lines.append("")
    return "\n".join(lines)


class SubagentSpawner:
    """Runs batches of episodes on the pool; one instance per parent agent."""

    def __init__(
        self,
        *,
        pool: SimulatorPool,
        parent: ComposedServer,
        workspace: Workspace,
        ledger: CostLedger,
        model_config: OpenAIModelConfig,
        preamble: str,
    ) -> None:
        """Initialize the spawner.

        :param pool: Simulators the episodes run on; `check` must have run
        :param parent: The parent's composed server, whose current descriptions episodes inherit
        :param workspace: The parent's workspace, shared with every episode
        :param ledger: The dollar ledger every episode spends against
        :param model_config: The parent's model, which every episode runs on
        :param preamble: System instructions given to every episode
        """
        self._pool = pool
        self._parent = parent
        self._workspace = workspace
        self._ledger = ledger
        self._model_config = model_config
        self._preamble = preamble
        self._batches_dir = workspace.logs_dir / SUBAGENT_LOGS_DIR
        self.batches_run = 0
        self.episodes_run = 0

    @property
    def model_name(self) -> str:
        """The model every episode runs on, as the results report it."""
        return self._model_config.model_id

    @property
    def permitted_tools(self) -> list[str]:
        """What an episode may be given: everything the pool serves, plus the file tools."""
        return [*self._pool.tool_names, *FILE_TOOL_NAMES]

    def validate(self, episodes: list[Episode]) -> None:
        """Refuse an episode naming a tool outside the permitted set or a missing state file.

        :raises ToolError: Saying which episode and listing every permitted tool by name
        """
        permitted = set(self.permitted_tools)
        for index, episode in enumerate(episodes):
            unknown = [name for name in episode.tools if name not in permitted]
            if unknown:
                raise ToolError(
                    f"Episode {index}: unknown tools {unknown}. An episode may use: "
                    f"{', '.join(self.permitted_tools)}."
                )
            self._initial_state(index, episode)

    def _initial_state(self, index: int, episode: Episode) -> SimState:
        """The episode's starting scene, read from the workspace when given as a path."""
        if isinstance(episode.initial_state, SimState):
            return episode.initial_state
        path = episode.initial_state
        file = self._workspace.resolve(path)
        if not file.is_file():
            raise ToolError(f"Episode {index}: no file at {path!r}.")
        try:
            return SimState.model_validate_json(file.read_text(encoding="utf-8"))
        except ValidationError as exc:
            raise ToolError(
                f"Episode {index}: {path} does not hold a state in the form get_sim_state "
                f"returns: {exc}"
            ) from None

    async def run_batch(
        self, episodes: list[Episode], batch_dir: Path | None = None
    ) -> list[EpisodeResult]:
        """Run every episode, each on a pool simulator, and wait for all of them.

        :param episodes: The episodes to run concurrently
        :param batch_dir: Where their logs go; the next numbered batch directory when omitted
            (the harness passes its experiment's directory)
        """
        self.validate(episodes)
        if batch_dir is None:
            batch_dir = self._batches_dir / f"batch_{self.batches_run:03d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        self.batches_run += 1
        results = await asyncio.gather(
            *(
                self._run_episode(index, episode, batch_dir)
                for index, episode in enumerate(episodes)
            )
        )
        (batch_dir / "results.json").write_text(
            json.dumps([result.model_dump() for result in results], indent=1), encoding="utf-8"
        )
        self.episodes_run += len(results)
        return list(results)

    async def _run_episode(self, index: int, episode: Episode, batch_dir: Path) -> EpisodeResult:
        out_dir = batch_dir / f"trial_{index:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "episode.json").write_text(episode.model_dump_json(indent=1), encoding="utf-8")
        log_path = out_dir / "log.md"
        transcript = Transcript(out_dir=out_dir)
        status = "finished"
        message = ""
        agent: SubagentLoop | None = None

        try:
            state = self._initial_state(index, episode)
            async with self._pool.acquire() as url:
                logger.info(f"Episode {index} of batch {batch_dir.name} on {url}.")
                composed = ComposedServer(
                    f"leap-subagent-{index}", simulator=url, snapshot_dir=None
                )
                register_file_tools(composed.local, self._workspace)
                async with composed:
                    for name, description in self._parent.descriptions().items():
                        if name in composed.tool_names:
                            composed.update_description(name, description)
                    async with MCPClient(composed.server) as client:
                        result = await client.call_tool(
                            SET_STATE_TOOL, {"state": state.model_dump()}
                        )
                        if result.is_error:
                            raise ToolError(
                                "Setting the initial state failed: "
                                + _text_of([b.model_dump() for b in result.content])
                            )
                        composed.set_allowlist(episode.tools)
                        agent = SubagentLoop(
                            client,
                            OpenAIModelClient(config=self._model_config),
                            transcript,
                            ledger=self._ledger,
                        )
                        await agent.enter_loop(
                            task=self._preamble,
                            history=[create_openai_message(episode.prompt, role="user")],
                            max_tool_calls=episode.max_tool_calls,
                        )
            if agent.budget_exceeded:
                status = "budget_exceeded"
            else:
                message = final_message_of(transcript.current_episode)
        except UnpricedModelCallError:
            raise
        except Exception as exc:
            logger.exception(f"Episode {index} of batch {batch_dir.name} failed.")
            status = "error"
            message = f"{type(exc).__name__}: {exc}"

        transcript_path = out_dir / "transcript.jsonl"
        if transcript_path.is_file():
            log_path.write_text(
                render_log(transcript_path, image_dir=self._workspace.relative(out_dir / "images")),
                encoding="utf-8",
            )
        else:
            log_path.write_text(
                "# Episode log\n\nThe episode made no model call.\n", encoding="utf-8"
            )

        return EpisodeResult(
            trial=index,
            status=status,
            final_message=message,
            log_path=self._workspace.relative(log_path),
            tool_calls_used=agent.tool_calls_made if agent else 0,
            cost_usd=round(agent.cost_usd, 4) if agent else 0.0,
            model=self.model_name,
        )


def register_spawn_tool(server: MCPServer, spawner: SubagentSpawner) -> str:
    """Register spawn_subagents on `server` as `RecursiveAgent` declares it.

    :return: The tool's name, for allowlists
    """

    async def spawn_subagents(episodes: list[Episode]) -> str:
        results = await spawner.run_batch(episodes)
        return json.dumps([result.model_dump() for result in results], indent=1)

    return register_declared_tool(server, RecursiveAgent, "spawn_subagents", spawn_subagents)
