"""One Rao episode in one of two modes, run as `agentic-robotics episode --plan-mode <mode>`:

    agentic-robotics episode --task rao_invert_5 --plan-mode closed \\
        --instance 0 --skills inline_oracle --model opus-5.5
    agentic-robotics episode --task rao_invert_5 --plan-mode online ...

`agentic-robotics` owns the boot; this module supplies the agent thread, as `leap.evaluate` does for
the LEAP evaluation. Both modes share the composed server (here only for its allowlist and local
tools; descriptions are the skill server's own, not a LEAP description set), the reset to
`--instance k`, the scoring (`leap.evaluate.score_scene`, the package goal classifier with the
gripper opened, settled by position) and the summary schema; what differs is the loop:

- **closed**: the LEAP `EvaluationLoop`, motor calls capped at the task's cutoff, the agent
  seeing every skill's image result. The plan-act loop.
- **online** ("online refinement"): the agent sees the motor skills' docstrings but
  cannot call them (the LEAP Harness refusal), and submits a symbolic plan, (skill, block
  color, direction); then for each step it is shown the current image and asked for that step's
  pixel through `execute_step`, which forwards to the skill named by the plan. It may `abort`.
  Steps that take no pixel run without asking. The plan length is capped at the task's
  motor-call cutoff, so both modes have the same ceiling on skill calls.

Everything the agent reads comes from `src`: the step model, the three
tools' names, schemas and docstrings (`OnlineRefinementAgent`, registered here through
`register_declared_tool`, which refuses an implementation whose signature drifts) and every
message, from `adapters/toyblocks/release/rao_prompts.py`; the mode prompt is that module's
behind the LEAP evaluation preamble, the way `tasks/rao.py` assembles the closed-loop one.

Perception calls are free up to the LEAP safety ceiling on calls of any kind. `PlanningLoop`
is the base loop's hooks; `OnlineLoop.run` is a loop over plan steps, each a short loop of base
turns, so it uses `take_turn` and `run_tool_calls` directly.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from mcp.server.mcpserver import Image as MCPImage
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import CallToolResult, ImageContent, TextContent
from mcp_types import Tool as MCPTool

from agentic_robotics.adapters.toyblocks.leap.declarations import register_declared_tool
from agentic_robotics.adapters.toyblocks.release.rao_prompts import (
    ABORT_TOOL,
    PLAN_TOOL,
    STEP_TOOL,
    OnlineRefinementAgent,
    OnlineStep,
    no_plan_note,
    no_step_note,
    online_auto_step_note,
    online_final_assessment,
    online_refinement_prompt,
    online_step_status,
    online_step_text,
    plan_accepted,
    planning_status,
)
from agentic_robotics.adapters.toyblocks.tasks.rao_problems import RaoProblem, problem_for
from agentic_robotics.core.agentic_loop import AgenticLoop
from agentic_robotics.core.foundation_models import History, create_openai_message
from agentic_robotics.core.images import PixelCoord  # noqa: TC001 (used for MCP tool registration)
from agentic_robotics.core.mcp_conversion import mcp_result_to_openai_output

if TYPE_CHECKING:
    import threading
    from collections.abc import Callable

    from mcp import Client as MCPClient

    from agentic_robotics.adapters.toyblocks.toyblocks_mcp_server import ToyBlocksMCPServer

logger = logging.getLogger("agentic_robotics.examples.rao.modes")

MODES = ("closed", "online")

MAX_REPROMPTS = 3
"""Turns without a tool call tolerated in a row before a planning or step loop gives up."""


class PlanBox:
    """Holds the submitted plan; shared between `submit_plan` and the loop that waits on it."""

    def __init__(self) -> None:
        self.steps: list[Any] | None = None


class OnlineExecution:
    """The state of a plan's execution, shared with `execute_step` and `abort`."""

    def __init__(self) -> None:
        self.steps: list[Any] = []
        self.index = 0
        self.awaiting = False
        self.records: list[dict[str, Any]] = []
        self.aborted = False
        self.abort_reason: str | None = None


def _enum_values(schema: dict[str, Any], prop: str) -> list[str] | None:
    """The enum a tool's input schema gives for a property, resolving one `$ref`."""
    spec = schema.get("properties", {}).get(prop)
    if spec is None:
        return None
    if "$ref" in spec:
        spec = schema.get("$defs", {}).get(spec["$ref"].split("/")[-1], {})
    for candidate in [spec, *spec.get("anyOf", [])]:
        if "enum" in candidate:
            return list(candidate["enum"])
    return None


def validate_steps(
    steps: list[Any],
    served: dict[str, MCPTool],
    motor: list[str],
    max_steps: int,
    colors: tuple[str, ...],
) -> None:
    """Refuse a plan the executor could not run as written.

    :raises ToolError: Naming the first problem, so the agent can resubmit
    """
    if not steps:
        raise ToolError("The plan is empty; submit at least one step.")
    if len(steps) > max_steps:
        raise ToolError(
            f"The plan has {len(steps)} steps; at most {max_steps} motor skill calls are allowed."
        )
    for index, step in enumerate(steps, start=1):
        if step.skill not in motor:
            raise ToolError(
                f"Step {index}: {step.skill!r} is not a motor skill. "
                f"The motor skills are: {', '.join(motor)}."
            )
        schema = served[step.skill].input_schema
        takes_pixel = "pixel" in schema.get("properties", {})
        directions = _enum_values(schema, "direction")
        if directions and step.direction not in directions:
            raise ToolError(
                f"Step {index}: {step.skill} needs a direction, one of {directions}; "
                f"got {step.direction!r}."
            )
        if not directions and step.direction is not None:
            raise ToolError(f"Step {index}: {step.skill} takes no direction.")
        needs_block = takes_pixel and step.skill != "place_on_table"
        if needs_block and step.block_color is None:
            raise ToolError(f"Step {index}: {step.skill} must name the block it acts on, by color.")
        if step.block_color is not None and step.block_color.lower() not in colors:
            raise ToolError(
                f"Step {index}: no {step.block_color!r} block; the blocks are {', '.join(colors)}."
            )


def result_text(result: CallToolResult) -> str:
    """The text blocks of a tool result, joined."""
    return " ".join(
        block.text for block in result.content if isinstance(block, TextContent)
    ).strip()


def result_image(result: CallToolResult) -> MCPImage | None:
    """The first image block of a tool result, as an MCP image a local tool can return."""
    for block in result.content:
        if isinstance(block, ImageContent):
            return MCPImage(
                data=base64.b64decode(block.data), format=block.mime_type.split("/")[-1]
            )
    return None


def step_arguments(step: OnlineStep, pixel: PixelCoord | None) -> dict[str, Any]:
    """The arguments a skill call takes, from a step and the pixel chosen for it."""
    arguments: dict[str, Any] = {}
    if pixel is not None:
        arguments["pixel"] = pixel.model_dump()
    if step.direction is not None:
        arguments["direction"] = step.direction
    return arguments


class PlanningLoop(AgenticLoop):
    """The planning stage: free perception until the plan is submitted.

    Hooks on the base loop: the loop ends when `box.steps` is set (`should_continue`), when the
    call ceiling is spent (`request_more_tool_calls`), or after `MAX_REPROMPTS` turns in a row
    without a tool call (`confirm_episode_end`); the last two set `gave_up`. There is no
    final-answer turn: planning ends on the submitted plan, and the final answer is asked for
    after execution.
    """

    def __init__(self, *args: Any, box: PlanBox, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._box = box
        self.calls_made = 0
        self.reprompts = 0
        self.gave_up = False
        self._without_calls = 0

    def status_text(self, tool_calls_left: int) -> str:
        return planning_status(tool_calls_left)

    def should_continue(self) -> bool:
        return self._box.steps is None and not self.gave_up

    async def request_more_tool_calls(self) -> int:
        self.gave_up = True
        return 0

    async def run_tool_calls(self, calls: list[Any]) -> list[Any]:
        outputs = await super().run_tool_calls(calls)
        self.calls_made += len(outputs)
        if outputs:
            self._without_calls = 0
        return outputs

    async def confirm_episode_end(self) -> bool:
        self._without_calls += 1
        self.reprompts += 1
        if self._without_calls > MAX_REPROMPTS:
            self.gave_up = True
            return True
        self._transcript.append(create_openai_message(no_plan_note(), role="system"))
        return False

    async def take_final_turn(self, task: str, tools: list[MCPTool]) -> History | None:
        return None


class OnlineLoop(PlanningLoop):
    """The execution stage: one image and one pixel per committed step."""

    def __init__(
        self,
        *args: Any,
        execution: OnlineExecution,
        simulator: MCPClient,
        served: dict[str, MCPTool],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._execution = execution
        self._simulator = simulator
        """An MCP client connected straight to the skill server, for the image and pixel-less
        steps."""
        self._served = served
        """Every served tool, before the allowlist narrowed what the agent sees: the motor
        skills' schemas say which steps take a pixel."""

    async def observation_message(self, text: str) -> dict[str, Any]:
        """A user message carrying `text` and the current image observation."""
        result = await self._simulator.call_tool("get_image_observation", {})
        blocks = mcp_result_to_openai_output(result)
        return {"role": "user", "content": [{"type": "input_text", "text": text}, *blocks]}

    async def run(self, task: str, calls_left: int) -> int:
        """Execute the committed plan step by step; return the calls of any kind left."""
        execution = self._execution
        tools = await self.list_mcp_tools()
        total = len(execution.steps)
        for index, step in enumerate(execution.steps):
            execution.index = index
            if "pixel" not in self._served[step.skill].input_schema.get("properties", {}):
                record = await run_skill(self._simulator, step, None, index)
                record.pop("result", None)
                execution.records.append(record)
                outcome = "succeeded" if record["ok"] else f"failed: {record['message']}"
                note = online_auto_step_note(index + 1, total, step, outcome)
                self._transcript.append(create_openai_message(note, role="system"))
                continue
            self._transcript.append(
                await self.observation_message(online_step_text(index + 1, total, step))
            )
            execution.awaiting = True
            without_calls = 0
            while execution.awaiting and not execution.aborted:
                if calls_left <= 0:
                    execution.aborted = True
                    execution.abort_reason = "call ceiling reached"
                    break
                response = await self.take_turn(
                    task, online_step_status(index + 1, total, calls_left), tools
                )
                made = len(await self.run_tool_calls(self._filter_tool_calls(response)))
                calls_left -= made
                if made:
                    without_calls = 0
                    continue
                without_calls += 1
                self.reprompts += 1
                if without_calls > MAX_REPROMPTS:
                    execution.aborted = True
                    execution.abort_reason = "no tool call in repeated turns"
                    break
                self._transcript.append(create_openai_message(no_step_note(), role="system"))
            if execution.aborted:
                break
        return calls_left


async def run_skill(
    simulator: MCPClient, step: OnlineStep, pixel: PixelCoord | None, index: int
) -> dict[str, Any]:
    """Call one skill on the skill server and record what came back."""
    arguments = step_arguments(step, pixel)
    started = time.monotonic()
    result = await simulator.call_tool(step.skill, arguments)
    return {
        "index": index,
        "skill": step.skill,
        "arguments": arguments,
        "step": step.model_dump(),
        "ok": not result.is_error,
        "message": result_text(result)[:500],
        "seconds": round(time.monotonic() - started, 3),
        "result": result,
    }


def register_online_tools(
    server: MCPServer,
    box: PlanBox,
    execution: OnlineExecution,
    simulator: MCPClient,
    validate: Callable[[list[OnlineStep]], None],
) -> dict[str, str]:
    """Register `submit_plan`, `execute_step` and `abort` as `OnlineRefinementAgent` declares them.

    :param validate: `validate(steps)` raises `ToolError` for a plan the executor could not run
    :return: The tool names by role: `plan`, `step`, `abort`, for the per-stage allowlists
    """

    async def submit_plan(steps: list[OnlineStep]) -> str:
        """Accepted once; see the declaration."""
        if box.steps is not None:
            raise ToolError("A plan was already submitted; it cannot be replaced.")
        validate(steps)
        box.steps = list(steps)
        execution.steps = list(steps)
        return plan_accepted(len(steps), "online-refinement")

    async def execute_step(pixel: PixelCoord) -> MCPImage:
        """Run the plan step awaiting a pixel; see the declaration."""
        if not execution.awaiting:
            raise ToolError("No step is awaiting a pixel right now.")
        step = execution.steps[execution.index]
        execution.awaiting = False
        record = await run_skill(simulator, step, pixel, execution.index)
        result = record.pop("result")
        execution.records.append(record)
        if result.is_error:
            raise ToolError(record["message"] or f"{step.skill} failed.")
        image = result_image(result)
        if image is None:
            raise ToolError(f"{step.skill} returned no image.")
        return image

    async def abort(reason: str) -> str:
        """End execution where it is; see the declaration."""
        execution.aborted = True
        execution.abort_reason = reason
        execution.awaiting = False
        return "Execution aborted. The episode is over."

    register_declared_tool(server, OnlineRefinementAgent, PLAN_TOOL, submit_plan)
    register_declared_tool(server, OnlineRefinementAgent, STEP_TOOL, execute_step)
    register_declared_tool(server, OnlineRefinementAgent, ABORT_TOOL, abort)
    return {"plan": PLAN_TOOL, "step": STEP_TOOL, "abort": ABORT_TOOL}


def online_prompt(problem: RaoProblem, max_steps: int) -> str:
    """The online mode's prompt: the LEAP evaluation preamble, then `src`'s mode prompt around
    the goal text, so both modes open with the same role and objective."""
    from agentic_robotics.adapters.toyblocks.leap.prompts.evaluation import EVAL_PROMPT_PREFIX

    return EVAL_PROMPT_PREFIX + online_refinement_prompt(problem.goal_text, max_steps)


async def run_rao_episode(server: ToyBlocksMCPServer, *, mode: str, instance: int | None) -> None:
    """One episode in `mode`, writing `evaluation_summary.json` (and `execution.json` for the
    online mode) in the run directory."""
    from mcp import Client as MCPClient

    from agentic_robotics.adapters.toyblocks.leap.composed_server import ComposedServer
    from agentic_robotics.adapters.toyblocks.leap.evaluate import (
        make_evaluation_loop_class,
        on_simulator_thread,
        reset_to_instance,
        score_scene,
        tool_call_counts,
    )
    from agentic_robotics.adapters.toyblocks.leap.leap_agents import NoOpMotorSkillsAgent
    from agentic_robotics.adapters.toyblocks.leap.perception_tools_agent import (
        PerceptionToolsAgent,
    )
    from agentic_robotics.adapters.toyblocks.leap.recursive_agent import MAX_EPISODE_TOOL_CALLS
    from agentic_robotics.adapters.toyblocks.leap.timing import tool_call_seconds
    from agentic_robotics.core.foundation_models import OpenAIModelClient, Transcript

    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; one of {MODES}.")
    config = server.config
    problem: RaoProblem = problem_for(config.task_config.task_slug)
    cutoff = config.task_config.max_tool_calls
    model_client = OpenAIModelClient(config=config.model_config)
    transcript = Transcript(out_dir=config.output_dir)
    composed = ComposedServer(f"rao-{mode}", simulator=server.mcp, snapshot_dir=None)
    metadata = lambda: {"frame_count": server.frame_count}  # noqa: E731

    summary: dict[str, Any] = {
        "mode": mode,
        "task": config.task_config.task_slug,
        "goal_family": problem.goal,
        "num_blocks": problem.num_blocks,
        "optimal_plan_length": problem.optimal_plan_length,
        "instance": instance,
        "model": config.model_config.model_id,
        "motor_cutoff": cutoff,
        "finished": False,
        "success": None,
    }
    started = time.time()
    served: dict[str, MCPTool] = {}
    motor: list[str] = []
    box = PlanBox()
    execution = OnlineExecution()

    perception_names = set(PerceptionToolsAgent.__abstractmethods__)
    async with MCPClient(server.mcp) as simulator:
        if mode == "online":
            prompt = online_prompt(problem, cutoff)
            names = register_online_tools(
                composed.local,
                box,
                execution,
                simulator,
                lambda steps: validate_steps(steps, served, motor, cutoff, problem.colors),
            )
            # The planner must know the skills it plans with, so they stay listed with their
            # docstrings while every call is refused, as the LEAP Harness agent's are; the
            # execution stage hides them behind `execute_step`. A planner that cannot see the
            # skills plans blind: a Sonnet planner submitted a one-step plan "because the skill
            # list wasn't visible".
            listing = await simulator.list_tools()
            motor_served = [
                tool.name for tool in listing.tools if tool.name not in perception_names
            ]
            composed.refuse_calls(motor_served, NoOpMotorSkillsAgent.get_refusal_text)
        else:
            prompt = config.task_config.prompt
        summary["prompt"] = prompt

        async with composed:
            skills = composed.skill_tool_names
            served.update({tool.name: tool for tool in composed.visible_tools()})
            perception = sorted(perception_names & set(skills))
            motor.extend(name for name in skills if name not in perception)
            if instance is not None:
                await on_simulator_thread(server, lambda: reset_to_instance(server, instance))
                logger.info(f"Scene reset to reset state {instance}.")
                if hasattr(server, "seed_quirks_for"):
                    # Quirk draws per task and instance, not per process.
                    server.seed_quirks_for(config.task_config.task_slug, instance)
            async with MCPClient(composed.server) as mcp_client:
                try:
                    if mode == "closed":
                        composed.set_allowlist(skills)
                        logger.info(f"Closed loop; tools: {composed.visible_tool_names}")
                        loop = make_evaluation_loop_class()(
                            mcp_client, model_client, transcript, metadata_callback=metadata
                        )
                        await loop.enter_loop(
                            task=prompt,
                            history=[create_openai_message(prompt, role="user")],
                            max_tool_calls=cutoff,
                        )
                        summary["motor_calls_made"] = loop.motor_calls_made
                        summary["hit_cutoff"] = loop.hit_cutoff
                    else:
                        composed.set_allowlist([*perception, *motor, names["plan"]])
                        logger.info(
                            f"Planning stage; tools: {composed.visible_tool_names} "
                            "(motor skills listed, calls refused)"
                        )
                        planner = PlanningLoop(
                            mcp_client,
                            model_client,
                            transcript,
                            metadata_callback=metadata,
                            box=box,
                        )
                        await planner.enter_loop(
                            task=prompt,
                            history=[create_openai_message(prompt, role="user")],
                            max_tool_calls=MAX_EPISODE_TOOL_CALLS,
                        )
                        summary["planning_calls"] = planner.calls_made
                        summary["planning_reprompts"] = planner.reprompts
                        summary["plan_submitted"] = box.steps is not None
                        summary["plan"] = (
                            [step.model_dump() for step in box.steps] if box.steps else None
                        )
                        summary["plan_length"] = len(box.steps) if box.steps else 0
                        calls_left = MAX_EPISODE_TOOL_CALLS - planner.calls_made
                        if box.steps is None:
                            logger.warning("No plan was submitted; nothing is executed.")
                        else:
                            composed.set_allowlist([*perception, names["step"], names["abort"]])
                            logger.info(f"Execution stage; tools: {composed.visible_tool_names}")
                            runner = OnlineLoop(
                                mcp_client,
                                model_client,
                                transcript,
                                metadata_callback=metadata,
                                box=box,
                                execution=execution,
                                simulator=simulator,
                                served=served,
                            )
                            await runner.run(prompt, calls_left)
                            summary["execution_reprompts"] = runner.reprompts
                            await runner.take_turn(
                                prompt,
                                online_final_assessment(prompt),
                                await runner.list_mcp_tools(),
                            )
                        records = execution.records
                        (config.output_dir / "execution.json").write_text(
                            json.dumps(records, indent=1), encoding="utf-8"
                        )
                        summary["steps_executed"] = len(records)
                        summary["steps_failed"] = sum(1 for r in records if not r["ok"])
                        summary["motor_calls_made"] = len(records)
                        summary["hit_cutoff"] = False
                        summary["aborted"] = execution.aborted
                        summary["abort_reason"] = execution.abort_reason
                    summary["finished"] = True
                except Exception as exc:
                    summary["error"] = f"{type(exc).__name__}: {exc}"[:1000]
                    logger.exception("Rao episode failed; ending after cleanup.")
                finally:
                    try:
                        summary.update(
                            await on_simulator_thread(
                                server, lambda: score_scene(server, config.output_dir)
                            )
                        )
                    except Exception:
                        logger.exception("Scoring the final scene failed.")
                    if not summary["finished"]:
                        summary["scene_success"] = summary.get("success")
                        summary["success"] = None
                    transcript_path = config.output_dir / "transcript.jsonl"
                    if transcript_path.is_file():
                        summary["tool_calls"] = tool_call_counts(transcript_path)
                        summary["tool_seconds"] = tool_call_seconds(transcript_path)
                    usage = model_client.usage_history
                    summary["model_calls"] = len(usage)
                    summary["cost_usd"] = round(sum(u.cost_usd or 0.0 for u in usage), 4)
                    summary["model_seconds"] = round(sum(u.duration_s or 0.0 for u in usage), 3)
                    summary["seconds"] = round(time.time() - started, 1)
                    (config.output_dir / "evaluation_summary.json").write_text(
                        json.dumps(summary, indent=1), encoding="utf-8"
                    )
                    logger.info(
                        f"Rao {mode} episode: success={summary['success']}, "
                        f"${summary['cost_usd']}, {summary['seconds']} s."
                    )


def rao_agentic_thread(
    server: ToyBlocksMCPServer, finished: threading.Event, **kwargs: Any
) -> None:
    """Run the episode on this thread, which is not the simulator's; release it when done."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_rao_episode(server, **kwargs))
    finally:
        loop.close()
        finished.set()
