"""Define prompts for the Blocksworld tasks in the PlanBench-based experiments."""

from abc import ABC, abstractmethod
from typing import Literal

from mcp.server.mcpserver import Image as MCPImage
from pydantic import BaseModel, Field

from agentic_robotics.adapters.toyblocks.config import PIXEL_COORDINATE_PROMPT
from agentic_robotics.core.images import PixelCoord

ExecutionMode = Literal["online-refinement", "closed-loop"]

PLAN_TOOL = "submit_plan"
STEP_TOOL = "execute_step"
ABORT_TOOL = "abort"


class OnlineStep(BaseModel):
    """One step of a committed symbolic plan; its pixel coordinate is selected online."""

    skill: str = Field(description="Name of the motor skill tool to call, exactly as listed.")
    block_color: str | None = Field(
        default=None,
        description="Color of the block the skill acts on. Omit for skills that act on no block.",
    )
    direction: str | None = Field(
        default=None,
        description=(
            "Direction argument of rotate (cw, ccw) or flip (front, back, left, right); "
            "omit otherwise."
        ),
    )
    note: str | None = Field(
        default=None,
        description=(
            "Optional; a few words describing the intent of this step, "
            "read back to you when the step comes up."
        ),
    )

    def as_text(self) -> str:
        """Render the online step as a string for the agent."""
        text = self.skill
        if self.block_color:
            text += f"{' onto' if self.skill == 'stack' else ''} the {self.block_color} block"
        if self.direction:
            text += f", direction {self.direction}"
        if self.note:
            text += f" ({self.note})"
        return text


class OnlineRefinementAgent(ABC):
    """An agent defining the MCP tools needed for the online refinement execution setting."""

    @abstractmethod
    async def submit_plan(self, steps: list[OnlineStep]) -> str:
        """Submit the complete, ordered plan of motor skills to execute.

        Only one plan may be submitted.
        """
        pass

    @abstractmethod
    async def execute_step(self, pixel: PixelCoord) -> MCPImage:
        """Execute the next step of the plan using the given pixel.

        :param pixel: Pixel passed as an argument to the next unexecuted skill in the plan
        :return: Updated image observation after the skill has terminated
        """
        ...

    @abstractmethod
    async def abort(self, reason: str) -> str:
        """Stop executing the plan and abort the current episode."""
        pass


def online_refinement_prompt(goal_text: str, max_steps: int) -> str:
    """Prompt for online refinement tasks, where the agent commits to a high-level plan."""
    return (
        goal_text
        + "\n\n"
        + "You act in two stages. First, inspect the scene using your perception tools and "
        "commit to a high-level plan: an ordered list of motor skills, each naming the "
        "block it acts on by color, and a direction if the skill takes one. Submit your "
        f"plan once using `{PLAN_TOOL}`; it may contain at most {max_steps} steps. "
        "Second, you execute the plan, one step at a time. Before each step, you "
        "will receive the current image observation, and may use perception tools. Then, "
        f"select the raw (u,v) pixel for the step's planned skill using `{STEP_TOOL}`. "
        "You cannot change the plan, reorder it, skip steps, or add steps. If you judge "
        f"that the plan can no longer succeed, call `{ABORT_TOOL}` to end the episode.\n\n"
        + PIXEL_COORDINATE_PROMPT
        + "The episode ends when every step of the plan has run, or when you abort."
    )


def planning_status(calls_left: int) -> str:
    """Build the system message sent to the agent before each planning turn."""
    return (
        f"You have {calls_left} tool calls remaining before you must submit your plan. "
        f"Call `{PLAN_TOOL}` once you are ready."
    )


def no_plan_note() -> str:
    """Warn the agent that the episode won't end until it submits a plan."""
    return f"You made no tool calls. The episode will not end until you call `{PLAN_TOOL}`."


def plan_accepted(num_steps: int, mode: ExecutionMode) -> str:
    """Output from `submit_plan` when the plan is accepted."""
    if mode != "online-refinement":
        raise RuntimeError(f"Unexpected execution mode for `submit_plan`: {mode}")

    return f"Plan of {num_steps} steps accepted. You will be asked for each step's pixel in turn."


def online_step_text(index: int, total: int, step: OnlineStep) -> str:
    """Build the user message, sent with the current image, that asks for a step's pixel."""
    return (
        f"Step {index} of {total}: {step.as_text()}. The current image observation is attached. "
        f"Give this step's pixel using `{STEP_TOOL}`, or call `{ABORT_TOOL}`."
    )


def online_step_status(index: int, total: int, calls_left: int) -> str:
    """Build the system message sent to the agent before each online step."""
    return (
        f"Awaiting the pixel for step {index} of {total}. You have {calls_left} tool calls of "
        f"any kind remaining in this episode. Call `{STEP_TOOL}` or `{ABORT_TOOL}`."
    )


def online_auto_step_note(index: int, total: int, step: OnlineStep, outcome: str) -> str:
    """Note to the agent whenever a step that takes no pixel was run without its input."""
    return (
        f"Step {index} of {total} ({step.as_text()}) needed no pixel input and was run: {outcome}"
    )


def no_step_note() -> str:
    """Note to the agent when an online step's turn made no tool call."""
    return f"You made no tool call. Call `{STEP_TOOL}` with this step's pixel, or `{ABORT_TOOL}`."


def online_final_assessment(task: str) -> str:
    """Ask the agent for a final answer at the end of an episode."""
    return (
        f"Recall your task:\n\n{task}\n\nExecution has ended. Respond with your final answer: "
        "whether you believe that the goal was achieved, and why."
    )
