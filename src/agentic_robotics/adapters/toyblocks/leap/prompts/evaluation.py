"""Define prompts for the evaluation phase of the LEAP experiments."""

from typing import Literal, get_args

from agentic_robotics.adapters.toyblocks.config import PIXEL_COORDINATE_PROMPT, TaskConfig

EVAL_PROMPT_PREFIX = (
    "You are the brain of an autonomous robot. You act solely through MCP tool calls, "
    "each corresponding to a robot motor skill or perception query. Your goal is to "
    "complete each task while minimizing the total wall clock time, including your "
    "reasoning and skill execution time. Some tasks may require long-horizon reasoning. "
    "Therefore, given a task, be thorough and form a coherent strategy before acting.\n\n"
    "Complete the following task using the available MCP tools:\n\n"
)

EVAL_PROMPT_SUFFIX = PIXEL_COORDINATE_PROMPT + (
    "You have a limited budget of motor skill tool calls available, and will be "
    "reminded of this budget before each conversation turn. When you are done, "
    "you may terminate the episode by returning zero tool calls. Double-check "
    "before exiting. Do not give up; persevere until you've exhausted your budget."
)

AI2_EASY_PROMPT = (
    "Stack the three blocks to spell 'AI2' from top to bottom, "
    "with each letter upright and facing the camera."
)

AI2_STACK_PROMPT = (
    "There are various letters and symbols on the faces of the blocks on the table. "
    "Your task is to use the blocks to spell the word 'AI2' in a tower, with 'A' "
    "on top and '2' on bottom, with all three letters upright and facing the camera. "
)

AI2_STACK_MOTOR_TOOL_CALL_BUDGET = 60

EvalRegion = Literal["feasible", "outside"]
EvalProximity = Literal["isolated", "adjacent"]

LEAP_OPTIMAL_PLAN_LENGTHS: dict[tuple[EvalRegion, EvalProximity], int] = {
    ("feasible", "isolated"): 4,
    ("outside", "isolated"): 6,
    ("feasible", "adjacent"): 8,
    ("outside", "adjacent"): 10,
}
"""Number of motor skills in the optimal plan for the longest of the relevant 'Ai2 easy' tasks."""

LEAP_EVAL_BUDGET_FACTOR = 3
"""During evaluation, permit this many motor skill calls per motor skill in the optimal plan."""


def leap_tasks() -> dict[str, TaskConfig]:
    """Generate task configurations for the LEAP workshop paper tasks."""
    tasks = {}
    for region in get_args(EvalRegion):
        for proximity in get_args(EvalProximity):
            for skill in ("flip", "rotate"):
                optimal_length = LEAP_OPTIMAL_PLAN_LENGTHS[(region, proximity)]

                slug = f"stack_{region}_{proximity}_{skill}"
                tasks[slug] = TaskConfig(
                    name=f"Stack AI2 ({region} region, {proximity}, {skill})",
                    task_slug=slug,
                    prompt=EVAL_PROMPT_PREFIX + AI2_EASY_PROMPT + EVAL_PROMPT_SUFFIX,
                    max_tool_calls=LEAP_EVAL_BUDGET_FACTOR * optimal_length,
                )

    ai2_stack_slug = "ai2_stacked"
    tasks[ai2_stack_slug] = TaskConfig(
        name="Ai2 Stack (w/ pre-stacked distractors)",
        task_slug=ai2_stack_slug,
        prompt=EVAL_PROMPT_PREFIX + AI2_STACK_PROMPT + EVAL_PROMPT_SUFFIX,
        max_tool_calls=AI2_STACK_MOTOR_TOOL_CALL_BUDGET,
    )

    return tasks
