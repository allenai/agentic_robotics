"""The Rao instances as `TaskConfig`s, for `agentic-robotics` to resolve like any other task.

`cli.py`'s `load_tasks` merges this table beside `config.TOYBLOCKS_TASKS` and the LEAP families
from `leap/prompts/evaluation.py`. The closed-loop prompt is the LEAP evaluation prompt around
the instance's goal text, so both modes and the LEAP suites share the role text, the pixel
convention and the "minimize wall-clock time" objective; the online mode's prompt is
`release/rao_prompts.py`'s behind the same preamble, assembled in `release/rao_modes.py`. The
motor-call cutoff is the LEAP rule, `LEAP_EVAL_BUDGET_FACTOR` calls per skill in the optimal
plan, applied to the plan length `tasks/rao_problems.py` computes.

Both the config module and the LEAP prompts module are loaded off disk through
`runner_bootstrap` when a function here is called, the way `load_tasks` reads
`TOYBLOCKS_TASKS`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentic_robotics.adapters.toyblocks.config import TaskConfig
from agentic_robotics.adapters.toyblocks.tasks.rao_problems import RaoProblem, all_problems

if TYPE_CHECKING:
    from types import ModuleType


def leap_prompts_module() -> ModuleType:
    """`leap/prompts/evaluation.py`: the LEAP evaluation prompt and its task sentence."""
    from agentic_robotics.adapters.toyblocks.leap.prompts import evaluation

    return evaluation


def closed_loop_prompt(problem: RaoProblem) -> str:
    """The closed-loop (plan-act) prompt: the LEAP evaluation prompt around the goal text."""
    prompts = leap_prompts_module()
    return prompts.EVAL_PROMPT_PREFIX + problem.goal_text + "\n\n" + prompts.EVAL_PROMPT_SUFFIX


def motor_call_cutoff(problem: RaoProblem) -> int:
    """Motor skill calls allowed: the LEAP factor times the optimal plan length."""
    return leap_prompts_module().LEAP_EVAL_BUDGET_FACTOR * problem.optimal_plan_length


def rao_tasks() -> dict[str, Any]:
    """Every Rao instance as a `TaskConfig`, keyed by slug."""
    tasks = {}
    for problem in all_problems():
        tasks[problem.slug] = TaskConfig(
            name=f"Rao {problem.goal} ({problem.num_blocks} blocks)",
            task_slug=problem.slug,
            prompt=closed_loop_prompt(problem),
            max_tool_calls=motor_call_cutoff(problem),
        )
    return tasks
