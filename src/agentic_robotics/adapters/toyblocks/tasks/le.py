"""One training scene of the LE checkpoint's dataset as a `TaskConfig`, for `agentic-robotics` to
resolve.

The minimum viable test of a ManiFlow checkpoint is a skill on a scene from its own training
distribution. `generate_task_from_dataset.py` builds such a package but samples no reset states,
so instead the env packages of some of the dataset's 1000 training problems are copies of the
dataset's own (with their 32 reset states and scene profiles) in `data/tasks/<problem>/`, with
each problem's PDDL in `data/problems/`. This table gives those packages task keys; `cli.py`'s
`load_tasks` merges it the way it merges the Rao instances. No agent runs on it, so its prompt is
empty and the budget is nominal; a skill probe drives it through `agentic-robotics pool`.
"""

from __future__ import annotations

from typing import Any

from agentic_robotics.adapters.toyblocks.config import TaskConfig

LE_TRAINING_PROBLEMS = [
    # letters c,l,q,u,w,x,z + 3 color cubes; u, l, z start stacked
    "problem_declarative_10_20261006_175602_738710_2591",
    # letters f,l,m,p,q,s clear on the table
    "problem_declarative_10_20261006_175604_653245_4117",
]
"""Training problems of `oct06_stack3_1000x32` whose env packages are installed here."""

LETTER_PROBE_TASK = "letter_probe"
"""Two scenes (`data/problems/letter_probe/`) of twelve letter blocks on the table, each at a
random orientation family: the six letters LEA's training data never names (a, b, i, o, r, t) and
six trained letters drawn as the control (d, j, n, u, v, y). For the per-letter skill probe; built
with `agentic-robotics generate-env --task letter_probe --usd-library train_blocks_large`."""


def le_tasks() -> dict[str, Any]:
    """The installed training scenes as `TaskConfig`s, keyed by problem slug."""
    return {
        slug: TaskConfig(
            name=f"LE training scene {slug.rsplit('_', 1)[-1]}",
            task_slug=slug,
            prompt="",
            max_tool_calls=1,
        )
        for slug in LE_TRAINING_PROBLEMS
    }


def letter_probe_tasks() -> dict[str, Any]:
    """The letter-probe task, whose two variant PDDLs become `letter_probe_a` and
    `letter_probe_b`."""
    return {
        LETTER_PROBE_TASK: TaskConfig(
            name="Letter probe: untrained letters beside trained ones",
            task_slug=LETTER_PROBE_TASK,
            prompt="",
            max_tool_calls=1,
        )
    }
