"""The Blocksworld task instances: goals over color cubes, with their optimal plan lengths.

Three goal families, each at five and six blocks ("three high-level
different goals x {5 or 6 blocks}", five sampled reset states each). Every family is a
classical Blocksworld problem: cubes carry no glyphs, so no orientation is asked for and the
skill inventory reduces to `pick` (pick-up or unstack), `stack` and `place_on_table`
(put-down); one classical action is one skill call.

- **tower**: every block starts on the table; the goal is one tower in a given order
  (PlanBench's table-to-stack instances).
- **invert**: two towers of equal height (plus one loose block at five); the goal is both
  towers reversed, the loose block on top of the second. Three table placements the agent
  chooses, which is where blind execution has to predict where it put things.
- **sussman**: the goal is one tower, and the start obstructs it the way the Sussman anomaly
  does: the goal's bottom block sits on its top block, the next-to-bottom on the second, and
  so on for half the blocks. Achieving the goal's conjuncts one at a time clobbers them.

The block order inside a family is a seeded shuffle of the colors, so goal statements are
not alphabetical; the seed is fixed so a regenerated file is identical. Optimal plan lengths
are computed by breadth-first search over the symbolic state (`optimal_plan_length`), not
counted by hand, and the generator refuses an unsolvable instance.

Colors are the cube colors `template/codegen.py` infers from a block's name (red, blue,
green, yellow, purple, orange); names follow `toyblocks.tamp.blocks.distractors.cube_name`
(`block_red1`, `block_blue`, ...). Six blocks add orange, which sits between red and yellow
in hue; whether the camera tells them apart is a thing the first renders should show.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

GOALS = ("tower", "invert", "sussman")
BLOCK_COUNTS = (5, 6)
COLOURS = ("red", "blue", "green", "yellow", "purple", "orange")
"""In the order blocks are taken for a problem of N blocks: the first N."""

TABLE = "workspace_table"
ORIENTATION = "o8"
"""The cube orientation `red_tower_3.pddl` uses; cubes look the same at any yaw."""

SHUFFLE_SEED = 20261013
"""Fixed so a regenerated problem file is byte-identical."""

Pair = tuple[str, str]
"""(upper, lower): the upper color's block rests on the lower color's block."""


def block_name(color: str) -> str:
    """The scene name of a color's cube, as `distractors.cube_name` spells it."""
    return "block_red1" if color == "red" else f"block_{color}"


def optimal_plan_length(
    blocks: tuple[str, ...], init_on: tuple[Pair, ...], goal_on: tuple[Pair, ...]
) -> int:
    """The fewest skill calls (pick, stack, place_on_table) from the start to any goal state.

    Breadth-first search over (on-relation, held block). A block is clear when nothing is on
    it; pick needs an empty hand and a clear block, stack a held block and a clear support,
    place_on_table a held block. The goal holds when every goal pair is in the on-relation and
    the hand is empty.

    :raises ValueError: If no plan exists
    """
    goal = frozenset(goal_on)
    start: tuple[frozenset[Pair], str | None] = (frozenset(init_on), None)

    def successors(
        state: tuple[frozenset[Pair], str | None],
    ) -> Iterator[tuple[frozenset[Pair], str | None]]:
        on, held = state
        covered = {lower for _, lower in on}
        if held is None:
            for block in blocks:
                if block not in covered:
                    yield frozenset(pair for pair in on if pair[0] != block), block
        else:
            yield on, None
            for block in blocks:
                if block != held and block not in covered:
                    yield on | {(held, block)}, None

    distance = {start: 0}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        if state[1] is None and goal <= state[0]:
            return distance[state]
        for nxt in successors(state):
            if nxt not in distance:
                distance[nxt] = distance[state] + 1
                queue.append(nxt)
    raise ValueError(f"No plan reaches {sorted(goal_on)} from {sorted(init_on)}.")


@dataclass(frozen=True)
class BlocksworldProblem:
    """One task instance: a family at a block count, with its symbolic start and goal."""

    goal: str
    """The family: tower, invert or sussman."""

    colors: tuple[str, ...]
    """The blocks in the scene, by color, in the family's (shuffled) order."""

    init_on: tuple[Pair, ...]
    """Stacks in the initial scene; blocks under no pair's upper are on the table."""

    goal_on: tuple[Pair, ...]
    """The goal conjuncts, in the order the goal text states them."""

    @property
    def num_blocks(self) -> int:
        return len(self.colors)

    @property
    def slug(self) -> str:
        """The problem name, file stem and autogen package slug."""
        return f"blocksworld_{self.goal}_{self.num_blocks}"

    @property
    def optimal_plan_length(self) -> int:
        return optimal_plan_length(self.colors, self.init_on, self.goal_on)

    @property
    def goal_text(self) -> str:
        """The goal as the agent reads it, in PlanBench's wording, with the colors present
        named (PlanBench names its objects too) and nothing about where anything starts."""
        present = ", ".join(self.colors[:-1]) + f" and {self.colors[-1]}"
        conjuncts = [
            f"the {upper} block is on top of the {lower} block" for upper, lower in self.goal_on
        ]
        if len(conjuncts) == 1:
            stated = conjuncts[0]
        else:
            stated = ", ".join(conjuncts[:-1]) + f", and {conjuncts[-1]}"
        return (
            f"There are {self.num_blocks} colored cube blocks on the table: {present}. "
            f"Your goal is to have that {stated}. "
            "Nothing else about the final arrangement matters."
        )

    def pddl(self) -> str:
        """The PDDL problem text."""
        names = {color: block_name(color) for color in self.colors}
        uppers = {upper for upper, _ in self.init_on}
        lowers = {lower for _, lower in self.init_on}
        on_table = [c for c in self.colors if c not in uppers]
        clear = [c for c in self.colors if c not in lowers]
        init = [f"        (on_table {names[c]} {TABLE})" for c in on_table]
        init += [f"        (on {names[u]} {names[low]})" for u, low in self.init_on]
        init.append("")
        init += [f"        (clear {names[c]})" for c in clear]
        init += ["", "        (handempty robot)", ""]
        init += [f"        (ori_of {names[c]} {ORIENTATION})" for c in self.colors]
        init += ["", "        (gripper_at_init)"]
        goal = [f"        (on {names[u]} {names[low]})" for u, low in self.goal_on]
        stacks = "; ".join(f"{u} on {low}" for u, low in self.init_on) or "every block on the table"
        return (
            f"(define (problem {self.slug})\n"
            "    (:domain blocks-tamp)\n"
            f"    ; Blocksworld release task, family {self.goal}, "
            f"{self.num_blocks} color cubes.\n"
            f"    ; Start: {stacks}. Optimal plan: {self.optimal_plan_length} skills.\n"
            "    ; Generated by examples/agent/task_gen/generate_blocksworld_problems.py; "
            "regenerate rather than edit.\n"
            "    (:objects\n"
            f"    {' '.join(names[c] for c in self.colors)} - block\n"
            f"    {TABLE} - table\n"
            "    robot - gripper    )\n"
            "    (:init\n" + "\n".join(init) + "\n    )\n"
            "    (:goal (and\n" + "\n".join(goal) + "\n    ))\n"
            ")\n"
        )


def _consecutive(order: tuple[str, ...]) -> tuple[Pair, ...]:
    """The on-pairs of a tower listed top to bottom."""
    return tuple((order[i], order[i + 1]) for i in range(len(order) - 1))


def _shuffled(goal: str, num_blocks: int) -> tuple[str, ...]:
    """The first `num_blocks` colors, shuffled by a seed fixed per (family, count)."""
    colors = list(COLOURS[:num_blocks])
    random.Random(f"{SHUFFLE_SEED}:{goal}:{num_blocks}").shuffle(colors)
    return tuple(colors)


def make_problem(goal: str, num_blocks: int) -> BlocksworldProblem:
    """The instance of a family at a block count."""
    if goal not in GOALS:
        raise ValueError(f"Unknown goal family {goal!r}; one of {GOALS}.")
    if num_blocks not in BLOCK_COUNTS:
        raise ValueError(f"Unsupported block count {num_blocks}; one of {BLOCK_COUNTS}.")
    colors = _shuffled(goal, num_blocks)
    if goal == "tower":
        return BlocksworldProblem(goal, colors, init_on=(), goal_on=_consecutive(colors))
    if goal == "invert":
        half = num_blocks // 2
        first, second, loose = colors[:half], colors[half : 2 * half], colors[2 * half :]
        init_on = _consecutive(first) + _consecutive(second)
        reversed_first, reversed_second = first[::-1], second[::-1]
        goal_on = _consecutive(reversed_first) + _consecutive(loose + reversed_second)
        return BlocksworldProblem(goal, colors, init_on=init_on, goal_on=goal_on)
    # sussman: the goal tower is the color order; the start pairs the ends inward.
    init_on = tuple((colors[num_blocks - 1 - i], colors[i]) for i in range(num_blocks // 2))
    return BlocksworldProblem(goal, colors, init_on=init_on, goal_on=_consecutive(colors))


def all_problems() -> list[BlocksworldProblem]:
    """Every instance: the families crossed with the block counts, in that order."""
    return [make_problem(goal, n) for goal in GOALS for n in BLOCK_COUNTS]


def problem_for(slug: str) -> BlocksworldProblem:
    """The instance a slug names.

    :raises KeyError: If no instance has that slug
    """
    for problem in all_problems():
        if problem.slug == slug:
            return problem
    raise KeyError(slug)
