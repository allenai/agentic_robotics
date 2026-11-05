"""The Blocksworld problems as a PlanBench domain: its config schema, its PDDL, its text, its plan parser.

The domain implements PlanBench's interfaces rather than recreating their Blocksworld domain,
except where the two are one to one, and those interfaces omit images entirely, because
PlanBench's do not support them. The skills are PlanBench's four Blocksworld
operators one to one (`pick` = pick-up from the table or unstack from a block, `stack`,
`place_on_table` = put-down), so the domain description is theirs verbatim and only the objects
differ: six colors, one of them (purple) absent from PlanBench's twelve.

Mirrors, from karthikv792/LLMs-Planning `plan-bench/` (main at its 2025-09-17 commit): `configs/blocksworld.yaml` (`domain_intro`, `actions`, `predicates`,
`predicate_mapping`, `encoded_objects`), `utils/pddl_to_text.py` (`parse_problem`: atoms sorted by
predicate name then argument names, joined ", " with " and " before the last; `fill_template`:
the `[STATEMENT]` block, then `replace("-", " ")` and `replace("ontable", "on the table")`),
`prompt_generation.py` `task_1_plan_generation` (one-shot: the domain intro, one solved example,
then the query ending at `[PLAN]`), and `utils/text_to_pddl.py` `text_to_plan_blocksworld` (a line
is an action when it contains the action's first word, objects in sentence order). Checking
replaces VAL with a replay of the same four operators, and optimality the Fast Downward plan with
`problems.optimal_plan_length`'s breadth-first search: equivalent in this domain.

The strings the model reads are drafts for a pilot; before a full run they belong in `src/`.

    agentic-robotics-planbench-domain write --out <dir>   # PlanBench layout
    agentic-robotics-planbench-domain prompt --instance 2  # one t1 query
"""

from __future__ import annotations

import argparse
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from agentic_robotics.adapters.toyblocks.tasks.blocksworld_problems import (
    COLOURS,
    BlocksworldProblem,
    Pair,
    all_problems,
)

DOMAIN_NAME = "blocksworld"
"""Contains "blocksworld", which is what PlanBench's code branches on for this domain's text."""

DOMAIN_INTRO = """I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do

Pick up a block
Unstack a block from on top of another block
Put down a block
Stack a block on top of another block

I have the following restrictions on my actions:
I can only pick up or unstack one block at a time.
I can only pick up or unstack a block if my hand is empty.
I can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.
I can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.
I can only unstack a block from on top of another block if the block I am unstacking is clear.
Once I pick up or unstack a block, I am holding the block.
I can only put down a block that I am holding.
I can only stack a block on top of another block if I am holding the block being stacked.
I can only stack a block on top of another block if the block onto which I am stacking the block is clear.
Once I put down or stack a block, my hand becomes empty.
Once you stack a block on top of a second block, the second block is no longer clear.
"""
"""PlanBench's `domain_intro`, verbatim (its YAML block scalar, so one trailing newline)."""

ACTIONS = {
    "pick-up": "pick up the {}",
    "put-down": "put down the {}",
    "stack": "stack the {} on top of the {}",
    "unstack": "unstack the {} from on top of the {}",
}
PREDICATES = {
    "ontable": "the {} is on the table",
    "clear": "the {} is clear",
    "handempty": "the hand is empty",
    "holding": "the hand is currently holding {}",
    "on": "the {} is on top of the {}",
}
PREDICATE_MAPPING = {
    "ontable": "on the table",
    "clear": "clear",
    "handempty": "hand is empty",
    "holding": "holding",
    "on": "on top of",
}

LETTERS = "abcdef"
ENCODED_OBJECTS = {
    letter: f"{colour} block" for letter, colour in zip(LETTERS, COLOURS, strict=True)
}
"""PDDL object name to text, alphabetical keys as PlanBench requires: a red, b blue, c green,
d yellow, e purple, f orange (PlanBench's colour order)."""
LETTER_OF = {colour: letter for letter, colour in zip(LETTERS, COLOURS, strict=True)}

DOMAIN_PDDL = """(define (domain blocksworld-4ops)
  (:requirements :strips)
  (:predicates (clear ?x) (ontable ?x) (handempty) (holding ?x) (on ?x ?y))
  (:action pick-up
    :parameters (?ob)
    :precondition (and (clear ?ob) (ontable ?ob) (handempty))
    :effect (and (holding ?ob) (not (clear ?ob)) (not (ontable ?ob)) (not (handempty))))
  (:action put-down
    :parameters (?ob)
    :precondition (holding ?ob)
    :effect (and (clear ?ob) (handempty) (ontable ?ob) (not (holding ?ob))))
  (:action stack
    :parameters (?ob ?underob)
    :precondition (and (clear ?underob) (holding ?ob))
    :effect (and (handempty) (clear ?ob) (on ?ob ?underob) (not (clear ?underob)) (not (holding ?ob))))
  (:action unstack
    :parameters (?ob ?underob)
    :precondition (and (on ?ob ?underob) (clear ?ob) (handempty))
    :effect (and (holding ?ob) (clear ?underob) (not (on ?ob ?underob)) (not (clear ?ob)) (not (handempty)))))
"""
"""The standard four-operator Blocksworld that PlanBench's `generated_domain.pddl` is."""


@dataclass(frozen=True)
class Instance:
    """One PlanBench instance: an id, the Blocksworld problem behind it, and its atoms over letters."""

    instance_id: int
    problem: BlocksworldProblem

    def _atoms(self, on: tuple[Pair, ...], *, full_state: bool) -> list[tuple[str, ...]]:
        """Ground atoms as (predicate, *letters). The goal lists only `on`, as PlanBench's do."""
        atoms = [("on", LETTER_OF[u], LETTER_OF[lo]) for u, lo in on]
        if full_state:
            uppers, lowers = {u for u, _ in on}, {lo for _, lo in on}
            atoms += [("ontable", LETTER_OF[c]) for c in self.problem.colors if c not in uppers]
            atoms += [("clear", LETTER_OF[c]) for c in self.problem.colors if c not in lowers]
            atoms.append(("handempty",))
        return atoms

    @property
    def init_atoms(self) -> list[tuple[str, ...]]:
        return self._atoms(self.problem.init_on, full_state=True)

    @property
    def goal_atoms(self) -> list[tuple[str, ...]]:
        return self._atoms(self.problem.goal_on, full_state=False)

    def pddl(self) -> str:
        """The instance in PlanBench's problem form, over objects a..f."""
        objects = " ".join(LETTER_OF[c] for c in self.problem.colors)
        init = " ".join(f"({' '.join(atom)})" for atom in self.init_atoms)
        goal = " ".join(f"({' '.join(atom)})" for atom in self.goal_atoms)
        return (
            f"(define (problem BW-rand-{self.problem.num_blocks})\n(:domain blocksworld-4ops)\n"
            f"(:objects {objects} )\n(:init\n{init}\n)\n(:goal\n(and\n{goal})\n)\n)\n"
            f"; {self.problem.slug}\n"
        )


def instances() -> list[Instance]:
    """The six Blocksworld problems as PlanBench instances 1-6, in `all_problems()` order."""
    return [Instance(i + 1, problem) for i, problem in enumerate(all_problems())]


# -- text, as `pddl_to_text.py` writes it ---------------------------------------------------------


def _sort_key(atom: tuple[str, ...]) -> str:
    """`get_sorted`'s key: predicate name, a space, then the argument names joined by spaces."""
    return atom[0] + " " + " ".join(atom[1:])


def _atoms_text(atoms: list[tuple[str, ...]]) -> str:
    """`parse_problem.parse`: each atom through `predicates`, joined ", " with " and " last."""
    phrases = [
        PREDICATES[atom[0]].format(*(ENCODED_OBJECTS[arg] for arg in atom[1:]))
        for atom in sorted(atoms, key=_sort_key)
    ]
    return phrases[0] if len(phrases) == 1 else ", ".join(phrases[:-1]) + f" and {phrases[-1]}"


def plan_text(plan: list[tuple[str, ...]]) -> str:
    """A plan as `instance_to_text` writes the example's: one action a line, then `[PLAN END]`."""
    lines = [ACTIONS[name].format(*(ENCODED_OBJECTS[arg] for arg in args)) for name, *args in plan]
    return "\n" + "".join(f"{line}\n" for line in lines) + "[PLAN END]\n"


def fill_template(init: str, goal: str, plan: str) -> str:
    """`fill_template` for a blocksworld domain, with `instruction=False`."""
    text = ""
    if init:
        text += "\n[STATEMENT]\n" + f"As initial conditions I have that, {init.strip()}."
    if goal:
        text += f"\nMy goal is to have that {goal}."
    text += f"\n\nMy plan is as follows:\n\n[PLAN]{plan}"
    return text.replace("-", " ").replace("ontable", "on the table")


def statement(instance: Instance, *, with_plan: bool) -> str:
    """One instance as text: the example carries its optimal plan, the query ends at `[PLAN]`."""
    plan = plan_text(optimal_plan(instance)) if with_plan else ""
    return fill_template(_atoms_text(instance.init_atoms), _atoms_text(instance.goal_atoms), plan)


def t1_query(query: Instance, example: Instance) -> str:
    """`task_1_plan_generation`'s one-shot prompt: domain intro, the solved example, the query."""
    return DOMAIN_INTRO + statement(example, with_plan=True) + statement(query, with_plan=False)


# -- plans: search, parsing, checking -----------------------------------------------------------

State = tuple[frozenset[tuple[str, str]], str | None]
"""(on-pairs over letters, held letter or None)."""


def _initial_state(instance: Instance) -> State:
    return frozenset((LETTER_OF[u], LETTER_OF[lo]) for u, lo in instance.problem.init_on), None


def _apply(state: State, action: tuple[str, ...], blocks: set[str]) -> State | None:
    """The four operators' preconditions and effects; None when the action is inapplicable."""
    on, held = state
    covered = {lower for _, lower in on}
    uppers = {upper for upper, _ in on}
    name, *args = action
    if any(arg not in blocks for arg in args):
        return None
    if name == "pick-up" and held is None and len(args) == 1:
        block = args[0]
        return (on, block) if block not in covered and block not in uppers else None
    if name == "unstack" and held is None and len(args) == 2:
        block, under = args
        return (
            (on - {(block, under)}, block)
            if (block, under) in on and block not in covered
            else None
        )
    if name == "put-down" and len(args) == 1 and held == args[0]:
        return on, None
    if (
        name == "stack"
        and len(args) == 2
        and held == args[0]
        and args[1] != held
        and args[1] not in covered
    ):
        return on | {(held, args[1])}, None
    return None


def _goal_met(state: State, instance: Instance) -> bool:
    goal = {(LETTER_OF[u], LETTER_OF[lo]) for u, lo in instance.problem.goal_on}
    return state[1] is None and goal <= state[0]


def optimal_plan(instance: Instance) -> list[tuple[str, ...]]:
    """A shortest plan by breadth-first search, in PlanBench's four operators."""
    blocks = {LETTER_OF[c] for c in instance.problem.colors}
    start = _initial_state(instance)
    parent: dict[State, tuple[State, tuple[str, ...]] | None] = {start: None}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        if _goal_met(state, instance):
            plan: list[tuple[str, ...]] = []
            while parent[state] is not None:
                state, action = parent[state]  # type: ignore[misc]
                plan.append(action)
            return plan[::-1]
        on, held = state
        candidates: list[tuple[str, ...]] = []
        if held is None:
            for block in sorted(blocks):
                below = [lo for u, lo in on if u == block]
                candidates.append(("unstack", block, below[0]) if below else ("pick-up", block))
        else:
            candidates.append(("put-down", held))
            candidates += [("stack", held, block) for block in sorted(blocks) if block != held]
        for action in candidates:
            nxt = _apply(state, action, blocks)
            if nxt is not None and nxt not in parent:
                parent[nxt] = (state, action)
                queue.append(nxt)
    raise ValueError(f"No plan for {instance.problem.slug}.")


def parse_plan(text: str) -> list[tuple[str, ...]]:
    """`text_to_plan_blocksworld`: lines naming an action word, objects in sentence order.

    Their parser lowercases the text, replaces each action's text word ("pick up", "put down",
    "stack", "unstack") with the PDDL name, takes a line as an action when a PDDL name is one of
    its words (the first in `ACTIONS` order wins), and reads the objects from the line in the
    order they occur, skipping a line whose object count is wrong. It stops at `[COST]`. It
    reads the whole reply; a model that writes `[PLAN END]` and then commentary has that
    commentary parsed too, as in PlanBench.
    """
    text = text.lower().strip()
    for name in ACTIONS:  # pick-up, put-down, stack, unstack: the order PlanBench's dict has
        word = ACTIONS[name].split(" ")[0]
        text_word = name.replace("-", " ") if word in name else word
        text = text.replace(text_word, name)
    names = list(ACTIONS)
    object_names = list(ENCODED_OBJECTS.values())
    letter_of_name = {v: k for k, v in ENCODED_OBJECTS.items()}
    plan: list[tuple[str, ...]] = []
    for line in (line.strip() for line in text.split("\n")):
        if "[cost]" in line:
            break
        words = line.split()
        hits = [name for name in names if name in words]
        if not hits:
            continue
        action = hits[0]
        arity = ACTIONS[action].count("{}")
        found = sorted(
            (m.start(), name) for name in object_names for m in re.finditer(re.escape(name), line)
        )
        objects = [letter_of_name[name] for _, name in found]
        if len(objects) != arity:
            continue
        plan.append((action, *objects))
    return plan


@dataclass(frozen=True)
class Verdict:
    """How a parsed plan fares: valid (every step applicable and the goal reached), its length,
    the optimum, and the first failing step (1-based) if any."""

    valid: bool
    length: int
    optimal_length: int
    first_failure: int | None

    @property
    def optimal(self) -> bool:
        return self.valid and self.length == self.optimal_length


def check(instance: Instance, plan: list[tuple[str, ...]]) -> Verdict:
    """Replay `plan` from the instance's start; VAL's verdict in this domain."""
    blocks = {LETTER_OF[c] for c in instance.problem.colors}
    state = _initial_state(instance)
    optimum = instance.problem.optimal_plan_length
    for step, action in enumerate(plan, start=1):
        nxt = _apply(state, action, blocks)
        if nxt is None:
            return Verdict(False, len(plan), optimum, step)
        state = nxt
    met = _goal_met(state, instance)
    return Verdict(met, len(plan), optimum, None if met else len(plan) + 1)


def config_yaml() -> str:
    """This domain's `configs/<domain>.yaml`, in PlanBench's schema."""
    indent = lambda text: "\n".join(f"   {line}" if line else "" for line in text.splitlines())  # noqa: E731
    pairs = lambda d: ",\n   ".join(f'"{k}": "{v}"' for k, v in d.items())  # noqa: E731
    return (
        f"instance_dir: {DOMAIN_NAME}/generated_basic\n"
        f"domain_file: {DOMAIN_NAME}/generated_domain.pddl\n"
        f"domain_name: {DOMAIN_NAME}\nn_instances: {len(instances())}\nstart: 1\nend: {len(instances())}\n"
        "instances_template: instance-{}.pddl\n"
        f"domain_intro: |\n{indent(DOMAIN_INTRO)}\n\n"
        f"actions:\n  {{{pairs(ACTIONS)}}}\n\n"
        f"predicates:\n  {{{pairs(PREDICATES)}}}\n\n"
        f"predicate_mapping:\n  {{{pairs(PREDICATE_MAPPING)}}}\n\n"
        "# encoded_objects: dictionary of object names, have to be alphabetical\n"
        f"encoded_objects:\n  {{{pairs(ENCODED_OBJECTS)}}}\n"
    )


def write_layout(out: Path) -> None:
    """Write `configs/`, `instances/<domain>/generated_basic/` and the domain, as PlanBench lays them out."""
    (out / "configs").mkdir(parents=True, exist_ok=True)
    (out / "configs" / f"{DOMAIN_NAME}.yaml").write_text(config_yaml(), encoding="utf-8")
    basic = out / "instances" / DOMAIN_NAME / "generated_basic"
    basic.mkdir(parents=True, exist_ok=True)
    (out / "instances" / DOMAIN_NAME / "generated_domain.pddl").write_text(
        DOMAIN_PDDL, encoding="utf-8"
    )
    for instance in instances():
        (basic / f"instance-{instance.instance_id}.pddl").write_text(
            instance.pddl(), encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    writer = sub.add_parser("write")
    writer.add_argument("--out", type=Path, required=True)
    prompter = sub.add_parser("prompt")
    prompter.add_argument("--instance", type=int, required=True, help="Query instance id, 1-6")
    prompter.add_argument(
        "--example",
        type=int,
        default=None,
        help="Example instance id (default: the previous one, cyclic)",
    )
    sub.add_parser("selftest")
    args = parser.parse_args()
    by_id = {instance.instance_id: instance for instance in instances()}
    if args.command == "write":
        write_layout(args.out)
        print(f"Wrote the {DOMAIN_NAME} layout under {args.out}.")
    elif args.command == "prompt":
        example = args.example or (args.instance - 2) % len(by_id) + 1
        print(t1_query(by_id[args.instance], by_id[example]))
    else:
        for instance in instances():
            plan = optimal_plan(instance)
            verdict = check(instance, parse_plan(plan_text(plan)))
            assert verdict.optimal and verdict.length == instance.problem.optimal_plan_length, (
                instance.problem.slug,
                verdict,
            )
            print(
                f"{instance.instance_id} {instance.problem.slug}: optimal {verdict.length}, round-trips through text"
            )


if __name__ == "__main__":
    main()
