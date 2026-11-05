"""Write the hand-holding setting's example tasks: one per evaluation variant, in the phase's scene.

    # Inside the container (reset_states.pkl holds numpy arrays):
    python examples/agent/handholding_tasks.py

The hand-holding experimentation phase starts with one example task
instance from each variant used in evaluation. The phase's simulators all run `ai2_stacked_09`, so
each example is that variant's reset state `--instance` (the suites use 0 and 1) carried into the
`ai2_stacked_09` scene: the stack families hold only blocks A, I and 2, all present there, and the
other blocks are left out as in `three_blocks.json`, whose robot base and configuration every
example takes. The 49th is the cluttered `ai2_stacked_09` state with the Ai2 Stack instruction.
Each file is `{"task": <instruction>, "motor_call_budget": <evaluation cap>, "state": <SimState>}`,
the budget from the evaluation's own `leap_tasks()`. Files are `example_00` to `example_48` in one
seeded shuffle: neither the name nor the number says the family; the key is
written beside the directory. `agentic-robotics leap --example-tasks DIR` copies the directory into the workspace's
read-only `logs/example_tasks/`.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parents[1]
sys.path.insert(0, str(EXAMPLES_DIR))

from agentic_robotics.adapters.toyblocks import task_variants
from agentic_robotics.adapters.toyblocks.leap.prompts.evaluation import (
    AI2_EASY_PROMPT,
    AI2_STACK_PROMPT,
    leap_tasks,
)

TEMPLATE = REPO_ROOT / "data" / "example_states" / "three_blocks.json"
"""A state read back from `ai2_stacked_09` holding only A, I and 2; its robot pose is reused."""
CLUTTERED = REPO_ROOT / "data" / "example_states" / "ai2_stacked.json"
"""The `ai2_stacked_09` reset state with all its blocks: the one Ai2 Stack example."""

TASKS_DIR = REPO_ROOT / "data" / "tasks"
OUT_DIR = EXAMPLES_DIR / "data" / "example_tasks"
KEY_FILE = EXAMPLES_DIR / "data" / "example_tasks_key.json"
"""Which variant each example file is; kept outside `OUT_DIR` so it never reaches the workspace."""
SHUFFLE_SEED = 0
BLOCKS = ("block_a", "block_i", "block_2")
STACK_FAMILIES = tuple(
    f"stack_{region}_{proximity}_{skill}"
    for region in ("feasible", "outside")
    for proximity in ("isolated", "adjacent")
    for skill in ("flip", "rotate")
)


def block_pose(root_pose: Any) -> dict[str, Any]:
    """A SimState pose from an Isaac root pose row `[x, y, z, qw, qx, qy, qz]`."""
    x, y, z, qw, qx, qy, qz = (round(float(v), 5) for v in root_pose)
    return {
        "position": {"x": x, "y": y, "z": z},
        "orientation": {"w": qw, "x": qx, "y": qy, "z": qz},
        "parent_frame": "world",
    }


def example_task(
    variant: str, instance: int, template: dict[str, Any], budget: int
) -> dict[str, Any]:
    """The example file for one variant: its task instruction and reset state `instance`."""
    with open(TASKS_DIR / variant / "reset_states.pkl", "rb") as f:
        objects = pickle.load(f)["states"]["rigid_object"]
    names = set(objects) - {"workspace_table"}
    if names != set(BLOCKS):
        raise SystemExit(
            f"{variant} holds {sorted(names)}, not {list(BLOCKS)}; it cannot be carried over."
        )
    state = dict(template)
    state["objects"] = [
        {"name": name, "pose": block_pose(objects[name]["root_pose"][instance])} for name in BLOCKS
    ]
    return {"task": AI2_EASY_PROMPT, "motor_call_budget": budget, "state": state}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--instance", type=int, default=2, help="Reset-state index to use (the suites use 0 and 1)"
    )
    parser.add_argument(
        "--out", type=Path, default=OUT_DIR, help="Directory to write the example files into"
    )
    args = parser.parse_args()

    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    for stale in args.out.glob("*.json"):
        stale.unlink()
    tasks = leap_tasks()
    examples: list[tuple[str, dict[str, Any]]] = []
    for family in STACK_FAMILIES:
        for path in sorted((task_variants.PROBLEMS_DIR / family).glob("*.pddl")):
            budget = tasks[family].max_tool_calls
            examples.append(
                (
                    f"{path.stem} instance {args.instance}",
                    example_task(path.stem, args.instance, template, budget),
                )
            )
    cluttered = {
        "task": AI2_STACK_PROMPT.strip(),
        "motor_call_budget": tasks["ai2_stacked"].max_tool_calls,
        "state": json.loads(CLUTTERED.read_text(encoding="utf-8")),
    }
    examples.append(("ai2_stacked_09, data/example_states/ai2_stacked.json", cluttered))
    random.Random(SHUFFLE_SEED).shuffle(examples)
    key: dict[str, str] = {}
    for index, (source, example) in enumerate(examples):
        name = f"example_{index:02d}"
        (args.out / f"{name}.json").write_text(
            json.dumps(example, indent=1) + "\n", encoding="utf-8"
        )
        key[name] = source
    KEY_FILE.write_text(json.dumps(key, indent=1) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(key)} example tasks (instance {args.instance}) to {args.out}; key in {KEY_FILE}"
    )


if __name__ == "__main__":
    main()
