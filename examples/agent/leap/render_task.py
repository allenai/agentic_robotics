"""Render the front camera's view of a task variant's sampled reset states to PNG files.

    python examples/agent/leap/render_task.py \\
        --pddl data/problems/side_effects/side_effects_00.pddl --states 4 \\
        --out-dir <renders dir>

Builds the variant's autogen package when it is missing (`--generate-env`; that is Stage-1
codegen, one Isaac boot in a subprocess), then boots Isaac again to load the scene the way an
episode would, sets reset states 0..N-1 in turn, settles physics, and writes one PNG each at the
camera's own resolution, named `<slug>_state<NN>.png`. No skill server and no policy server:
nothing here runs a skill, so it works for a task that `config.py` does not register yet.

`--check-goal` also runs the package's generated PDDL-goal success classifier
(`_compute_intermediate_values` and `evaluate` in its `env.py`) on each rendered state and logs
`success` and every conjunct, so a package whose classifier cannot run fails here rather than at
the end of an evaluation episode. A freshly reset state is expected to read false.

`--solved` (with `--check-goal`) then writes a solved configuration over each reset state and
classifies that too, which is the check that a classifier can read *true*: the goal's
`(on A B)` literals give the stack order, its `(ori_of X oN)` literals give each block's
orientation from the adapter's `ORI_QUATERNIONS` table, the stack sits where the bottom block already is, and
the block height is measured from the reset state's own tower (two blocks within 2 cm in x-y and
2 to 9 cm apart in z) rather than assumed. The state goes in through `leap.sim_state`,
the same path `set_sim_state` uses, so physics settles it before the classifier reads it.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from agentic_robotics.adapters.toyblocks import runner_bootstrap as boot
from agentic_robotics.adapters.toyblocks import task_variants

logger = logging.getLogger("agentic_robotics.examples.leap.render_task")

REEXEC_MARKER = "AGENTIC_ROBOTICS_REEXECED"
DEFAULT_GLYPH_LIBRARY = "train_blocks_large"
"""What `agentic-robotics` builds against when a task names no library; kept in step with it by hand."""


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pddl", type=Path, required=True, help="The variant's problem file")
    parser.add_argument("--states", type=int, default=4, help="Reset states to render, from 0")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--python", default=None, help="Interpreter with Isaac, if not on PATH")
    parser.add_argument("--generate-env", action="store_true")
    parser.add_argument("--force", action="store_true", help="Regenerate an existing package")
    parser.add_argument("--reset-states", type=int, default=16)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--usd-library", default=DEFAULT_GLYPH_LIBRARY)
    parser.add_argument("--runs-dir", default=None, help="Where the run directory is created")
    parser.add_argument(
        "--experiment", default="default", help="Experiment whose settings the render uses"
    )
    parser.add_argument("--verbose-isaac", action="store_true")
    parser.add_argument(
        "--check-goal",
        action="store_true",
        help="Run the package's PDDL-goal success classifier on each rendered state and log it",
    )
    parser.add_argument(
        "--solved",
        action="store_true",
        help="With --check-goal: also write the goal configuration over each state and classify it",
    )
    parser.add_argument(
        "--settle-syncs",
        type=int,
        default=5,
        help="Extra simulator syncs after writing the solved state, so the classifier's settled "
        "check (block velocities near zero) sees a stack at rest; a single sync can leave one "
        "still moving",
    )
    return parser.parse_args()


def goal_literals(pddl_text: str) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """The goal's `(on top bottom)` pairs and `{block: orientation}` from a problem file."""
    goal = pddl_text[pddl_text.index("(:goal") :]
    on_pairs = re.findall(r"\(on\s+(\S+)\s+(\S+)\)", goal)
    orientations = dict(re.findall(r"\(ori_of\s+(\S+)\s+(o\d+)\)", goal))
    return on_pairs, orientations


def stack_order(on_pairs: list[tuple[str, str]]) -> list[str]:
    """Blocks from the bottom up, following the `on` chain."""
    above = dict((bottom, top) for top, bottom in on_pairs)
    tops = {top for top, _ in on_pairs}
    bottoms = [bottom for _, bottom in on_pairs if bottom not in tops]
    if len(bottoms) != 1:
        raise ValueError(f"The goal's on-literals {on_pairs} do not form one stack.")
    order = [bottoms[0]]
    while order[-1] in above:
        order.append(above[order[-1]])
    return order


def measured_block_height(state: Any) -> float:
    """The z gap between two stacked blocks in the state: the tower the scene starts with."""
    objects = state.objects
    for a in objects:
        for b in objects:
            pa, pb = a.pose.position, b.pose.position
            if (
                a is not b
                and abs(pa.x - pb.x) < 0.02
                and abs(pa.y - pb.y) < 0.02
                and 0.02 < pa.z - pb.z < 0.09
            ):
                return round(pa.z - pb.z, 4)
    raise ValueError("No stacked pair in the reset state to measure the block height from.")


def solved_state(state: Any, on_pairs: list[tuple[str, str]], orientations: dict[str, str]) -> Any:
    """The state with the goal's stack built where its bottom block is, at the goal orientations."""
    from agentic_robotics.adapters.toyblocks.block_orientations import ORI_QUATERNIONS
    from agentic_robotics.core.states import Pose, Position, Quaternion

    order = stack_order(on_pairs)
    height = measured_block_height(state)
    by_name = {obj.name: obj for obj in state.objects}
    base = by_name[order[0]].pose.position
    solved = state.model_copy(deep=True)
    for obj in solved.objects:
        if obj.name not in order and obj.name not in orientations:
            continue
        position = obj.pose.position
        if obj.name in order:
            position = Position(x=base.x, y=base.y, z=base.z + height * order.index(obj.name))
        orientation = obj.pose.orientation
        if obj.name in orientations:
            w, x, y, z = ORI_QUATERNIONS[orientations[obj.name]]
            orientation = Quaternion(w=w, x=x, y=y, z=z)
        obj.pose = Pose(position=position, orientation=orientation)
    return solved


def reexec_under_isaac_python_if_needed(args: argparse.Namespace) -> None:
    """Hand this command to an interpreter with ToyBlocks in it, if this one lacks it."""
    if os.environ.get(REEXEC_MARKER) or importlib.util.find_spec("blocksuite") is not None:
        return
    python = boot.resolve_isaac_python(args.python)
    logger.info(f"{sys.executable} cannot import blocksuite; re-running under {python}.")
    os.execve(
        str(python),
        [str(python), str(Path(__file__).resolve()), *sys.argv[1:]],
        {**os.environ, REEXEC_MARKER: "1"},
    )


def main() -> None:
    """Build the package if asked, boot the scene, and write the renders."""
    args = parse_args()
    reexec_under_isaac_python_if_needed(args)
    boot.prepare_import_path()

    from agentic_robotics.core.logging import configure_logging

    configure_logging()

    pddl = args.pddl.resolve()
    if not pddl.is_file():
        raise SystemExit(f"No problem file at {pddl}.")
    slug = task_variants.variant_slug(pddl)

    from agentic_robotics.adapters.toyblocks import experiment
    from agentic_robotics.adapters.toyblocks.config import TaskConfig

    task = TaskConfig(name=slug, task_slug=slug, prompt="", max_tool_calls=0)

    boot.ensure_task_package(
        slug,
        pddl,
        generate_env=args.generate_env,
        reset_states=args.reset_states,
        envs=args.envs,
        seed=args.seed,
        codegen_python=boot.resolve_isaac_python(args.python),
        force=args.force,
        usd_library=args.usd_library,
    )
    config = experiment.build_config(
        experiment.load_experiment(args.experiment),
        task_config=task,
        output_dir=boot.create_run_dir(
            f"render-{slug.replace('_', '-')}", Path(args.runs_dir) if args.runs_dir else None
        ),
    )

    boot.require_eula_acceptance()
    if not args.verbose_isaac:
        boot.quiet_isaac_logs()

    from agentic_robotics.adapters.toyblocks.toyblocks_env import ToyBlocksEnv  # Boots Isaac
    from agentic_robotics.core.filesystem import export_image

    if not args.verbose_isaac:
        boot._quiet_isaac_logs_post_launch()

    import torch

    env = ToyBlocksEnv(task, viewer=False)
    base = env.base_env
    available = int(getattr(base, "num_reset_states", 0) or 0)
    if available == 0:
        raise SystemExit(f"Package {slug} carries no reset states; nothing to render.")
    count = min(args.states, available)

    out_dir = args.out_dir / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        env_ids = torch.tensor([0], device=base.device)
        instance_ids = torch.tensor([i], device=base.device)
        base._reset_idx(env_ids, instance_ids=instance_ids)
        env._sync_simulator()
        image = env.get_current_image(config.camera_name)
        path = out_dir / f"{slug}_state{i:02d}.png"
        export_image(image, path)
        logger.info(f"Wrote {path} ({image.shape[1]}x{image.shape[0]}).")
        if args.check_goal:
            base._compute_intermediate_values()
            result = base.evaluate()
            stats = {key: value.tolist() for key, value in result.get("stats", {}).items()}
            logger.info(
                f"Goal classifier on state {i}: success={result['success'].tolist()} {stats}"
            )
        if args.check_goal and args.solved:
            from agentic_robotics.adapters.toyblocks.leap.sim_state import (
                read_sim_state,
                write_sim_state,
            )

            on_pairs, orientations = goal_literals(pddl.read_text(encoding="utf-8"))
            solved = solved_state(read_sim_state(env), on_pairs, orientations)
            write_sim_state(env, solved)
            for _ in range(args.settle_syncs):
                env._sync_simulator()
            image = env.get_current_image(config.camera_name)
            export_image(image, out_dir / f"{slug}_state{i:02d}_solved.png")
            base._compute_intermediate_values()
            result = base.evaluate()
            stats = {key: value.tolist() for key, value in result.get("stats", {}).items()}
            logger.info(
                f"Goal classifier on SOLVED state {i}: success={result['success'].tolist()} {stats}"
            )
    logger.info(f"Rendered {count} of {available} reset states of {slug} to {out_dir}.")

    # Closing the Isaac app can hang for minutes after the last image, which is why Stage-1
    # codegen exits the process directly too. The images are on disk; nothing else needs a
    # clean shutdown.
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
