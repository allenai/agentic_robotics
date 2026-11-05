"""Build the Search & Spell task instances: one autogen package per instance, sampled by Stage-1
codegen under the training data's settings, checked against the specification, trimmed to one
reset state, and copied under `data/tasks/`.

    # In the container (codegen boots Isaac); two builds at a time, each ~7.6 GB of GPU memory
    agentic-robotics-spell-build build --jobs 2
    agentic-robotics-spell-build build --tasks spell_auk_patterned

    # Re-check built packages (numpy only, either interpreter)
    agentic-robotics-spell-build check

**Sampling.** Each instance is its own PDDL (`search_spell_env.py`), so `ensure_task_package`
builds it alone, through the same Stage-1 command `agentic-robotics generate-env` uses, with `--envs 1`
so that every candidate state is sampled in env slot 0 and carries slot 0's scene profile (the
block sizes, table height and friction codegen drew for this package), which is the slot an
episode runs in. `--candidates` states are sampled per attempt (the specification: a state
with blocks closer than 7 cm is rejected and resampled), the first state that passes every
check below is kept as reset state 0, the others are dropped from `reset_states.pkl`, and when
none passes the package is rebuilt with the next seed, up to `--attempts` times. The seed is
`runner_bootstrap.seed_for_task` of the slug plus the attempt, so a rebuild is reproducible.

**Checks**, per candidate state, against the instance's family (`families.json`):

- separation: every pair of blocks not in the same stack is at least `MIN_SEPARATION_M` apart,
  center to center in the table plane (the specification's rule; the sampler's own test is the
  weaker "not within 7 cm on both axes");
- stacks: each `(on top bottom)` holds by the classifier's rule (`predicates/on.py`: within 2 cm
  in x and y, 0-9 cm higher), and no other block sits on any block;
- table: each on-table block rests at the table's height (`reset_state_check.TABLE_Z_TOLERANCE`);
- orientation: each block is within 5 degrees of tilt and 40 degrees of yaw of its `ori_of`
  (the `ori_of` sampler's own acceptance; it jitters yaw by up to 30 degrees);
- reach: every block is within the arm's reach radius (`reset_state_check`).

A kept package carries `search_spell_instance.json` (seed, attempt, which candidate was kept,
and the checks' numbers), so a package's provenance is readable without rebuilding it.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

from agentic_robotics.adapters.toyblocks import runner_bootstrap as boot

boot.prepare_import_path()

from agentic_robotics.adapters.toyblocks.block_orientations import (
    ORI_QUATERNIONS,
    orientation_errors,
)
from agentic_robotics.adapters.toyblocks.release.search_spell_families import (
    MIN_SEPARATION_M,
    PROBLEMS_DIR,
    Family,
    TaskFamilies,
    task_keys,
)
from agentic_robotics.adapters.toyblocks.reset_state_check import (
    MAX_REACH_RADIUS,
    ROBOT_BASE_X,
    TABLE_Z_TOLERANCE,
    _block_sizes,
)

logger = logging.getLogger("agentic_robotics.examples.search_spell_build")

TABLE = "workspace_table"
ON_XY_TOLERANCE_M = 0.02
ON_DZ_MAX_M = 0.09
TILT_TOLERANCE_DEG = 5.0
YAW_TOLERANCE_DEG = 40.0
INSTANCE_RECORD = "search_spell_instance.json"

DEFAULT_CANDIDATES = 8
DEFAULT_ATTEMPTS = 3


@dataclass
class StateCheck:
    """One candidate state against its family."""

    index: int
    closest_pair_m: float = float("inf")
    closest_pair: str = ""
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def check_state(
    family: Family, poses: dict[str, np.ndarray], table_z: float, sizes: dict[str, float], i: int
) -> StateCheck:
    """Check candidate state `i` of a package against the specification and its family."""
    check = StateCheck(index=i)
    p = {name: poses[name][i] for name in family.blocks}
    stacked_pairs = {frozenset((top, bottom)) for top, bottom in family.on.items()}

    for a, b in combinations(sorted(p), 2):
        if frozenset((a, b)) in stacked_pairs:
            continue
        distance = float(np.hypot(p[a][0] - p[b][0], p[a][1] - p[b][1]))
        if distance < check.closest_pair_m:
            check.closest_pair_m, check.closest_pair = distance, f"{a}/{b}"
        if distance < MIN_SEPARATION_M:
            check.failures.append(f"{a} and {b} are {distance:.3f} m apart")

    def on(top: str, bottom: str) -> bool:
        dx, dy = abs(p[top][0] - p[bottom][0]), abs(p[top][1] - p[bottom][1])
        dz = p[top][2] - p[bottom][2]
        return dx <= ON_XY_TOLERANCE_M and dy <= ON_XY_TOLERANCE_M and 0 < dz <= ON_DZ_MAX_M

    for top, bottom in family.on.items():
        if not on(top, bottom):
            check.failures.append(f"{top} is not on {bottom}")
    for a, b in combinations(sorted(p), 2):
        for top, bottom in ((a, b), (b, a)):
            if family.on.get(top) != bottom and on(top, bottom):
                check.failures.append(f"{top} sits on {bottom}, which the family does not say")

    for name in family.on_table:
        resting_z = table_z + sizes.get(name, 0.04) / 2
        if abs(p[name][2] - resting_z) > TABLE_Z_TOLERANCE:
            check.failures.append(f"{name} is not resting on the table (z {p[name][2]:.3f})")

    for name, ori in family.ori.items():
        tilt, yaw = orientation_errors(tuple(p[name][3:7]), ORI_QUATERNIONS[ori])
        if tilt > TILT_TOLERANCE_DEG or yaw > YAW_TOLERANCE_DEG:
            check.failures.append(
                f"{name} is {tilt:.1f} deg tilted, {yaw:.1f} deg yawed from {ori}"
            )

    for name in p:
        reach = float(np.hypot(p[name][0] - ROBOT_BASE_X, p[name][1]))
        if reach > MAX_REACH_RADIUS:
            check.failures.append(f"{name} is out of reach ({reach:.3f} m)")
    return check


def load_states(package_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], float, int]:
    """The package's reset-state payload, block poses (N, 7), table z and profile count."""
    with (package_dir / "reset_states.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    states = payload.get("states", payload)
    rigid = states["rigid_object"]
    poses = {name: np.asarray(entry["root_pose"]) for name, entry in rigid.items() if name != TABLE}
    table_z = float(np.asarray(rigid[TABLE]["root_pose"])[0, 2])
    return payload, poses, table_z, int(payload.get("num_profiles", 8))


def check_package_states(package_dir: Path, family: Family) -> list[StateCheck]:
    """Every candidate state of a package against its family."""
    _, poses, table_z, _ = load_states(package_dir)
    missing = set(family.blocks) - set(poses)
    if missing:
        raise RuntimeError(f"{package_dir.name} lacks blocks {sorted(missing)}")
    sizes = _block_sizes(package_dir)
    n = next(iter(poses.values())).shape[0]
    return [check_state(family, poses, table_z, sizes, i) for i in range(n)]


def _slice_leaves(tree: Any, index: int) -> Any:
    """`tree` with every array leaf cut to its row `index`, keeping the leading axis."""
    if isinstance(tree, dict):
        return {key: _slice_leaves(value, index) for key, value in tree.items()}
    return np.asarray(tree)[index : index + 1]


def keep_one_state(package_dir: Path, index: int) -> None:
    """Rewrite the package's `reset_states.pkl` with candidate `index` as its only state."""
    payload, _, _, num_profiles = load_states(package_dir)
    states = payload.get("states", payload)
    profile_ids = np.asarray(payload.get("profile_ids", np.zeros(1, dtype=np.int64)))
    kept = {
        "states": _slice_leaves(states, index),
        "profile_ids": profile_ids[index : index + 1],
        "num_profiles": num_profiles,
    }
    with (package_dir / "reset_states.pkl").open("wb") as handle:
        pickle.dump(kept, handle)


def build_instance(slug: str, family: Family, args: argparse.Namespace) -> dict[str, Any]:
    """Build one instance's package: sample, check, keep one state, record. Returns the record."""
    pddl = PROBLEMS_DIR / _task_of(slug) / f"{slug}.pddl"
    package_dir = boot.task_package_dir(slug)
    started = time.time()
    for attempt in range(args.attempts):
        seed = boot.seed_for_task(slug) + attempt
        boot.ensure_task_package(
            slug,
            pddl,
            generate_env=True,
            reset_states=args.candidates,
            envs=1,
            seed=seed,
            codegen_python=boot.resolve_isaac_python(args.python),
            force=True,
            usd_library=family.library,
        )
        checks = check_package_states(package_dir, family)
        passing = [check for check in checks if check.ok]
        logger.info(
            f"{slug}: attempt {attempt}, seed {seed}: {len(passing)} of {len(checks)} candidate "
            f"states pass" + ("" if passing else "; " + "; ".join(checks[0].failures[:3]))
        )
        if passing:
            kept = passing[0]
            keep_one_state(package_dir, kept.index)
            record = {
                "slug": slug,
                "family": family.slug,
                "library": family.library,
                "seed": seed,
                "attempt": attempt,
                "candidates": len(checks),
                "kept_candidate": kept.index,
                "closest_pair_m": round(kept.closest_pair_m, 4),
                "closest_pair": kept.closest_pair,
                "rejected": [
                    {"index": check.index, "failures": check.failures}
                    for check in checks
                    if not check.ok
                ],
                "seconds": round(time.time() - started, 1),
            }
            (package_dir / INSTANCE_RECORD).write_text(json.dumps(record, indent=1) + "\n")
            return record
    raise RuntimeError(f"{slug}: no candidate state passed in {args.attempts} attempts")


def _task_of(slug: str) -> str:
    """The task key an instance slug belongs to (`spell_<word>_<collection>`)."""
    return "_".join(slug.split("_")[:3])


def instances(tasks: list[str], only: list[str] | None = None) -> list[tuple[str, Family]]:
    """(instance slug, family) for every instance of the named tasks, or only the slugs in `only`."""
    out: list[tuple[str, Family]] = []
    for task in tasks:
        for family in TaskFamilies.read(task).families.values():
            out.extend((slug, family) for slug in family.instances if not only or slug in only)
    return out


def run_build(args: argparse.Namespace) -> int:
    todo = [
        (slug, family)
        for slug, family in instances(args.tasks, args.instances)
        if args.force or not (boot.task_package_dir(slug) / INSTANCE_RECORD).is_file()
    ]
    logger.info(f"{len(todo)} instances to build, {args.jobs} at a time.")
    failures = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:

        def one(pair: tuple[str, Family]) -> bool:
            try:
                record = build_instance(pair[0], pair[1], args)
            except Exception:
                logger.exception(f"{pair[0]} failed")
                return False
            logger.info(
                f"{pair[0]}: kept candidate {record['kept_candidate']} (closest pair "
                f"{record['closest_pair_m']} m), {record['seconds']} s"
            )
            return True

        for ok in pool.map(one, todo):
            failures += not ok
    logger.info(f"Built {len(todo) - failures} of {len(todo)} instances; {failures} failed.")
    return 1 if failures else 0


def run_check(args: argparse.Namespace) -> int:
    bad = 0
    for slug, family in instances(args.tasks, args.instances):
        package_dir = boot.task_package_dir(slug)
        if not (package_dir / "reset_states.pkl").is_file():
            print(f"{slug}: not built")
            bad += 1
            continue
        checks = check_package_states(package_dir, family)
        verdict = "OK" if all(c.ok for c in checks) else "FAIL"
        bad += verdict == "FAIL"
        print(
            f"{slug}: {verdict}, {len(checks)} state(s), closest pair "
            f"{min(c.closest_pair_m for c in checks):.3f} m"
            + "".join(f"\n  state {c.index}: {'; '.join(c.failures)}" for c in checks if not c.ok)
        )
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("build", "check"))
    parser.add_argument(
        "--tasks", nargs="+", default=task_keys(), help="Task keys (default: all ten)"
    )
    parser.add_argument("--instances", nargs="+", default=None, help="Only these instance slugs")
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--jobs", type=int, default=2, help="build: Isaac boots at a time")
    parser.add_argument("--force", action="store_true", help="build: rebuild recorded packages")
    parser.add_argument("--python", default=None, help="build: interpreter with Isaac")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "build":
        return run_build(args)
    return run_check(args)


if __name__ == "__main__":
    sys.exit(main())
