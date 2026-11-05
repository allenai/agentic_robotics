"""Check a generated task package's reset states for layouts the arm cannot act in.

`OnTable.sample_vectorized` rejects a proposed block position within `MIN_BLOCK_SEPARATION` of
another block in both axes, or outside the arm's reach. But when 5000 rounds of rejection
sampling fail, it falls through to an **unvalidated** uniform random pose and prints a warning.
So a package can carry states whose blocks overlap or sit outside the workspace, and nothing
downstream notices: the scene loads, physics shoves any overlap apart while settling, and the
result looks plausible, so inspecting renders by eye passes such a package (`wasp_tower`'s
overlapping states looked correct).

Numpy and stdlib only, deliberately. Reset states are plain numpy arrays, so this runs under the
light interpreter as well as the Isaac one -- which is what lets `ensure_task_package` call it
straight after codegen instead of hopping through a subprocess.

`MAX_REACH_RADIUS` and `ROBOT_BASE_X` are copies of ToyBlocks's `OnTable` predicate's values as
the shipped task packages were generated (0.55 m; ToyBlocks's `main` has since widened its own to
0.65 m), since importing that module would pull in torch. Block *sizes* are not duplicated: they
ship with the package in `non_sim_state_randomizations.json`, and they matter -- the procedural
colour cuboids are 0.04 m while the USD glyph blocks render larger, so a single assumed size
either over-reports overlap for one kind or under-reports it for the other.

A block stacked on another (centres within a block of each other horizontally and one block
apart vertically, as `on` places them) is neither an overlap nor off the table, so a scene that
starts with a tower, as the LEAP task families' scenes start with a two-block one, passes.

Run it directly on one or more package directories::

    python -m agentic_robotics.adapters.toyblocks.reset_state_check \\
        data/tasks/ai2_letters
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np

MAX_REACH_RADIUS = 0.55
"""Furthest a block centre may sit from the robot base, in metres: the `OnTable` predicate's
`MAX_REACH_RADIUS` the shipped packages were sampled under. Kept in step by hand."""

ROBOT_BASE_X = 0.0
"""`OnTable.ROBOT_BASE_X`, the frame the reach radius is measured in."""

TABLE_Z_TOLERANCE = 0.02
"""How far a block centre may sit off the expected resting height before it counts as off-table."""

DEFAULT_BLOCK_SIZE = 0.04
"""Used only for a block the package records no size for."""


@dataclass
class ResetStateReport:
    """What one package's reset states look like geometrically."""

    package: str
    n_states: int = 0
    n_blocks: int = 0
    overlapping_pairs: int = 0
    states_with_overlap: int = 0
    unreachable_blocks: int = 0
    off_table_blocks: int = 0
    tightest_separation: float = float("inf")
    tightest_pair: str = ""
    furthest_reach: float = 0.0
    furthest_block: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether every state is one the arm could actually work in."""
        return not (self.overlapping_pairs or self.unreachable_blocks or self.off_table_blocks)

    def summary(self) -> str:
        """One line, suitable for a log."""
        if self.n_states == 0:
            return f"{self.package}: no reset states"
        verdict = "OK" if self.ok else "PROBLEMS"
        return (
            f"{self.package}: {verdict} -- {self.n_blocks} blocks, {self.n_states} states, "
            f"{self.overlapping_pairs} overlapping pairs (in {self.states_with_overlap} states), "
            f"{self.unreachable_blocks} unreachable, {self.off_table_blocks} off-table, "
            f"tightest gap {self.tightest_separation:.4f} m ({self.tightest_pair}), "
            f"furthest reach {self.furthest_reach:.4f} m ({self.furthest_block})"
        )


def _block_sizes(package_dir: Path) -> dict[str, float]:
    """Read each block's side length from the package, falling back to a default.

    :param package_dir: Autogen package directory
    :return: Block name to side length in metres
    """
    path = package_dir / "non_sim_state_randomizations.json"
    if not path.is_file():
        return {}

    sizes: dict[str, float] = {}
    for name, per_env in json.loads(path.read_text()).get("block_sizes", {}).items():
        # One size triple per randomization slot; they are equal by construction, so take the max
        # over slots and axes rather than assuming which.
        sizes[name] = max((max(triple) for triple in per_env), default=DEFAULT_BLOCK_SIZE)
    return sizes


def check_package(package_dir: Path) -> ResetStateReport:
    """Check one generated package's reset states.

    :param package_dir: Autogen package directory holding `reset_states.pkl`
    :return: What the states look like; `report.ok` is False if any is unusable
    """
    report = ResetStateReport(package=package_dir.name)

    states_path = package_dir / "reset_states.pkl"
    if not states_path.is_file():
        report.notes.append("no reset_states.pkl; the env would use its codegen spawn layout")
        return report

    with states_path.open("rb") as handle:
        state_dict = pickle.load(handle)

    # Codegen writes `{"states": {...}, "profile_ids", "num_profiles"}`; some packages saved
    # under `data/tasks/` hold the bare state dict.
    state_dict = state_dict.get("states", state_dict)
    rigid = state_dict.get("rigid_object", {})
    poses = {
        name: np.asarray(entry["root_pose"])
        for name, entry in rigid.items()
        if name != "workspace_table"
    }
    if not poses:
        report.notes.append("no blocks in the reset states")
        return report

    sizes = _block_sizes(package_dir)
    report.n_blocks = len(poses)
    report.n_states = next(iter(poses.values())).shape[0]

    overlap_per_state = np.zeros(report.n_states, dtype=int)
    supported = {name: np.zeros(report.n_states, dtype=bool) for name in poses}
    """Per block and state, whether it rests on another block rather than on the table."""
    for a, b in combinations(sorted(poses), 2):
        pa, pb = poses[a][:, :3], poses[b][:, :3]
        dx, dy, dz = (np.abs(pa[:, i] - pb[:, i]) for i in range(3))

        # Two axis-aligned cubes intersect when their centres are closer than the mean of their
        # side lengths on every axis; one resting on the other sits exactly that far above it.
        limit = (sizes.get(a, DEFAULT_BLOCK_SIZE) + sizes.get(b, DEFAULT_BLOCK_SIZE)) / 2
        aligned = (dx < limit) & (dy < limit)
        stacked = aligned & (np.abs(dz - limit) <= TABLE_Z_TOLERANCE)
        overlap_per_state += (aligned & (dz < limit - TABLE_Z_TOLERANCE)).astype(int)
        supported[a] |= stacked & (pa[:, 2] > pb[:, 2])
        supported[b] |= stacked & (pb[:, 2] > pa[:, 2])

        # The sampler's own test: a placement is rejected when it is close in *both* axes, so the
        # separation a pair actually achieved is the larger of the two gaps. A stacked pair has
        # none and is left out.
        gaps = np.maximum(dx, dy)[~stacked]
        gap = float(gaps.min()) if gaps.size else report.tightest_separation
        if gap < report.tightest_separation:
            report.tightest_separation = gap
            report.tightest_pair = f"{a}/{b}"

    report.overlapping_pairs = int(overlap_per_state.sum())
    report.states_with_overlap = int((overlap_per_state > 0).sum())

    table = (
        np.asarray(rigid["workspace_table"]["root_pose"]) if "workspace_table" in rigid else None
    )
    for name, pose in poses.items():
        reach = np.hypot(pose[:, 0] - ROBOT_BASE_X, pose[:, 1])
        report.unreachable_blocks += int((reach > MAX_REACH_RADIUS).sum())
        if float(reach.max()) > report.furthest_reach:
            report.furthest_reach = float(reach.max())
            report.furthest_block = name

        if table is not None:
            resting_z = table[:, 2] + sizes.get(name, DEFAULT_BLOCK_SIZE) / 2
            on_table = np.abs(pose[:, 2] - resting_z) <= TABLE_Z_TOLERANCE
            report.off_table_blocks += int((~on_table & ~supported[name]).sum())

    return report


def main(argv: list[str]) -> int:
    """Check every package named on the command line.

    :param argv: Package directories
    :return: 0 if all are usable, 1 otherwise
    """
    if not argv:
        print(__doc__)
        return 0

    reports = [check_package(Path(arg)) for arg in argv]
    for report in reports:
        print(report.summary())
        for note in report.notes:
            print(f"  note: {note}")
    return 0 if all(r.ok and r.n_states for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
