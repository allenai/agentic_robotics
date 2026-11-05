"""The scripted skills with the LEAP quirk space, as a second skill module.

The LEAP experiments run over *modified* scripted skills rather than the learned ManiFlow
policies, with a known set of deficiencies for the agents to discover:

- **rotate** may confuse directions;
- **flip** may not flip, may flip the wrong way, or may be nominal;
- **pick** won't pick up next to a stack, and needs the gripper open;
- **place** (both `stack` and `place_on_table`) adds positional noise, or drops the block;
- **rotate** and **flip** need the block in a certain area of the table, and on the table;
- nothing may be on top of the block, for every skill that acts on one.

This module is `inline_oracle.py` plus those quirks and nothing else. It is a separate file so that
the scripted skills stay as they are: `inline_oracle.py` is the ground truth the quirks are measured
against, and other work builds on it. Everything here subclasses or calls what that module already
provides. `--skills quirk_skills` selects it (`agentic-robotics` derives `--backend oracle` from the
module name; see `runner_bootstrap.SKILL_MODULE_BACKENDS`).

What the model sees is unchanged by design: `ToyBlocksMCPServer._tool_description` serves the
docstring of the *declaring* class, `OracleMCPServer`, so the overrides below can say what they
really do without that text reaching the agent. The description set an agent is evaluated with
comes from the LEAP baseline (Control's declared docstrings, the developer's `ORACLE_DOCSTRINGS`,
or a Naive/Harness snapshot), exactly as for the scripted skills.

Two things here are research decisions rather than mechanism, and are parameters so the developer
can set them rather than inherit mine:

- **The probabilities and distances** in :class:`QuirkConfig`. The defaults are placeholders.
- **Whether a failed precondition is silent or reported.** The scripted skills fail loudly
  (`OracleSkillError`); the ManiFlow skills, which the LEAP design was built around, fail
  silently -- the arm moves and the scene is unchanged, and the agent has only the image. The
  default here is silent (`privileged_errors=False`): the skill returns the current image with
  nothing done. With `privileged_errors=True` the skill raises instead, naming the unmet
  precondition.

**No skill here reports a motion-planning failure to the agent.** The scripted skills raise
`OracleSkillError` with the planner's account of what went wrong (which block, how many grasp
candidates, whether the arm is wedged), and the ManiFlow skills never raise from a skill body:
whatever happened, the agent gets the next image. The quirk skills behave like the ManiFlow
ones: every `OracleSkillError` from the scripted motion, the pointing
resolver or the held-block bookkeeping is swallowed, logged, written to `quirks.jsonl` as a
`skill_error` event, and answered with the current image. The one exception is a precondition
decline under `privileged_errors=True`, which is the agent-facing message it asks for.

Every quirk that fires is written to `quirks.jsonl` in the run directory, one line per event,
so an episode can be scored against what the skills actually did. Override any field without
editing this file through the `AGENTIC_QUIRKS` environment variable, a JSON object of field
names to values, e.g. ``AGENTIC_QUIRKS='{"flip_none_p": 0.5, "privileged_errors": true}'``.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
import os
import zlib
from typing import TYPE_CHECKING, Any

import numpy as np
from mcp.server.mcpserver import Image as MCPImage  # noqa: TC002 (used for MCP tool registration)

# isort: off
# Boots Isaac Sim, like every skill module; nothing Isaac-dependent may be imported above it.
# curobo in particular must come after it (inline_oracle says why: its USD import before the
# app launch is a native crash, and warp's runtime is None until Isaac has loaded it, so
# `MotionGenConfig` then fails with "'NoneType' object has no attribute 'cuda_devices'"). The
# isort fence keeps a formatter from sorting curobo above this import, which kills every boot.
import torch
from agentic_robotics.adapters.toyblocks.skills.inline_oracle import (
    GRASP_CLEARANCE_M,
    HOME_ARM_JOINT_POSITIONS,
    HOME_STEPS,
    FlipDirectionArg,
    InlineOracleMCPServer,
    grasp_poses,
    OracleSkillError,
    RotateDirection,
    RotateDirectionArg,
    SimThreadOracleServer as _ScriptedSimThreadOracleServer,
    place_on_block_pose,
    place_on_table_pose,
    rotation_of,
)
from agentic_robotics.core.images import PixelCoord  # noqa: TC001 (used for MCP tool registration)
from curobo.types.math import Pose
# isort: on

if TYPE_CHECKING:
    from collections.abc import Callable

    from agentic_robotics.adapters.toyblocks.skills.inline_oracle import CuroboArm

logger = logging.getLogger("agentic_robotics.examples.quirk_skills")

QUIRKS_ENV_VAR = "AGENTIC_QUIRKS"
"""JSON object of `QuirkConfig` field overrides, read once at import."""


@dataclasses.dataclass(frozen=True)
class QuirkConfig:
    """The quirk space's parameters. Every default is a placeholder for the developer to set."""

    rotate_confuse_p: float = 0.25
    """Probability a rotate turns the block the other way from the direction asked."""

    flip_none_p: float = 0.1
    """Probability a flip does nothing at all."""

    flip_wrong_p: float = 0.2
    """Probability a flip tips the block in one of the three other directions."""

    place_noise_m: float = 0.005
    """Standard deviation, in metres, of the planar offset added to every placement."""

    place_drop_p: float = 0.1
    """Probability a placement releases the block from `place_drop_height_m` above the target
    instead of setting it down."""

    place_drop_height_m: float = 0.06
    """How far above the placement a dropped block is released from."""

    stack_radius_m: float = 0.03
    """Edge-to-edge gap, in metres, below which a block beside the target stops a pick. A gap,
    not a centre distance: two 5.7 cm blocks are 5.7 cm apart centre to centre when touching,
    so a centre radius this small would never fire. 0.03 rather than the 0.04 the ManiFlow
    pick's measured limit suggests (0-3 of 5 at gaps of 1-3 cm, 5 of 5 from 5 cm): a scripted
    flip moves a block half a block width, 2.85 cm (the ManiFlow flip moved about a whole one),
    and the `near` predicate places neighbours at 1-3 cm, so at 0.04 a block flipped once away
    from a 1 cm neighbour would still be declined."""

    pick_neighbour_must_be_raised: bool = True
    """Whether only a raised neighbour (one resting on another block: a tower) blocks a pick.
    True is the meeting's phrasing, "won't pick up next to a stack". False is what the
    2026-10-05 probe measured for the ManiFlow pick: beside a single block on the table it
    lifted 0, 0, 1 of 5 at gaps of 1, 2, 3 cm, the same as beside a tower. True is the default
    because with False, a loose block beside a tower outside the reorient area has no legal
    first move (its pick declined for the gap, its rotate and flip for the area, and the
    tower's top block declined for the loose block beside it), and the outside adjacent
    families are meant to be solvable. With True the tower's top block can be
    picked off, after which the loose block can be too."""

    reorient_x_m: tuple[float, float] = (0.29, 0.72)
    """World-x band a block must lie in for rotate and flip to act. Set here, not inherited:
    the scripted flip generator's own band (`inline_oracle.REORIENT_X_LIMITS_M`, the same
    numbers today) is replaced by this one for the flip targets too, so the quirk skills have
    one x band, for the precondition and for where a flip may land."""

    reorient_y_m: tuple[float, float] = (-0.30, 0.30)
    """World-y band a block must lie in for rotate and flip to act. 0.30 keeps the band and the
    outside regions disjoint: the far boxes of ToyBlocks's `in_region.py` start at |y| 0.32,
    and at 0.35 a third of the outside instances would start inside the band. 0.30 leaves the
    same 2 cm margin as `near_robot` (x up to 0.27) has to the x band, and is nearer the
    ManiFlow flip/rotate training distribution (|y| up to 0.25). Every feasible isolated
    instance starts within it."""

    on_table_tolerance_m: float = 0.01
    """How far a block's underside may sit above the table top and still count as on it."""

    rotate_max_steps: int = 250
    """RL rollout cutoff for the rotate policy, in simulator steps (plus the scripted slack of
    20). Each skill has its own cutoff. The policy was trained with a 100-step horizon, but from
    the scripted approach pose a quarter turn of a square block can run past 100+20 and stop at
    45 degrees (3 of 8 rotates on 2026-10-06); the ManiFlow rotate tool has 270 steps. The
    rollout exits at the executor's goal test, so a longer cutoff cannot over-rotate."""

    flip_max_steps: int = 100
    """RL rollout cutoff for the flip policy, in simulator steps (plus the slack of 20). The
    flip policy's training horizon was 64 steps; flips that reach the cutoff have failed, not
    run out of time (successes exited at 43-57 steps on 2026-10-06)."""

    feint_pick: bool = True
    """Whether a declined pick still does what the ManiFlow pick did when it found nothing to
    lift (2026-09-29 gripper check; the meta-oracle's picks beside a tower, 2026-10-05): the
    arm goes down to the block, the gripper closes on nothing, the arm lifts, and the gripper
    stays closed, so the next pick needs `open_gripper` first. With the gripper already
    closed the ManiFlow pick never opened it and pushed the block; here the closed gripper
    cannot be planned onto the block, so the arm stays put. False returns the unchanged image."""

    feint_reorient: bool = True
    """Whether a declined rotate or flip still moves the arm the way the ManiFlow policies did
    when asked to act out of bounds (probe of 2026-10-06, `ood_reorient_probe.py`, five
    near-robot instances): the arm flies to the skill's starting pose above the block and stays
    there, the block untouched; a flip also closes the gripper on nothing, a rotate leaves it
    open. False returns the unchanged image with the arm where it was."""

    pick_requires_open_gripper: bool = True
    """Whether a pick with the gripper closed is declined (True: LEAP as submitted, where a declined
    pick's feint left the gripper closed and every later pick was declined until the agent opened
    it). False: the pick opens the gripper itself first, treating a closed gripper as something
    the skill can resolve on its own (the Rao suite's setting)."""

    correct_placement: bool = False
    """Whether a placement nudges the held block over its (noisy) target before releasing, as
    `InlineTampMotion.place_on_block` does (without it, stacked cubes land 1.3-1.7 cm short).
    False: LEAP as submitted. The 2026-10-16 Rao suite with noise 0 and this False solved 0/30
    online: towers fell from the short placements alone."""

    pick_gap_rule: bool = True
    """Whether a pick is declined beside a block within `stack_radius_m` (True: LEAP). False: the
    pick goes ahead, and a tower it knocks over is the agent's to deal with (the Rao suite's
    setting)."""

    privileged_errors: bool = False
    """Whether an unmet precondition raises `OracleSkillError` (True) or silently returns the
    unchanged image (False, the ManiFlow-like default)."""


def load_quirk_config() -> QuirkConfig:
    """`QuirkConfig` with any `AGENTIC_QUIRKS` overrides applied.

    :raises SystemExit: If the variable is not a JSON object of known fields
    """
    raw = os.environ.get(QUIRKS_ENV_VAR)
    if not raw:
        return QuirkConfig()
    try:
        overrides = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{QUIRKS_ENV_VAR} is not valid JSON: {exc}") from exc
    known = {field.name for field in dataclasses.fields(QuirkConfig)}
    unknown = sorted(set(overrides) - known)
    if not isinstance(overrides, dict) or unknown:
        raise SystemExit(
            f"{QUIRKS_ENV_VAR} must be an object of QuirkConfig fields; unknown: {unknown}"
        )
    for name in ("reorient_x_m", "reorient_y_m"):
        if name in overrides:
            overrides[name] = tuple(overrides[name])
    return QuirkConfig(**overrides)


QUIRKS = load_quirk_config()
"""The parameters this process's quirk skills run with."""

# ---------------------------------------------------------------------------------------------
# Scene geometry the preconditions read
# ---------------------------------------------------------------------------------------------

STOWED_BELOW_Z = -0.5
"""Blocks below this are stowed under the table by `leap/sim_state.py` and are not in play."""

RAISED_M = 0.02
"""A block whose underside is this far above the table is resting on something else."""


def vertical_half_extent(arm: CuroboArm, name: str) -> float:
    """Half of a block's world-vertical extent, for whichever face it lies on."""
    size = arm.get_size(name).detach().cpu().numpy()
    axes = rotation_of(arm.get_pose(name)[3:]).as_matrix()
    return 0.5 * float(np.abs(axes[2]) @ size)


def planar_half_extent(arm: CuroboArm, name: str) -> float:
    """Half of a block's longest side: a conservative planar footprint radius."""
    return 0.5 * float(arm.get_size(name).max())


def blocks_in_play(arm: CuroboArm) -> list[str]:
    """Names of the blocks on or above the table, leaving out any stowed beneath it."""
    return [name for name in arm.block_names if float(arm.get_pose(name)[2]) > STOWED_BELOW_Z]


def table_height(arm: CuroboArm) -> float:
    """World z of the table top: the lowest underside among the blocks in play.

    `block_pointing.find_table_z` reads the lowest block's centre and size, which is wrong once
    a scene has blocks stowed under the table and for a block lying on its side.
    """
    names = blocks_in_play(arm)
    if not names:
        raise OracleSkillError("There are no blocks on the table.")
    return min(float(arm.get_pose(name)[2]) - vertical_half_extent(arm, name) for name in names)


def underside_height(arm: CuroboArm, name: str, table_z: float) -> float:
    """How far a block's underside sits above the table top."""
    return float(arm.get_pose(name)[2]) - vertical_half_extent(arm, name) - table_z


def blocks_on_top(arm: CuroboArm, name: str) -> list[str]:
    """Blocks resting on the named one: overlapping it in plan and sitting above its top face."""
    pose = arm.get_pose(name)
    top_z = float(pose[2]) + vertical_half_extent(arm, name)
    footprint = planar_half_extent(arm, name)
    above = []
    for other in blocks_in_play(arm):
        if other == name:
            continue
        other_pose = arm.get_pose(other)
        overlap = footprint + planar_half_extent(arm, other)
        in_plan = bool((other_pose[:2] - pose[:2]).abs().max() < overlap)
        underside = float(other_pose[2]) - vertical_half_extent(arm, other)
        if in_plan and underside > top_z - 0.01:
            above.append(other)
    return above


def blocking_neighbours(
    arm: CuroboArm, name: str, table_z: float, gap_m: float, *, raised_only: bool
) -> list[str]:
    """Blocks whose edge is within `gap_m` of the named block's; only raised ones if asked.

    The gap is measured along whichever axis separates the two footprints more (the Chebyshev
    distance between centres, less both half-extents), so it is the width of the space between
    two side-by-side blocks. A block in the named one's own column -- the tower it is part of --
    does not count: the quirk is about picking *beside* another block, and `blocks_on_top`
    covers the tower itself.
    """
    pose = arm.get_pose(name)
    own_footprint = planar_half_extent(arm, name)
    near = []
    for other in blocks_in_play(arm):
        if other == name:
            continue
        other_pose = arm.get_pose(other)
        gap = (
            float((other_pose[:2] - pose[:2]).abs().max())
            - own_footprint
            - planar_half_extent(arm, other)
        )
        if gap < 0.0:
            continue  # Overlapping footprints: the same column, which `blocks_on_top` judges
        if gap >= gap_m:
            continue
        if raised_only and underside_height(arm, other, table_z) <= RAISED_M:
            continue
        near.append(other)
    return near


def shifted(pose: Pose, dx_m: float, dy_m: float) -> Pose:
    """`pose` translated in the world's horizontal plane, keeping its orientation."""
    position = pose.position.clone()
    position[..., 0] += dx_m
    position[..., 1] += dy_m
    return Pose(position=position, quaternion=pose.quaternion)


class DeclinedLoudly(OracleSkillError):
    """A precondition decline under `privileged_errors=True`: the one error the agent is meant
    to see."""


def silent_like_maniflow(skill: Callable) -> Callable:
    """Answer any `OracleSkillError` from a motor skill with the current image instead.

    The ManiFlow skills never raise from a skill body, so neither do these; the scripted
    motion's failure text (planner-specific and naming blocks the agent cannot see) goes to
    the log and `quirks.jsonl` only. A `DeclinedLoudly` passes through, since it is the
    agent-facing message `privileged_errors=True` asks for. `functools.wraps` keeps the declared
    signature visible to the registration check and the MCP schema.
    """

    @functools.wraps(skill)
    async def wrapped(self: QuirkOracleMCPServer, *args: Any, **kwargs: Any) -> MCPImage:
        try:
            return await skill(self, *args, **kwargs)
        except DeclinedLoudly:
            raise
        except OracleSkillError as exc:
            return self._swallow(skill.__name__, exc)

    return wrapped


# ---------------------------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------------------------


class QuirkOracleMCPServer(InlineOracleMCPServer):
    """`InlineOracleMCPServer` with the quirk space applied in front of each motor skill.

    The perception tools, `return_home`, `open_gripper` and `close_gripper` are inherited
    unchanged. The five motor skills check the meeting's preconditions first, then either decline
    (see `QuirkConfig.privileged_errors`) or run the scripted motion, with the skill's own quirk
    applied to what is asked of it.
    """

    def _build_skill_backend(self) -> None:
        """Stand up the scripted motion backend, then the quirk state beside it."""
        super()._build_skill_backend()
        self._quirks = QUIRKS
        self._motion.flip_x_limits_m = self._quirks.reorient_x_m
        self._motion.rl_max_steps_by_skill = {
            "rotate": self._quirks.rotate_max_steps,
            "flip": self._quirks.flip_max_steps,
        }
        self._quirk_rng = np.random.default_rng(self.config.rng_seed)
        self._quirk_log = (
            self.config.output_dir / "quirks.jsonl" if self.config.output_dir else None
        )
        logger.info(f"Quirk skills active: {dataclasses.asdict(self._quirks)}")

    @property
    def table_z(self) -> float:
        """The table top from the blocks in play, so a scene with blocks stowed under the table
        (`leap/sim_state.py` writes subsets that way) still places and deprojects on the table."""
        return table_height(self._motion.arm)

    def seed_for_episode(self, episode_index: int) -> None:
        """Seed the quirk draws with the episode, the way the scene's own RNGs are."""
        super().seed_for_episode(episode_index)
        self._quirk_rng = np.random.default_rng(
            (self.config.rng_seed * 1_000_003 + episode_index + 1) % (2**32)
        )

    def seed_quirks_for(self, task: str, instance: int) -> None:
        """Reseed the quirk draws from the task and its reset-state index.

        :meth:`seed_for_episode` sees only the episode index, which is 0 in every one-episode
        process the suites launch, so every task at every instance would draw the same quirk
        sequence (in the 2026-10-16 Rao pilot the second placement dropped in all six episodes).
        Seeding by task and instance avoids that. CRC-32 rather than `hash()`, which Python salts
        per process.
        """
        key = zlib.crc32(f"{task}:{instance}".encode())
        seed = (self.config.rng_seed * 1_000_003 + key) % (2**32)
        self._quirk_rng = np.random.default_rng(seed)
        logger.info(f"Quirk draws seeded for {task} instance {instance} with {seed}.")

    # -- bookkeeping ---------------------------------------------------------------------------

    def _record_quirk(self, skill: str, kind: str, **details: Any) -> None:
        """Log one quirk event and append it to the run's `quirks.jsonl`."""
        event = {"skill": skill, "quirk": kind, "call": self.frame_count, **details}
        logger.info(f"Quirk fired: {event}")
        if self._quirk_log is not None:
            with self._quirk_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")

    def _decline(self, skill: str, block_name: str, reasons: list[str]) -> MCPImage:
        """What a skill does when a precondition is unmet: nothing, silently or reported.

        A declined rotate or flip first feints (see `QuirkConfig.feint_reorient`), so the
        agent sees the arm go to the block and come back with nothing changed, as it did under
        the learned skills. The feint's own planning failure, if any, is swallowed by
        `silent_like_maniflow` like every other scripted error.
        """
        self._record_quirk(skill, "precondition", block=block_name, reasons=reasons)
        if self._quirks.privileged_errors:
            raise DeclinedLoudly(f"{skill} of '{block_name}' did nothing: " + "; ".join(reasons))
        if self._quirks.feint_reorient and skill in ("rotate", "flip"):
            self._feint_reorient(skill, block_name)
        if self._quirks.feint_pick and skill == "pick":
            self._feint_pick(block_name)
        return self.current_image_obs

    FEINT_PICK_STANDOFF_M = 0.06
    """How far above the grasp pose the feint pick closes: a block height plus clearance, so the
    fingertips shut in the air above the block instead of around it."""

    def _feint_pick(self, block_name: str) -> None:
        """Move the arm as the ManiFlow pick did when it came up empty: down, close, lift.

        The 2026-10-05 adjacency probe found the ManiFlow pick lifted nothing at 1-2 cm from a
        neighbour, so the feint must not lift either. Closing at the grasp pose itself shuts the
        fingers around the block and carries it up unrecorded (in the 2026-10-06 Naive pilot,
        stack and place then "silently did nothing" with the block visibly held, and the agent
        learned close_gripper as the fix). So the scripted grasp poses are raised by
        `FEINT_PICK_STANDOFF_M`, the gripper closes on air above the block, the arm retracts,
        and the gripper is left closed. If no raised pose plans the error is swallowed like any
        other.
        """
        self._record_quirk("pick", "feint", block=block_name)
        motion = self._motion
        candidates = [
            motion._raised(pose, self.FEINT_PICK_STANDOFF_M)
            for pose in grasp_poses(motion.arm, block_name)[: self.config.oracle.max_samples]
        ]
        if not motion._try_candidates(candidates):
            raise OracleSkillError(
                f"Feint of a pick of '{block_name}': no pose above a grasp could be planned to."
            )
        motion.arm.close_gripper()
        motion._retract()

    FEINT_SETTLE_STEPS = 24
    """Steps the arm holds at the feint pose before the skill returns."""

    def _feint_reorient(self, skill: str, block_name: str) -> None:
        """Move the arm as the ManiFlow rotate or flip did on an out-of-bounds block.

        The policies drove the arm to their starting pose over the block and held there with
        the block untouched; the flip closed the gripper on nothing on the way (every trial of
        the 2026-10-06 probe ended with the finger at 0.84), the rotate left it open. So: the
        scripted approach to the same pre-skill pose, the gripper closed for a flip, a short
        hold, and no retraction, which is where the policy left the arm.
        """
        self._record_quirk(skill, "feint", block=block_name)
        motion = self._motion
        motion._approach_for_rl(block_name, "feint", skill=skill)
        if skill == "flip":
            motion.arm.close_gripper()
        motion.arm.step(self.FEINT_SETTLE_STEPS)

    def _swallow(self, skill: str, error: OracleSkillError) -> MCPImage:
        """What the agent gets when the scripted motion fails: the scene as it now stands."""
        self._record_quirk(skill, "skill_error", message=str(error))
        self._motion.arm.refresh_world()  # The arm stopped somewhere no plan chose
        return self.current_image_obs

    def _common_blockers(self, block_name: str) -> list[str]:
        """The precondition every motor skill shares: nothing on top of the block."""
        above = blocks_on_top(self._motion.arm, block_name)
        return [f"'{', '.join(above)}' on top of it"] if above else []

    def _pick_blockers(self, block_name: str) -> list[str]:
        """Unmet pick preconditions: gripper open, no block within the gap beside it.

        Nothing on top is not checked here: `pick` has already redirected to the top of the
        column, so the block it asks about has nothing above it.
        """
        arm = self._motion.arm
        reasons: list[str] = []
        if self._motion.gripper_state != "open" and self._quirks.pick_requires_open_gripper:
            reasons.append("gripper not open")
        near = (
            blocking_neighbours(
                arm,
                block_name,
                table_height(arm),
                self._quirks.stack_radius_m,
                raised_only=self._quirks.pick_neighbour_must_be_raised,
            )
            if self._quirks.pick_gap_rule
            else []
        )
        if near:
            reasons.append(f"another block within the gap ({', '.join(near)})")
        return reasons

    def _reorient_blockers(self, block_name: str) -> list[str]:
        """Unmet rotate/flip preconditions: nothing on top, on the table, inside the area, and
        nothing held.

        The empty gripper is required because the oracle text promises it for both skills and
        the ManiFlow policies were never measured with a block in hand. Declined with the feint
        like the other preconditions.
        """
        arm = self._motion.arm
        reasons = self._common_blockers(block_name)
        if self._held_block is not None:
            reasons.append(f"holding '{self._held_block}'")
        if underside_height(arm, block_name, table_height(arm)) > self._quirks.on_table_tolerance_m:
            reasons.append("not on the table")
        x, y = (float(v) for v in arm.get_pose(block_name)[:2])
        (x_min, x_max), (y_min, y_max) = self._quirks.reorient_x_m, self._quirks.reorient_y_m
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            reasons.append(f"outside the workable area at x={x:.2f}, y={y:.2f}")
        return reasons

    # -- motor skills --------------------------------------------------------------------------

    @silent_like_maniflow
    async def pick(self, pixel: PixelCoord) -> MCPImage:
        """The scripted pick: the top of the column if the block is buried, declined beside
        another block or with the gripper closed."""
        self._raise_if_shutdown(skill_name="Pick")
        if self._held_block is not None:
            raise OracleSkillError(
                f"The gripper is already holding '{self._held_block}'; place that block before "
                "picking up another."
            )
        block_name = self._resolve_block(pixel)
        above = blocks_on_top(self._motion.arm, block_name)
        if above:
            # The oracle text: "Attempting to pick any other block in the stack will result in
            # the top block being picked." The ManiFlow pick came from above and took whatever
            # was on top of the column (the 2026-10-05 meta-oracle runs).
            top = max(above, key=lambda name: float(self._motion.arm.get_pose(name)[2]))
            self._record_quirk("pick", "picked_top_of_stack", asked=block_name, picked=top)
            block_name = top
        blockers = self._pick_blockers(block_name)
        if blockers:
            return self._decline("pick", block_name, blockers)
        if self._motion.gripper_state != "open":
            # Only reachable with pick_requires_open_gripper False: the skill opens it itself.
            self._record_quirk("pick", "opened_gripper", block=block_name)
            self._motion.open_gripper()
        self._motion.grasp(block_name)
        self._held_block = block_name
        return self.current_image_obs

    @silent_like_maniflow
    async def stack(self, pixel: PixelCoord) -> MCPImage:
        """The scripted stack, with placement noise or a drop."""
        self._raise_if_shutdown(skill_name="PlaceOnBlock")
        held_block = self._require_held_block()
        support_name = self._resolve_block(pixel)
        support = self._motion.arm.get_pose(support_name)
        self._place_with_quirks(
            "stack",
            place_on_block_pose(self._motion.arm, held_block, support_name),
            f"'{held_block}' on '{support_name}'",
            held_block=held_block,
            block_xy=(float(support[0]), float(support[1])),
        )
        self._held_block = None
        return self.current_image_obs

    @silent_like_maniflow
    async def place_on_table(self, pixel: PixelCoord) -> MCPImage:
        """The scripted table placement, with placement noise or a drop."""
        self._raise_if_shutdown(skill_name="PlaceOnTable")
        held_block = self._require_held_block()
        world_xyz = self._deproject_to_table(pixel)
        self._place_with_quirks(
            "place_on_table",
            place_on_table_pose(self._motion.arm, held_block, world_xyz),
            f"'{held_block}' on the table",
            held_block=held_block,
            block_xy=(float(world_xyz[0]), float(world_xyz[1])),
        )
        self._held_block = None
        return self.current_image_obs

    @silent_like_maniflow
    async def rotate(self, pixel: PixelCoord, direction: RotateDirectionArg) -> MCPImage:
        """The scripted rotate, sometimes in the other direction, declined off the table or area
        or while a block is held."""
        self._raise_if_shutdown(skill_name="Rotate")
        block_name = self._resolve_block(pixel)
        blockers = self._reorient_blockers(block_name)
        if blockers:
            return self._decline("rotate", block_name, blockers)

        asked = RotateDirection.parse(direction)
        executed = asked
        if self._quirk_rng.random() < self._quirks.rotate_confuse_p:
            executed = RotateDirection.CCW if asked is RotateDirection.CW else RotateDirection.CW
            self._record_quirk(
                "rotate",
                "confused_direction",
                block=block_name,
                asked=str(asked),
                executed=str(executed),
            )
        self._motion.rotate(block_name, executed)
        return self.current_image_obs

    @silent_like_maniflow
    async def flip(self, pixel: PixelCoord, direction: FlipDirectionArg) -> MCPImage:
        """The scripted flip; sometimes no flip, sometimes another direction; declined off the
        table or area or while a block is held."""
        self._raise_if_shutdown(skill_name="Flip")
        block_name = self._resolve_block(pixel)
        blockers = self._reorient_blockers(block_name)
        if blockers:
            return self._decline("flip", block_name, blockers)

        asked = str(direction)
        roll = self._quirk_rng.random()
        if roll < self._quirks.flip_none_p:
            self._record_quirk("flip", "no_flip", block=block_name, asked=asked)
            return self.current_image_obs

        executed = asked
        if roll < self._quirks.flip_none_p + self._quirks.flip_wrong_p:
            others = [name for name in self.TOP_FACE_ENDS_UP if name != asked]
            executed = str(self._quirk_rng.choice(others))
            self._record_quirk(
                "flip", "wrong_direction", block=block_name, asked=asked, executed=executed
            )
        self._motion.flip(block_name, self.TOP_FACE_ENDS_UP[executed])
        # The ManiFlow flip left the gripper closed whether or not it turned the block (0.72
        # on nothing in every 2026-09-29 trial, 0.84 in 9 of 10 on 2026-10-06); the scripted
        # flip opens it after the push, so close it again.
        self._motion.close_gripper()
        return self.current_image_obs

    async def return_home(self) -> MCPImage:
        """The arm home, the gripper and anything it holds left as they are.

        The ManiFlow `return_home` ramps the joints home with the gripper command kept as it
        reads; the scripted one opens the gripper first and drops what it holds. This follows
        the ManiFlow one: a held block comes along, modelled as attached for the planner.
        """
        self._raise_if_shutdown(skill_name="ReturnHome")
        arm = self._motion.arm
        arm.move_arm_to_joint_positions(
            torch.tensor(HOME_ARM_JOINT_POSITIONS, device=arm.device), steps=HOME_STEPS
        )
        arm.refresh_world()
        return self.current_image_obs

    GRASP_REGISTER_TOLERANCE_M = 0.03
    """How close a block's centre must be to the point between the fingertips for a close of
    the gripper to count as a grasp of it."""

    async def close_gripper(self) -> MCPImage:
        """Close the gripper; a block between the fingertips counts as picked from here on.

        The scripted close only moves the fingers, so without this a block it closed on would
        never be modeled as held and the next stack or place would be silently declined. The
        scripted skills already read ground-truth state, so the fairest reading is that a block
        between the fingers is grasped, as `pick` would record it. The point
        between the fingertips is the grasp geometry of `grasp_poses` inverted: the gripper
        advances along its own -Z, so the block centre a closed gripper holds lies
        `GRASP_CLEARANCE_M` plus a half-block along +Z of the end-effector frame.
        """
        self._raise_if_shutdown(skill_name="CloseGripper")
        self._motion.close_gripper()
        if self._held_block is None:
            grasped = self._block_between_fingertips()
            if grasped is not None:
                self._motion.arm.attach(grasped)
                self._held_block = grasped
                self._record_quirk("close_gripper", "grasped", block=grasped)
        return self.current_image_obs

    def _block_between_fingertips(self) -> str | None:
        """The block whose centre sits where a grasp would hold it, or None."""
        arm = self._motion.arm
        ee = arm.get_curr_pose()
        ee_position = ee.position.reshape(-1).detach().cpu().numpy()
        ee_z_axis = rotation_of(ee.quaternion).as_matrix()[:, 2]
        best: tuple[float, str] | None = None
        for name in blocks_in_play(arm):
            half = 0.5 * float(arm.get_size(name).min())
            expected = ee_position + ee_z_axis * (GRASP_CLEARANCE_M + half)
            centre = arm.get_pose(name)[:3].detach().cpu().numpy()
            distance = float(np.linalg.norm(centre - expected))
            if distance < self.GRASP_REGISTER_TOLERANCE_M and (best is None or distance < best[0]):
                best = (distance, name)
        return best[1] if best else None

    # -- placement -----------------------------------------------------------------------------

    def _place_with_quirks(
        self,
        skill: str,
        target: Pose,
        description: str,
        *,
        held_block: str,
        block_xy: tuple[float, float],
    ) -> None:
        """Run a placement the way `InlineTampMotion.place_on_*` does, with the place quirks.

        The target is first shifted by the planar noise. Then either the block is dropped from
        `place_drop_height_m` above it -- the arm flies to that height and opens -- or it is set
        down there by the scripted standoff-and-descend, released and retracted.
        """
        motion = self._motion
        dx, dy = (0.0, 0.0)
        if self._quirks.place_noise_m > 0:
            dx, dy = (float(v) for v in self._quirk_rng.normal(0.0, self._quirks.place_noise_m, 2))
            target = shifted(target, dx, dy)

        if self._quirk_rng.random() < self._quirks.place_drop_p:
            self._record_quirk(skill, "dropped", offset_m=[round(dx, 4), round(dy, 4)])
            release_from = motion._raised(target, self._quirks.place_drop_height_m)

            def fly_to_drop_height() -> None:
                if not motion.arm.plan_and_execute(release_from):
                    raise OracleSkillError(
                        f"Could not reach above where {description} would go; the approach may "
                        "be blocked or out of reach." + motion._planning_failure_hint()
                    )

            self._after_clearing_contact(fly_to_drop_height)
            motion._release()
            motion._retract()
            return

        self._record_quirk(skill, "noise", offset_m=[round(dx, 4), round(dy, 4)])
        self._after_clearing_contact(lambda: motion._place_at(target, description))
        if self._quirks.correct_placement:
            motion._correct_placement(held_block, (block_xy[0] + dx, block_xy[1] + dy))
        motion._release()
        motion._retract()

    def _after_clearing_contact(self, approach: Callable[[], None]) -> None:
        """Run an approach; if it cannot be planned, lift clear of contact once and retry.

        The scripted `grasp` lifts straight after closing, by IK when the planner refuses the
        start state, so its placements begin from a clear pose. A block grasped through
        `close_gripper` leaves the arm down among the blocks, and in the Naive pilot of
        2026-10-06 the standoff above the placement then could not be planned (start state in
        world collision with the neighbour the fingers touch). The retract's IK fallback is
        what breaks that contact; if even it moves nothing the original error stands.
        """
        try:
            approach()
        except OracleSkillError:
            if self._motion._retract() <= 0.0:
                raise
            approach()


class SimThreadQuirkServer(_ScriptedSimThreadOracleServer, QuirkOracleMCPServer):
    """`SimThreadOracleServer`'s dispatch onto the simulator's thread, over the quirk skills."""


SimThreadOracleServer = SimThreadQuirkServer
"""What `runner_bootstrap.sim_thread_server_class` looks up for the oracle backend."""
