"""The scripted (oracle) skills, run on ToyBlocks's vecgen engine.

Every motion here is a ToyBlocks vecgen skill -- ``Pick``, ``Place`` (on a block, or on the table),
``Flip`` and ``Rotate`` -- planned with vecgen's batched IK and cuRobo v2 (``Geometry``) and played
out through its segment table: the same skills that record ToyBlocks's training demonstrations,
and the same engine wiring as ToyBlocks's evaluation low-level oracle
(``blocksuite.eval.ll_oracle``), which runs one skill per high-level command. This module is that
wiring for a single environment, stepped synchronously: a skill starts, the simulator is stepped
one control tick at a time until the engine reports its outcome, and every tick passes through
:meth:`VecgenMotion.step`, so the video, the viewer and a human's stop see all of them.

A skill runs once: no retries, no snapshot restores, no re-planning. Its outcome is either a
success or an :class:`OracleSkillError` naming what failed, which the MCP server hands the agent.

**Importing this module needs a running Isaac Sim app** (``blocksuite.eval.ll_oracle`` imports
Isaac Lab); ``inline_oracle.py`` imports it only after its own boot import.
"""

from __future__ import annotations

import logging
import re
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from blocksuite.eval.ll_oracle import (
    DOMAIN_PDDL,
    _EvalSim,
    _OracleEngine,
    _PixelTablePlace,
)
from blocksuite.tamp.blocks.pddl_goal_tensor_conversions import (
    ORI_QUATERNIONS,
    _quat_close_batched,
    front_camera_pixels,
    resolve_root_pose_w,
)
from blocksuite.vecgen.engine import EngineConfig
from blocksuite.vecgen.geometry import Geometry
from blocksuite.vecgen.program import SkillCall
from blocksuite.vecgen.skills.flip import Flip
from blocksuite.vecgen.skills.pnp import MAX_REACH_RADIUS, Pick, Place
from blocksuite.vecgen.skills.rotate import Rotate

if TYPE_CHECKING:
    from collections.abc import Callable

    import gymnasium as gym

    from agentic_robotics.adapters.toyblocks.config import OracleConfig

logger = logging.getLogger(__name__)

PROFILE = "medium"
"""The arm stiffness profile vecgen's skills are tuned for (feed-forward and settle times)."""

TABLE = "workspace_table"
"""Name of the table asset the skills place onto."""

CLOSED_TARGET = 0.05
"""Finger-joint target above which the gripper counts as closed (vecgen closes to 0.7, and a
flip pushes at 0.9)."""

FACE_YAW_TOL_DEG = 45.0
"""Yaw tolerance when reading which face a block rests on: any yaw has a nearest face, and the
skills correct for the block's actual yaw."""


class SkillStopped(RuntimeError):
    """Raised inside a skill when a human asked the robot to stop.

    Unwinds whatever motion was in flight from wherever it had got to. The text the *model* is
    shown is not this exception's -- it is the loop's, beside the one it shows for a call an
    interjection skipped -- because both are prompts.
    """


class OracleSkillError(RuntimeError):
    """Raised when an oracle skill cannot be executed as requested.

    Deliberately distinct from a pointing failure, which the server raises with its own message:
    the oracle exists to separate perception failures from motion failures, and one exception type
    covering both collapses the distinction it is meant to measure.
    """


class RotateDirection(StrEnum):
    """Yaw-rotation direction, as spelled by `rot_dir` in `domain.pddl`."""

    CW = "cw"
    CCW = "ccw"

    @classmethod
    def parse(cls, value: RotateDirection | str) -> RotateDirection:
        """Coerce a caller-supplied value.

        :param value: A direction or its spelling
        :return: The direction
        :raises OracleSkillError: If the value names no direction
        """
        try:
            return cls(value)
        except ValueError:
            raise OracleSkillError(
                f"Rotate direction must be one of {[d.value for d in cls]}; got '{value}'."
            ) from None


class FlipDirection(StrEnum):
    """Flip direction, as spelled by `flip_dir` in `domain.pddl`.

    The domain names the lateral flips `left_flip`/`right_flip`, so those are the member values;
    the friendlier "left"/"right" spellings are accepted as aliases.
    """

    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT_FLIP = "left_flip"
    RIGHT_FLIP = "right_flip"

    @classmethod
    def _missing_(cls, value: object) -> FlipDirection | None:
        """Accept the shorthand spellings of the two lateral flips.

        :param value: The spelling that matched no member value
        :return: The lateral flip it abbreviates, or None
        """
        return {"left": cls.LEFT_FLIP, "right": cls.RIGHT_FLIP}.get(str(value))

    @classmethod
    def parse(cls, value: FlipDirection | str) -> FlipDirection:
        """Coerce a caller-supplied value.

        :param value: A direction or its spelling
        :return: The direction
        :raises OracleSkillError: If the value names no direction
        """
        try:
            return cls(value)
        except ValueError:
            raise OracleSkillError(
                f"Flip direction must be one of {[d.value for d in cls]} "
                f"(or 'left'/'right'); got '{value}'."
            ) from None


def _load_next_faces() -> dict[tuple[str, str], str]:
    """Parse the `yaw_next` / `flip_next` transitions out of ToyBlocks's `domain.pddl`.

    :return: Map from (direction, face) to the face a rotate or flip in that direction reaches
    """
    pattern = r"\((?:flip_next|yaw_next)\s+(\w+)\s+(\w+)\s+(\w+)\s*\)"
    return {(d, o): o2 for d, o, o2 in re.findall(pattern, DOMAIN_PDDL.read_text())}


NEXT_FACE = _load_next_faces()
"""Face reached from (direction, face): the reorientation transitions the skills are asked for."""


class VecgenMotion:
    """The oracle's motion layer for one environment, on vecgen's engine and skills.

    Skills take block names and world-frame points; resolving what the agent pointed at is the MCP
    server's job. The held block is tracked by the engine (``ctx.held``), which the server's own
    ``_held_block`` mirrors.
    """

    def __init__(self, env: gym.Env, config: OracleConfig, *, on_step: Callable[[], None]) -> None:
        """Build the engine, geometry and skills against the live scene.

        :param env: The ToyBlocks gym environment to drive (one environment)
        :param config: Configuration for the oracle skills
        :param on_step: Called once after each simulated tick
        :raises RuntimeError: If the environment holds more than one parallel env
        """
        self.should_stop: Callable[[], bool] = lambda: False
        """Asked before every simulated tick. Set by whatever owns the human's stop control; the
        default never stops, so a runner with no page is unaffected."""

        self._config = config
        self._on_step = on_step
        self._base_env = env.unwrapped
        device = torch.device(self._base_env.device)
        self.sim = _EvalSim(env, device)
        if self.sim.num_envs != 1:
            raise RuntimeError(f"The oracle drives one environment; got {self.sim.num_envs}.")
        sizes, _centers = self.sim.measure_block_dims()
        self.geo = Geometry(1, self.sim.block_names, sizes, device)
        self.geo.table_top = self.sim.measure_table_top(TABLE)
        skills = {
            "pick": Pick(),
            "place_on": Place("on"),
            "place_table_px": _PixelTablePlace(),
            "flip": Flip(profile=PROFILE),
            "rotate": Rotate(),
        }
        # Synchronous: plans and skill builds resolve inside after_step, on this thread.
        engine_config = EngineConfig(
            max_attempts=1,
            max_replans=0,
            profile=PROFILE,
            async_workers=0,
            async_plans=False,
            async_programs=False,
            plan_gather_ticks=0,
        )
        self.engine = _OracleEngine(self.sim, self.geo, skills, engine_config)
        self._ctx = self.engine.ctx
        self._env_ids = torch.zeros(1, dtype=torch.long, device=device)
        self._home = self.sim.art.data.default_joint_pos[0, self.sim.arm_ids].clone()
        self.sim.apply_profile(PROFILE)

    # -- simulator ------------------------------------------------------------------------------

    def step(self, num_steps: int = 1) -> None:
        """Advance the simulator by one control tick per step, then run the step callback.

        Every tick a skill takes comes through here, which makes this the one place a stop can
        land. It is checked before each tick, so a stop costs at most the tick in flight.

        :param num_steps: Ticks to simulate
        :raises SkillStopped: If :attr:`should_stop` says a human asked the robot to halt
        """
        for _ in range(num_steps):
            if self.should_stop():
                self.hold_position()
                raise SkillStopped("A human stopped the robot mid-skill.")
            self.sim.step()
            self._on_step()

    def hold_position(self) -> None:
        """Command the arm to stay where it is, and take one tick so the command lands."""
        self.engine.cancel(0)
        art = self.sim.art
        positions = art.data.joint_pos[:, self.sim.arm_ids].clone()
        art.set_joint_position_target(positions, joint_ids=self.sim.arm_ids, env_ids=self._env_ids)
        art.set_joint_velocity_target(
            torch.zeros_like(positions), joint_ids=self.sim.arm_ids, env_ids=self._env_ids
        )
        self.sim.flush()
        self.sim.step()
        self._on_step()
        logger.info("Arm holding position after a stop.")

    def reset(self) -> None:
        """Forget the engine's state after the environment was reset under it."""
        self.engine.cancel(0)
        self.detach()
        self.sim._gripper_target[self._env_ids] = 0.0
        self.sim.apply_profile(PROFILE)

    # -- state ----------------------------------------------------------------------------------

    @property
    def held_block(self) -> str | None:
        """The block the engine believes the gripper holds, or None."""
        index = int(self._ctx.held[0])
        return self.sim.block_names[index] if index >= 0 else None

    @property
    def gripper_state(self) -> str:
        """``"closed"`` or ``"open"``, from the gripper's commanded target."""
        return "closed" if float(self.sim._gripper_target[0, 0]) > CLOSED_TARGET else "open"

    def detach(self) -> None:
        """Drop any held block from the planning world, without moving the gripper."""
        self.geo.detach(self._env_ids)
        self._ctx.held[0] = -1

    def current_face(self, block_name: str) -> str | None:
        """The face ``block_name`` rests on, at the nearest of that face's yaws.

        :param block_name: Block to classify
        :return: The face (``o1``..``o24``), or None if the block rests on no face (tipped)
        """
        quaternion = resolve_root_pose_w(self._base_env, block_name)[0:1, 3:7]
        for face, face_quaternion in ORI_QUATERNIONS.items():
            target = torch.tensor(
                [face_quaternion], device=quaternion.device, dtype=quaternion.dtype
            )
            if bool(_quat_close_batched(quaternion, target, yaw_tol_deg=FACE_YAW_TOL_DEG)[0]):
                return face
        return None

    # -- skills ---------------------------------------------------------------------------------

    def grasp(self, block_name: str) -> None:
        """Pick up a block: approach, close the gripper and lift.

        :param block_name: Block to pick up
        :raises OracleSkillError: If the gripper is holding a block or the pick fails
        """
        self._require_empty_gripper("pick up another block")
        self._run_skill(SkillCall("pick", (block_name,), hl_name="pick"), {}, "pick up that block")

    def place_on_block(self, held_block: str, support_block: str) -> None:
        """Stack the held block onto another block and release it.

        :param held_block: The block in the gripper
        :param support_block: The block to stack it on
        :raises OracleSkillError: If the gripper is not holding ``held_block`` or the place fails
        """
        self._require_holding(held_block)
        self._run_skill(
            SkillCall("place_on", (held_block, support_block, "")),
            {},
            "stack the held block there",
            settle=True,
        )

    def place_on_table(self, held_block: str, world_xyz: np.ndarray) -> None:
        """Place the held block on the table at a world-frame point and release it.

        The point is used as given when it is feasible -- in reach, in the camera's view and clear
        of other blocks; otherwise the nearest feasible point within a few pixels of it is taken.

        :param held_block: The block in the gripper
        :param world_xyz: World-frame point on the table, shape (3,)
        :raises OracleSkillError: If the point is out of reach, the gripper is not holding
            ``held_block``, or the place fails
        """
        self._require_holding(held_block)
        point = torch.as_tensor(np.asarray(world_xyz), dtype=torch.float32, device=self.sim.device)
        local = point - self.sim.env_origins[0]
        if float(torch.hypot(local[0], local[1])) > MAX_REACH_RADIUS:
            raise OracleSkillError("That point is out of the arm's reach; place closer to the arm.")
        pixel = front_camera_pixels(self._base_env, point.reshape(1, 3))[0]
        self._run_skill(
            SkillCall("place_table_px", (held_block, "")),
            {"table_uv": [float(pixel[0]), float(pixel[1])]},
            "place the held block there",
            settle=True,
        )

    def rotate(self, block_name: str, direction: RotateDirection | str) -> None:
        """Turn a block 90 degrees about the vertical axis, in place.

        :param block_name: Block to rotate
        :param direction: ``cw`` or ``ccw``, viewed from above
        :raises OracleSkillError: If the gripper is holding a block, the block rests on no face,
            or the rotate fails
        """
        rotate_dir = RotateDirection.parse(direction)
        self._reorient("rotate", block_name, rotate_dir.value)

    def flip(self, block_name: str, direction: FlipDirection | str) -> None:
        """Tip a block onto an adjacent face, then open the gripper.

        :param block_name: Block to flip
        :param direction: How the block tips, as spelled by `flip_dir` in `domain.pddl`
        :raises OracleSkillError: If the gripper is holding a block, the block rests on no face,
            or the flip fails
        """
        flip_dir = FlipDirection.parse(direction)
        self._reorient("flip", block_name, flip_dir.value)
        # vecgen's flip pushes with a closed gripper and ends closed, holding nothing.
        self._release()
        self.step(self._config.release_settle_steps)

    def return_home(self) -> None:
        """Open the gripper, dropping anything held, and drive the arm to its home configuration.

        Joint space, not a plan: this is the escape hatch from a configuration the planner cannot
        leave, so it must not itself depend on planning.
        """
        self.engine.cancel(0)
        self._release()
        self.step(self._config.release_settle_steps)
        art = self.sim.art
        start = art.data.joint_pos[0, self.sim.arm_ids].clone()
        steps = self._config.return_home_steps
        for i in range(1, steps + 1):
            blended = start + (self._home - start) * (i / steps)
            art.set_joint_position_target(
                blended.unsqueeze(0), joint_ids=self.sim.arm_ids, env_ids=self._env_ids
            )
            self.sim.flush()
            self.step()
        self.step(self._config.settle_steps)

    def open_gripper(self) -> None:
        """Open the gripper, releasing anything it holds."""
        self._release()
        self.step(self._config.release_settle_steps)

    def close_gripper(self) -> None:
        """Close the gripper.

        The engine is not told about a block that ends up between the fingers, so a block grasped
        this way is not "held" for a later stack or table place; `pick` is the skill for that.
        """
        self.sim.command_gripper(torch.ones(1, dtype=torch.bool), self._env_ids)
        self.sim.flush()
        self.step(self._config.release_settle_steps)

    # -- internals ------------------------------------------------------------------------------

    def _reorient(self, skill: str, block_name: str, direction: str) -> None:
        """Run a flip or rotate from the block's current face to the face the domain names.

        :param skill: ``"flip"`` or ``"rotate"``
        :param block_name: Block to reorient
        :param direction: The PDDL direction
        :raises OracleSkillError: If the gripper is holding a block, the block rests on no face,
            the domain has no such transition, or the skill fails
        """
        self._require_empty_gripper(f"{skill} a block")
        face = self._require_face(block_name, skill)
        next_face = NEXT_FACE.get((direction, face))
        if next_face is None:
            raise OracleSkillError(f"That block cannot {skill} {direction} from where it rests.")
        self._run_skill(
            SkillCall(skill, (block_name, direction, face, next_face)),
            {},
            f"{skill} that block",
            settle=True,
        )

    def _run_skill(
        self, call: SkillCall, params: dict[str, Any], intent: str, *, settle: bool = False
    ) -> None:
        """Run one skill to its outcome, stepping the simulator tick by tick.

        :param call: The skill call to run
        :param params: Per-call parameters the skill reads from the engine context
        :param intent: What the skill was asked to do, for the error message
        :param settle: Whether to let the scene settle after a success
        :raises OracleSkillError: If the skill fails or runs out of ticks
        :raises SkillStopped: If a human stopped the robot mid-skill
        """
        self.engine.results.pop(0, None)
        self.engine.start(0, call, params)
        for _ in range(self._config.skill_timeout_steps):
            self.engine.after_step()
            if 0 in self.engine.results:
                break
            self.engine.control()
            self.step()
        else:
            self.engine.cancel(0)
            self._after_failure()
            raise OracleSkillError(f"The arm could not {intent}: the skill did not finish in time.")
        success, reason = self.engine.results.pop(0)
        if not success:
            logger.warning(f"Skill {call} failed: {reason}")
            self._after_failure()
            raise OracleSkillError(f"The arm could not {intent} ({reason}).")
        if settle:
            self.step(self._config.release_settle_steps)

    def _after_failure(self) -> None:
        """Keep a block that is still in a closed gripper; otherwise open and forget it."""
        if self.held_block is not None and self.gripper_state == "closed":
            return
        self._release()

    def _release(self) -> None:
        """Command the gripper open and drop any held block from the planning world."""
        self.sim.command_gripper(torch.zeros(1, dtype=torch.bool), self._env_ids)
        self.sim.flush()
        self.detach()

    def _require_empty_gripper(self, action: str) -> None:
        """Raise unless the gripper is empty.

        :param action: What was asked, for the error message
        :raises OracleSkillError: If the gripper holds a block
        """
        held = self.held_block
        if held is not None:
            raise OracleSkillError(
                f"The gripper is holding '{held}'; place it before trying to {action}."
            )

    def _require_holding(self, block_name: str) -> None:
        """Raise unless the gripper holds ``block_name``.

        :param block_name: The block the caller believes is held
        :raises OracleSkillError: If the engine believes something else (or nothing) is held
        """
        if self.held_block != block_name:
            raise OracleSkillError("The gripper is not holding a block; pick one up first.")

    def _require_face(self, block_name: str, skill: str) -> str:
        """Return the face a block rests on.

        :param block_name: Block to classify
        :param skill: The skill asked for, for the error message
        :return: The face
        :raises OracleSkillError: If the block rests on no face
        """
        face = self.current_face(block_name)
        if face is None:
            raise OracleSkillError(
                f"That block is not resting flat on one of its faces, so it cannot {skill}."
            )
        return face
