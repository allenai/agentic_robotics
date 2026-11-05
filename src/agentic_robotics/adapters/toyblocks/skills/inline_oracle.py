"""The oracle skill surface, on a motion stack written directly against curobo.

The only implementation. It talks to ``curobo.wrap.reacher.motion_gen.MotionGen`` and the Isaac
Lab scene itself, and inlines the handful of ToyBlocks pose generators the skills actually use.
It registers exactly the nine skills ``OracleMCPServer`` declares.

It is not built on ToyBlocks's vectorized TAMP backend (``IsaacLabSimulatorVectorized`` /
``IsaacRobotVectorized`` / ``BatchedRLCoordinator``); "Why this exists" below says why.

Skills executed by curobo plans and RL reorientation policies rather than by a learned policy
server, so a failure can be attributed to the *agent's* planning rather than to an undertrained
checkpoint. That is the whole of what the oracle is for; it cannot appear in a demo, because it
reads simulator state the physical robot does not have.

**Two things in the ToyBlocks tree are called "oracle", and this drives the second.**
``src/blocksuite/eval/oracle/`` (``OracleHLController``, ``OracleHighLevelPolicy``) is a high-level
*symbolic* policy: it FF-plans a PDDL problem and emits ``verb + pixel`` as conditioning for a
learned low-level policy, never moves the arm, never calls ``env.step``, and has a pixel as its
**output**. The ground-truth *motion* stack that produced the training demonstrations is the other
one -- ``src/blocksuite/tamp/blocks/{generators,executors}`` plus ``TAMP/.../Backends`` -- and a
pixel is its **input**, which is the inverse relationship.

**Importing this module boots Isaac Sim.** ``cli.py`` is the caller, and passes the run's
`ToyBlocksConfig` to the server it builds from here. The imports below are ordered against the
boot and must not be hoisted.


Why this exists
---------------
Two API shapes meet if the skills are built on ToyBlocks's vectorized backend. The ToyBlocks
pose generators were written against a world-frame, no-``env_idx``, ``execute_plan_in_sim``
interface; the vectorized backend offers a base-frame, per-``env_idx``, no-executor one. Building
on that backend takes a translation layer that reconciles the two, and reconciling is all it does
-- it contributes no motion behavior of its own. Two costs follow from that, and they fall on
whoever touches the skills next rather than on whoever writes the layer: a change on either side
has to be traced through the other before its effect is knowable, and a defect *in* the layer
presents as a geometry bug while living nowhere near the geometry. Talking to curobo and the Isaac
Lab scene directly removes the mismatch instead of translating it, so the geometry a skill
executes is the geometry written here.

Portability follows from the same property. Everything except the two reorientation skills carries
its own kinematics, planning and pose generation, so pointing this surface at another simulator
means writing a new arm backend rather than a new translation of ToyBlocks's TAMP tree -- which
matters if a second domain is ever added, and does not otherwise.

Three properties of the vectorized backend also fit one environment poorly, all of them
consequences of its being built for a different job -- 16+ parallel environments at datagen
throughput, where they are the right calls:

- **Obstacle sync.** ``_get_collision_world_config`` plus ``_sync_obstacle_poses_for_env`` and
  ``_resolve_scene_obj_name`` do two things: walk the USD stage for collision geometry, and then
  overwrite every obstacle's pose from the live scene tensors. The second part is not a throughput
  optimisation for 16 environments -- it is what makes the world correct at all. With fabric
  enabled, Isaac Lab's default and how this env is created, PhysX never writes simulated transforms
  back to USD, so the stage holds each block at its spawn pose (one metre apart along x, on the
  ground plane) for the whole run. :meth:`CuroboArm.refresh_world` walks the stage once and then
  does the pose overwrite in :meth:`CuroboArm._place_at_live_poses`; the caching TAMP layers on top
  of that is what one environment can skip.
- **Attachment.** ``attach_obj`` resolves the held object's collision prim by prim path, and misses
  it for the alphabet-block USD assets, leaving curobo unaware of the block while it is
  carried. ``MotionGen.attach_objects_to_robot`` attaches by world-object name against the
  ``WorldConfig`` just read off the stage, and does its own sphere fitting, so that resolution step
  does not occur. :meth:`CuroboArm.attach` still reports when a name does not match.
- **Per-env machinery.** Locks, an IK cache, a shared attached-object sphere cache, deferred
  voxelization and a planning coordinator all collapse at one environment.

Two workarounds that translation would need are also unnecessary here. The robot prim path is
resolved from the articulation config and passed to ``get_obstacles_from_stage`` as its ignore
filter, rather than being corrected after a hardcoded ``/World/envs/env_0/Franka``; and poses are
``(7,)`` throughout because nothing produces a batched ``(1,7)``.

What is inlined, and what it was narrowed from
----------------------------------------------
The ToyBlocks generators are ~60% planner protocol -- ``get_next(flag)``, ``generate_function``,
``generate_function_err``, ``reset``, ``get_values`` -- which exists for TAMP's refinement graph and
its backtracking. The oracle wants a ranked list of poses, so the protocol is dropped and each
generator becomes a function. ``GraspPoseGenerator``'s three unused entry points
(``get_top_grasp`` / ``get_side_grasp`` / ``get_angled_grasp``) and its rerun logging go the same
way; the surviving geometry is :func:`grasp_poses`.

Candidates are also returned in the **world frame**. The generators return object-relative poses
because TAMP generates at plan time and executes later, so the object may have moved in between;
here generation and execution are one call apart, and the round trip through the object frame is
pure loss.

``tmp3.Utils.pose_utils`` has no counterpart here: ``get_relative_pose`` and
``transform_pose_to_world`` are exactly ``curobo.types.math.Pose.compute_local_pose`` and
``compute_offset_pose``. Rotations that are not poses -- composing two, inverting one, building
one from angles, reading a block's axes out of one -- go through
``scipy.spatial.transform.Rotation``, which the adapter already uses to read camera extrinsics in
``toyblocks_env.py``, so no quaternion algebra is written out here. :func:`rotation_of` and
:func:`quaternion_of` are the two converters between that and the ``(w, x, y, z)`` tensors the
scene and curobo speak.

``ORI_QUATERNIONS`` in tmp3 and ``RotatePoseGenerator.FACE_QUATERNIONS`` in ToyBlocks are the same
24 quaternions; :data:`FACE_QUATERNIONS` is the one copy.

The RL-trained skills
---------------------
``rotate`` and ``flip`` roll out RL checkpoints through :mod:`skills.rl_executors`'s
``RotateTrajectory`` / ``ReOrientTrajectory``, whose ``BaseRLTrajectory`` base instantiates the
ToyBlocks ``rotate`` / ``flip`` gym environments to build the policy's observation vector, so the
trained task's own observation semantics run against the live scene. The checkpoints, the gym
tasks and the agent networks are ToyBlocks's; the executor layer is this repo's, carried over
from the TAMP library's executors. :meth:`InlineTampMotion._run_rl_skill` is the single-environment
rollout. Everything else here -- pick, both places, retract, return-home, the gripper tools, and
all planning and collision behind them -- is curobo and Isaac Lab only.

Robot kinematics assets
-----------------------
curobo ships ``franka_robotiq_2f_85_test.yml`` and its collision spheres, but TAMP overrides the
kinematics to point at its own URDF and hand-tuned spheres, which are what planning actually uses.
Those two files total ~10 KB of text and are read from :data:`ROBOT_ASSET_DIR`, inside this repo.

The MotionGen tuning in :meth:`CuroboArm._build_motion_gen` is copied from TAMP's
``_init_motion_gen`` deliberately and should be changed only against measurements.
``interpolation_dt=0.04`` in particular equals one render step, which is what lets
:meth:`CuroboArm.execute_plan` replay a trajectory at real speed by stepping once per waypoint.

Verification
------------
Read run status off the artifacts, not off this docstring: every run writes ``run.json`` beside
its transcript under ``<runs dir>/tamp-inline-*``.

The judgment, as distinct from the log: **every tool in this file has executed successfully**, and
the geometry behind them -- curobo planning, the grasp and place pose generators, the collision
world refresh, and the RL-backed ``rotate``/``flip`` path -- has run on Isaac. What that establishes
is that the surface *works*, not that it is *reliable*: each skill has succeeded, and nothing here
measures how often. The two-conjunct ``r_on_red`` goal has been reached on its stacking conjunct,
with the orientation conjunct missed on agent strategy rather than on a skill failing.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import math
import os
import re
from collections import deque
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, TypeVar

import numpy as np
import torch
from mcp.server.mcpserver import Image as MCPImage  # noqa: TC002 (used for MCP tool registration)
from pydantic import Field
from scipy.spatial.transform import Rotation

# This import boots Isaac Sim, on whichever thread imports this module -- which is the only
# thread allowed to step it. Everything Isaac-dependent must follow it.
from agentic_robotics.adapters.toyblocks.toyblocks_env import ToyBlocksEnv

# isort: split
from agentic_robotics.adapters.toyblocks.block_pointing import find_table_z, resolve_block
from agentic_robotics.adapters.toyblocks.oracle_mcp_server import (
    FlipDirection as FlipDirectionArg,
)
from agentic_robotics.adapters.toyblocks.oracle_mcp_server import (
    OracleMCPServer,
)
from agentic_robotics.adapters.toyblocks.oracle_mcp_server import (
    RotateDirection as RotateDirectionArg,
)
from agentic_robotics.core.images import (
    PixelCoord,
    array_to_mcp_image,
    crop_around,
    resize_to_width,
)

# isort: split
# curobo lands after the Isaac boot above, not before it: `curobo.wrap.reacher.motion_gen` pulls
# in `pxr`, and a USD loaded before the app launch leaves Isaac's own extensions unable to
# register their schema wrappers. That failure is a native crash during startup, not an
# exception -- every extension logs "class wrapper ... has not been created yet" and the process
# dies in the crash reporter.
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    MotionGenResult,
)

from agentic_robotics.adapters.toyblocks import runner_bootstrap
from agentic_robotics.adapters.toyblocks.ik import solve_ik_joint_target

if TYPE_CHECKING:
    from collections.abc import Callable

    import gymnasium as gym

    from agentic_robotics.adapters.toyblocks.config import (
        OracleConfig,
        TaskConfig,
        ToyBlocksConfig,
    )
    from agentic_robotics.adapters.toyblocks.skills.rl_executors import BaseRLTrajectory
    from agentic_robotics.adapters.toyblocks.toyblocks_runner import SimulatorDispatcher

logger = logging.getLogger("agentic_robotics.examples.inline_oracle")

TOYBLOCKS_ROOT = runner_bootstrap.toyblocks_root()
"""The ToyBlocks checkout. Resolved once, in `runner_bootstrap`, which asks the installed package
and falls back to the nested checkout; two copies of that logic would drift on one machine or
both."""

DOMAIN_PDDL = TOYBLOCKS_ROOT / "src/blocksuite/tamp/blocks/domain.pddl"
"""The PDDL domain whose `yaw_next`/`flip_next` facts define the reorientation transitions."""


class SkillStopped(RuntimeError):
    """Raised inside a skill when a human asked the robot to stop.

    Unwinds whatever motion was in flight from wherever it had got to. The text the *model* is
    shown is not this exception's -- it is the loop's, beside the one it shows for a call an
    interjection skipped -- because both are prompts.
    """


RESET_RERENDERS = 30
"""Renders after a reset or scene build before the first image is read; see
`InlineOracleMCPServer.settle_and_rerender`."""

WARMUP = True
"""Whether to warm curobo's MotionGen up at boot rather than on the first plan.

Costs ~30 s once and saves it on the first plan. Set False when iterating on something that
fails before the arm ever moves.
"""

ROBOT_ASSET_DIR = runner_bootstrap.AGENTIC_ROBOTICS_ROOT / "data/robot/fr3_robotiq_2f85/curobo"
"""Directory holding the FR3 + Robotiq kinematics assets curobo plans against.

Two text files, ~10 KB together: ``..._kinematics.urdf`` and ``..._collision_spheres.yml``, carried
over from the TAMP library's ``assets/fr3_robotiq_2f85/curobo``. They describe the arm curobo
plans for; the ``.usd`` that spawns the articulation in Isaac is ToyBlocks's and stays there.
"""

ROBOT_URDF_PATH = ROBOT_ASSET_DIR / "fr3_model_full_robotiq_2f85_kinematics.urdf"
ROBOT_SPHERES_PATH = ROBOT_ASSET_DIR / "fr3_model_full_robotiq_2f85_collision_spheres.yml"


# ---------------------------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------------------------


class OracleSkillError(RuntimeError):
    """Raised when an oracle skill cannot be executed as requested.

    Deliberately distinct from a pointing failure, which the server raises with its own message:
    the oracle exists to separate perception failures from motion failures, and one exception type
    covering both collapses the distinction it is meant to measure.
    """


# ---------------------------------------------------------------------------------------------
# Quaternion and pose helpers
#
# A pose is curobo's `Pose`, because that is what planning consumes: `compute_local_pose` and
# `compute_offset_pose` are exactly `pose_utils.get_relative_pose` and `transform_pose_to_world`.
# A bare rotation is scipy's `Rotation`, which composes with `*`, inverts with `.inv()`, builds
# from angles with `.from_euler`, and hands back a block's world-frame axes as `.as_matrix()`
# columns. Composing rotations through `Pose.multiply` instead means wrapping each quaternion in a
# pose at the origin, which reads as pose arithmetic when it is quaternion arithmetic.
# ---------------------------------------------------------------------------------------------


def rotation_of(quaternion: torch.Tensor) -> Rotation:
    """Read a (w, x, y, z) quaternion as a rotation.

    :param quaternion: Quaternion (w, x, y, z), shape (4,) or (1, 4)
    :return: The rotation it encodes
    """
    return Rotation.from_quat(quaternion.reshape(-1).detach().cpu().numpy(), scalar_first=True)


def quaternion_of(rotation: Rotation, device: torch.device | str) -> torch.Tensor:
    """Write a rotation back out as the (w, x, y, z) quaternion the scene and curobo speak.

    :param rotation: Rotation to write out
    :param device: Torch device to build the tensor on
    :return: Quaternion (w, x, y, z), shape (4,)
    """
    return torch.tensor(rotation.as_quat(scalar_first=True), dtype=torch.float32, device=device)


def pose_from_state(state: torch.Tensor) -> Pose:
    """Build a `Pose` from a (7,) `(x, y, z, qw, qx, qy, qz)` state vector."""
    flat = state.reshape(-1).to(dtype=torch.float32)
    return Pose(position=flat[:3].reshape(1, 3), quaternion=flat[3:7].reshape(1, 4))


def yaw_of(quaternion: torch.Tensor) -> float:
    """Extract the world-z rotation angle a (w, x, y, z) quaternion encodes, in radians."""
    flat = quaternion.reshape(-1)
    return 2.0 * math.atan2(float(flat[3]), float(flat[0]))


# ---------------------------------------------------------------------------------------------
# Block-orientation vocabulary
#
# `domain.pddl` treats a block's orientation as one of 24 canonical faces and defines
# `yaw_next` / `flip_next` as the transition tables a rotate or flip moves between. Those tables
# are ~150 lines of ground facts, and are parsed from the domain rather than transcribed, so this
# cannot drift out of sync with the domain being planned against. A wrong table plans the *wrong*
# reorientation while every skill still runs, which looks like policy noise rather than a bug.
# ---------------------------------------------------------------------------------------------

BlockFace = StrEnum("BlockFace", {f"O{n}": f"o{n}" for n in range(1, 25)})
"""One of the 24 canonical block orientations (`O1`..`O24`), matching `ori_of` in `domain.pddl`."""

FACE_QUATERNIONS: dict[str, tuple[float, float, float, float]] = {
    "o1": (1.0, 0.0, 0.0, 0.0),
    "o2": (0.7071, 0.0, 0.0, -0.7071),
    "o3": (0.0, 0.0, 0.0, -1.0),
    "o4": (0.7071, 0.0, 0.0, 0.7071),
    "o5": (0.7071, 0.0, -0.7071, 0.0),
    "o6": (0.5, -0.5, -0.5, -0.5),
    "o7": (0.0, 0.7071, 0.0, 0.7071),
    "o8": (0.5, 0.5, -0.5, 0.5),
    "o9": (0.0, 0.0, 1.0, 0.0),
    "o10": (0.0, 0.7071, 0.7071, 0.0),
    "o11": (0.0, -1.0, 0.0, 0.0),
    "o12": (0.0, -0.7071, 0.7071, 0.0),
    "o13": (0.7071, 0.0, 0.7071, 0.0),
    "o14": (0.5, 0.5, 0.5, -0.5),
    "o15": (0.0, -0.7071, 0.0, 0.7071),
    "o16": (0.5, -0.5, 0.5, 0.5),
    "o17": (0.5, 0.5, -0.5, -0.5),
    "o18": (0.0, 0.0, 0.7071, 0.7071),
    "o19": (0.5, 0.5, 0.5, 0.5),
    "o20": (0.7071, 0.7071, 0.0, 0.0),
    "o21": (0.5, -0.5, 0.5, -0.5),
    "o22": (0.0, 0.0, 0.7071, -0.7071),
    "o23": (0.5, -0.5, -0.5, 0.5),
    "o24": (0.7071, -0.7071, 0.0, 0.0),
}
"""Canonical world-frame quaternion `(w, x, y, z)` for each block face.

The single copy of a table that exists twice in the tree today, as `tmp3.Utils.orientations`'s
``ORI_QUATERNIONS`` and as ``RotatePoseGenerator.FACE_QUATERNIONS``.
"""

FLIP_PUSH_QUATERNIONS: dict[str, tuple[float, float, float, float]] = {
    "backward": (-0.7071, 0.0, -0.7071, 0.0),
    "forward": (-0.7071, 0.0, 0.7071, 0.0),
    "left_flip": (0.7071, 0.7071, 0.0, 0.0),
    "right_flip": (0.7071, -0.7071, 0.0, 0.0),
}
"""Per-direction quaternion the flip policy reads its push direction out of.

``ReOrientTrajectory`` applies this to world up to get ``flip_push_dir_world``, so it encodes
*where the current top face ends up*, not a pose the arm moves to. It is ``FACE_FAKE_QUATERNIONS``
in ``ReOrientPoseGenerator``; the name here says what it is used for.
"""

FLIP_OFFSETS_M: dict[str, tuple[float, float]] = {
    "forward": (-0.06, 0.0),
    "backward": (0.06, 0.0),
    "left_flip": (0.0, -0.06),
    "right_flip": (0.0, 0.06),
}
"""Planar offset (dx, dy) the block travels as it tips onto its adjacent face."""


class RotateDirection(StrEnum):
    """Yaw-rotation direction, as spelled by `rot_dir` in `domain.pddl`."""

    CW = "cw"
    CCW = "ccw"

    @classmethod
    def parse(cls, value: RotateDirection | str) -> RotateDirection:
        """Coerce a caller-supplied value, raising `OracleSkillError` on an invalid one."""
        try:
            return cls(value)
        except ValueError:
            raise OracleSkillError(
                f"Rotate direction must be one of {[d.value for d in cls]}; got '{value}'."
            ) from None


class FlipDirection(StrEnum):
    """Flip direction, as spelled by `flip_dir` in `domain.pddl`.

    The domain names the lateral flips `left_flip`/`right_flip`, so those are the member values;
    they are passed straight through as PDDL argument values. The friendlier "left"/"right"
    spellings are accepted as aliases, including by Pydantic when this annotates a tool parameter.
    """

    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT_FLIP = "left_flip"
    RIGHT_FLIP = "right_flip"

    @classmethod
    def _missing_(cls, value: object) -> FlipDirection | None:
        """Accept the shorthand spellings of the two lateral flips."""
        return {"left": cls.LEFT_FLIP, "right": cls.RIGHT_FLIP}.get(str(value))

    @classmethod
    def parse(cls, value: FlipDirection | str) -> FlipDirection:
        """Coerce a caller-supplied value, raising `OracleSkillError` on an invalid one."""
        try:
            return cls(value)
        except ValueError:
            raise OracleSkillError(
                f"Flip direction must be one of {[d.value for d in cls]} "
                f"(or 'left'/'right'); got '{value}'."
            ) from None


_DirectionT = TypeVar("_DirectionT", bound=StrEnum)


def _load_next_table(
    predicate: str, direction_cls: type[_DirectionT]
) -> dict[_DirectionT, dict[str, str]]:
    """Parse `(<predicate> <direction> <face> <next_face>)` ground facts out of `domain.pddl`.

    :param predicate: Name of the PDDL predicate holding the transitions
    :param direction_cls: Enum of the directions the predicate is indexed by
    :return: Mapping from each direction, then each face, to the resulting face
    :raises RuntimeError: If the parsed table does not cover every direction and all 24 faces
    """
    table: dict[_DirectionT, dict[str, str]] = {}
    pattern = rf"\({predicate}\s+(\w+)\s+(\w+)\s+(\w+)\)"
    faces = {face.value for face in BlockFace}

    for direction, face, next_face in re.findall(pattern, DOMAIN_PDDL.read_text()):
        table.setdefault(direction_cls(direction), {})[face] = next_face

    if set(table) != set(direction_cls):
        raise RuntimeError(
            f"Directions {sorted(table)} of '{predicate}' in {DOMAIN_PDDL} don't match the "
            f"{direction_cls.__name__} members {sorted(direction_cls)}; the domain may have "
            "changed shape."
        )

    for direction, transitions in table.items():
        if set(transitions) != faces:
            raise RuntimeError(
                f"'{predicate} {direction}' in {DOMAIN_PDDL} only covers "
                f"{len(transitions)}/{len(faces)} block faces; the domain may have changed shape."
            )

    return table


YAW_NEXT = _load_next_table("yaw_next", RotateDirection)
"""Face reached by rotating a block 90 degrees in each direction."""

FLIP_NEXT = _load_next_table("flip_next", FlipDirection)
"""Face reached by flipping a block onto its adjacent face in each direction."""


def nearest_face(quaternion: torch.Tensor) -> str:
    """Classify a block's orientation as the canonical face closest to the given quaternion.

    :param quaternion: The block's world-frame orientation, shape (4,) as `(w, x, y, z)`
    :return: Name of the nearest canonical block face
    """
    normalized = quaternion.reshape(-1) / torch.linalg.norm(quaternion.reshape(-1))

    def face_error(face: str) -> float:
        """Geodesic-ish distance between the block's orientation and a canonical face."""
        face_quaternion = torch.tensor(
            FACE_QUATERNIONS[face], device=normalized.device, dtype=normalized.dtype
        )
        face_quaternion = face_quaternion / torch.linalg.norm(face_quaternion)
        return 1.0 - float(torch.abs(torch.dot(normalized, face_quaternion)))

    return min(FACE_QUATERNIONS, key=face_error)


# ---------------------------------------------------------------------------------------------
# FR3 + Robotiq kinematics overrides
#
# curobo's stock `franka_robotiq_2f_85_test.yml` describes a different link naming than the
# model-full FR3 asset ToyBlocks spawns, so these patch its kinematics to match. Copied verbatim
# from `tmp3.Config.RobotConfig`, which is the only place they exist -- they are hand-derived
# collision data, and re-deriving them would be inventing a self-collision model.
# ---------------------------------------------------------------------------------------------

FR3_COLLISION_LINK_NAMES = [
    "panda_link0",
    "panda_link1",
    "panda_link2",
    "panda_link3",
    "panda_link4",
    "panda_link5",
    "panda_link6",
    "panda_link7",
    "robotiq_arg2f_base_link",
    "left_outer_knuckle",
    "right_outer_knuckle",
    "left_outer_finger",
    "right_outer_finger",
    "left_inner_knuckle",
    "right_inner_knuckle",
    "left_inner_finger",
    "right_inner_finger",
    "attached_object",
]

FR3_SELF_COLLISION_IGNORE = {
    "panda_link0": ["panda_link1", "panda_link2"],
    "panda_link1": ["panda_link2", "panda_link3"],
    "panda_link2": ["panda_link3", "panda_link4"],
    "panda_link3": ["panda_link4", "panda_link5"],
    "panda_link4": ["panda_link5", "panda_link6", "panda_link7"],
    "panda_link5": ["panda_link6", "panda_link7"],
    "panda_link6": ["panda_link7", "robotiq_arg2f_base_link"],
    "panda_link7": ["robotiq_arg2f_base_link"],
    "robotiq_arg2f_base_link": [
        "left_outer_knuckle",
        "right_outer_knuckle",
        "left_outer_finger",
        "right_outer_finger",
        "left_inner_knuckle",
        "right_inner_knuckle",
        "left_inner_finger",
        "right_inner_finger",
        "attached_object",
    ],
    "left_outer_knuckle": [
        "right_outer_knuckle",
        "left_outer_finger",
        "left_inner_knuckle",
        "right_inner_knuckle",
        "left_inner_finger",
        "attached_object",
    ],
    "right_outer_knuckle": [
        "right_outer_finger",
        "left_inner_knuckle",
        "right_inner_knuckle",
        "right_inner_finger",
        "attached_object",
    ],
    "left_outer_finger": ["left_inner_knuckle", "left_inner_finger", "attached_object"],
    "right_outer_finger": ["right_inner_knuckle", "right_inner_finger", "attached_object"],
    "left_inner_knuckle": ["right_inner_knuckle", "left_inner_finger", "attached_object"],
    "right_inner_knuckle": ["right_inner_finger", "attached_object"],
    "left_inner_finger": ["attached_object"],
    "right_inner_finger": ["attached_object"],
    "attached_object": [],
}

FR3_SELF_COLLISION_BUFFER = dict.fromkeys(FR3_COLLISION_LINK_NAMES, 0.0)
FR3_SELF_COLLISION_BUFFER.update({"panda_link0": 0.1, "panda_link1": 0.05})

ARM_JOINT_NAMES = [f"panda_joint{n}" for n in range(1, 8)]
GRIPPER_JOINT_NAMES = ["finger_joint"]

ROBOTIQ_OPEN_TARGET = 0.0
ROBOTIQ_CLOSE_TARGET = 0.7
"""Robotiq ``finger_joint`` targets. The close value matches tmp3's
`ROBOTIQ_DEFAULT_CLOSE_TARGET`."""

PHYS_STEPS_PER_RENDER = 4
"""Physics substeps per render interval, as `PHYS_STEP_PER_RENDER_INTERVAL` in TAMP's simulator.

At the scene's 0.01 s physics dt this makes one :meth:`CuroboArm.step` advance 0.04 s, which is
what ``interpolation_dt`` in :meth:`CuroboArm._build_motion_gen` is matched to.
"""


def build_robot_config() -> dict:
    """Build the curobo robot kinematics config for the FR3 + Robotiq arm.

    Starts from curobo's own shipped yaml, then applies the two things that are not curobo's: the
    kinematics assets under :data:`ROBOT_ASSET_DIR`, and the model-full link naming that the FR3
    asset uses. Both come from `tmp3.Config.RobotConfig`.

    :return: The ``robot_cfg`` dict, ready for `MotionGenConfig.load_from_robot_config`
    :raises RuntimeError: If either kinematics asset is missing
    """
    for asset in (ROBOT_URDF_PATH, ROBOT_SPHERES_PATH):
        if not asset.is_file():
            raise RuntimeError(
                f"Robot kinematics asset not found: {asset}. curobo plans against this file; see "
                "the ROBOT_ASSET_DIR note in this module's header."
            )

    stock = join_path(get_robot_configs_path(), "franka_robotiq_2f_85_test.yml")
    robot_cfg = load_yaml(stock)["robot_cfg"]
    kinematics = robot_cfg["kinematics"]

    kinematics["base_link"] = "panda_link0"
    kinematics["ee_link"] = "robotiq_arg2f_base_link"
    kinematics["urdf_path"] = str(ROBOT_URDF_PATH)
    kinematics["asset_root_path"] = str(ROBOT_ASSET_DIR)
    kinematics["collision_spheres"] = str(ROBOT_SPHERES_PATH)
    kinematics["extra_collision_spheres"] = {"attached_object": 100}
    kinematics["collision_link_names"] = list(FR3_COLLISION_LINK_NAMES)
    kinematics["mesh_link_names"] = list(FR3_COLLISION_LINK_NAMES)
    kinematics["self_collision_ignore"] = {k: list(v) for k, v in FR3_SELF_COLLISION_IGNORE.items()}
    kinematics["self_collision_buffer"] = dict(FR3_SELF_COLLISION_BUFFER)
    kinematics["lock_joints"] = {"finger_joint": 0.0}
    kinematics["extra_links"] = {
        "attached_object": {
            "parent_link_name": "robotiq_arg2f_base_link",
            "link_name": "attached_object",
            "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
            "joint_type": "FIXED",
            "joint_name": "attach_joint",
        },
    }

    return robot_cfg


# ---------------------------------------------------------------------------------------------
# The motion backend
# ---------------------------------------------------------------------------------------------

StepCallback = Any
"""Called once after each simulated step, e.g. to record a video frame."""


class CuroboArm:
    """Collision-aware planning and execution for one Isaac Lab arm, straight onto curobo.

    The single-environment counterpart of TAMP's `IsaacRobotVectorized`. Poses crossing
    this boundary are **world-frame** and shaped ``(1, 3)`` / ``(1, 4)`` inside a curobo `Pose`;
    curobo plans in the robot's base frame, and the conversion happens here rather than being a
    caller's concern.

    Every step of the simulator taken by a skill goes through :meth:`step`, so ``on_step`` observes
    all of them. That is what drives per-waypoint video frames: recording once per completed skill
    would make a grasp that failed mid-trajectory and one that succeeded look identical in the
    video.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        on_step: StepCallback,
        seed: int | None = 0,
        warmup: bool = True,
    ) -> None:
        """Stand up curobo's MotionGen against the live Isaac Lab scene.

        :param env: The ToyBlocks gym environment to drive
        :param on_step: Called once after each simulated step
        :param seed: Seed for curobo's IK and graph samplers, or None to leave them unseeded
        :param warmup: Whether to warm MotionGen up now rather than on the first plan
        """
        self.should_stop: Any = lambda: False
        """Asked before every simulated step. Set by whatever owns the human's stop control; the
        default never stops, so a runner with no page is unaffected."""

        self._env = env
        self._base_env = env.unwrapped
        self._scene = self._base_env.scene
        self._on_step = on_step

        self.device = self._scene.device
        self._tensor_args = TensorDeviceType(device=torch.device(self.device))

        self._articulation = self._base_env.robot.articulation
        self._arm_joint_ids = self._articulation.find_joints(ARM_JOINT_NAMES)[0]
        self._gripper_joint_ids = self._articulation.find_joints(GRIPPER_JOINT_NAMES)[0]
        self.joint_names = list(ARM_JOINT_NAMES)

        self._robot_prim_path = self._resolve_robot_prim_path()
        logger.info(f"Robot prim path for curobo: {self._robot_prim_path}")

        self._usd_helper = UsdHelper()
        self._usd_helper.load_stage(self._scene.stage)

        # Read the scene before building MotionGen: warmup traces CUDA graphs against whatever
        # world it was configured with, so warming up on an empty one and swapping the real scene
        # in afterwards is a worse starting point than configuring it correctly once.
        self._world = self._read_world()
        self._motion_gen = self._build_motion_gen(seed=seed, warmup=warmup)
        self._attached: str | None = None
        self._gripper_open = True

    # -- setup ---------------------------------------------------------------------------------

    def _resolve_robot_prim_path(self) -> str:
        """Resolve the arm's prim path from its articulation config.

        TAMP hardcodes ``/World/envs/env_0/Franka`` and then corrects it, because ToyBlocks mounts
        the arm at ``.../Robot``; left uncorrected, the robot's own links enter curobo's collision
        world and every plan fails with ``INVALID_START_STATE_WORLD_COLLISION``. Reading it off the
        config removes the hardcode and the correction together.

        :return: Absolute prim path of the arm in environment 0
        """
        prim_expr = getattr(self._articulation.cfg, "prim_path", None)
        if not prim_expr:
            raise RuntimeError("The robot articulation config carries no prim_path.")

        leaf = prim_expr.rstrip("/").split("/")[-1]
        return f"/World/envs/env_0/{leaf}"

    def _build_motion_gen(self, *, seed: int | None, warmup: bool) -> MotionGen:
        """Build and optionally warm up curobo's MotionGen.

        The tuning here is copied from TAMP's ``_init_motion_gen`` rather than re-derived. It is
        tuned against this scene and this robot, and getting it wrong does not raise -- it lowers
        the fraction of poses that plan, which with no verifier is indistinguishable from the agent
        planning badly. ``interpolation_dt=0.04`` does more than shape planning quality: it
        equals one render step, which is what lets :meth:`execute_plan` step once per waypoint.

        :param seed: Seed for the IK and graph samplers, or None to leave them unseeded
        :param warmup: Whether to warm up now rather than on the first plan
        :return: The configured motion generator
        """
        seed_kwargs = {} if seed is None else {"ik_seed": seed, "graph_seed": seed}

        config = MotionGenConfig.load_from_robot_config(
            build_robot_config(),
            self._world,
            self._tensor_args,
            trajopt_tsteps=32,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            interpolation_dt=0.04,
            collision_cache={"obb": 30, "mesh": 30},
            store_ik_debug=False,
            store_trajopt_debug=False,
            minimize_jerk=True,
            maximum_trajectory_dt=0.25,
            **seed_kwargs,
        )
        motion_gen = MotionGen(config)

        if warmup:
            logger.info("Warming up curobo's MotionGen (once, ~30 s)...")
            motion_gen.warmup(parallel_finetune=True)

        self._plan_config = MotionGenPlanConfig(
            enable_graph=False,
            max_attempts=128,
            timeout=20.0,
            enable_graph_attempt=None,
            enable_finetune_trajopt=True,
            partial_ik_opt=False,
            parallel_finetune=True,
            pose_cost_metric=None,
        )
        return motion_gen

    # -- collision world -----------------------------------------------------------------------

    def refresh_world(self) -> None:
        """Re-read the scene's collision geometry and hand it to curobo.

        With nothing held the world is reloaded whole. While a block is attached the obstacles
        the checker already knows, other than the held block's own, are moved to their live poses
        in place and nothing is reloaded. A reload with a block attached leaves curobo's
        start-state check reading a world collision that no listed obstacle accounts for (every
        obstacle disabled, still colliding; see `held_refresh_probe.py`), and re-reading the held
        block puts it back as an obstacle around the closed fingers. Either way every plan then
        fails with ``INVALID_START_STATE_WORLD_COLLISION`` until the block is released, which is
        how the Naive pilot of 2026-10-06 lost nine placements in a row. Blocks never appear or
        vanish mid-episode, so a pose update is the whole of what a refresh while holding needs.
        """
        debug = self._attached is not None and os.environ.get("AGENTIC_WORLD_DEBUG") == "1"
        if debug:
            logger.info(f"[world debug] before refresh, start state: {self._start_state_status()}")
        world = self._read_world()
        if self._attached is None:
            self._world = world
            self._motion_gen.update_world(self._world)
            return

        checker = self._motion_gen.world_coll_checker
        held = set(self._obstacle_names_for(self._attached))
        known = {name for name in checker.get_obstacle_names() if name}
        for obstacle in world.objects:
            if obstacle.name in held:
                continue
            if obstacle.name not in known:
                logger.warning(
                    f"Obstacle '{obstacle.name}' appeared while a block is held; "
                    "it is not modelled until the next full refresh."
                )
                continue
            checker.update_obstacle_pose(
                name=obstacle.name,
                w_obj_pose=Pose.from_list(obstacle.pose, self._tensor_args),
                env_idx=0,
            )
        if debug:
            logger.info(
                f"[world debug] after in-place refresh, start state: {self._start_state_status()}"
            )

    def _start_state_status(self) -> str | None:
        """curobo's verdict on the current joint state, None when valid (debugging aid)."""
        valid, status = self._motion_gen.check_start_state(self.joint_state())
        return None if valid else str(status)

    def _read_world(self) -> WorldConfig:
        """Read the scene's collision geometry off the USD stage, at the poses the simulator has.

        This is the whole of what TAMP spends ``_get_collision_world_config``,
        ``_sync_obstacle_poses_for_env`` and ``_resolve_scene_obj_name`` on. The stage walk gives
        the geometry only: with fabric on, the stage's transforms are the spawn poses, never the
        simulated ones (see the module docstring). :meth:`_place_at_live_poses` then moves each
        obstacle to where its scene object actually is and expresses it in the robot's base frame,
        the frame curobo plans in. Passing the robot as ``reference_prim_path`` instead, without the
        re-posing, checks every plan against blocks sitting on the ground plane a meter or more
        from the table.

        Unlike TAMP's version this does **not** filter out the table prims and then synthesize a
        replacement cuboid. curobo reads whatever real table collider the scene provides; only if
        it finds none does :meth:`_synthesized_table` add one, at the height measured from the
        resting blocks. TAMP's own comment records the failure the unconditional version caused --
        a phantom table box floating ~0.3 m above the real surface when both were present.

        :return: The scene's obstacles, in the robot's base frame
        """
        obstacles = self._usd_helper.get_obstacles_from_stage(
            only_paths=[self._ENV_PRIM_ROOT],
            ignore_substring=[
                self._robot_prim_path,
                "/World/target",
                "/World/defaultGroundPlane",
                "/curobo",
                "material",
                "Visuals",  # Visual meshes are not collision geometry
                "Sites",  # Registration/site bounding boxes are not collision geometry
            ],
        ).get_collision_check_world()
        self._place_at_live_poses(obstacles)

        if not self._has_table(obstacles):
            table = self._synthesized_table()
            if table is not None:
                obstacles.add_obstacle(table)
                logger.info("No table collider on the stage; added a synthesized table cuboid.")

        names = [obstacle.name for obstacle in obstacles.objects]
        logger.debug(f"Read {len(names)} collision obstacles from the stage: {names}")

        return obstacles

    _ENV_PRIM_ROOT = "/World/envs/env_0"

    def _place_at_live_poses(self, obstacles: WorldConfig) -> None:
        """Move each obstacle to where its scene object is now, and express it in the base frame.

        Obstacles arrive in the world frame as the USD stage has them. For an obstacle under a
        scene object's prim, the rigid offset between the object's root prim and the collider prim
        is read off the stage -- both poses there are stale, but the offset between them is not --
        and re-applied to the object's live root pose. Re-applying the offset, rather than writing
        the root pose over the collider pose as TAMP does, keeps colliders that are not centred on
        their root (the alphabet blocks' nested ``Collisions`` meshes) where they belong.
        Obstacles under no scene object (the table) are static, so their stage pose is right.

        :param obstacles: World-frame obstacles as read from the stage; poses are rewritten in place
        """
        stale_roots: dict[str, Pose] = {}
        moved: list[str] = []
        for obstacle in obstacles.objects:
            world_pose = Pose.from_list(list(obstacle.pose), self._tensor_args)
            key = self._scene_key_for(obstacle.name)
            if key is not None:
                if key not in stale_roots:
                    matrix = torch.as_tensor(
                        np.asarray(self._usd_helper.get_pose(f"{self._ENV_PRIM_ROOT}/{key}")),
                        device=self._tensor_args.device,
                        dtype=self._tensor_args.dtype,
                    )
                    stale_roots[key] = Pose.from_matrix(matrix)
                live_root = pose_from_state(self._scene[key].data.root_pose_w[0].detach().cpu()).to(
                    self._tensor_args
                )
                root_to_collider = stale_roots[key].inverse().multiply(world_pose)
                world_pose = live_root.multiply(root_to_collider)
                moved.append(obstacle.name)
            obstacle.pose = self._to_base(world_pose).to_list()
        logger.debug(f"Re-posed {len(moved)} obstacles from live scene poses: {moved}")

    def _scene_key_for(self, obstacle_name: str) -> str | None:
        """The scene object an obstacle belongs to, or None for a static prim.

        curobo names obstacles after their prim paths, and the env spawns every scene object at
        ``/World/envs/env_.*/<scene key>``, so the path segment after the env root is the key.
        Only objects with a simulated root pose count; a static prim has no live pose to move to.

        :param obstacle_name: The obstacle's name, a prim path
        :return: The Isaac Lab scene key, or None
        """
        prefix = f"{self._ENV_PRIM_ROOT}/"
        if not obstacle_name.startswith(prefix):
            return None
        segment = obstacle_name[len(prefix) :].split("/", 1)[0]
        # `InteractiveScene` has `keys()` and `__getitem__` but no `__contains__`
        if segment not in self._scene.keys():  # noqa: SIM118
            return None
        data = getattr(self._scene[segment], "data", None)
        return segment if hasattr(data, "root_pose_w") else None

    @staticmethod
    def _has_table(world: WorldConfig) -> bool:
        """Whether the world config already carries a table-surface collider."""
        return any("table" in obstacle.name.lower() for obstacle in world.objects)

    def _synthesized_table(self) -> Cuboid | None:
        """Build a table cuboid in the robot's base frame, for a scene with no table collider.

        The top surface is placed at the height the resting blocks imply, which is the same
        measurement `find_table_z` makes for pointing.

        :return: The cuboid, or None if the table height could not be measured
        """
        try:
            block_names = sorted(
                name for name in self._scene.rigid_objects if "block" in name.lower()
            )
            if not block_names:
                return None

            lowest = min(
                block_names,
                key=lambda name: float(self._scene.rigid_objects[name].data.root_pos_w[0, 2]),
            )
            asset = self._scene.rigid_objects[lowest]
            from blocksuite.utils.obs_utils import get_bounding_box

            bbox = get_bounding_box(asset)  # (n, 2, 3)
            half_height = float((bbox[0, 1] - bbox[0, 0]).abs().max()) / 2.0
            table_top_z = float(asset.data.root_pos_w[0, 2]) - half_height
        except Exception:
            logger.exception("Could not measure the table height; planning without a table.")
            return None

        thickness = 0.1
        centre_w = torch.tensor(
            [0.5, 0.0, table_top_z - thickness / 2.0, 1.0, 0.0, 0.0, 0.0], device=self.device
        )
        centre_base = self._to_base(pose_from_state(centre_w))

        position = centre_base.position.reshape(-1).tolist()
        quaternion = centre_base.quaternion.reshape(-1).tolist()

        return Cuboid(
            name="synthesized_table",
            pose=[*position, *quaternion],
            dims=[1.2, 0.9, thickness],
        )

    # -- frames --------------------------------------------------------------------------------

    @property
    def base_pose_w(self) -> Pose:
        """The arm's root pose in the world frame."""
        return pose_from_state(self._base_env.robot.data.root_pose_w[0].detach().cpu())

    def _to_base(self, world_pose: Pose) -> Pose:
        """Express a world-frame pose in the robot's base frame."""
        return self.base_pose_w.to(self._tensor_args).compute_local_pose(
            world_pose.to(self._tensor_args)
        )

    def _to_world(self, base_pose: Pose) -> Pose:
        """Express a base-frame pose in the world frame."""
        return self.base_pose_w.to(self._tensor_args).compute_offset_pose(
            base_pose.to(self._tensor_args)
        )

    # -- state ---------------------------------------------------------------------------------

    def step(self, num_steps: int = 1) -> None:
        """Advance the simulator by one render interval per step, then run the step callback.

        The scene clock is advanced by ``physics_dt * PHYS_STEPS_PER_RENDER``, matching what TAMP's
        vectorized simulator does, because that product is the 0.04 s render interval curobo's
        ``interpolation_dt`` is set to. Advancing by the bare physics dt instead would leave the
        scene's clock running at a quarter speed relative to the trajectories being replayed.

        Every simulated step a skill takes comes through here, which is what makes this the one
        place a stop can land. It is checked before each step rather than after, so a stop costs
        at most the render interval already in flight.

        :raises SkillStopped: If :attr:`should_stop` says a human asked the robot to halt
        """
        for _ in range(num_steps):
            if self.should_stop():
                self.hold_position()
                raise SkillStopped("A human stopped the robot mid-skill.")
            self._base_env.sim.step()
            self._scene.update(self._base_env.physics_dt * PHYS_STEPS_PER_RENDER)
            self._on_step()

    def hold_position(self) -> None:
        """Command the arm to stay where it is, and let that command take effect.

        Without this the arm keeps tracking the last waypoint it was sent, so a stop would be a
        pause in the commanding rather than in the motion. One step is taken so the hold reaches
        the simulator; it is the step this method exists to make the last one.
        """
        positions = self._articulation.data.joint_pos[0, self._arm_joint_ids].clone()
        self._articulation.set_joint_position_target(
            positions.unsqueeze(0),
            joint_ids=self._arm_joint_ids,
            env_ids=torch.tensor([0], device=self.device),
        )
        self._articulation.write_data_to_sim()
        self._base_env.sim.step()
        self._scene.update(self._base_env.physics_dt * PHYS_STEPS_PER_RENDER)
        self._on_step()
        logger.info("Arm holding position after a stop.")

    def joint_state(self) -> JointState:
        """The arm's current joint configuration, as curobo expects it for planning."""
        positions = self._articulation.data.joint_pos[0, self._arm_joint_ids].clone()
        return JointState.from_position(
            positions.unsqueeze(0).to(self._tensor_args.device), joint_names=self.joint_names
        )

    def get_pose(self, obj_name: str) -> torch.Tensor:
        """World-frame pose of a named rigid object, shape (7,) as `(x, y, z, qw, qx, qy, qz)`."""
        return self._scene.rigid_objects[obj_name].data.root_pose_w[0].clone()

    def get_size(self, obj_name: str) -> torch.Tensor:
        """Local-frame (x, y, z) size of a named rigid object, in metres, shape (3,)."""
        from blocksuite.utils.obs_utils import get_bounding_box

        bbox = get_bounding_box(self._scene.rigid_objects[obj_name])  # (n, 2, 3)
        return (bbox[0, 1] - bbox[0, 0]).abs()

    def get_curr_pose(self) -> Pose:
        """The end-effector's current world-frame pose.

        curobo's forward kinematics reports the pose in the robot's base frame, and its sequence
        tensors carry a trajectory dimension even for a single state, so both are flattened back
        to the ``(1, 3)`` / ``(1, 4)`` shape every other pose here uses.
        """
        ee_pose = self._motion_gen.compute_kinematics(self.joint_state()).ee_pose
        base_pose = Pose(
            position=ee_pose.position.reshape(-1, 3)[:1],
            quaternion=ee_pose.quaternion.reshape(-1, 4)[:1],
        )
        return self._to_world(base_pose)

    @property
    def block_names(self) -> list[str]:
        """Names of the blocks in the scene."""
        return sorted(name for name in self._scene.rigid_objects if "block" in name.lower())

    @property
    def gripper_state(self) -> str:
        """Current gripper state, either "open" or "closed"."""
        return "open" if self._gripper_open else "closed"

    # -- planning ------------------------------------------------------------------------------

    def collision_free_ik(self, world_pose: Pose) -> bool:
        """Whether a world-frame end-effector pose has a collision-free IK solution.

        `MotionGen` exposes its `IKSolver` directly, so this is curobo's own solver against the
        world config already loaded by :meth:`refresh_world` -- no cache, and no separate
        collision model to keep in sync with the planner's.

        :param world_pose: The end-effector pose to test, in the world frame
        :return: True if a collision-free solution exists
        """
        try:
            result = self._motion_gen.ik_solver.solve_single(self._to_base(world_pose))
        except Exception:
            logger.exception("IK solve raised; treating the pose as unreachable.")
            return False

        return bool(torch.any(result.success).item())

    def plan(self, world_pose: Pose) -> MotionGenResult | None:
        """Plan a collision-free motion to a world-frame end-effector pose.

        :param world_pose: Target end-effector pose, in the world frame
        :return: The planning result if one succeeded, else None
        """
        try:
            result = self._motion_gen.plan_single(
                self.joint_state(), self._to_base(world_pose), self._plan_config.clone()
            )
        except Exception:
            logger.exception("Motion planning raised; treating the target as unreachable.")
            return None

        if bool(result.success.item()):
            return result

        # curobo rejects an invalid start state before it plans anything, and says which of joint
        # limits, world collision and self-collision it was. That distinction is the one a failure
        # here cannot be read back from the transcript without.
        logger.info(f"Motion planning failed: {result.status}")
        return None

    def execute_plan(self, plan: MotionGenResult) -> None:
        """Replay an interpolated plan onto the simulator.

        curobo's ``interpolation_dt`` is 0.04 s, which is one :meth:`step` render interval, so one
        step per interpolated waypoint plays the trajectory back at real speed. Stepping more than
        once per waypoint would play it back in slow motion.

        :param plan: A successful planning result
        """
        trajectory = plan.get_interpolated_plan().get_ordered_joint_state(self.joint_names)
        env_ids = torch.tensor([0], device=self.device)

        for waypoint in range(trajectory.position.shape[0]):
            self._articulation.set_joint_position_target(
                trajectory.position[waypoint].unsqueeze(0).to(self.device),
                joint_ids=self._arm_joint_ids,
                env_ids=env_ids,
            )
            self._articulation.write_data_to_sim()
            self.step()

    def plan_and_execute(self, world_pose: Pose) -> bool:
        """Plan to a world-frame pose and, if that succeeds, run it.

        :param world_pose: Target end-effector pose, in the world frame
        :return: True if a plan was found and executed
        """
        plan = self.plan(world_pose)
        if plan is None:
            return False

        self.execute_plan(plan)
        return True

    def ik_joint_target(self, world_pose: Pose) -> torch.Tensor | None:
        """Joint configuration reaching a world-frame pose, seeded from where the arm is now.

        ToyBlocks's ``IKInterpTrajectory`` is the implementation, and is the same solve backing
        the ``grasp_delta_traj`` and ``place_delta_traj`` segments of its pnp actions. Its
        acceptance rules are the ones wanted here: a solution that converged on the pose is taken
        even when CuRobo flags it as in collision, and one that folds the arm into itself or sits
        within 0.15 rad of a joint limit is refused.

        :param world_pose: Target end-effector pose, in the world frame
        :return: Arm joint positions ordered as :attr:`joint_names`, shape (7,), or None
        """
        try:
            return solve_ik_joint_target(
                self._motion_gen, self._to_base(world_pose), self.joint_state(), self.joint_names
            )
        except Exception:
            logger.exception("Seeded IK raised; treating the pose as unreachable.")
            return None

    def ik_interp_to(self, world_pose: Pose, *, steps: int) -> bool:
        """Move to a world-frame pose by IK at the target and a joint-space ramp, not by planning.

        For the short segments where contact is the intent -- setting a block down on another,
        lifting it back off -- and where the motion planner therefore refuses, because its
        collision checker sees that contact and rejects the goal.

        Nothing checks the world on the way, so this is only for a few centimetres along a
        direction already known to be clear. Anything longer is :meth:`plan_and_execute`'s job.

        :param world_pose: Target end-effector pose, in the world frame
        :param steps: Simulated steps the ramp is spread over
        :return: True if IK solved and the ramp ran
        """
        target = self.ik_joint_target(world_pose)
        if target is None:
            return False

        self.move_arm_to_joint_positions(target, steps=steps)
        return True

    # -- attachment ----------------------------------------------------------------------------

    def _obstacle_names_for(self, block_name: str) -> list[str]:
        """Find the collision-world obstacles belonging to a named block.

        curobo names obstacles after the USD prims they were read from, so a block's obstacles are
        those under its prim: the path segment after the env root, which :meth:`_scene_key_for`
        already extracts. Matching on the world config -- rather than resolving prim paths as
        TAMP's ``attach_obj`` does -- is what makes this work for the alphabet-block USD assets
        whose collision prim that resolution misses.

        The match is on that segment, not a substring of the whole path. The letter blocks' asset
        root prim is ``Alphabet_Block_Large``, which contains ``block_l``, so a substring match
        attaches every letter block in the scene to the gripper when ``block_l`` is attached and
        removes them all from the world. That is the run ``menu-ai2-stacked-04-20260915-171421``:
        every placement of that block failed, the "wedged" probe fired after each pick of it, and
        no other block's placement failed.

        :param block_name: Name of the block in the Isaac Lab scene
        :return: Names of the matching obstacles, possibly empty
        """
        return [
            obstacle.name
            for obstacle in self._world.objects
            if self._scene_key_for(obstacle.name) == block_name
        ]

    def attach(self, block_name: str) -> bool:
        """Attach a grasped block to the gripper, so curobo plans around it while it is carried.

        :param block_name: Name of the block now held
        :return: True if the block was attached to the planning model
        """
        obstacle_names = self._obstacle_names_for(block_name)
        if not obstacle_names:
            logger.warning(
                f"No collision obstacle matches '{block_name}', so curobo will not model it while "
                f"it is carried. The block is still physically grasped. Obstacles present: "
                f"{[obstacle.name for obstacle in self._world.objects]}"
            )
            return False

        try:
            # The block rests on the table when it is attached, so spheres sampled on its
            # underside sit at zero distance from the table collider and curobo's start-state
            # check rejects every plan out of the grasp as a world collision; the planned
            # retract then fails and only the IK lift is left (every pick in the 2026-10-06
            # pilot log, and the pick that stayed down in `held_refresh_probe.py`). Lifting the
            # modelled block `ATTACH_LIFT_M` clears that; the value is TAMP's `attach_obj` one.
            self._motion_gen.attach_objects_to_robot(
                self.joint_state(),
                obstacle_names,
                world_objects_pose_offset=Pose.from_list(
                    [0.0, 0.0, ATTACH_LIFT_M, 1.0, 0.0, 0.0, 0.0], self._tensor_args
                ),
                remove_obstacles_from_world_config=True,
            )
        except Exception:
            logger.exception(f"Could not attach '{block_name}' to the planning model.")
            return False

        self._attached = block_name
        return True

    def detach(self) -> None:
        """Detach whatever is attached to the gripper, and re-read the collision world."""
        if self._attached is None:
            return

        try:
            self._motion_gen.detach_object_from_robot()
        except Exception:
            logger.exception("Could not detach the held block from the planning model.")

        self._attached = None
        self.refresh_world()  # The released block is an obstacle again.

    # -- actuation -----------------------------------------------------------------------------

    def _write_gripper(self, target: float, *, settle_steps: int) -> None:
        """Drive the gripper joint to a target and let it settle."""
        env_ids = torch.tensor([0], device=self.device)
        self._articulation.set_joint_position_target(
            torch.tensor([[target]], device=self.device),
            joint_ids=self._gripper_joint_ids,
            env_ids=env_ids,
        )
        self._articulation.write_data_to_sim()
        self.step(settle_steps)

    def open_gripper(self, *, settle_steps: int = 24) -> None:
        """Open the gripper."""
        self._write_gripper(ROBOTIQ_OPEN_TARGET, settle_steps=settle_steps)
        self._gripper_open = True

    def close_gripper(self, *, settle_steps: int = 24) -> None:
        """Close the gripper."""
        self._write_gripper(ROBOTIQ_CLOSE_TARGET, settle_steps=settle_steps)
        self._gripper_open = False

    GRIPPER_OPEN_BELOW = 0.25
    """`finger_joint` reading below which the gripper counts as open (0.0 open, 0.5 on a block,
    0.7 closed on nothing)."""

    def sync_gripper_flag(self) -> None:
        """Set the open/closed bookkeeping from the finger joint as it actually is.

        The RL reorientation policies drive the gripper themselves, so after a rotate the
        fingers are open while the flag still says whatever the last `open_gripper` or
        `close_gripper` left it: closed after a flip or a feinted pick. The quirky pick reads
        the flag, and in the 2026-10-06 Oracle suite a pick straight after flip-then-rotate was
        declined for a closed gripper that was open (10 episodes).
        """
        finger = float(self._articulation.data.joint_pos[0, self._gripper_joint_ids[0]])
        self._gripper_open = finger < self.GRIPPER_OPEN_BELOW

    def set_gripper_opening(self, opening: float) -> None:
        """Set the gripper joint target directly, without stepping.

        The RL reorientation checkpoints expect a specific opening at handoff, and set it as part
        of their own start state rather than as a gripper action.
        """
        env_ids = torch.tensor([0], device=self.device)
        self._articulation.set_joint_position_target(
            torch.tensor([[opening]], device=self.device),
            joint_ids=self._gripper_joint_ids,
            env_ids=env_ids,
        )
        self._articulation.write_data_to_sim()

    def move_arm_to_joint_positions(self, positions: torch.Tensor, *, steps: int) -> None:
        """Ramp the arm joints to `positions` over `steps` simulated steps.

        Joint space, and deliberately so: this is the escape hatch from a configuration curobo
        cannot plan out of, so it must not itself depend on planning. Interpolating rather than
        writing the target once keeps the PD controller tracking a reachable setpoint instead of
        flinging the arm at a distant one.

        :param positions: Target arm joint positions, shape (7,)
        :param steps: Simulated steps to spread the motion over
        """
        env_ids = torch.tensor([0], device=self.device)
        start = self._articulation.data.joint_pos[0, self._arm_joint_ids].clone()
        target = positions.to(device=start.device, dtype=start.dtype)

        for step in range(1, steps + 1):
            blended = start + (target - start) * (step / steps)
            self._articulation.set_joint_position_target(
                blended.unsqueeze(0), joint_ids=self._arm_joint_ids, env_ids=env_ids
            )
            self._articulation.write_data_to_sim()
            self.step()


# ---------------------------------------------------------------------------------------------
# Pose generators, inlined
#
# Each is the geometry of the ToyBlocks generator of the same name, with the planner protocol
# removed and world-frame poses returned directly. See this module's header for why.
# ---------------------------------------------------------------------------------------------

GRASP_CLEARANCE_M = 0.14
"""Distance the gripper is backed off the block's surface along its approach direction."""

ATTACH_LIFT_M = 0.015
"""How far above its resting pose a grasped block is modelled when attached to the gripper."""

GRASP_YAW_COUNT = 4
"""Yaw candidates generated per grasp, one per vertical face of the block."""

GRASP_PITCH = math.pi
"""Approach pitch used for grasps. `pi` is top-down.

``GraspPoseGenerator.get_all_grasps`` generates only this pitch: its side and angled variants are
either unused entry points or commented out. Adding a pitch here is how they would come back.
"""


def grasp_poses(arm: CuroboArm, block_name: str) -> list[Pose]:
    """Generate world-frame grasp poses for a block, ranked nearest-first.

    The gripper yaw tracks the block's own world-frame z-rotation, so the fingers align with its
    faces: ``R_grasp = R_yaw(block) @ Rz(i * pi/2) @ Ry(pitch)``. Candidates are filtered by
    collision-free IK and ranked by how far the arm has to move to reach them, in position and
    orientation together.

    The block's heading is recovered by projecting its local axes onto the xy-plane rather than
    from Euler angles, which gimbal-lock when the block's pitch is +/-90 degrees -- the case for a
    block resting on a side face, which is exactly what a flip produces.

    :param arm: The motion backend, used for scene state and IK
    :param block_name: Name of the block to grasp
    :return: Ranked world-frame end-effector poses, possibly empty
    """
    block_pose = arm.get_pose(block_name)
    block_position, block_quaternion = block_pose[:3], block_pose[3:]
    block_size = arm.get_size(block_name)

    ee_pose = arm.get_curr_pose()
    ee_position = ee_pose.position.reshape(-1)
    ee_quaternion = ee_pose.quaternion.reshape(-1)

    # The columns of a rotation matrix are the block's own axes, in the world frame.
    block_axes_w = rotation_of(block_quaternion).as_matrix()
    local_x_w, local_y_w = block_axes_w[:, 0], block_axes_w[:, 1]

    if np.linalg.norm(local_x_w[:2]) >= np.linalg.norm(local_y_w[:2]):
        heading = math.atan2(local_x_w[1], local_x_w[0])
    else:
        heading = math.atan2(local_y_w[1], local_y_w[0]) - math.pi / 2

    heading_rotation = Rotation.from_euler("z", heading)

    sin_pitch, cos_pitch = abs(math.sin(GRASP_PITCH)), abs(math.cos(GRASP_PITCH))
    vertical_half = float(block_size[2]) / 2.0
    unit_z = np.array([0.0, 0.0, 1.0])

    ranked: list[tuple[float, Pose]] = []

    for face in range(GRASP_YAW_COUNT):
        grasp_rotation = heading_rotation * Rotation.from_euler(
            "xyz", (0.0, GRASP_PITCH, face * math.pi / 2)
        )
        grasp_quaternion = quaternion_of(grasp_rotation, arm.device)

        # The gripper advances along its own -Z toward the block.
        approach = -torch.as_tensor(
            grasp_rotation.apply(unit_z), dtype=block_position.dtype, device=arm.device
        )

        horizontal_half = float(block_size[face % 2]) / 2.0
        surface_distance = sin_pitch * horizontal_half + cos_pitch * vertical_half

        position = block_position + approach * (surface_distance + GRASP_CLEARANCE_M)
        candidate = Pose(
            position=position.reshape(1, 3).float(), quaternion=grasp_quaternion.reshape(1, 4)
        )

        if not arm.collision_free_ik(candidate):
            continue

        position_distance = float(torch.norm(position - ee_position))
        alignment = float(torch.abs(torch.dot(ee_quaternion, grasp_quaternion)).clamp(max=1.0))
        ranked.append((position_distance + 2.0 * math.acos(alignment), candidate))

    ranked.sort(key=lambda scored: scored[0])
    return [pose for _, pose in ranked]


def place_on_block_pose(arm: CuroboArm, held_block: str, support_block: str) -> Pose:
    """Generate the world-frame end-effector pose that stacks one block squarely on another.

    The placed block is aligned with the support block's yaw rather than with the world: the
    support's own perturbation away from its canonical face is measured and re-applied to the goal
    orientation, so a tower stays square even when its base is not.

    The grasp offset is preserved, so the placement respects how the block is actually held rather
    than assuming it sits centred in the gripper.

    :param arm: The motion backend, used for scene state and IK
    :param held_block: Name of the block currently held
    The returned pose is in contact with the support block by construction, so it is deliberately
    not screened for a collision-free IK solution: that screen rejects sound placements, which
    reach the model as "the target may be blocked". Reaching it is
    `InlineTampMotion._place_at`'s job, and it approaches through a standoff for that reason.

    :param support_block: Name of the block to stack it on
    :return: The world-frame end-effector pose
    """
    support_pose = arm.get_pose(support_block)
    held_pose = arm.get_pose(held_block)

    support_half_height = float(arm.get_size(support_block)[2]) / 2.0
    held_half_height = float(arm.get_size(held_block)[2]) / 2.0

    # Where the held block should end up: centred over the support, resting on its top face.
    target_position = support_pose[:3].clone()
    target_position[2] += support_half_height + held_half_height + 0.005

    # Keep the held block's current orientation as the goal, corrected by however far the support
    # has been perturbed from the canonical face it is nearest.
    support_canonical = Rotation.from_quat(
        FACE_QUATERNIONS[nearest_face(support_pose[3:])], scalar_first=True
    )
    perturbation = rotation_of(support_pose[3:]) * support_canonical.inv()
    target_quaternion = quaternion_of(perturbation * rotation_of(held_pose[3:]), arm.device)

    target_block_pose = Pose(
        position=target_position.reshape(1, 3).float(),
        quaternion=target_quaternion.reshape(1, 4).float(),
    )

    ee_in_block = pose_from_state(held_pose).compute_local_pose(arm.get_curr_pose())
    return target_block_pose.compute_offset_pose(ee_in_block)


def place_on_table_pose(arm: CuroboArm, held_block: str, world_xyz: np.ndarray) -> Pose:
    """Generate the end-effector pose that puts the held block down at a world point on the table.

    Preserves how the block is currently grasped, and how it is currently oriented, so the face
    that ends up down is whichever one is down now. The local half-extents are therefore projected
    onto world z rather than read off the block's z axis.

    :param arm: The motion backend, used for scene state
    :param held_block: Name of the block currently held
    :param world_xyz: World-frame point on the table to place the block at, shape (3,)
    :return: The world-frame end-effector pose to plan to
    """
    held_pose = arm.get_pose(held_block)
    half_extents = arm.get_size(held_block) / 2.0

    # |R[2, :]| . half_extents is the world-z half-extent of the rotated box.
    world_z_row = torch.as_tensor(
        rotation_of(held_pose[3:7]).as_matrix()[2],
        dtype=half_extents.dtype,
        device=half_extents.device,
    )
    half_height_world = float(world_z_row.abs() @ half_extents)

    target_block_pose = held_pose.clone()
    target_block_pose[0] = float(world_xyz[0])
    target_block_pose[1] = float(world_xyz[1])
    target_block_pose[2] = float(world_xyz[2]) + half_height_world

    ee_in_block = pose_from_state(held_pose).compute_local_pose(arm.get_curr_pose())
    return pose_from_state(target_block_pose).compute_offset_pose(ee_in_block)


CLUTTER_RADIUS_M = 0.10
"""Planar distance within which another block blocks a reorientation target."""

REORIENT_X_LIMITS_M = (0.2, 0.72)
"""World-x band a flip target must land in, from `ReOrientPoseGenerator.validate_pose`: the
scripted skills' own default. `InlineTampMotion.flip_x_limits_m` is the live value, which a
skill module built on this one sets (the quirk skills own theirs; `quirk_skills.QuirkConfig`)."""


def _target_is_clear(arm: CuroboArm, block_name: str, target_xy: torch.Tensor) -> bool:
    """Whether no other block sits within `CLUTTER_RADIUS_M` of a planar target point."""
    for other in arm.block_names:
        if other == block_name:
            continue
        other_xy = arm.get_pose(other)[:2]
        if bool((other_xy - target_xy).abs().max() < CLUTTER_RADIUS_M):
            return False
    return True


def rotate_target(
    arm: CuroboArm, block_name: str, target_face: str, *, ignore_collision: bool = False
) -> tuple[Pose, Pose] | None:
    """Generate the goal pose for rotating a block onto a canonical face, in place.

    :param arm: The motion backend, used for scene state
    :param block_name: Name of the block to rotate
    :param target_face: Canonical face the block should end up on
    :param ignore_collision: Whether to accept a target another block sits near
    :return: ``(absolute, relative)`` goal poses for the RL executor, or None if none is valid
    """
    block_pose = arm.get_pose(block_name)

    target = block_pose.clone()
    target[3:] = torch.tensor(FACE_QUATERNIONS[target_face], device=arm.device)

    if not ignore_collision and not _target_is_clear(arm, block_name, target[:2]):
        return None

    # `RotateTrajectory` reads only `target_pose`; it takes both slots because the flip executor's
    # second slot carries real information and the two share a call shape.
    goal = pose_from_state(target)
    return goal, goal


def flip_target(
    arm: CuroboArm,
    block_name: str,
    current_face: str,
    target_face: str,
    direction: str,
    *,
    ignore_collision: bool = False,
    x_limits_m: tuple[float, float] = REORIENT_X_LIMITS_M,
) -> tuple[Pose, Pose] | None:
    """Generate the goal pose for tipping a block onto an adjacent face.

    The goal orientation carries the block's own yaw: the block's rotation away from the canonical
    face it currently occupies is measured and applied to the canonical target face, so a block
    resting at an angle flips to a correspondingly angled result rather than snapping square.

    The second returned pose is not a pose the arm moves to. ``ReOrientTrajectory`` applies its
    quaternion to world up to recover ``flip_push_dir_world`` -- where the current top face ends up
    -- which is how the policy is told which way to push.

    :param arm: The motion backend, used for scene state
    :param block_name: Name of the block to flip
    :param current_face: Canonical face the block currently occupies
    :param target_face: Canonical face it should end up on
    :param direction: Flip direction, one of the `FlipDirection` values
    :param ignore_collision: Whether to accept a target another block sits near
    :param x_limits_m: World-x band the target must land in, unless collisions are ignored
    :return: ``(absolute, push-direction)`` poses for the RL executor, or None if none is valid
    """
    block_pose = arm.get_pose(block_name)

    canonical_from = Rotation.from_quat(FACE_QUATERNIONS[current_face], scalar_first=True)
    canonical_to = Rotation.from_quat(FACE_QUATERNIONS[target_face], scalar_first=True)
    block_yaw = rotation_of(block_pose[3:]) * canonical_from.inv()

    target = block_pose.clone()
    target[3:] = quaternion_of(block_yaw * canonical_to, arm.device)
    offset_x, offset_y = FLIP_OFFSETS_M[direction]
    target[0] += offset_x
    target[1] += offset_y

    if not ignore_collision:
        if not _target_is_clear(arm, block_name, target[:2]):
            return None
        if not x_limits_m[0] <= float(target[0]) <= x_limits_m[1]:
            return None

    push_quaternion = torch.tensor(FLIP_PUSH_QUATERNIONS[direction], device=arm.device)
    push = Pose(
        position=torch.zeros(1, 3, device=arm.device),
        quaternion=push_quaternion.reshape(1, 4).float(),
    )
    return pose_from_state(target), push


PRE_RL_HEIGHT_M = {"rotate": 0.2, "flip": 0.2111}
"""Height above the block the arm is placed at before handing off to an RL checkpoint.

Two values because the two ToyBlocks generators use two, and neither is derived from anything
here. Worth checking against what the checkpoints trained on: ``rotate_env._reset_idx`` starts the
end-effector 0.07 m above the block with xy noise uniform over +/-0.04 and a uniformly random yaw,
where these place it ~0.2 m above with xy noise on a +/-0.01 grid and yaw snapped to the nearest
cardinal. If that gap is real, the policies begin each rollout off-distribution.
"""

PRE_RL_XY_NOISE_M = (-0.01, 0.0, 0.01)
"""Planar offsets tried around the block, as a grid, when looking for a reachable pre-pose."""

PRE_RL_BASE_QUATERNION = (0.0, -0.7071, 0.7071, 0.0)
"""End-effector orientation the RL checkpoints expect at handoff."""


def pre_rl_poses(arm: CuroboArm, block_name: str, *, skill: str) -> list[Pose]:
    """Generate world-frame poses the arm can hand off to an RL reorientation skill from.

    The pose sits above the block, oriented so the gripper points down at it with its yaw snapped
    to the block's nearest cardinal heading. Candidates are filtered by collision-free IK.

    :param arm: The motion backend, used for scene state and IK
    :param block_name: Name of the block the skill will act on
    :param skill: Either "rotate" or "flip", selecting the handoff height
    :return: World-frame end-effector poses, possibly empty
    """
    block_pose = arm.get_pose(block_name)
    height = PRE_RL_HEIGHT_M[skill]

    yaw_delta = yaw_of(block_pose[3:])
    nearest_cardinal = round(yaw_delta / (math.pi / 2)) * (math.pi / 2)
    snap_to_cardinal = Rotation.from_euler("z", yaw_delta - nearest_cardinal)
    base_rotation = Rotation.from_quat(PRE_RL_BASE_QUATERNION, scalar_first=True)
    quaternion = quaternion_of(snap_to_cardinal * base_rotation, arm.device)

    poses = []
    for x_noise in PRE_RL_XY_NOISE_M:
        for y_noise in PRE_RL_XY_NOISE_M:
            position = block_pose[:3].clone()
            position[0] += x_noise
            position[1] += y_noise
            position[2] += height

            candidate = Pose(
                position=position.reshape(1, 3).float(), quaternion=quaternion.reshape(1, 4)
            )
            if arm.collision_free_ik(candidate):
                poses.append(candidate)

    return poses


# ---------------------------------------------------------------------------------------------
# The skills
# ---------------------------------------------------------------------------------------------

HOME_ARM_JOINT_POSITIONS = (0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, 0.0)
"""Arm joint targets for the home configuration, matching the autogen env's own reset pose."""

HOME_STEPS = 60
"""Simulated steps the return-home motion is spread over."""

RETRACT_MULTIPLES = (4.0, 2.0, 1.0)
"""Multiples of `retract_height_m` tried in turn, tallest reachable lift winning."""

PLACE_STANDOFF_M = 0.03
"""Height above a placement the arm is planned to, before the last stretch is run by IK.

Matches `STANDOFF_HEIGHT` in ToyBlocks's `PrePlacePoseGenerator`, which splits its pnp place the
same way.
"""

IK_INTERP_STEPS = 80

PLACE_CORRECTION_TOL_M = 0.003
"""A placement is left alone once the held block is within this of its target in x-y."""

PLACE_CORRECTION_ROUNDS = 3
"""Lateral IK nudges tried before releasing wherever the block is."""

IK_CORRECTION_STEPS = 40

RECOVERY_LIFT_M = 0.10
"""Straight-up IK lift before ramping home with a block in hand, to clear whatever is near."""
"""Simulated steps one lateral nudge is ramped over; the move is a centimetre or two."""
"""Simulated steps an IK-interpolated segment is spread over.

ToyBlocks's `IKInterpTrajectory` holds each of 20 joint-space waypoints for 4 sim steps, and this
is that product: the descent takes the same wall-clock time there and here.
"""

WEDGE_PROBE_M = 0.01
"""Distance of the nudge used to test whether the arm can be planned at all from here."""

RL_STEP_SLACK = 20
"""Extra loop iterations allowed past an RL skill's step budget, to observe its termination."""

RL_TASK_MODULES = ("blocksuite.tasks.rotate", "blocksuite.tasks.flip")
"""Modules imported for the side effect of registering the gym tasks the RL executors look up."""


class InlineTampMotion:
    """Ground-truth motion over a ToyBlocks environment, on the curobo backend.

    Grasp and place are planned and executed here end to end. Rotate and flip plan their approach
    here and then hand off to a trained RL checkpoint, which is the one part of this file that
    still reaches into TAMP -- see the module header.
    """

    def __init__(
        self, env: gym.Env, config: OracleConfig, *, on_step: StepCallback, warmup: bool = True
    ) -> None:
        """Build the motion backend over a ToyBlocks gym environment.

        Constructing this does **not** reset the scene: unlike ToyBlocks's vectorized simulator,
        whose constructor drops objects through the floor, nothing here has to be settled
        afterwards, so task setup may be applied either side of it.

        :param env: The ToyBlocks gym environment to drive
        :param config: Oracle parameters for planning, reorientation and the RL rollouts
        :param on_step: Called once per simulated step, e.g. to record a video frame
        :param warmup: Whether to warm curobo up at construction rather than on the first plan
        """
        self._config = config
        self._arm = CuroboArm(env, on_step=on_step, seed=0, warmup=warmup)
        self._checkpoint_dir: Path | None = None
        self.flip_x_limits_m: tuple[float, float] = REORIENT_X_LIMITS_M
        """World-x band a flip target must land in; a skill module may set its own after
        construction."""
        self.rl_max_steps_by_skill: dict[str, int] = {}
        """Rollout step cutoff per RL checkpoint name, overriding `config.rl_max_steps` (the rotate
        policy was trained with a 100-step horizon and the flip with 64, and from the scripted
        approach pose a quarter turn of a square block can run past 120 steps)."""

    @property
    def arm(self) -> CuroboArm:
        """The underlying motion backend."""
        return self._arm

    @property
    def gripper_state(self) -> str:
        """Current gripper state, either "open" or "closed"."""
        return self._arm.gripper_state

    def open_gripper(self) -> None:
        """Open the gripper, releasing anything held and telling the planner it was released.

        Opening by hand is a release like any other: whatever was held drops, so it has to be
        detached from the planning model and become a world obstacle again. Skipping that leaves
        curobo planning around a block attached to a gripper that no longer holds it, and
        through the space where the block actually landed.
        """
        self._release()

    def close_gripper(self) -> None:
        """Close the gripper."""
        self._arm.close_gripper()

    # -- pick and place ------------------------------------------------------------------------

    def grasp(self, block_name: str) -> None:
        """Pick up the named block.

        :param block_name: Name of the block to grasp
        :raises OracleSkillError: If none of the generated grasp poses can be planned to, or if
            the block stays behind when the arm lifts
        """
        candidates = grasp_poses(self._arm, block_name)[: self._config.max_samples]
        start_z = float(self._arm.get_pose(block_name)[2])

        if not self._try_candidates(candidates):
            # The message does not guess "occluded": the same failure covers a block out of the
            # arm's reach and an arm already parked where the grasp poses live.
            raise OracleSkillError(
                f"Could not plan a grasp of '{block_name}': none of {len(candidates)} candidate "
                f"grasp poses could be reached." + self._planning_failure_hint()
            )

        # The grasp poses approach from above, so the end-effector belongs over the block's
        # centre; centre it before closing, see `_nudge_xy`.
        self._nudge_xy(f"Grasp of '{block_name}'", self._block_xy(block_name), self._ee_xy)
        self._arm.close_gripper()
        self._arm.attach(block_name)
        lift_m = self._retract()  # Attached first: the lift is planned with the block modelled

        # A block that is really between the fingers rises with them, so the retraction doubles as
        # the one available test of whether the gripper closed on the block or on air. Only asked
        # when a retraction ran: with none planned the arm is still down on the block, and the
        # block's height then says nothing either way. Failing here rather than returning lets the
        # caller keep its own bookkeeping honest -- a grasp recorded as held when it is not makes
        # that block unnameable for the rest of the episode.
        if lift_m > 0.0 and float(self._arm.get_pose(block_name)[2]) - start_z < lift_m / 2.0:
            self._release()
            raise OracleSkillError(
                f"The gripper closed on '{block_name}' and lifted {lift_m:.3f} m, but the block "
                "stayed where it was. The gripper is now open and empty; try pointing at a "
                "different part of the block."
            )

    def place_on_block(self, held_block: str, support_block: str) -> None:
        """Stack the held block on top of another block, then release and retract.

        :param held_block: Name of the block currently held by the gripper
        :param support_block: Name of the block to stack it on
        :raises OracleSkillError: If the placement cannot be reached
        """
        target = place_on_block_pose(self._arm, held_block, support_block)
        self._place_at(target, f"'{held_block}' on '{support_block}'")
        support = self._arm.get_pose(support_block)
        self._correct_placement(held_block, (float(support[0]), float(support[1])))
        self._release()
        self._retract()

    def place_on_table(self, held_block: str, world_xyz: np.ndarray) -> None:
        """Place the held block at an exact world point on the table, then release and retract.

        :param held_block: Name of the block currently held by the gripper
        :param world_xyz: World-frame point on the table to place the block at, shape (3,)
        :raises OracleSkillError: If the placement cannot be reached
        """
        self._place_at(
            place_on_table_pose(self._arm, held_block, world_xyz),
            f"'{held_block}' on the table",
        )
        self._correct_placement(held_block, (float(world_xyz[0]), float(world_xyz[1])))

        self._release()
        self._retract()

    # -- reorientation -------------------------------------------------------------------------

    def rotate(self, block_name: str, direction: RotateDirection | str) -> None:
        """Rotate a block in place about the vertical axis, using the trained RL skill.

        :param block_name: Name of the block to rotate
        :param direction: Direction to rotate it by one 90-degree step
        :raises OracleSkillError: If the pre-rotate pose or the target pose cannot be reached
        """
        from agentic_robotics.adapters.toyblocks.skills.rl_executors import RotateTrajectory

        rotate_direction = RotateDirection.parse(direction)
        current_face = self._current_face(block_name)
        target_face = YAW_NEXT[rotate_direction][current_face]

        self._approach_for_rl(block_name, rotate_direction, skill="rotate")

        target = self._reorient_target(
            lambda ignore: rotate_target(
                self._arm, block_name, target_face, ignore_collision=ignore
            )
        )
        if target is None:
            raise OracleSkillError("Rotation not possible; no valid target pose was generated.")

        absolute, relative = target

        def block_yaw_deg() -> float:
            return float(
                rotation_of(self._arm.get_pose(block_name)[3:]).as_euler("zyx", degrees=True)[0]
            )

        goal_yaw = float(
            rotation_of(absolute.quaternion.reshape(-1)).as_euler("zyx", degrees=True)[0]
        )
        logger.info(
            f"Rotate of {block_name}: yaw {block_yaw_deg():+.1f} deg before, "
            f"goal yaw {goal_yaw:+.1f} deg."
        )
        self._run_rl_skill(
            RotateTrajectory(),
            {
                "obj": block_name,
                "target_pose": absolute,
                "rotate_pose_rel": relative,
                "robot": "robot",
                "initial_gripper_opening": 0.0,
            },
            checkpoint_name="rotate",
        )
        logger.info(f"Rotate of {block_name}: yaw {block_yaw_deg():+.1f} deg at rollout exit.")
        self._retract()
        logger.info(f"Rotate of {block_name}: yaw {block_yaw_deg():+.1f} deg after the retract.")

    def flip(self, block_name: str, direction: FlipDirection | str) -> None:
        """Flip a block onto an adjacent face, using the trained RL skill.

        :param block_name: Name of the block to flip
        :param direction: Direction to flip it in
        :raises OracleSkillError: If the pre-flip pose or the target pose cannot be reached
        """
        from agentic_robotics.adapters.toyblocks.skills.rl_executors import ReOrientTrajectory

        flip_direction = FlipDirection.parse(direction)
        current_face = self._current_face(block_name)
        target_face = FLIP_NEXT[flip_direction][current_face]

        self._approach_for_rl(block_name, flip_direction, skill="flip")

        target = self._reorient_target(
            lambda ignore: flip_target(
                self._arm,
                block_name,
                current_face,
                target_face,
                str(flip_direction),
                ignore_collision=ignore,
                x_limits_m=self.flip_x_limits_m,
            )
        )
        if target is None:
            raise OracleSkillError("Flip not possible; no valid target pose was generated.")

        absolute, push = target
        self._run_rl_skill(
            ReOrientTrajectory(),
            {
                "obj": block_name,
                "target_pose": absolute,
                "reorient_pose_rel": push,
                "robot": "robot",
                "initial_gripper_opening": 0.72,
            },
            checkpoint_name="flip",
        )
        self._arm.open_gripper()  # The flip closes the gripper to push, so release afterwards.
        self._retract()

    def _current_face(self, block_name: str) -> str:
        """Canonical orientation face nearest to a block's current orientation."""
        return nearest_face(self._arm.get_pose(block_name)[3:])

    def _approach_for_rl(self, block_name: str, direction: str, *, skill: str) -> None:
        """Plan and run the curobo approach to an RL skill's starting pose.

        :param block_name: Name of the block the skill will act on
        :param direction: Direction the skill will move the block in, for the error message
        :param skill: Either "rotate" or "flip"
        :raises OracleSkillError: If none of the generated pre-poses can be planned to
        """
        candidates = pre_rl_poses(self._arm, block_name, skill=skill)[: self._config.max_samples]

        if not self._try_candidates(candidates):
            raise OracleSkillError(
                f"Could not plan the pose the {skill} skill starts from for '{block_name}' "
                f"({direction}): none of {len(candidates)} candidate pre-poses could be reached. "
                "The pre-pose sits above the block, so a different direction may be reachable "
                "where this one is not." + self._planning_failure_hint()
            )

    @staticmethod
    def _reorient_target(
        generate: Callable[[bool], tuple[Pose, Pose] | None],
    ) -> tuple[Pose, Pose] | None:
        """Generate a reorientation target, retrying with collisions ignored (a backtrack)."""
        for ignore_collision in (False, True):
            target = generate(ignore_collision)
            if target is not None:
                return target
        return None

    def _register_rl_checkpoints(self) -> Path:
        """Point the RL executors at the checkpoints bundled with ToyBlocks.

        Also imports the gym tasks the executors look up by name.

        :return: Directory holding the RL skills' checkpoints
        """
        if self._checkpoint_dir is not None:
            return self._checkpoint_dir

        import blocksuite

        from agentic_robotics.adapters.toyblocks.skills import rl_executors

        checkpoint_dir = Path(blocksuite.__file__).resolve().parent / "tamp/blocks/checkpoints"
        rl_executors.set_checkpoint_dir(checkpoint_dir)

        for task_module in RL_TASK_MODULES:
            try:
                importlib.import_module(task_module)
            except Exception:
                logger.exception(f"Could not import {task_module}; its RL skill will not run.")

        self._checkpoint_dir = checkpoint_dir
        return checkpoint_dir

    def _run_rl_skill(
        self, executor: BaseRLTrajectory, env_data: dict, *, checkpoint_name: str
    ) -> None:
        """Roll out an RL reorientation policy on the single environment.

        This is what `BatchedRLCoordinator` reduces to at one environment: load the checkpoint,
        prepare the pseudo-environment the observation is built from, set the handoff gripper
        opening, then loop inference and stepping until the executor reports success or the budget
        runs out. Its registration barrier, batching threshold, per-env state records and
        vectorized termination mask all have nothing to act on here.

        The environment's table and block friction tensors are left at the zeros it allocates:
        they belong to the privileged observation group, which only the critic reads, and the
        actor discards that tail at inference.

        :param executor: Trajectory executor wrapping the RL policy
        :param env_data: Skill arguments the executor's observation is built from
        :param checkpoint_name: Name of the checkpoint directory to run
        """
        self._register_rl_checkpoints()

        base_env = self._arm._base_env
        max_steps = self.rl_max_steps_by_skill.get(checkpoint_name, self._config.rl_max_steps)

        executor.load_rl_agent(_ExecutorSimulator(base_env))

        keyed = {0: env_data}
        executor._setup_asset_references(base_env, executor._pseudo_env.rl_env, keyed, env_ids=[0])
        executor.prepare_rl_env(base_env, executor._pseudo_env.rl_env, keyed, [0])
        self._arm.set_gripper_opening(env_data["initial_gripper_opening"])

        self._apply_stiffness_profile(checkpoint_name)

        simulator = _ExecutorSimulator(base_env)
        env_ids = torch.tensor([0], device=self._arm.device)

        succeeded = False
        steps_run = 0
        try:
            for steps_run in range(1, max_steps + RL_STEP_SLACK + 1):  # noqa: B007 (logged below)
                delta = executor.step(simulator=simulator, env_idx=[0], env_data=env_data)
                delta = delta.squeeze(0) if delta.dim() > 1 else delta

                joint_pos = self._arm._articulation.data.joint_pos[0]
                target = delta + joint_pos[self._arm._arm_joint_ids].clone()
                self._arm._articulation.set_joint_position_target(
                    target.unsqueeze(0), joint_ids=self._arm._arm_joint_ids, env_ids=env_ids
                )
                self._arm._articulation.write_data_to_sim()

                # One RL action is one render interval, which is what the TAMP orchestrator does.
                # Stepping `decimation` times here would run the motion roughly 4x too fast.
                self._arm.step()

                if bool(executor.evaluate(simulator, 0, env_data)["success"]):
                    succeeded = True
                    break
        finally:
            logger.info(
                f"RL rollout of {checkpoint_name}: {steps_run} of {max_steps + RL_STEP_SLACK} "
                "steps, "
                f"{'the executor reported success' if succeeded else 'ran to the cutoff'}."
            )
            self._restore_stiffness_profile()
            self._arm.sync_gripper_flag()  # The policy drove the gripper; the flag follows it.
            self._arm.refresh_world()  # The block has moved; the planner's world is now stale.

    def _apply_stiffness_profile(self, checkpoint_name: str) -> None:
        """Soften the arm's gains to the profile a checkpoint expects, and let it settle.

        The profile table (:mod:`skills.stiffness_profiles`) is plain data -- a stiffness,
        damping, effort and velocity limit per named profile.

        :param checkpoint_name: Name of the checkpoint being run, for the log line
        """
        from agentic_robotics.adapters.toyblocks.skills.stiffness_profiles import (
            get_profile,
            get_profile_tensors,
            profile_for_robot_id,
        )

        name = profile_for_robot_id(str(getattr(self._arm._base_env.robot, "id", "")))
        logger.info(f"Applying the '{name}' stiffness profile for the {checkpoint_name} skill.")
        self._write_gains(name, get_profile_tensors)
        self._arm.step(int(get_profile(name).settle_steps))

    def _restore_stiffness_profile(self) -> None:
        """Restore stiff gains, so the motion-planned skills behave as they did before."""
        try:
            from agentic_robotics.adapters.toyblocks.skills.stiffness_profiles import (
                get_profile_tensors,
            )

            self._write_gains("stiff", get_profile_tensors)
        except Exception:
            logger.exception("Could not restore stiff gains after the RL skill.")

    def _write_gains(
        self,
        name: str,
        get_profile_tensors: Callable[
            ..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ],
    ) -> None:
        """Write a named stiffness profile onto the arm joints.

        Position targets are snapped to the current actual positions and velocity targets zeroed
        first: a gain change otherwise leaves the controller chasing a stale target, which spikes
        torque when stiffening and lets the arm drift when softening.

        :param name: Name of a registered stiffness profile
        :param get_profile_tensors: The profile table's accessor for the profile's four tensors
        """
        articulation = self._arm._articulation
        joint_ids = self._arm._arm_joint_ids
        env_ids = torch.tensor([0], device=self._arm.device)

        current = articulation.data.joint_pos[0, joint_ids].unsqueeze(0).clone()
        articulation.set_joint_position_target(current, joint_ids=joint_ids, env_ids=env_ids)
        articulation.set_joint_velocity_target(
            torch.zeros_like(current), joint_ids=joint_ids, env_ids=env_ids
        )

        stiffness, damping, effort, velocity_limit = get_profile_tensors(
            name, self._arm.device, batch_size=1
        )
        articulation.write_joint_velocity_limit_to_sim(
            velocity_limit, joint_ids=joint_ids, env_ids=env_ids
        )
        articulation.write_joint_stiffness_to_sim(stiffness, joint_ids=joint_ids, env_ids=env_ids)
        articulation.write_joint_damping_to_sim(damping, joint_ids=joint_ids, env_ids=env_ids)
        articulation.write_joint_effort_limit_to_sim(effort, joint_ids=joint_ids, env_ids=env_ids)
        articulation.write_data_to_sim()

    # -- recovery and shared helpers -------------------------------------------------------------

    @staticmethod
    def _raised(pose: Pose, height_m: float) -> Pose:
        """`pose` translated straight up in the world frame, keeping its orientation."""
        position = pose.position.clone()
        position[..., 2] += height_m
        return Pose(position=position, quaternion=pose.quaternion)

    def _place_at(self, target: Pose, description: str) -> None:
        """Fly to a standoff above a placement, then close the last stretch by IK.

        The planner is never asked to reach the placement itself. A block set down on another is
        touching it by definition, and CuRobo's collision checker rejects that as a goal -- so a
        geometrically sound stack would come back to the model as "the target may be blocked",
        and send the model looking for an obstruction that is not there. Only the standoff is
        planned to, and it is clear of everything; the descent is IK-interpolated and tolerates
        the contact.

        ToyBlocks's pnp actions split their place the same way, into `pre_place_pose` reached by
        `place_traj` and `place_pose` reached by `place_delta_traj`.

        :param target: World-frame end-effector pose the placement finishes at
        :param description: What is being placed where, for the error the model reads
        :raises OracleSkillError: If the standoff cannot be planned to, or the descent has no IK
        """
        standoff = self._raised(target, PLACE_STANDOFF_M)
        if not self._arm.plan_and_execute(standoff):
            # About one pick in ten on the Rao cubes (cube_stack_probe, 2026-10-14) leaves the
            # arm where the planner can plan nothing, not even the retraction after the grasp,
            # so the standoff fails from there. Lifting clear by IK and ramping the joints to
            # the home configuration, block in hand, gives the planner a start it accepts.
            logger.info(
                f"Could not plan to the standoff above {description}; recovering through home, "
                "block in hand."
            )
            self._recover_holding()
            if not self._arm.plan_and_execute(standoff):
                raise OracleSkillError(
                    f"Could not reach the standoff above where {description} would go; "
                    "the approach may be blocked or out of reach." + self._planning_failure_hint()
                )

        if not self._arm.ik_interp_to(target, steps=IK_INTERP_STEPS):
            raise OracleSkillError(
                f"Reached the standoff above where {description} would go, but no arm "
                f"configuration puts it down there. The block is still held."
            )

    def _nudge_xy(
        self,
        what: str,
        target_xy: tuple[float, float],
        current_xy: Callable[[], tuple[float, float]],
    ) -> None:
        """Close a lateral error by IK, from where a move actually left the arm.

        Measured on the Rao color cubes (cube_stack_probe, 2026-10-13/14): the planner's targets
        were right every time, and a stacked cube still landed 1.3-1.7 cm short toward the robot,
        because a 4 cm cube slides about 1 cm back toward the gripper's centre during the carry
        (its in-hand offset at planning time is stale by then) and the arm stops about 0.5 cm
        short of the pose it was ramped to. The same arrival error at a grasp has one finger
        shove the cube 1-2.6 cm before the other arrives, and in one position pushed it out of
        the gripper altogether. Neither is visible to the planner, so the oracle closes the last
        centimetres on the simulator's own poses: up to `PLACE_CORRECTION_ROUNDS` lateral moves
        at the current height, each by the remaining x-y error of `current_xy` against
        `target_xy`, until it is within `PLACE_CORRECTION_TOL_M`. A nudge that cannot be solved
        leaves things where they are.

        :param what: What is being lined up, for the log
        :param target_xy: World-frame x-y to reach
        :param current_xy: Reads the world-frame x-y that should end up at the target
        """
        for attempt in range(PLACE_CORRECTION_ROUNDS):
            x, y = current_xy()
            dx, dy = target_xy[0] - x, target_xy[1] - y
            error = math.hypot(dx, dy)
            if error <= PLACE_CORRECTION_TOL_M:
                logger.info(f"{what} within {error * 100:.2f} cm after {attempt} nudge(s).")
                return
            ee = self._arm.get_curr_pose()
            position = ee.position.clone()
            position[..., 0] += dx
            position[..., 1] += dy
            if not self._arm.ik_interp_to(
                Pose(position=position, quaternion=ee.quaternion), steps=IK_CORRECTION_STEPS
            ):
                logger.warning(
                    f"{what} is {error * 100:.2f} cm off and the nudge has no IK; continuing as is."
                )
                return
        x, y = current_xy()
        residual = math.hypot(target_xy[0] - x, target_xy[1] - y)
        logger.info(
            f"{what} still {residual * 100:.2f} cm off after {PLACE_CORRECTION_ROUNDS} nudges; "
            "continuing."
        )

    def _block_xy(self, block_name: str) -> tuple[float, float]:
        pose = self._arm.get_pose(block_name)
        return float(pose[0]), float(pose[1])

    def _ee_xy(self) -> tuple[float, float]:
        position = self._arm.get_curr_pose().position.reshape(-1)
        return float(position[0]), float(position[1])

    def _correct_placement(self, held_block: str, target_xy: tuple[float, float]) -> None:
        """Nudge the held block over its target before releasing; see `_nudge_xy`."""
        self._nudge_xy(
            f"Placement of '{held_block}'", target_xy, lambda: self._block_xy(held_block)
        )

    def _recover_holding(self) -> None:
        """Lift straight up by IK, then ramp the joints home, keeping whatever is held.

        `return_home` releases first, since a block dropped from home height lands anywhere; this
        keeps the grasp and the planner's attached model, so the block travels with the arm.
        The lift is skipped when its IK has no solution.
        """
        if not self._arm.ik_interp_to(
            self._raised(self._arm.get_curr_pose(), RECOVERY_LIFT_M), steps=IK_INTERP_STEPS
        ):
            logger.info("Recovery lift has no IK; ramping home from here.")
        self._arm.move_arm_to_joint_positions(
            torch.tensor(HOME_ARM_JOINT_POSITIONS, device=self._arm.device), steps=HOME_STEPS
        )

    def _try_candidates(self, candidates: list[Pose]) -> bool:
        """Try each candidate pose in order until one plans, then run it.

        :param candidates: Ranked world-frame end-effector poses
        :return: True if a candidate was planned to and executed
        """
        return any(self._arm.plan_and_execute(candidate) for candidate in candidates)

    def _release(self) -> None:
        """Open the gripper and stop modelling the released block as held."""
        self._arm.open_gripper()
        self._arm.detach()

    def return_home(self) -> None:
        """Release whatever is held and drive the arm back to its home configuration.

        The recovery path for a wedged arm. After an RL reorientation skill the arm can end in a
        configuration curobo cannot plan *anything* from, and nothing in the tool surface could
        otherwise recover: on 2026-08-26 a rotate was followed by four blocks in a row failing
        every candidate pose, and by a retraction that could not be planned at any height.

        So this deliberately avoids the planner and ramps joint targets straight to the pose the
        generated env resets to, which is available regardless of what curobo thinks of the current
        state. The gripper is opened *before* the move, so anything held drops from where it is
        rather than from over the table at home height, and the attachment is cleared so a stale
        held-object model cannot poison later plans.
        """
        self._release()
        self._arm.move_arm_to_joint_positions(
            torch.tensor(HOME_ARM_JOINT_POSITIONS, device=self._arm.device), steps=HOME_STEPS
        )
        self._arm.refresh_world()

    def _arm_is_wedged(self) -> bool:
        """Whether the arm cannot be planned anywhere from its current configuration.

        Plans a 1-cm nudge from the current pose. If even that fails, the failure is not about
        whichever target was asked for -- the start configuration itself is unusable, and every
        skill will fail identically until the arm is moved by other means.

        In the 2026-08-26 run four blocks in a row failed with "none of 12 candidates could be
        reached", including one in the middle of the workspace. A single block being out of reach
        cannot explain that; a wedged start can.
        """
        try:
            current = self._arm.get_curr_pose()
            position = current.position.clone()
            position[..., 2] += WEDGE_PROBE_M
            return self._arm.plan(Pose(position=position, quaternion=current.quaternion)) is None
        except Exception:
            logger.exception("Could not run the wedged-arm probe; assuming the arm is fine.")
            return False

    def _planning_failure_hint(self) -> str:
        """Sentence to append to a "nothing could be planned" error, naming the likelier cause."""
        if self._arm_is_wedged():
            return (
                " The arm also cannot plan a 1-cm move from where it is, so every motion-planning "
                "skill will fail the same way until the arm is moved. Call `return_home`, then "
                "retry."
            )
        return (
            " The arm can plan other moves from here, so this target specifically may be outside "
            "the arm's workspace, or blocked by another block."
        )

    def _retract(self) -> float:
        """Raise the end-effector straight up, to clear whatever block it was acting on.

        Tries several lifts, tallest first, because a single failed plan would leave the arm draped
        over the block it has just acted on, and every subsequent skill would then have to plan out
        of that cramped configuration.

        Every height is planned before anything unchecked is tried, so the lift stays
        collision-checked wherever planning works at all. Only once all of them have failed does the
        shortest get a second attempt as an IK-interpolated segment, and that is for the one case
        planning cannot express: the gripper is still touching what it just acted on -- a block it
        has closed on, or one it has just set down and opened around -- so the planner refuses a
        start state it reads as in collision and no height helps. Breaking that contact is enough to
        give the next skill something it can plan from, which is why the fallback takes the shortest
        lift rather than the tallest reachable one.

        :return: Height the end-effector actually rose, in metres; zero if no lift was executed
        """
        if self._config.retract_height_m <= 0:
            return 0.0

        current = self._arm.get_curr_pose()
        attempted = []

        for multiple in RETRACT_MULTIPLES:
            height = self._config.retract_height_m * multiple
            attempted.append(height)

            if self._arm.plan_and_execute(self._raised(current, height)):
                return height

        heights = ", ".join(f"{height:.3f}" for height in attempted)
        shortest = self._config.retract_height_m * min(RETRACT_MULTIPLES)
        logger.info(
            f"Could not plan a retraction at any of {heights} m; lifting {shortest:.3f} m by IK "
            "instead, to break contact with whatever the gripper is still touching."
        )
        if self._arm.ik_interp_to(self._raised(current, shortest), steps=IK_INTERP_STEPS):
            return shortest

        logger.warning(
            f"Could not retract at any of {heights} m, by planning or by IK; leaving the arm "
            "where it is. Skills acting on this block next will have to plan from here."
        )
        return 0.0


class _ExecutorSimulator:
    """The one attribute the ToyBlocks RL executors want from a simulator.

    ``BaseRLTrajectory.load_rl_agent``, ``.step`` and ``.evaluate`` reach for ``simulator.base_env``
    and nothing else, so the vectorized simulator they were written against is not needed to drive
    them -- only something that carries the Isaac Lab environment.
    """

    def __init__(self, base_env: gym.Env) -> None:
        """Wrap the base environment as the simulator handle the executors expect."""
        self.base_env = base_env


# ---------------------------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------------------------


class OracleToyBlocksEnv(ToyBlocksEnv):
    """A `ToyBlocksEnv` prepared for the oracle: viewer trajectory open, denoiser settled.

    Reached through `InlineOracleMCPServer._build_env` rather than by rebinding the name the base
    server constructs.
    """

    def __init__(self, task_config: TaskConfig, *, viewer: bool = False) -> None:
        """Open the viewer's first trajectory, which nothing else in this run will open.

        `Viewer._frames` is a deque *of trajectories*, and the only thing that appends one is the
        `_after_reset_idx` hook `Viewer.hook_to_env` installs. `ToyBlocksEnv.__init__` resets
        before it hooks the viewer, so that hook never fires here, and `InlineTampMotion`
        deliberately does not reset the scene a second time, so `_frames` stays empty and the first
        `_refresh_viewer` -- reached from this server's `record_current_frame`, which every
        simulator step fires -- raises `IndexError: deque index out of range`.

        :param task_config: Configuration specifying which task to load
        :param viewer: Whether to launch a viewer for the simulation environment
        """
        super().__init__(task_config, viewer=viewer)
        if self._viewer is not None and len(self._viewer._frames) == 0:
            self._viewer._frames.append(deque(maxlen=self._viewer.cfg.max_frames_per_trajectory))

        # ToyBlocks re-renders once after a reset. The RTX denoiser keeps several frames of
        # history, so one render leaves ghosts of the blocks' old positions in the first image.
        self.base_env.cfg.num_rerenders_on_reset = max(
            self.base_env.cfg.num_rerenders_on_reset, RESET_RERENDERS
        )


class InlineOracleMCPServer(OracleMCPServer):
    """The oracle tool surface, over the curobo motion backend.

    Three things about the surface are decisions rather than mechanism, and are recorded here
    because the surface *is* the abstraction handed to the model.

    - **Placement is two tools, not one.** `stack` stacks on the block nearest the pixel;
      `place_on_table` deprojects the exact pixel onto the table plane and places there. The agent
      states its intent instead of the server inferring block-vs-table from a distance threshold,
      and `place_on_table` honours the point given rather than resampling a nearby free spot.
    - **Pick and place actuate the gripper themselves.** A `pick` approaches, closes and lifts; a
      place positions and opens. `open_gripper`/`close_gripper` are kept for skills that separate
      motion from grasping, and are rarely needed in this loop.
    - **Skills fail loudly.** The oracle can know a plan failed, so it raises `OracleSkillError`
      instead of returning a silently-unchanged frame. `pick` only raises once *every* ranked
      grasp candidate fails to plan, or once the arm has lifted and the block has not.

    The surface is wider than ManiFlow's, which is `pick` and `place`. The two are not reconciled,
    so a cross-condition comparison has to account for it.
    """

    VIEWER_STRIDE = 4
    """Sim steps between viewer pushes: about 6 Hz against the 25 Hz simulation.

    Each push copies every link, block and marker pose off the GPU, stores a frame in the viewer's
    history and sets position and orientation on every viser handle, and the browser applies all of
    it and re-renders. At 1 that runs on every simulated step and the view drags; the recorded video
    is unaffected, since `record_current_frame` still captures every step.
    """

    RESET_SETTLE_STEPS = 200
    """Sim steps run after a scene is built or reset, before it is observed, so the blocks have
    come to rest where the reset put them."""

    RESET_RERENDERS = RESET_RERENDERS
    """Extra renders after those steps, so the RTX denoiser's temporal history holds only the new
    scene; ghosts of the blocks' old positions otherwise survive into the first image."""

    def set_should_stop(self, should_stop: Callable[[], bool]) -> None:
        """Give the motion layer something to ask before every simulated step.

        The page's stop control lives on the other side of the MCP boundary from the arm, and this
        is the one wire between them. Passed as a callable rather than a flag so the caller keeps
        ownership of what "stopped" means; `web_ui/server.py` hands it the channel's property.

        :param should_stop: Called with no arguments, truthy when the robot should halt
        """
        self._motion.arm.should_stop = should_stop

    def settle_and_rerender(self) -> None:
        """Let physics settle and flush the renderer's history. Runs on the simulator's thread."""
        self._motion.arm.step(self.RESET_SETTLE_STEPS)
        sim = self._toyblocks_env.base_env.sim
        for _ in range(self.RESET_RERENDERS):
            sim.render()

    def _build_env(self, config: ToyBlocksConfig) -> ToyBlocksEnv:
        """Build the oracle's environment: viewer trajectory opened, denoiser settled.

        The base names `ToyBlocksEnv` inside `_build_env`, so overriding the hook is how the oracle
        substitutes its environment. Rebinding the module-level name instead would have to cover
        both the base server's namespace and `toyblocks_env`'s, since which one the base reads
        depends on how it imports the name, and would reach every server in the process. A hook
        cannot be missed that way, and it cannot reach a server that did not ask for it.

        :param config: The `ToyBlocksConfig` this server was built with
        :return: The environment wrapper the oracle skills drive
        """
        return OracleToyBlocksEnv(config.task_config, viewer=config.viewer)

    def _build_skill_backend(self) -> None:
        """Stand up the motion backend, then settle the scene it will plan against.

        One ordering inside this hook matters: settling comes before the world refresh and before
        the base records the first image, which happens as soon as this returns -- without it
        curobo plans against blocks mid-fall, and that first image carries ghosts of their spawn
        poses.

        Nothing here prepares the base's frame state. curobo's warmup and the first world refresh
        can step the simulator, and every step fires ``on_step=self.record_current_frame``, so
        that state has to exist before this hook runs; `ToyBlocksMCPServer.__init__` opens the
        video writer and zeroes the counter before calling it.
        """
        self._motion = InlineTampMotion(
            self._toyblocks_env.env,
            self.config.oracle,
            on_step=self.record_current_frame,
            warmup=WARMUP,
        )
        self._held_block: str | None = None

        self.settle_and_rerender()
        self._toyblocks_env.refresh_observation()
        self._motion.arm.refresh_world()

    @property
    def toyblocks_env(self) -> ToyBlocksEnv:
        """The adapter's environment wrapper, which the base class holds privately."""
        return self._toyblocks_env

    @property
    def table_z(self) -> float:
        """World-frame height of the table's top surface, derived on each use."""
        table_z = find_table_z(self._toyblocks_env)
        logger.debug(f"Placing blocks on a table at z={table_z:.4f} m.")
        return table_z

    def record_current_frame(self, camera_name: str | None = None) -> None:
        """Record a video frame, and push the current state to the viewer.

        Viser only advances what the browser shows when something calls `view_frame`, and these
        skills step PhysX directly rather than going through `ToyBlocksEnv.step`. Without this the
        viewer stays frozen on the boot state for the whole episode.
        """
        super().record_current_frame(camera_name)
        if self.frame_count % self.VIEWER_STRIDE == 0:
            self._toyblocks_env._refresh_viewer()

    def viewer_url(self) -> str | None:
        """Where this scene's viser server is listening, or None without a viewer.

        Asked for rather than assumed, because viser silently takes the next free port when its
        default is busy -- which it can be for a moment after the previous scene's server stopped.
        """
        viewer = self._toyblocks_env._viewer
        if viewer is None:
            return None
        # viser reports its bind address; 0.0.0.0 is not something a browser can dial, and the
        # page knows to swap "localhost" for the host it reached this machine by.
        return viewer.get_server_url().replace("//0.0.0.0:", "//localhost:")

    def dispose(self) -> None:
        """Tear this scene down so another can be built in the same process.

        Stops the viser server first, so the next scene's viewer can take its port; then the base
        `shutdown` exports the video and closes the gym environment. Runs on the simulator's thread.
        """
        viewer = self._toyblocks_env._viewer
        if viewer is not None:
            try:
                viewer._server.stop()
            except Exception:
                logger.exception(
                    "Could not stop the viser server; the next one may take another port."
                )
        self.shutdown()
        logger.info(
            f"Scene '{self.config.task_config.name}' disposed; "
            f"video and transcript in {self.config.output_dir}"
        )

    def seed_for_episode(self, episode_index: int) -> None:
        """Make the next reset's layout a function of the episode index rather than of history.

        The autogen env picks its reset state with `torch.randint` off the ambient device RNG, and
        the oracle's pose generators draw from the same stream, so without this an episode's layout
        depends on how many draws the previous episode happened to make. Two runs of the same task
        at the same seed then diverge from the second episode onward, which is exactly the
        comparison the memory metric needs to hold fixed.

        Seeding here rather than passing `instance_ids` keeps the fix on this side: `env.reset()`
        reaches `_reset_idx` with `instance_ids=None` and there is no argument to thread through it.

        :param episode_index: Which episode of this run is about to start, counting from zero
        """
        import random

        import numpy as np

        seed = self.config.rng_seed * 1_000_003 + episode_index
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        logger.info(f"Seeded episode {episode_index} with {seed}.")

    async def reset_scene_on_sim_thread(self, episode_index: int = 0) -> None:
        """Put the scene back to a fresh episode. Runs on the simulator's thread and never awaits.

        Reached from the page's "reset episode" control through `SimThreadOracleServer.reset_scene`.
        The order matters: the planning model must forget any held block before the environment
        moves it, and curobo's obstacle world is re-read after the blocks have landed somewhere
        else. The gym reset draws a fresh reset state from the autogen package, so the blocks come
        back in a new random layout.

        The gripper is *not* opened first. `set_state_dict` writes the whole articulation --
        `write_joint_state_to_sim` takes no joint filter, and both position and velocity targets
        are set -- so the reset restores the fingers along with the arm. Actuating them beforehand
        would only play out a held block falling to the table, several seconds before the arm it
        fell from is reset out from under it.

        Then :meth:`settle_and_rerender` runs before anything reads the scene: right after a reset
        the RTX renderer still shows the blocks where they were, as ghosts, until enough frames
        have gone by, and a first image with ghost blocks in it would mislead the model.
        """
        self.seed_for_episode(episode_index)
        self._motion.arm.detach()
        self._held_block = None

        def block_positions() -> dict[str, list[float]]:
            return {
                name: [round(v, 3) for v in self._toyblocks_env.get_block_center(name).tolist()]
                for name in self._toyblocks_env.block_names
            }

        logger.info(f"Block positions before reset: {block_positions()}")
        self._toyblocks_env.env.reset()
        logger.info(f"Block positions after reset: {block_positions()}")
        self.settle_and_rerender()
        self._toyblocks_env.refresh_observation()
        self._motion.arm.refresh_world()

        self._show_fresh_episode_in_viewer()
        self.record_current_frame()
        logger.info("Scene reset to a fresh episode.")

    def _show_fresh_episode_in_viewer(self) -> None:
        """Point viser at the trajectory the reset just opened, so the page shows the new scene.

        `Viewer.hook_to_env` appends a new trajectory on every env reset, but only views the very
        first one; later resets leave the browser on the old trajectory's last frame.
        """
        viewer = self._toyblocks_env._viewer
        if viewer is None:
            return
        slider = viewer._trajectory_id_slider
        before = (slider.value, slider.max, [len(t) for t in viewer._frames][-4:])
        try:
            slider.value = len(viewer._frames) - 1
        except Exception:
            logger.exception("Could not switch the viewer to the new trajectory.")
        self._toyblocks_env._refresh_viewer()
        logger.info(
            f"Viewer after reset: trajectories={len(viewer._frames)} slider(value,max,lens) "
            f"before={before} after=({slider.value}, {slider.max}) "
            f"viewed={viewer.viewed_trajectory_id} "
            f"last_viewed_set={viewer._last_viewed_set}"
        )

    # -- tools ---------------------------------------------------------------------------------

    async def get_detailed_image(self) -> MCPImage:
        """Retrieve the whole scene at the camera's full resolution.

        The image you normally receive is reduced in size; this is the same view at full size, and
        costs several times as much to look at. It is worth that when you want a more faithful
        view of the entire scene at once.

        To identify something small -- a letter or a marking on a block, or which face of a block
        points towards you -- use the zoomed view instead. Even at full resolution a block is a
        small part of this image, so this is usually still too coarse to read such things.

        Locations you identify in this image can be given to any other tool unchanged.
        """
        self._raise_if_shutdown(skill_name="GetDetailedImage")
        return array_to_mcp_image(self._current_frame(self.config.camera_name))

    async def get_zoomed_image(
        self,
        pixel: PixelCoord,
        zoom: Annotated[float, Field(ge=1.0, le=20.0)],
    ) -> MCPImage:
        """Retrieve a magnified view of one part of the current image observation.

        Use this whenever something is too small to identify: reading a letter or a marking on a
        block, or telling which face of a block is pointing towards you. This is the reliable way
        to see fine detail. Moving a block closer to the camera is not: the camera is low and
        distant, so carrying a block across the table barely changes how large it appears.

        The view returned is centred on the point you give.

        Coordinates read off this magnified view are meaningless to other tools. Points for
        picking, placing, rotating and flipping must be read off the standard image observation.

        :param point: Point in the image to centre the magnified view on, in the same coordinates
            as every other tool
        :param zoom: How much larger things should appear than in the standard image. 4 shows a
            quarter of the image's width and height around your point, at four times the size;
            10 magnifies ten times. Higher values magnify further; 1 returns the whole image.
        """
        self._raise_if_shutdown(skill_name="GetZoomedImage")

        frame = self._current_frame(self.config.camera_name)
        window = crop_around(frame, self.convert_obs_pixel_to_camera(pixel), 1.0 / zoom)

        return array_to_mcp_image(resize_to_width(window, self.observation_resolution.width))

    async def pick(self, pixel: PixelCoord) -> MCPImage:
        """Pick up the block at the given location in the current image observation.

        The gripper closes on the block and lifts it, so no separate gripper call is needed.

        Precondition: The gripper must be empty.
        """
        self._raise_if_shutdown(skill_name="Pick")
        if self._held_block is not None:
            raise OracleSkillError(
                f"The gripper is already holding '{self._held_block}'; place that block before "
                "picking up another."
            )

        block_name = self._resolve_block(pixel)
        self._motion.grasp(block_name)
        self._held_block = block_name

        return self.current_image_obs

    async def stack(self, pixel: PixelCoord) -> MCPImage:
        """Stack the held block onto the block at the given location in the current image.

        The held block is centered on top of the one pointed at, then released.

        Precondition: The gripper must be holding a block.
        """
        self._raise_if_shutdown(skill_name="PlaceOnBlock")
        held_block = self._require_held_block()

        support_name = self._resolve_block(pixel)
        self._motion.place_on_block(held_block, support_name)
        self._held_block = None

        return self.current_image_obs

    async def place_on_table(self, pixel: PixelCoord) -> MCPImage:
        """Place the held block on the table, at the given location in the current image.

        The block is placed at exactly the point given, not at a nearby free spot.

        Precondition: The gripper must be holding a block, and the point must be on the table.
        """
        self._raise_if_shutdown(skill_name="PlaceOnTable")
        held_block = self._require_held_block()

        world_xyz = self._deproject_to_table(pixel)
        self._motion.place_on_table(held_block, world_xyz)
        self._held_block = None

        return self.current_image_obs

    async def rotate(self, pixel: PixelCoord, direction: RotateDirectionArg) -> MCPImage:
        """Rotate the block at the given location in the current image observation.

        The block is turned 90 degrees about the vertical axis, in place, and stays on the table.

        Precondition: The gripper must be empty.
        """
        self._raise_if_shutdown(skill_name="Rotate")

        block_name = self._resolve_block(pixel)
        self._motion.rotate(block_name, direction)

        return self.current_image_obs

    TOP_FACE_ENDS_UP: ClassVar[dict[str, FlipDirection]] = {
        "back": FlipDirection.FORWARD,
        "front": FlipDirection.BACKWARD,
        "left": FlipDirection.LEFT_FLIP,
        "right": FlipDirection.RIGHT_FLIP,
    }
    """The interface's flip names, mapped onto the domain's `flip_dir` values.

    The interface asks where the *top face* ends up; `domain.pddl` names how the block *tips*, so
    the two are inverted for the fore-aft pair. `back` and `front` are settled empirically: in run
    `ai2-colors20260908-152044` a `forward` flip moved front->top and bottom->front, leaving the
    top face at the back. `left` and `right` are mapped by name and are **unverified** -- the
    in-plane experiment compared an observed state against a remembered one twenty calls older.
    """

    async def flip(self, pixel: PixelCoord, direction: FlipDirectionArg) -> MCPImage:
        """Flip the block at the given location in the current image observation.

        The block is tipped onto the adjacent face in the given direction, so the face that was
        pointing up ends up pointing sideways.

        Precondition: The gripper must be empty.
        """
        self._raise_if_shutdown(skill_name="Flip")

        block_name = self._resolve_block(pixel)
        self._motion.flip(block_name, self.TOP_FACE_ENDS_UP[direction])

        return self.current_image_obs

    async def return_home(self) -> MCPImage:
        """Open the gripper, drop anything held, and move the arm back to its resting pose.

        Use this when a skill reports that the arm cannot plan a move from where it is: every
        pick, place, rotate and flip will keep failing until the arm is moved, and this is the
        only tool that can move it, because it drives the joints directly instead of planning.
        It is also worth using to clear the arm out of the camera's view.

        Anything currently held is dropped where it is, so it lands on the table rather than
        being carried; pick it up again afterwards if it is still needed.
        """
        self._raise_if_shutdown(skill_name="ReturnHome")
        self._motion.return_home()
        self._held_block = None
        return self.current_image_obs

    async def toggle_gripper(self) -> MCPImage:
        """Toggle the gripper between an 'open' and a 'closed' state.

        The picking and placing skills operate the gripper themselves, so this is rarely needed.
        """
        self._raise_if_shutdown(skill_name="ToggleGripper")

        if self._motion.gripper_state == "open":
            self._motion.close_gripper()
        else:
            self._motion.open_gripper()
            self._held_block = None  # Opening drops whatever was held.

        return self.current_image_obs

    async def open_gripper(self) -> MCPImage:
        """Open the gripper, releasing anything it holds."""
        self._raise_if_shutdown(skill_name="OpenGripper")
        self._motion.open_gripper()
        self._held_block = None  # Opening drops whatever was held.
        return self.current_image_obs

    async def close_gripper(self) -> MCPImage:
        """Close the gripper, grasping whatever lies between its fingers."""
        self._raise_if_shutdown(skill_name="CloseGripper")
        self._motion.close_gripper()
        return self.current_image_obs

    # -- internals -----------------------------------------------------------------------------

    def _current_frame(self, camera_name: str) -> np.ndarray:
        """Retrieve the current image, re-reading the observation from the simulator first.

        These skills step the simulator directly rather than through `ToyBlocksEnv.step`, so the
        environment's cached observation is otherwise stale.
        """
        self._toyblocks_env.refresh_observation()
        return super()._current_frame(camera_name)

    def _deproject_to_table(self, pixel: PixelCoord) -> np.ndarray:
        """Intersect the camera ray through a pointed pixel with the table plane.

        :param pixel: Pixel in the observation the agent asked to place at
        :return: The world-frame point on the table plane, shape (3,)
        :raises OracleSkillError: If the ray is parallel to, or points away from, the plane
        """
        camera_model = self._toyblocks_env.read_camera_model(camera_name=self.config.camera_name)
        # The camera model carries the render's intrinsics, so the pixel has to be in the
        # render's raster, not the downscaled one the agent read it off.
        origin_w, direction_w = camera_model.pixel_to_ray(self.convert_obs_pixel_to_camera(pixel))

        if abs(direction_w[2]) < 1e-6:
            raise OracleSkillError("That point does not lie on the table; try another.")

        distance_m = (self.table_z - origin_w[2]) / direction_w[2]
        if distance_m <= 0:
            raise OracleSkillError("That point does not lie on the table; try another.")

        return origin_w + distance_m * direction_w

    def _resolve_block(self, pixel: PixelCoord) -> str:
        """Identify which block the agent pointed at.

        :param pixel: Raw pixel in the observation the model was shown
        :return: Name of the block whose projected center is nearest to that point
        :raises OracleSkillError: If no block projects close enough to the given point
        """
        held = frozenset() if self._held_block is None else frozenset({self._held_block})

        block_name = resolve_block(
            self._toyblocks_env,
            self.convert_obs_pixel_to_camera(pixel),
            camera_name=self.config.camera_name,
            config=self.config.oracle,
            exclude=held,
        )
        if block_name is None:
            # Deliberately says nothing more. Reporting the miss distance and the tracked block
            # positions hands the agent privileged state it cannot read out of the image, and
            # paying that out only on a miss makes "probe anywhere, read the answer off the error"
            # a cheaper strategy than looking. The pixel echoed back is the caller's own, in the
            # raster it named it in -- not the render-space pixel `convert_obs_pixel_to_camera`
            # produced, which is twice the size and would read as the arm having misheard the
            # request.
            raise OracleSkillError(
                f"No block found at ({pixel.u},{pixel.v}). Point at a block visible in the image."
            )

        return block_name

    def _require_held_block(self) -> str:
        """Return the name of the held block.

        :raises OracleSkillError: If the gripper is not currently holding a block
        """
        if self._held_block is None:
            raise OracleSkillError("The gripper is not holding a block; pick one up first.")
        return self._held_block


class SimThreadOracleServer(InlineOracleMCPServer):
    """Runs every registered skill on the simulator's thread.

    Isaac must be stepped from the thread that created it, so the simulator runs on the main thread
    and the agentic loop on a second one. This mirrors `SimThreadMCPServer` in
    `toyblocks_runner._boot_mcp_server`, which cannot be imported: it is defined inside that
    function and closed over a `SkillBackend` match that resolves to the `src` stub server. Stepping
    PhysX from the wrong thread crashes rather than raising, so the body is kept identical
    rather than restructured.
    """

    def __init__(self, dispatcher: SimulatorDispatcher, *args: Any, **kwargs: Any) -> None:
        """Initialize the simulator-thread MCP server."""
        self._dispatcher = dispatcher  # Set before super().__init__() registers the MCP tools
        self.registered_tools: list[str] = []  # Likewise; the tools append to it
        super().__init__(*args, **kwargs)

    def _register_tool(self, title: str, skill: Callable) -> None:
        """Register an MCP tool whose body will be executed on the simulator's thread.

        A stop raised inside the skill is caught here rather than escaping as a tool error,
        because a stopped skill is not a failed one: the arm is somewhere legitimate, and what the
        model needs is the scene as it now stands. The world is re-read first -- the arm halted
        somewhere no plan chose, possibly holding a block -- so the next plan is made against it.
        """
        import asyncio
        import functools

        dispatcher = self._dispatcher

        @functools.wraps(skill)
        async def dispatched(*args: Any, **kwargs: Any) -> object:
            async def body() -> object:
                try:
                    return await skill(*args, **kwargs)
                except SkillStopped:
                    logger.info(f"'{title}' stopped by a human; refreshing the world.")
                    self._motion.arm.refresh_world()
                    self.record_current_frame()
                    return self.current_image_obs

            return await asyncio.wrap_future(dispatcher.submit(body()))

        self.registered_tools.append(title)
        super()._register_tool(title, dispatched)

    async def reset_scene(self, episode_index: int = 0) -> None:
        """Reset the scene on the simulator's thread. `web_ui/server.py` finds this by name.

        :param episode_index: Which episode of this run is about to start, counting from zero
        """
        import asyncio

        await asyncio.wrap_future(
            self._dispatcher.submit(self.reset_scene_on_sim_thread(episode_index))
        )
