"""Read and write the simulator's state as `agentic_robotics.core.states.SimState`.

Built on ToyBlocks's own state dictionary (`get_state_dict` and `set_state_dict` on the base
env), which is what the environment's resets use, so a state read here and written back lands the
scene exactly. Poses are world-frame in both directions, so the "world" parent frame the type
defaults to is true by construction. Velocities and joint targets are dropped on read and zeroed
on write, so a written state is always at rest with the arm holding its joints. Block sizes and
colours come from the task package, not the state. The state carries no task name: which task a
scene belongs to is not something the agent should be able to read off it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from agentic_robotics.core.states import (
    Configuration,
    JointType,
    ObjectPose,
    Pose,
    Position,
    Quaternion,
    SimState,
    _rounded,
)

if TYPE_CHECKING:
    import torch
    from isaaclab.assets import Articulation

    from agentic_robotics.adapters.toyblocks.toyblocks_env import ToyBlocksEnv

ROBOT = "robot"

STOWED_Z = -1.02
"""World z at which a block a written state leaves out rests: on the ground plane the scene
spawns at z = -1.05 (`base_env.py`), under the table and out of the camera's view. Isaac cannot
add or remove objects once a scene exists, so "a scene with fewer blocks" is every other block
stowed here. A read reports only the blocks above the table."""

STOWED_X, STOWED_Y0, STOWED_SPACING = 0.5, -0.6, 0.12
"""Stowed blocks sit in a row under the table's working area, spaced so they do not touch."""

TABLE_Z = -0.5
"""Blocks below this world z are stowed and left out of a read."""
"""The articulation key ToyBlocks's base env registers the arm under."""

DOF_TYPE_NAMES: dict[int, JointType] = {0: "revolute", 1: "prismatic"}
"""PhysX `DofType` values (Rotation, Translation) by their meaning in `core.states`."""


def _pose_from_row(row: torch.Tensor) -> Pose:
    """A world-frame `Pose` from one (x, y, z, qw, qx, qy, qz) row of a pose tensor."""
    p = _rounded(row[:3].tolist())
    q = _rounded(row[3:7].tolist())
    return Pose(
        position=Position(x=p[0], y=p[1], z=p[2]),
        orientation=Quaternion(w=q[0], x=q[1], y=q[2], z=q[3]),
    )


def _joint_types(articulation: Articulation) -> list[JointType]:
    """The articulation's joint types, read from PhysX (one row per articulation; env 0's)."""
    types: list[JointType] = []
    for value in articulation.root_physx_view.get_dof_types()[0].tolist():
        if value not in DOF_TYPE_NAMES:
            raise ValueError(f"Joint type {value} is not a revolute or prismatic joint.")
        types.append(DOF_TYPE_NAMES[value])
    return types


def read_sim_state(toyblocks_env: ToyBlocksEnv) -> SimState:
    """Read the scene's state from the simulator, in world-frame poses.

    :param toyblocks_env: The adapter's `ToyBlocksEnv`
    :return: The state, rounded to four decimals
    """
    base_env = toyblocks_env.base_env
    state = base_env.get_state_dict(is_relative=False)
    robot = state["articulation"][ROBOT]
    articulation = base_env.scene.articulations[ROBOT]
    return SimState(
        robot_base_pose=_pose_from_row(robot["root_pose"][0]),
        robot_configuration=Configuration(
            joint_names=list(articulation.joint_names),
            joint_types=_joint_types(articulation),
            joint_values=_rounded(robot["joint_position"][0].tolist()),
        ),
        objects=[
            ObjectPose(name=name, pose=pose)
            for name in toyblocks_env.block_names
            for pose in [_pose_from_row(state["rigid_object"][name]["root_pose"][0])]
            if pose.position.z > TABLE_Z
        ],
    )


def _unit_quaternion(q: Quaternion) -> list[float]:
    """The quaternion as a normalized (w, x, y, z) list; rejects anything far from unit length."""
    components = [q.w, q.x, q.y, q.z]
    norm = math.sqrt(sum(c * c for c in components))
    if not 0.9 < norm < 1.1:
        raise ValueError(f"Quaternion {components} is not close to unit length (norm {norm:.3f}).")
    return [c / norm for c in components]


def _assign_pose(tensor: torch.Tensor, pose: Pose) -> None:
    """Overwrite env 0's row of a (num_envs, 7) world-frame pose tensor in place."""
    if pose.parent_frame != "world":
        raise ValueError(
            f"Poses must be relative to the world frame; got parent_frame {pose.parent_frame!r}."
        )
    position = pose.position
    tensor[0] = tensor.new_tensor(
        [position.x, position.y, position.z, *_unit_quaternion(pose.orientation)]
    )


def write_sim_state(toyblocks_env: ToyBlocksEnv, state: SimState) -> None:
    """Write a state into the simulator and let it settle.

    Starts from the current state dictionary so that assets the state does not describe keep
    their values, then overwrites the robot and every block, zeroes all velocities, and sets the
    arm's joint targets to hold the written positions. The state may name any subset of the
    scene's blocks: the ones it leaves out are stowed on the floor under the table (`STOWED_Z`),
    so an agent can set up a one-block scene in a twelve-block package.

    :param toyblocks_env: The adapter's `ToyBlocksEnv`
    :param state: The state to write, with world-frame poses
    :raises ValueError: If the state names a block the scene lacks, names one twice, names the
        wrong joints, or uses a non-world frame
    """
    base_env = toyblocks_env.base_env
    current = base_env.get_state_dict(is_relative=False)  # Cloned tensors, edited in place below

    robot = current["articulation"][ROBOT]
    joint_names = list(base_env.scene.articulations[ROBOT].joint_names)
    configuration = state.robot_configuration
    if configuration.joint_names != joint_names:
        raise ValueError(
            f"joint_names must match the robot's order {joint_names}; got "
            f"{configuration.joint_names}."
        )
    if len(configuration.joint_values) != len(joint_names):
        raise ValueError(
            f"joint_values must have {len(joint_names)} values, in the order {joint_names}; got "
            f"{len(configuration.joint_values)}."
        )
    _assign_pose(robot["root_pose"], state.robot_base_pose)
    robot["root_velocity"].zero_()
    robot["joint_position"][0] = robot["joint_position"].new_tensor(configuration.joint_values)
    robot["joint_velocity"].zero_()
    robot["joint_position_target"].copy_(robot["joint_position"])

    scene_blocks = list(toyblocks_env.block_names)
    given = [obj.name for obj in state.objects]
    unknown = sorted(set(given) - set(scene_blocks))
    if unknown or len(given) != len(set(given)):
        raise ValueError(
            f"objects may name each of this scene's blocks at most once; the scene has "
            f"{scene_blocks}, got {given}."
        )
    for obj in state.objects:
        rigid_object = current["rigid_object"][obj.name]
        _assign_pose(rigid_object["root_pose"], obj.pose)
        rigid_object["root_velocity"].zero_()
    for index, name in enumerate(block for block in scene_blocks if block not in given):
        rigid_object = current["rigid_object"][name]
        rigid_object["root_pose"][0] = rigid_object["root_pose"].new_tensor(
            [STOWED_X, STOWED_Y0 + index * STOWED_SPACING, STOWED_Z, 1.0, 0.0, 0.0, 0.0]
        )
        rigid_object["root_velocity"].zero_()

    base_env.set_state_dict(current, is_relative=False)
    toyblocks_env._sync_simulator()  # Settle physics and refresh the observation
