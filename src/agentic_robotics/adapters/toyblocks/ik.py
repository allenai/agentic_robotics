"""Seeded cuRobo inverse kinematics for the scripted skills.

The scripted skills move the end effector over short segments (setting a block down, lifting it
off, retracting) by solving IK for the target pose and interpolating in joint space. These
helpers hold the IK part: a solve seeded with the current configuration, so cuRobo returns the
solution nearest to it, and the acceptance rules for that solution.

The code was carried over from ToyBlocks's ``leap`` branch (``tamp/blocks/executors/
IKInterpTrajectory.py``) and the TAMP library's ``tmp3.Utils.ik_seed``, so that the adapter
depends on ToyBlocks's released branch only, which carries neither.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from curobo.types.state import JointState

if TYPE_CHECKING:
    from collections.abc import Sequence

    from curobo.types.math import Pose
    from curobo.wrap.reacher.ik_solver import IKResult
    from curobo.wrap.reacher.motion_gen import MotionGen

logger = logging.getLogger("agentic_robotics.examples.ik")

IK_POSITION_TOLERANCE: float = 0.005
"""Meters. Contact with the reference object is intended on the short segments these solutions
serve (setting a block down, lifting it off), so a solution that converged on the pose is accepted
even when cuRobo flags it as in collision; this is how close "converged" has to be."""

IK_ROTATION_TOLERANCE: float = 0.05
"""cuRobo geodesic distance; the rotation counterpart of :data:`IK_POSITION_TOLERANCE`."""

JOINT_LIMIT_MARGIN: float = 0.15
"""Radians. Solutions closer than this to a joint bound are rejected: the seeded IK drifts wrist
joints onto their limits over successive short segments, and trajectory optimization then refuses
to start from there. Also keeps joint 7 under the real FR3 limit (2.897 rad), which is tighter than
the cuRobo config's 3.016 rad."""


def _ik_joint_names(motion_gen: MotionGen) -> list[str] | None:
    """Joint order the IK solver expects, if it can be read off ``motion_gen``."""
    try:
        return list(motion_gen.ik_solver.joint_names)
    except Exception:
        return None


def _seed_position(
    motion_gen: MotionGen, joint_state: JointState | torch.Tensor | None
) -> torch.Tensor | None:
    """Flatten ``joint_state`` into a ``(1, dof)`` seed ordered for the IK solver, or None."""
    if joint_state is None:
        return None
    if isinstance(joint_state, JointState):
        ik_joint_names = _ik_joint_names(motion_gen)
        if (
            ik_joint_names is not None
            and joint_state.joint_names is not None
            and list(joint_state.joint_names) != ik_joint_names
        ):
            if set(ik_joint_names) - set(joint_state.joint_names):
                # The solver optimizes joints there is no value for; a partial seed would be
                # silently misaligned, so skip seeding.
                return None
            joint_state = joint_state.get_ordered_joint_state(ik_joint_names)
        position = joint_state.position
    else:
        position = joint_state
    if position is None:
        return None
    position = position.detach().reshape(1, -1).to(dtype=torch.float32)
    dof = getattr(getattr(motion_gen, "ik_solver", None), "dof", None)
    if dof is not None and position.shape[-1] != dof:
        return None
    return position


def solve_ik_seeded(
    motion_gen: MotionGen,
    pose: Pose,
    joint_state: JointState | torch.Tensor | None,
    **kwargs: object,
) -> IKResult:
    """``motion_gen.solve_ik(pose)`` seeded with the current joint configuration.

    cuRobo returns the solution closest to the current configuration when both ``retract_config``
    and ``seed_config`` are set to it. The seed is only the first seed; cuRobo still draws its
    other seeds at random, so seeding biases which solution comes back without lowering the
    success rate. Falls back to an unseeded solve when the seed cannot be built or the installed
    cuRobo rejects it, so IK never fails purely because of seeding.

    :param motion_gen: cuRobo ``MotionGen``
    :param pose: Target pose, in the robot's base frame
    :param joint_state: Current configuration, as the seed
    :param kwargs: Passed on to ``solve_ik``
    :return: cuRobo's IK result
    """
    position = _seed_position(motion_gen, joint_state)
    if position is not None:
        try:
            return motion_gen.solve_ik(
                pose, retract_config=position, seed_config=position.unsqueeze(1), **kwargs
            )
        except (TypeError, RuntimeError) as exc:
            logger.warning(f"Seeded IK rejected ({exc}); solving unseeded.")
    return motion_gen.solve_ik(pose, **kwargs)


def joint_limit_margin(
    motion_gen: MotionGen, q_arm: torch.Tensor, joint_names: Sequence[str]
) -> float:
    """Smallest distance, in radians, from ``q_arm`` (ordered as ``joint_names``) to any joint
    bound."""
    limits = motion_gen.kinematics.get_joint_limits()
    kin_names = list(motion_gen.kinematics.joint_names)
    order = [kin_names.index(n) for n in joint_names]
    lower = limits.position[0][order].to(q_arm.device)
    upper = limits.position[1][order].to(q_arm.device)
    return float(torch.minimum(q_arm - lower, upper - q_arm).min().item())


def self_collision_free(motion_gen: MotionGen, q_arm: torch.Tensor) -> bool:
    """Self-collision and joint-limit check of the ``(dof,)`` arm configuration, world muted.

    Mirrors ``MotionGen.check_start_state``. Returns True when the check cannot run (CUDA-graph
    rollouts), so it never rejects on its own failure.
    """
    rollout_fn = motion_gen.rollout_fn
    if getattr(rollout_fn, "cuda_graph_instance", False):
        return True
    q = q_arm.detach().reshape(1, 1, -1)
    try:
        rollout_fn.primitive_collision_constraint.disable_cost()
        metrics = rollout_fn.rollout_constraint(q, use_batch_env=False)
        return bool(metrics.feasible.reshape(-1)[0].item())
    except Exception as exc:
        logger.warning(f"Self-collision check unavailable ({exc}); accepting the IK solution.")
        return True
    finally:
        rollout_fn.primitive_collision_constraint.enable_cost()


def solve_ik_joint_target(
    motion_gen: MotionGen, goal_pose: Pose, seed_state: JointState, joint_names: Sequence[str]
) -> torch.Tensor | None:
    """IK for ``goal_pose`` seeded with ``seed_state``, as a ``(dof,)`` joint vector, or None.

    A solution is refused when it sits within :data:`JOINT_LIMIT_MARGIN` of a joint bound, when
    it did not converge within the position and rotation tolerances, or when a converged but
    collision-flagged solution turns out to be a self-collision rather than the intended contact
    with the world.

    :param motion_gen: cuRobo ``MotionGen``
    :param goal_pose: Target end-effector pose, in the robot's base frame
    :param seed_state: Current joint state, the seed
    :param joint_names: Order the returned vector follows
    :return: Arm joint positions ordered as ``joint_names``, or None if no acceptable solution
    """
    ik_result = solve_ik_seeded(motion_gen, goal_pose, seed_state)
    if ik_result is None or ik_result.js_solution is None:
        return None
    success = ik_result.success is not None and bool(ik_result.success.flatten()[0].item())
    goal_js = ik_result.js_solution.get_ordered_joint_state(list(joint_names))
    q_goal = goal_js.position.detach().reshape(-1)
    margin = joint_limit_margin(motion_gen, q_goal, joint_names)
    if margin < JOINT_LIMIT_MARGIN:
        logger.warning(
            f"IK solution is {margin:.3f} rad from a joint limit (< {JOINT_LIMIT_MARGIN}); "
            "rejecting."
        )
        return None
    if not success:
        position_error = float(ik_result.position_error.flatten()[0].item())
        rotation_error = float(ik_result.rotation_error.flatten()[0].item())
        if position_error > IK_POSITION_TOLERANCE or rotation_error > IK_ROTATION_TOLERANCE:
            logger.warning(
                f"IK did not converge (position_error={position_error:.4f}, "
                f"rotation_error={rotation_error:.4f}); rejecting."
            )
            return None
        if not self_collision_free(motion_gen, q_goal):
            logger.warning("IK solution converged but is in self-collision; rejecting.")
            return None
    return q_goal
