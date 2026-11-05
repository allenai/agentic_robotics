"""The RL-trained rotate and flip skills: loading a ToyBlocks checkpoint and rolling it out on
the live scene, one environment at a time.

ToyBlocks ships the checkpoints (``tamp/blocks/checkpoints/<name>/``), the gym tasks the policies
were trained in (``blocksuite.tasks.rotate``, ``blocksuite.tasks.flip``) and the agent networks
(``blocksuite.rl.agents``). What it does not ship on its released branch is the executor layer
that builds a policy's observation from a scene it was not trained in: a pseudo environment of
the trained task is instantiated without a scene and pointed at the live scene's robot and block,
so the task's own observation and evaluation code runs against the live state. That layer is
here, carried over from the TAMP library's executors (``RotateTrajectory``, ``ReOrientTrajectory``
and their ``BaseRLTrajectory`` base) with the orchestration the TAMP planner needed left out.
:class:`InlineTampMotion` in ``inline_oracle.py`` drives it: load, prepare, then step until the
executor reports success or the step budget runs out.
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import gymnasium as gym
import numpy as np
import torch
from blocksuite.rl.agents import _AGENTS, AttentionMLPAgent
from blocksuite.utils import common_utils, tree_utils
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from collections.abc import Sequence

    from blocksuite.tasks.base_env import BaseEnv
    from blocksuite.tasks.base_env_cfg import BaseEnvCfg


class SimulatorLike(Protocol):
    """What the executors need of the caller: the live Isaac Lab environment."""

    @property
    def base_env(self) -> BaseEnv: ...


logger = logging.getLogger("agentic_robotics.examples.rl_executors")

CHECKPOINT_DIR: Path | None = None
"""Directory holding one subdirectory per checkpoint name (``final_ckpt.pt`` and
``config.json`` in each); set by :func:`set_checkpoint_dir` before a policy is loaded."""


def set_checkpoint_dir(path: Path) -> None:
    """Point the executors at the directory the RL checkpoints live in."""
    global CHECKPOINT_DIR
    CHECKPOINT_DIR = Path(path)


class _FakeContactData:
    """Mimics the ``.data`` of an Isaac Lab contact sensor."""

    def __init__(self, num_envs: int, device: torch.device) -> None:
        self.force_matrix_w = torch.zeros(num_envs, 1, 1, 3, device=device)


class FakeContactSensor:
    """A finger contact sensor that always reports zero force."""

    def __init__(self, num_envs: int, device: torch.device) -> None:
        self.data = _FakeContactData(num_envs, device)


class PseudoEnv:
    """The trained task's environment, built without a scene and pointed at the live one.

    ``rl_env`` was constructed with ``config_setup_only=True``, which returns before the scene is
    set up, so the attributes its observation code reads are bound here to the live environment's.
    """

    def __init__(self, rl_env: BaseEnv, base_env: BaseEnv) -> None:
        self.rl_env = rl_env
        self.rl_env._tracker_stats = {}
        self.rl_env.scene = base_env.scene
        # The observation code reads ``self.robot`` for the end effector; the two robots are
        # asserted identical in ``load_rl_agent``.
        self.rl_env.robot = base_env.robot
        self.rl_env._finger_contact_sensors = [
            FakeContactSensor(base_env.num_envs, base_env.device),
            FakeContactSensor(base_env.num_envs, base_env.device),
        ]
        # The privileged observation group reads these; it reaches only the critic, which is
        # unused at inference.
        self.rl_env.table_friction = base_env.table_friction
        self.rl_env.block_friction = base_env.block_friction
        self._base_env = base_env

    def _compute_intermediate_values(self) -> None:
        self.rl_env.joint_pos = self._base_env.robot.data.joint_pos
        self.rl_env.joint_vel = self._base_env.robot.data.joint_vel
        self.rl_env.joint_targets = self._base_env.robot.data.joint_pos_target
        self.rl_env._compute_intermediate_values()

    def get_observations(self) -> dict:
        rl_obs_dict = self.rl_env._get_observations()
        # This path bypasses the flattening wrapper, so the agent-group allowlist is applied here.
        rl_obs_dict["agent"] = self.rl_env.rl_agent_obs(rl_obs_dict["agent"])
        return rl_obs_dict

    def evaluate(self) -> dict:
        return self.rl_env.evaluate()


class BaseRLTrajectory:
    """One RL skill: a checkpoint, the task it was trained in, and how to run it on the live scene.

    Subclasses name the task and the checkpoint and say how the trained task's goal attributes
    are set from the skill's arguments (``env_data``: the block's name under ``obj``, the target
    pose under ``target_pose``, and for a flip the relative rotation under ``reorient_pose_rel``).
    """

    rl_task_id: str
    ckpt_name: str
    gripper_hold_target: float = 0.0
    """Target the gripper joints are held at during the rollout."""

    def load_rl_agent(self, simulator: SimulatorLike) -> None:
        """Load the checkpoint and build the pseudo environment the observations come from.

        :param simulator: The caller's handle on the live Isaac Lab environment
        :raises RuntimeError: If :data:`CHECKPOINT_DIR` is unset
        """
        from blocksuite.tasks.base_env_cfg import rl_agent_obs_keys_from_recorded_env_cfg

        if CHECKPOINT_DIR is None:
            raise RuntimeError("set_checkpoint_dir() has not been called; no checkpoints to load.")

        self.env_spec = gym.registry[self.rl_task_id]
        module_path, class_name = self.env_spec.entry_point.split(":")
        rl_env_class: type[BaseEnv] = getattr(importlib.import_module(module_path), class_name)
        module_path, class_name = self.env_spec.kwargs["env_cfg_entry_point"].split(":")
        cfg_class: type[BaseEnvCfg] = getattr(importlib.import_module(module_path), class_name)

        cfg = cfg_class()
        cfg = self.modify_env_cfg(cfg)
        rl_env = rl_env_class(cfg, config_setup_only=True)
        self._pseudo_env = PseudoEnv(rl_env, simulator.base_env)

        ckpt_path = CHECKPOINT_DIR / self.ckpt_name
        policy_state_dict = torch.load(str(ckpt_path / "final_ckpt.pt"))
        with open(ckpt_path / "config.json") as f:
            rl_config = json.load(f)

        # Safe to mutate ``cfg`` after the ``config_setup_only`` construction: it returned before
        # any observation space was cached.
        cfg.rl_agent_obs_keys = rl_agent_obs_keys_from_recorded_env_cfg(rl_config["env_cfg"])
        # An asymmetric-critic checkpoint was trained on a flat observation whose tail is the
        # privileged group, and the agent strips exactly ``priv_dim`` features off that tail for
        # the actor, so the pseudo environment has to emit the group or the actor's slice eats
        # real features. Read from ``priv_dim`` so the width emitted and the width stripped
        # cannot disagree.
        cfg.asymmetric_critic = rl_config["agent_kwargs"].get("priv_dim", 0) > 0
        agent_class = _AGENTS[rl_config["args"]["agent_type"]]
        if agent_class == AttentionMLPAgent:
            agent_kwargs = {
                k: v
                for k, v in rl_config["agent_kwargs"].items()
                if k not in ("dist_start_idx", "K_obs")
            }
            self.rl_agent = AttentionMLPAgent(
                **agent_kwargs,
                dist_start_idx=rl_config["agent_kwargs"]["dist_start_idx"],
                K_obs=rl_config["env_cfg"]["num_distractors"],
                device=simulator.base_env.device,
            )
        else:
            self.rl_agent = agent_class(
                **rl_config["agent_kwargs"], device=simulator.base_env.device
            )
        self.rl_agent.load_state_dict(policy_state_dict)
        self.rl_agent.eval()
        # Widths the checkpoint was trained at, validated against the first real observation in
        # ``step``; ``n_obs`` is the total across all groups, ``dist_start_idx`` (attention
        # checkpoints only) the agent group's width.
        self._expected_n_obs = rl_config["agent_kwargs"]["n_obs"]
        self._expected_agent_obs_width = rl_config["agent_kwargs"].get("dist_start_idx")
        self._obs_widths_validated = False
        self._rl_agent_control_mode = rl_config["env_cfg"]["control_mode"]
        if simulator.base_env.robot.id != rl_config["env_cfg"]["robot_id"]:
            raise RuntimeError(
                f"The checkpoint was trained with robot {rl_config['env_cfg']['robot_id']!r}; the "
                f"environment has {simulator.base_env.robot.id!r}."
            )

    def modify_env_cfg(self, env_cfg: BaseEnvCfg) -> BaseEnvCfg:
        """The trained task's configuration, adjusted for the rollout (a loosened success
        threshold, for instance)."""
        return env_cfg

    def prepare_rl_env(
        self, base_env: BaseEnv, rl_env: BaseEnv, env_data: dict, active_env_ids: Sequence[int]
    ) -> None:
        """Allocate the goal attributes on the pseudo environment and set them for the active
        environments, once before the rollout."""
        raise NotImplementedError

    def _setup_asset_references(
        self, base_env: BaseEnv, rl_env: BaseEnv, env_data: dict, env_ids: Sequence[int]
    ) -> None:
        """Point the pseudo environment at the live scene's block, once before the rollout."""
        raise NotImplementedError

    def _compute_goals(
        self, base_env: BaseEnv, rl_env: BaseEnv, env_data: dict, env_idx: int
    ) -> None:
        """Refresh the goal attributes for one environment, before each observation."""
        raise NotImplementedError

    def _sync_visualization_data(self, base_env: BaseEnv, rl_env: BaseEnv, env_idx: int) -> None:
        """Publish this environment's goal to the live environment's viewer, if any."""

    def _validate_obs_widths(self, obs_dict: dict, obs: torch.Tensor) -> None:
        """Check the live observation matches the widths the checkpoint was trained at.

        Runs once, on the first real observation, since the widths depend on the scene. A
        mismatch means the observation keys were derived wrongly from the checkpoint's config,
        which would otherwise feed a silently misaligned vector into a wrongly sized layer.

        :raises ValueError: On a mismatch, naming both widths
        """
        if self._obs_widths_validated:
            return
        self._obs_widths_validated = True
        n_obs = obs.shape[-1]
        if n_obs != self._expected_n_obs:
            raise ValueError(
                f"{type(self).__name__}: checkpoint {self.ckpt_name!r} expects "
                f"n_obs={self._expected_n_obs} but the environment produced {n_obs}. "
                f"Derived rl_agent_obs_keys={list(self._pseudo_env.rl_env.cfg.rl_agent_obs_keys)}."
            )
        if self._expected_agent_obs_width is None:
            return
        agent_width = common_utils.flatten_state_dict(obs_dict["agent"]).shape[-1]
        if agent_width != self._expected_agent_obs_width:
            raise ValueError(
                f"{type(self).__name__}: checkpoint {self.ckpt_name!r} expects an agent "
                f"observation group of width {self._expected_agent_obs_width} "
                f"(dist_start_idx) but the environment produced {agent_width}. "
                f"Derived rl_agent_obs_keys={list(self._pseudo_env.rl_env.cfg.rl_agent_obs_keys)}."
            )

    def step(
        self, simulator: SimulatorLike, env_idx: int | Sequence[int], env_data: dict
    ) -> torch.Tensor:
        """One policy step for one environment: the arm joint deltas the policy commands.

        The gripper joints are held at :attr:`gripper_hold_target`; the caller applies the
        returned targets and steps the simulation.
        """
        if isinstance(env_idx, (list, tuple)):
            env_idx = env_idx[0]
        base_env = simulator.base_env
        robot = base_env.robot
        original_joint_targets = robot.data.joint_pos_target.clone()
        original_joint_targets[env_idx, 7:] = self.gripper_hold_target
        robot.articulation.set_joint_position_target(original_joint_targets)
        base_env.set_robot_control_mode(self._rl_agent_control_mode)
        self._compute_goals(base_env, self._pseudo_env.rl_env, env_data, env_idx)
        self._pseudo_env._compute_intermediate_values()
        self._sync_visualization_data(base_env, self._pseudo_env.rl_env, env_idx)
        obs_dict = self._pseudo_env.get_observations()
        # Slice rather than index so the batch dimension survives: the agents index
        # ``obs[:, ...]`` to strip the privileged tail or cut the distractor slice.
        obs = common_utils.flatten_state_dict(obs_dict)[env_idx : env_idx + 1]
        self._validate_obs_widths(obs_dict, obs)
        with torch.no_grad():
            action = self.rl_agent.get_deterministic_action(obs)
            # Reuses the environment's own controllers to turn the action into joint targets.
            base_env._pre_physics_step(action.repeat(base_env.num_envs, 1))
            base_env._apply_action()
        robot.articulation.set_joint_position_target(original_joint_targets)
        robot.write_data_to_sim()
        return base_env.robot.controller["arm"].processed_actions.clone()

    def evaluate(self, simulator: SimulatorLike, env_idx: int, env_data: dict) -> dict:
        """The trained task's evaluation of one environment's current state (``success`` among
        its keys)."""
        base_env = simulator.base_env
        base_env.set_robot_control_mode(self._rl_agent_control_mode)
        self._compute_goals(base_env, self._pseudo_env.rl_env, env_data, env_idx)
        self._pseudo_env._compute_intermediate_values()
        return tree_utils.slice(self._pseudo_env.evaluate(), env_idx)


def _block_sizes_from_scene(base_env: BaseEnv, rl_env: BaseEnv, env_ids: Sequence[int]) -> None:
    """Allocate the pseudo environment's block sizes if needed and read them off the live block."""
    from blocksuite.utils import obs_utils

    if not hasattr(rl_env, "block_sizes"):
        block_size = torch.tensor(
            rl_env.cfg.block_size, device=base_env.device, dtype=torch.float32
        )
        rl_env.block_sizes = block_size.unsqueeze(0).expand(base_env.num_envs, -1).clone()
    bbox = obs_utils.get_bounding_box(rl_env.block)
    rl_env.block_sizes[env_ids] = bbox[env_ids, 1] - bbox[env_ids, 0]


class RotateTrajectory(BaseRLTrajectory):
    """Rotate a block in place about the vertical axis, by the trained ``rotate`` policy."""

    rl_task_id: str = "Blocksuite-Rotate-v0"
    ckpt_name: str = "rotate"

    def modify_env_cfg(self, env_cfg: BaseEnvCfg) -> BaseEnvCfg:
        env_cfg.orientation_threshold = np.deg2rad(10.0)
        env_cfg.distance_threshold = 9999  # the distance threshold is disabled
        return env_cfg

    def prepare_rl_env(
        self, base_env: BaseEnv, rl_env: BaseEnv, env_data: dict, active_env_ids: Sequence[int]
    ) -> None:
        device = base_env.device
        if not hasattr(rl_env, "goal_pos"):
            rl_env.goal_pos = torch.zeros(rl_env.num_envs, 3, device=device)
        if not hasattr(rl_env, "goal_quat"):
            rl_env.goal_quat = torch.zeros(rl_env.num_envs, 4, device=device)
        if not hasattr(rl_env, "table_height_offset"):
            rl_env.table_height_offset = torch.zeros(rl_env.num_envs, device=device)
        if not hasattr(rl_env, "_reach_achieved"):
            rl_env._reach_achieved = torch.zeros(base_env.num_envs, dtype=torch.bool, device=device)
        if not hasattr(rl_env, "actions"):
            rl_env.actions = torch.zeros(base_env.num_envs, 3, device=device)

        target_pos = torch.stack(
            [env_data[env_idx]["target_pose"].position for env_idx in active_env_ids]
        ).squeeze(1)
        target_quat = torch.stack(
            [env_data[env_idx]["target_pose"].quaternion for env_idx in active_env_ids]
        ).squeeze(1)
        # The target position is in the environment's local frame; the goal is in the world frame.
        rl_env.goal_pos[active_env_ids] = target_pos + rl_env.scene.env_origins[active_env_ids]
        rl_env.goal_quat[active_env_ids] = target_quat
        rl_env.table_height_offset[active_env_ids] = base_env.scene.rigid_objects[
            "workspace_table"
        ].data.root_pose_w[active_env_ids, 2]
        rl_env.goal_pos[active_env_ids, 2] = rl_env.block.data.root_pos_w[active_env_ids, 2]
        rl_env.table_penetration = torch.zeros(rl_env.num_envs, device=device)

    def _setup_asset_references(
        self, base_env: BaseEnv, rl_env: BaseEnv, env_data: dict, env_ids: Sequence[int]
    ) -> None:
        rl_env.block = base_env.scene[env_data[env_ids[0]]["obj"]]
        _block_sizes_from_scene(base_env, rl_env, env_ids)

    def _sync_visualization_data(self, base_env: BaseEnv, rl_env: BaseEnv, env_idx: int) -> None:
        base_env.block = rl_env.block
        base_env.block_corner_pts = rl_env.block_corner_pts
        base_env.set_goal_visualization(
            env_idx, rl_env.goal_corner_pts[env_idx], rl_env.goal_quat[env_idx]
        )

    def _compute_goals(
        self, base_env: BaseEnv, rl_env: BaseEnv, env_data: dict, env_idx: int
    ) -> None:
        rl_env.block = base_env.scene[env_data["obj"]]
        _block_sizes_from_scene(base_env, rl_env, [env_idx])
        target_pose = env_data["target_pose"]
        rl_env.goal_pos[env_idx] = target_pose.position + rl_env.scene.env_origins[env_idx]
        rl_env.goal_quat[env_idx] = target_pose.quaternion
        rl_env.goal_pos[env_idx, 2] = rl_env.block.data.root_pos_w[env_idx, 2]
        rl_env.table_penetration = torch.zeros(rl_env.num_envs, device=base_env.device)


class ReOrientTrajectory(BaseRLTrajectory):
    """Flip a block onto another face, by the trained ``flip`` policy."""

    rl_task_id: str = "Blocksuite-Flip-v0"
    ckpt_name: str = "flip"
    gripper_hold_target: float = 0.0

    def modify_env_cfg(self, env_cfg: BaseEnvCfg) -> BaseEnvCfg:
        env_cfg.orientation_threshold = 0.15
        env_cfg.distance_threshold = 9999  # the distance threshold is disabled
        return env_cfg

    def prepare_rl_env(
        self, base_env: BaseEnv, rl_env: BaseEnv, env_data: dict, active_env_ids: Sequence[int]
    ) -> None:
        device = base_env.device
        if not hasattr(rl_env, "flip_push_dir_world"):
            rl_env.flip_push_dir_world = torch.zeros(rl_env.num_envs, 3, device=device)
        if not hasattr(rl_env, "goal_pos"):
            rl_env.goal_pos = torch.zeros(rl_env.num_envs, 3, device=device)
        if not hasattr(rl_env, "goal_quat"):
            rl_env.goal_quat = torch.zeros(rl_env.num_envs, 4, device=device)
        if not hasattr(rl_env, "table_height_offset"):
            rl_env.table_height_offset = torch.zeros(rl_env.num_envs, device=device)
        if not hasattr(rl_env, "robot"):
            rl_env.robot = rl_env._robot_class(cfg=rl_env._robot_cfg, env=rl_env)
            rl_env.robot.articulation = base_env.scene["robot"]
        if not hasattr(rl_env, "_finger_joint_ids"):
            rl_env._finger_joint_ids, _ = rl_env.robot.articulation.find_joints(["finger_joint"])
        if not hasattr(rl_env, "_inner_finger_body_ids"):
            rl_env._inner_finger_body_ids, _ = rl_env.robot.articulation.find_bodies(
                ["left_inner_finger", "right_inner_finger"]
            )
        if not hasattr(rl_env, "_finger_tip_local"):
            rl_env._finger_tip_local = torch.tensor([0.0433, 0.0, 0.1388], device=device)
        if not hasattr(rl_env, "_gripper_close_target"):
            rl_env._gripper_close_target = (
                rl_env.cfg.gripper_close_fraction
                * rl_env.robot.articulation.data.joint_pos_limits[0, rl_env._finger_joint_ids, 1]
            )

        # The trained task holds the gripper closed by a fraction of the finger joint's travel.
        finger_upper = rl_env.robot.articulation.data.joint_pos_limits[
            0, rl_env._finger_joint_ids, 1
        ]
        self.gripper_hold_target = rl_env.cfg.gripper_close_fraction * finger_upper

        target_pos = torch.stack(
            [env_data[env_idx]["target_pose"].position for env_idx in active_env_ids]
        ).squeeze(1)
        target_quat = torch.stack(
            [env_data[env_idx]["target_pose"].quaternion for env_idx in active_env_ids]
        ).squeeze(1)
        rl_env.goal_pos[active_env_ids] = target_pos + rl_env.scene.env_origins[active_env_ids]
        rl_env.goal_quat[active_env_ids] = target_quat
        rl_env.table_height_offset[active_env_ids] = base_env.scene.rigid_objects[
            "workspace_table"
        ].data.root_pose_w[active_env_ids, 2]
        rl_env.goal_pos[active_env_ids, 2] = rl_env.block.data.root_pos_w[active_env_ids, 2]
        rl_env.table_penetration = torch.zeros(rl_env.num_envs, device=device)

        # Where the current top face goes after the flip: the relative rotation stored with the
        # skill's arguments, applied to world up.
        n = len(active_env_ids)
        world_up = torch.zeros(n, 3, device=device)
        world_up[:, 2] = 1.0
        relative_quats = torch.stack(
            [
                env_data[env_idx]["reorient_pose_rel"].quaternion.squeeze(0)
                for env_idx in active_env_ids
            ]
        )
        rl_env.flip_push_dir_world[active_env_ids] = quat_apply(relative_quats, world_up)

    def _setup_asset_references(
        self, base_env: BaseEnv, rl_env: BaseEnv, env_data: dict, env_ids: Sequence[int]
    ) -> None:
        rl_env.block = base_env.scene[env_data[env_ids[0]]["obj"]]
        _block_sizes_from_scene(base_env, rl_env, env_ids)

    def _sync_visualization_data(self, base_env: BaseEnv, rl_env: BaseEnv, env_idx: int) -> None:
        base_env.block = rl_env.block
        base_env.block_corner_pts = rl_env.block_corner_pts
        base_env.set_goal_visualization(
            env_idx, rl_env.goal_corner_pts[env_idx], rl_env.goal_quat[env_idx]
        )

    def _compute_goals(
        self, base_env: BaseEnv, rl_env: BaseEnv, env_data: dict, env_idx: int
    ) -> None:
        rl_env.block = base_env.scene[env_data["obj"]]
        _block_sizes_from_scene(base_env, rl_env, [env_idx])
        target_pose = env_data["target_pose"]
        rl_env.goal_pos[env_idx] = target_pose.position + rl_env.scene.env_origins[env_idx]
        rl_env.goal_quat[env_idx] = target_pose.quaternion
        rl_env.goal_pos[env_idx, 2] = rl_env.block.data.root_pos_w[env_idx, 2]
        rl_env.table_penetration = torch.zeros(rl_env.num_envs, device=base_env.device)

        world_up = torch.zeros(1, 3, device=base_env.device)
        world_up[:, 2] = 1.0
        relative_quat = env_data["reorient_pose_rel"].quaternion
        if len(relative_quat.shape) == 1:
            relative_quat = relative_quat.unsqueeze(0)
        rl_env.flip_push_dir_world[env_idx] = quat_apply(relative_quat, world_up)
