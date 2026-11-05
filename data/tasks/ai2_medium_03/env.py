# GENERATED CODE
from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING
import numpy as np

import numpy as np
import torch

from blocksuite.tasks.base_env import BaseEnv
from blocksuite.tasks.base_env_cfg import BaseEnvCfg
from blocksuite.utils import common_utils, tree_utils
from blocksuite.utils.randomization_utils import set_rigid_body_friction
from blocksuite.utils.reset_states_io import load_reset_states

from .env_cfg import ai2_medium_03_EnvCfg

if TYPE_CHECKING:
    from blocksuite.utils.episode_queue import EpisodeQueueClient


class ai2_medium_03_Env(BaseEnv):
    cfg: ai2_medium_03_EnvCfg
    reset_states: dict[str, dict[str, dict[str, torch.Tensor]]] | None = None
    reset_state_profile_ids: torch.Tensor | None = None
    num_scene_profiles: int | None = None
    queue_client: EpisodeQueueClient | None = None

    def __init__(self, cfg: ai2_medium_03_EnvCfg, **kwargs):
        if os.path.exists(cfg.reset_states_path):
            reset_states, profile_ids, num_profiles = load_reset_states(cfg.reset_states_path)
            self.reset_states = common_utils.to_torch(reset_states, device=cfg.sim.device)
            self.num_reset_states = tree_utils.shape(reset_states, first_only=True)[0]
            if profile_ids is not None:
                self.reset_state_profile_ids = torch.as_tensor(profile_ids, device=cfg.sim.device)
            self.num_scene_profiles = num_profiles
        super().__init__(cfg, **kwargs)

    def _merge_instance_layout_json_into_cfg(self, cfg: BaseEnvCfg) -> None:
        from blocksuite.utils import instance_layout_json

        self._instance_json_blocks = instance_layout_json.merge_instance_json_into_env_cfg(
            cfg,
            block_names=frozenset(("block_2", "block_a", "block_i")),
        )

    def _write_instance_layout_json_block_poses(self, env_ids: Sequence[int] | None) -> None:
        from blocksuite.utils import instance_layout_json

        layout = getattr(self, "_instance_json_blocks", None)
        if layout is None or env_ids is None:
            return
        instance_layout_json.write_block_layout_to_sim(self, layout, env_ids)

    def _setup_scene_objects(self):
        from blocksuite.utils import rigid_objects
        from blocksuite import OBJECT_ASSET_DIR

        block_names = ["block_a", "block_i", "block_2"]
        block_specs = {
            "block_a": {
                "type": "usd",
                "color": (1.0, 0.0, 0.0),
                "usd_path": "train_blocks/usd/A.usd",
                "usd_scale": (1.65, 1.65, 1.65),
            },
            "block_i": {
                "type": "usd",
                "color": (1.0, 0.0, 0.0),
                "usd_path": "train_blocks/usd/I.usd",
                "usd_scale": (1.65, 1.65, 1.65),
            },
            "block_2": {
                "type": "usd",
                "color": (1.0, 0.0, 0.0),
                "usd_path": "train_blocks/usd/2.usd",
                "usd_scale": (1.65, 1.65, 1.65),
            },
        }
        randomizations = {
            "block_sizes": {
                "block_a": [
                    [0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04],
                ],
                "block_i": [
                    [0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04],
                ],
                "block_2": [
                    [0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04],
                ],
            },
            "table_heights": [],
            "friction": {
                "static_friction": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                "dynamic_friction": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                "restitution": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            "cube_scale_slots": [],
            "table_height": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        MAX_NON_SIM_STATE_RANDOMIZATIONS = 8
        derive_block_scale_from_size = False
        self.randomizations = randomizations
        self.MAX_NON_SIM_STATE_RANDOMIZATIONS = MAX_NON_SIM_STATE_RANDOMIZATIONS
        # NOTE (stao): I think we might want to just get rid most of the codegen code here and stick most of the code into the template_env.py file
        if derive_block_scale_from_size:
            # Exact per-episode cube-scale matching: every distinct recorded
            # scale is a spawn slot, so num_envs must cover all of them (a
            # positive multiple of MAX) or some episodes have no size-matched env.
            assert self.num_envs % MAX_NON_SIM_STATE_RANDOMIZATIONS == 0, (
                f"num_envs ({self.num_envs}) must be a positive multiple of MAX_NON_SIM_STATE_RANDOMIZATIONS ({MAX_NON_SIM_STATE_RANDOMIZATIONS} = number of distinct recorded cube scales) so every cube-scale slot is spawned for exact matching."
            )
        else:
            assert (
                self.num_envs % MAX_NON_SIM_STATE_RANDOMIZATIONS == 0
                or self.num_envs < MAX_NON_SIM_STATE_RANDOMIZATIONS
            ), (
                f"Number of parallel envs has to be a multiple of {MAX_NON_SIM_STATE_RANDOMIZATIONS} due to isaac asset randomization limitations"
            )
        block_sizes_list = dict()

        replay_side = getattr(self.cfg, "replay_block_side_length", None)
        if replay_side is not None and self.num_envs == 1:
            side_vec = [float(replay_side), float(replay_side), float(replay_side)]
            for name in block_names:
                if len(randomizations["block_sizes"][name]) > 0:
                    block_sizes_list[name] = [side_vec]
        else:
            # randomize block sizes if needed and format the list
            for name in block_names:
                if len(randomizations["block_sizes"][name]) > 0:
                    assert isinstance(randomizations["block_sizes"][name], list), (
                        "block_size randomization should be given as a list"
                    )
                    if self.num_envs < MAX_NON_SIM_STATE_RANDOMIZATIONS:
                        block_sizes_list[name] = randomizations["block_sizes"][name][
                            : self.num_envs
                        ]
                    else:
                        block_sizes_list[name] = randomizations["block_sizes"][name] * (
                            self.num_envs // MAX_NON_SIM_STATE_RANDOMIZATIONS
                        )
        # randomize block frictions
        fric = randomizations["friction"]
        sf_list = fric["static_friction"]
        df_list = fric["dynamic_friction"]
        rest_list = fric["restitution"]
        n = len(sf_list)
        if self.num_envs <= n:
            sf_list = sf_list[: self.num_envs]
            df_list = df_list[: self.num_envs]
            rest_list = rest_list[: self.num_envs]
        else:
            reps = self.num_envs // n
            sf_list = sf_list * reps
            df_list = df_list * reps
            rest_list = rest_list * reps

        for i, name in enumerate(block_names):
            spec = block_specs[name]
            if spec["type"] == "cuboid":
                block = rigid_objects.create_cuboid(
                    prim_path=f"/World/envs/env_.*/{name}",
                    size=block_sizes_list[name],
                    init_pos=(i + 1, 0.0, 0.05 / 2),
                    init_quat=(1, 0, 0, 0),
                    diffuse_color=spec["color"],
                    metallic=0.2,
                    static_friction=sf_list,
                    dynamic_friction=df_list,
                    restitution=rest_list,
                )
            else:
                usd_kwargs: dict = {
                    "usd_path": str(OBJECT_ASSET_DIR / spec["usd_path"]),
                }
                if derive_block_scale_from_size:
                    # When matching recorded per-episode sizes, let create_toy_block
                    # derive each env's scale from its size, and disable size
                    # bucketing so every distinct recorded scale is its own USD
                    # prototype (exact match; more prototypes to load).
                    usd_kwargs["max_unique_assets"] = None
                else:
                    # Legacy default: fixed scale from the USD override registry.
                    usd_kwargs["scale"] = spec["usd_scale"]
                block = rigid_objects.create_toy_block(
                    prim_path=f"/World/envs/env_.*/{name}",
                    size=block_sizes_list[name],
                    init_pos=(i + 1, 0.0, 0.05 / 2),
                    init_quat=(1, 0, 0, 0),
                    static_friction=sf_list,
                    dynamic_friction=df_list,
                    restitution=rest_list,
                    **usd_kwargs,
                )
            self.scene.rigid_objects[name] = block
        for k in self.randomizations["friction"].keys():
            self.randomizations["friction"][k] = torch.tensor(
                self.randomizations["friction"][k], device=self.device
            )
        self.randomizations["table_height"] = torch.tensor(
            self.randomizations["table_height"], device=self.device
        )
        return

    def _after_scene_setup(self):
        # before the first environment is reset, we call this function once
        # useful for initializing anything that may need access to instantiated objects e.g. kinematics
        return

    _GOAL_TEXTS: list = [
        "(and\n        (on block_a block_i)\n        (on block_i block_2)\n        (ori_of block_a o2)\n        (ori_of block_i o2)\n        (ori_of block_2 o2)\n    )"
    ]

    @staticmethod
    def _normalize_goal_text(text: str) -> str:
        """Whitespace/case-insensitive form used to match a queue goal to a compiled one."""
        return " ".join(text.split()).lower()

    def _init_goal_state(self):
        self._per_env_goal_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def set_episode_goal(self, env_idx: int, pddl_goal: str) -> None:
        normalized = self._normalize_goal_text(pddl_goal)
        for i, gt in enumerate(self._GOAL_TEXTS):
            if normalized == self._normalize_goal_text(gt):
                self._per_env_goal_idx[env_idx] = i
                return
        raise ValueError(
            "set_episode_goal env=%d: goal %r is not one of the goals this task "
            "was compiled for (%s). Falling back to another goal would grade the "
            "episode against the wrong target and fabricate the result; "
            "regenerate the task from the same episode set the queue serves."
            % (env_idx, pddl_goal, self._GOAL_TEXTS)
        )

    def _compute_intermediate_values(self):
        # Auto-generated PDDL goal success: ai2_medium_03.pddl
        # 1 goal(s); per-env selection via _per_env_goal_idx
        from blocksuite.tamp.blocks.pddl_goal_tensor_conversions import (
            batched_goal_conjunct,
            scene_settled_and_gripper_open,
        )

        if not hasattr(self, "_per_env_goal_idx"):
            self._init_goal_state()

        # --- goal 0: '(and\n        (on block_a block_i)\n        (on block_i block_2)\n        (ori_of block_a o2)\n        (ori_of block_i o2)\n        (ori_of block_2 o2)\n    )'
        self._g0_conj_0 = batched_goal_conjunct(self, "on", ("block_a", "block_i"))
        self._g0_conj_1 = batched_goal_conjunct(self, "on", ("block_i", "block_2"))
        self._g0_conj_2 = batched_goal_conjunct(self, "ori_of", ("block_a", "o2"))
        self._g0_conj_3 = batched_goal_conjunct(self, "ori_of", ("block_i", "o2"))
        self._g0_conj_4 = batched_goal_conjunct(self, "ori_of", ("block_2", "o2"))
        self._goal_0_success = (
            self._g0_conj_0 & self._g0_conj_1 & self._g0_conj_2 & self._g0_conj_3 & self._g0_conj_4
        ) & scene_settled_and_gripper_open(self, ["block_2", "block_a", "block_i"])

        _all_goals = torch.stack([self._goal_0_success], dim=-1)
        self._goal_success = _all_goals.gather(1, self._per_env_goal_idx.unsqueeze(1)).squeeze(1)

    def _get_obs_robot(self) -> dict:
        return super()._get_obs_robot()

    def _get_obs_extra(self) -> dict:
        # By default observations only include the robot's information returned by _get_obs_robot
        # to add more information like ground truth object poses return them here as a dictionary
        # Whatever is returned here will also be used to automatically create the observation space
        observations = {"random_pose": torch.rand(self.num_envs, 7, device=self.device)}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        return self._goal_success.float()

    def evaluate(self) -> dict:
        # return optionally additional metrics here
        # include the keys "success" and/or "fail" to indicate if the environment state is in a success or failed state
        # and will update some recorded metrics
        # (success from PDDL goal is injected by scripts/generate_task_from_pddl.py)
        result = super().evaluate()
        result["success"] = self._goal_success
        stats = result.setdefault("stats", {})
        stats["g0_conj_0"] = self._g0_conj_0.float()
        stats["g0_conj_1"] = self._g0_conj_1.float()
        stats["g0_conj_2"] = self._g0_conj_2.float()
        stats["g0_conj_3"] = self._g0_conj_3.float()
        stats["g0_conj_4"] = self._g0_conj_4.float()
        stats["goal_0_success"] = self._goal_0_success.float()
        return result

    def _reset_idx(self, env_ids: Sequence[int] | None, instance_ids: Sequence[int] | None = None):
        super()._reset_idx(env_ids)
        if self.queue_client is not None:
            from blocksuite.utils.episode_queue import apply_setup

            self._write_instance_layout_json_block_poses(env_ids)
            setups = self.queue_client.fetch_reset_states(env_ids)
            for eid_t, setup in zip(env_ids, setups):
                if setup is not None:
                    eid = int(eid_t) if isinstance(eid_t, torch.Tensor) else eid_t
                    apply_setup(self, eid, setup)
            return
        if self.reset_states is not None:
            if instance_ids is not None:
                self.set_state_dict(
                    tree_utils.slice(self.reset_states, instance_ids),
                    env_ids=env_ids,
                    is_relative=True,
                )
            else:
                self.set_state_dict(
                    tree_utils.slice(
                        self.reset_states,
                        torch.randint(
                            0, self.num_reset_states, (len(env_ids),), device=self.device
                        ),
                    ),
                    env_ids=env_ids,
                    is_relative=True,
                )
        else:
            pose = torch.tensor(
                [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, 0.0], device=self.device
            )
            ids = [0, 1, 2, 3, 4, 5, 6]
            self.robot.articulation.write_joint_position_to_sim(
                pose, joint_ids=ids, env_ids=env_ids
            )
            self.robot.articulation.set_joint_position_target(pose, joint_ids=ids, env_ids=env_ids)
        # randomize table friction
        table = self.scene.rigid_objects["workspace_table"]
        set_rigid_body_friction(
            asset=table,
            static_friction=self.randomizations["friction"]["static_friction"][
                env_ids % self.MAX_NON_SIM_STATE_RANDOMIZATIONS
            ],
            dynamic_friction=self.randomizations["friction"]["dynamic_friction"][
                env_ids % self.MAX_NON_SIM_STATE_RANDOMIZATIONS
            ],
            restitution=self.randomizations["friction"]["restitution"][
                env_ids % self.MAX_NON_SIM_STATE_RANDOMIZATIONS
            ],
            env_ids=env_ids,
        )
        # randomize table height (delta relative to scene spawn pose)
        # Only apply when reset_states are NOT used: during generation, the table
        # shift is applied here, and the resulting state (table + blocks) is saved
        # into reset_states.pkl. When reset_states ARE loaded, set_state_dict
        # already restores the table and blocks in correct relative positions —
        # overriding the table z again would double-apply the offset.
        if self.reset_states is None:
            pose = table.data.root_pose_w[env_ids].clone()
            if not hasattr(self, "_workspace_table_spawn_root_z"):
                self._workspace_table_spawn_root_z = table.data.root_pose_w[:, 2].clone()
            env_idx_t = (
                env_ids
                if isinstance(env_ids, torch.Tensor)
                else torch.tensor(env_ids, device=self.device, dtype=torch.long)
            )
            slot = env_idx_t % self.MAX_NON_SIM_STATE_RANDOMIZATIONS
            dz = self.randomizations["table_height"][slot].to(device=pose.device, dtype=pose.dtype)
            pose[:, 2] = (
                self._workspace_table_spawn_root_z[env_idx_t].to(
                    device=pose.device,
                    dtype=pose.dtype,
                )
                + dz
            )
            table.write_root_pose_to_sim(pose, env_ids=env_ids)
        self._write_instance_layout_json_block_poses(env_ids)
