"""Define a class to configure, create, and manage the ToyBlocks simulated environment."""

# ========================================================================
# WARNING: Omniverse/Isaac Sim import statements must take place after the
# `SimulationApp` class has been instantiated. It is a requirement of the
# Carbonite framework's extension/runtime plugin system.
# ========================================================================
from isaaclab.app import AppLauncher

_SIMULATION_APP = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app

# ruff: noqa: E402 (because Isaac Sim needs to launch before the other imports)


import importlib
import logging
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from blocksuite.utils.obs_utils import get_bounding_box
from blocksuite.viewer.viewer import Viewer, ViewerCfg
from isaaclab.sensors import Camera as IsaacCamera
from isaaclab_tasks.utils import parse_env_cfg
from scipy.spatial.transform import Rotation

from agentic_robotics.adapters.toyblocks import runner_bootstrap
from agentic_robotics.adapters.toyblocks.config import TaskConfig
from agentic_robotics.core.cameras import CameraModel

logger = logging.getLogger(__name__)


def _to_numpy(value: object) -> np.ndarray:
    """Convert a possibly-tensor value into a float64 NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


class ToyBlocksEnv:
    """Handles the creation and management of the ToyBlocks simulation environment."""

    def __init__(
        self, task_config: TaskConfig, obs_mode: str = "state+rgb", *, viewer: bool = False
    ) -> None:
        """Initialize the class by creating a simulation app for ToyBlocks.

        :param task_config: Configuration specifying which task to load
        :param viewer: Whether to launch a viewer for the simulation environment
        """
        self.simulation_app = _SIMULATION_APP

        # Register the Gym environment
        importlib.import_module(runner_bootstrap.task_package_module(task_config.task_slug))

        self._env_cfg = parse_env_cfg(task_config.env_name)
        self._env_cfg.scene.num_envs = 1
        self._env_cfg.sim.device = "cuda:0"
        self._env_cfg.control_mode = "arm_joint_gripper_binary"
        self._env_cfg.obs_mode = obs_mode
        self._env_cfg.replay_gripper_close_target = 0.865
        self._env_cfg.include_extended_state_in_obs = True  # Includes joint_vel + ee_pos

        self.env = gym.make(task_config.env_name, cfg=self._env_cfg)

        obs, _ = self.env.reset()
        self._obs = obs

        logger.info(f"Gym environment '{task_config.env_name}' is up.")

        self._viewer: Viewer | None = None
        if viewer:
            viewer_cfg = ViewerCfg(env_ids=[0])
            if hasattr(viewer_cfg, "minimal_gui"):  # a field of some ToyBlocks versions only
                viewer_cfg.minimal_gui = True
            self._viewer = Viewer(cfg=viewer_cfg)
            self._viewer.hook_to_env(self.env)
            logger.debug("Viewer successfully hooked to the environment.")

    @property
    def base_env(self) -> gym.Env:
        """Retrieve the base Gym environment."""
        return self.env.unwrapped

    @property
    def current_obs(self) -> dict[str, Any]:
        """Retrieve the current observation of the environment."""
        return self._obs

    @property
    def block_names(self) -> list[str]:
        """Retrieve the names of the blocks in the environment."""
        return sorted(name for name in self.base_env.scene.rigid_objects if "block" in name.lower())

    def get_block_center(self, block_name: str, env_idx: int = 0) -> np.ndarray:
        """Retrieve the world-frame position of the center of the named block."""
        return _to_numpy(self.base_env.scene.rigid_objects[block_name].data.root_pos_w[env_idx])

    def get_block_size_xyz_m(self, block_name: str, env_idx: int = 0) -> np.ndarray:
        """Retrieve the (x, y, z) size of the named block (in meters), measured from its geometry.

        :param block_name: Name of the block to measure
        :param env_idx: Index of the environment to read (defaults to 0)
        :return: Size in meters (x, y, z) in the block's local frame; shape (3,)
        """
        bbox = get_bounding_box(self.base_env.scene.rigid_objects[block_name])  # (n, 2, 3)
        return _to_numpy(bbox[env_idx, 1] - bbox[env_idx, 0])  # (3,)

    def refresh_observation(self) -> None:
        """Re-read the current observation from the simulator."""
        if self.base_env.sim.has_rtx_sensors():
            self.base_env.sim.render()
        self._obs = self.base_env._get_observations()

    def get_current_image(self, camera_name: str, env_idx: int = 0) -> np.ndarray:
        """Retrieve the current image observation from the requested camera.

        :param camera_name: Name of the camera stream to retrieve
        :param env_idx: Index of the environment to read (defaults to 0)
        :return: NumPy array containing the requested image data
        """
        image = self._obs["images"][camera_name]["rgb"][env_idx]

        if hasattr(image, "detach"):
            image = image.detach().cpu().numpy()

        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        return image

    def _get_camera(self, camera_name: str) -> IsaacCamera:
        """Retrieve the requested camera sensor from Isaac Lab.

        The environment's `cameras` attribute is tried first, then the scene's sensors by bare
        name and with the `camera:` prefix, since which of the three holds a camera depends on
        how the environment registered it.
        """
        cameras = getattr(self.base_env, "cameras", None)
        if cameras and camera_name in cameras:
            return cameras[camera_name]

        sensors = self.base_env.scene.sensors
        for key in (camera_name, f"camera:{camera_name}"):
            if key in sensors:
                return sensors[key]

        raise KeyError(f"No camera '{camera_name}' in scene; its sensors are {sorted(sensors)}.")

    def read_camera_model(self, camera_name: str, env_idx: int = 0) -> CameraModel:
        """Read the live intrinsics and extrinsics of the requested camera.

        :param camera_name: Name of the camera to read
        :param env_idx: Index of the environment to read (defaults to 0)
        :return: Live pinhole model (intrinsics + extrinsics) of the camera
        """
        camera_data = self._get_camera(camera_name).data
        intrinsics = _to_numpy(camera_data.intrinsic_matrices[env_idx])
        quat_ros = _to_numpy(camera_data.quat_w_ros[env_idx])
        rotation = Rotation.from_quat(quat_ros, scalar_first=True)

        return CameraModel(
            fx=float(intrinsics[0, 0]),
            fy=float(intrinsics[1, 1]),
            x0=float(intrinsics[0, 2]),
            y0=float(intrinsics[1, 2]),
            position_w_c=_to_numpy(camera_data.pos_w[env_idx]),
            rotation_w_c=rotation.as_matrix(),
        )

    def _sync_simulator(self) -> None:
        """Push state updates to the simulator, settle its physics, and refresh its sensors.

        Reference: `set_initial_state()` from `toyblocks/eval/replay.py`
        """
        scene = self.base_env.scene
        sim = self.base_env.sim
        dt_s = self.base_env.physics_dt

        scene.write_data_to_sim()
        scene.update(dt_s)
        sim.step()
        scene.update(dt_s)

        for _ in range(10):
            sim.step()
            scene.update(dt_s)

        self.refresh_observation()

    def _refresh_viewer(self) -> None:
        """Refresh the viewer to reflect the current simulator state.

        Reference: `_after_sim_step(substep)` from `toyblocks/viewer/viewer.py`
        """
        if self._viewer is None:
            return

        self.base_env._update_visualization_markers()
        frame = self._viewer._add_trajectory_frame_from_env(
            self.base_env, env_ids=self._viewer._tracked_env_ids
        )
        frame.timestep += 1

        traj_id = self._viewer.viewed_trajectory_id
        frame_idx = len(self._viewer._frames[traj_id]) - 1
        self._viewer.view_frame(frame_idx, self._viewer.viewed_env_id, traj_id)

    def step(self, action: torch.Tensor) -> tuple[Any, ...]:
        """Step the environment, keeping the viewer and current observation up-to-date."""
        obs, *rest = self.env.step(action)
        self._obs = obs

        if self._viewer is not None:
            traj_id = self._viewer.viewed_trajectory_id
            frame_idx = len(self._viewer._frames[traj_id]) - 1
            self._viewer.view_frame(frame_idx, self._viewer.viewed_env_id, traj_id)

        return obs, *rest
