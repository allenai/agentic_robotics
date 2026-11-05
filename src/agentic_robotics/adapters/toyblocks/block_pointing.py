"""Define ToyBlocks-specific utility code for pointing to blocks within images."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from agentic_robotics.adapters.toyblocks.config import OracleConfig
    from agentic_robotics.adapters.toyblocks.toyblocks_env import ToyBlocksEnv
    from agentic_robotics.core.cameras import CameraModel
    from agentic_robotics.core.images import PixelCoord


logger = logging.getLogger(__name__)


def get_block_pixels(
    env: ToyBlocksEnv, camera_model: CameraModel, env_idx: int = 0
) -> dict[str, np.ndarray]:
    """Compute the pixel each block's center projects to, ignoring blocks behind the camera.

    :param env: Interface to access the ToyBlocks environment
    :param camera_model: Pinhole model of the camera used for the projection
    :param env_idx: Index of the environment to read (defaults to 0)
    :return: Map from each block name to its projected pixel, shape (2,)
    """
    block_names = env.block_names

    positions_w = np.array([env.get_block_center(name, env_idx) for name in block_names])  # (n, 3)
    pixels = camera_model.points_to_pixels(positions_w)  # (n, 2)

    return {
        name: pixel
        for name, pixel in zip(block_names, pixels, strict=True)
        if np.isfinite(pixel).all()  # Omit blocks whose pixels have NaN values
    }


def resolve_block(
    env: ToyBlocksEnv,
    target_pixel: PixelCoord,
    *,
    camera_name: str,
    config: OracleConfig,
    exclude: frozenset[str] = frozenset(),
    env_idx: int = 0,
) -> str | None:
    """Identify the block whose center is nearest to the given pixel.

    :param env: Interface to access the ToyBlocks environment
    :param target_pixel: Pixel coordinate the agent pointed at
    :param camera_name: Name of the camera used for the projection
    :param config: Configuration parameters for the oracle skills
    :param exclude: Set of block names to exclude from consideration (defaults to empty)
    :param env_idx: Index of the environment to read (defaults to 0)
    :return: Name of the matched block, or None if no block was in front of the camera
    """
    camera_model = env.read_camera_model(camera_name=camera_name, env_idx=env_idx)
    block_pixels = get_block_pixels(env, camera_model, env_idx=env_idx)
    filtered_block_pixels = {name: px for name, px in block_pixels.items() if name not in exclude}

    target_pixel_arr = np.array([target_pixel.u, target_pixel.v], dtype=np.float64)

    def pixels_from_target(block_name: str) -> float:
        """Find the pixel distance between the named block's center and the target pixel."""
        center_pixel = block_pixels[block_name]
        return np.linalg.norm(target_pixel_arr - center_pixel)

    closest_name = min(filtered_block_pixels, key=pixels_from_target, default=None)
    if closest_name is None:
        return None

    closest_center = env.get_block_center(closest_name, env_idx=env_idx)
    closest_radius_m = np.max(env.get_block_size_xyz_m(closest_name, env_idx=env_idx)) / 2.0
    closest_radius_px = camera_model.projected_ray_length_px(closest_center, closest_radius_m)

    if closest_radius_px is None:
        logger.warning(f"Unable to find the projected radius of block '{closest_name}'.")
        return closest_name

    error_tolerance_px = closest_radius_px * config.pointing_error_ratio
    if pixels_from_target(closest_name) > error_tolerance_px:
        return None

    return closest_name


def find_table_z(env: ToyBlocksEnv, env_idx: int = 0) -> float:
    """Find the table-top height based on the resting blocks' positions and sizes.

    :param env: Interface to access the ToyBlocks environment
    :param env_idx: Index of the environment to read (defaults to 0)
    :return: World-frame height (meters) of the table-top surface
    """
    block_names = env.block_names
    if not block_names:
        raise RuntimeError("Cannot derive table height because there are no blocks in the scene.")

    lowest_block_name = min(
        block_names, key=lambda name: env.get_block_center(name, env_idx=env_idx)[2]
    )
    lowest_center_z = env.get_block_center(lowest_block_name, env_idx=env_idx)[2]
    lowest_block_size_m = np.max(env.get_block_size_xyz_m(lowest_block_name, env_idx=env_idx))

    return lowest_center_z - (lowest_block_size_m / 2.0)
