"""Define a class providing an MCP server to call ManiFlow-based skills as MCP tools."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from math import nan
from typing import Any, TypeVar

import numpy as np
from blocksuite.eval.pd_mode_controller import PDModeController
from blocksuite.serving import PolicyClientAdapter, WebsocketClientPolicy
from blocksuite.serving.action_chunk_broker import ActionChunkBroker
from blocksuite.serving.maniflow_serving_utils import _build_heatmap, format_task_name
from blocksuite.serving.obs_transforms import ObsTransformPolicy
from blocksuite.serving.obs_transforms.maniflow import transform_kwargs_from_server_metadata
from blocksuite.serving.obs_transforms.maniflow_blocks_table import ManiflowBlocksTableObsTransform
from mcp.server.mcpserver import Image as MCPImage

from agentic_robotics.adapters.toyblocks.leap.constants import (
    FLIP_STEPS,
    PICK_STEPS,
    PLACE_STEPS,
    ROTATE_STEPS,
    STACK_STEPS,
)
from agentic_robotics.adapters.toyblocks.leap.leap_agents import PerceptionToolsAgent
from agentic_robotics.adapters.toyblocks.leap.maniflow_skills.maniflow_motor_skills_agent import (
    MotorSkillsAgent,
)
from agentic_robotics.adapters.toyblocks.toyblocks_mcp_server import (
    FlipDirection,
    RotateDirection,
    ToyBlocksMCPServer,
)
from agentic_robotics.core.images import PixelCoord

logger = logging.getLogger(__name__)

ActionT = TypeVar("ActionT")

StopRule = Callable[[ActionT], bool]


@dataclass(frozen=True)
class ManiFlowConditioning:
    """Conditioning variables given alongside each call to the ManiFlow policy server."""

    prompt: str
    pixel_arr: np.ndarray
    """Pixel in the camera image. Note that [NaN, NaN] is used to denote an all-zero heatmap."""


GRIPPER_CONDITIONING = ManiFlowConditioning(prompt="task", pixel_arr=np.array([nan, nan]))


def _stop_rule(action: np.ndarray) -> bool:
    """Determine whether a policy rollout should be terminated early.

    Assumes that when the action's 9th slot is -1, the action should terminate.
    """
    mode = action[..., 8]
    return bool((mode <= -1).all())


class ManiFlowMCPServer(ToyBlocksMCPServer, MotorSkillsAgent, PerceptionToolsAgent):
    """An MCP server providing MCP tools to call ManiFlow-based skills."""

    def _build_skill_backend(self) -> None:
        """Initialize the ManiFlow backend for the ToyBlocks MCP server."""
        self._conditioning = GRIPPER_CONDITIONING

        device = self._toyblocks_env.base_env.device
        mf_config = self.config.maniflow
        mf_config.validate()

        # Construct an adapter for the remote policy server
        # Reference: `_make_policy_client(...)` from `toyblocks/utils/dataset_recording_utils.py`
        ws = WebsocketClientPolicy(host=mf_config.policy_host, port=mf_config.policy_port)

        language_mode = ws.metadata.get("language_mode")
        if language_mode is None:
            raise RuntimeError(
                f"Policy server metadata doesn't define a language mode: {ws.metadata}."
            )
        self._language_mode = str(language_mode)

        self._obs_transform = ManiflowBlocksTableObsTransform(
            **transform_kwargs_from_server_metadata(ws.metadata)
        )
        policy = ActionChunkBroker(
            ObsTransformPolicy(ws, self._obs_transform), action_horizon=self.config.action_horizon
        )

        self._policy_adapter = PolicyClientAdapter(
            policy, action_key="actions", device=device, batched=True
        )

        self._pd_controller = PDModeController(
            self._toyblocks_env.base_env,
            device,
            num_envs=1,
            stiff_gains=self.config.maniflow.stiff_gains.as_dict(),
            compliant_gains=self.config.maniflow.compliant_gains.as_dict(),
        )
        self._pd_controller.initialize_after_reset()

    def _register_skills(self) -> None:
        """Register ManiFlow-specific skills as MCP tools provided by the server."""
        self._register_tool(title="Pick Up Block", skill=self.pick)
        self._register_tool(title="Stack on Block", skill=self.stack)
        self._register_tool(title="Place Block on Table", skill=self.place_on_table)
        self._register_tool(title="Rotate Block", skill=self.rotate)
        self._register_tool(title="Flip Block", skill=self.flip)

    async def pick(self, pixel: PixelCoord) -> MCPImage:
        """Pick up the block at the given pixel in the current image."""
        self._raise_if_shutdown(skill_name="Pick")

        # Agent-predicted pixels are relative to the observation resolution, so we must
        #   convert them into the camera's resolution before conditioning ManiFlow
        camera_pixel = self.convert_obs_pixel_to_camera(pixel)

        self._conditioning = ManiFlowConditioning(prompt="pick", pixel_arr=camera_pixel.as_array())
        self.rollout(num_steps=PICK_STEPS)

        return self.current_image_obs

    async def stack(self, pixel: PixelCoord) -> MCPImage:
        """Stack the held block onto the block at the given pixel in the current image."""
        self._raise_if_shutdown(skill_name="Stack")

        # Agent-predicted pixels are relative to the observation resolution, so we must
        #   convert them into the camera's resolution before conditioning ManiFlow
        camera_pixel = self.convert_obs_pixel_to_camera(pixel)

        self._conditioning = ManiFlowConditioning(prompt="place", pixel_arr=camera_pixel.as_array())
        self.rollout(num_steps=STACK_STEPS)

        return self.current_image_obs

    async def place_on_table(self, pixel: PixelCoord) -> MCPImage:
        """Place the held block on the table, at the given pixel in the current image."""
        self._raise_if_shutdown(skill_name="Place on Table")

        # Agent-predicted pixels are relative to the observation resolution, so we must
        #   convert them into the camera's resolution before conditioning ManiFlow
        camera_pixel = self.convert_obs_pixel_to_camera(pixel)

        self._conditioning = ManiFlowConditioning(
            prompt="place on table", pixel_arr=camera_pixel.as_array()
        )
        self.rollout(num_steps=PLACE_STEPS)

        return self.current_image_obs

    async def rotate(self, pixel: PixelCoord, direction: RotateDirection) -> MCPImage:
        """Rotate the block at the given pixel in the current image.

        :param pixel: Raw pixel coordinate in the current image observation
        :param direction: Direction to rotate the block, as viewed from above ("cw" or "ccw")
        :return: Updated image observation after the skill has finished
        """
        self._raise_if_shutdown(skill_name="Rotate Block")

        # Agent-predicted pixels are relative to the observation resolution, so we must
        #   convert them into the camera's resolution before conditioning ManiFlow
        camera_pixel = self.convert_obs_pixel_to_camera(pixel)

        self._conditioning = ManiFlowConditioning(
            prompt=f"rotate {direction}", pixel_arr=camera_pixel.as_array()
        )
        self.rollout(num_steps=ROTATE_STEPS)

        return self.current_image_obs

    async def flip(self, pixel: PixelCoord, direction: FlipDirection) -> MCPImage:
        """Flip the block at the given pixel in the current image.

        :param pixel: Raw pixel coordinate in the current image observation
        :param direction: Specifies where the top face of the block ends up after the flip
        :return: Updated image observation after the skill has finished
        """
        self._raise_if_shutdown(skill_name="Flip Block")

        # Agent-predicted pixels are relative to the observation resolution, so we must
        #   convert them into the camera's resolution before conditioning ManiFlow
        camera_pixel = self.convert_obs_pixel_to_camera(pixel)

        # The tool's directions are the front camera's: `front` is where the top face ends up
        # when the block rolls toward the camera. The policy's labels are ToyBlocks's world-frame
        # ones ("flip forward" rolls the block along -x, "flip backward" along +x), and the front
        # camera sits on the +x side, so the front/back pair maps to the opposite label; left (-y)
        # and right (+y) coincide with the camera's. Measured on 109 flips, 2026-10-23.
        label = {"front": "backward", "back": "forward"}.get(direction, direction)
        logger.info(f"flip {direction!r} -> policy prompt 'flip {label}'")
        self._conditioning = ManiFlowConditioning(
            prompt=f"flip {label}", pixel_arr=camera_pixel.as_array()
        )
        self.rollout(num_steps=FLIP_STEPS)

        return self.current_image_obs

    def _build_conditioning(self) -> dict[str, Any]:
        """Construct a dictionary of values used to condition the ManiFlow policy."""
        target_xy = self._conditioning.pixel_arr
        width = float(self.camera_resolution.width)
        height = float(self.camera_resolution.height)
        target_sigma = self.config.maniflow.target_sigma

        # Expect the heatmap to have shape (3, 224, 224), then pad to account for batching
        heatmap: np.ndarray = _build_heatmap(target_xy, width, height, target_sigma)
        batch = heatmap[None, None, ...].astype(np.float32)

        optional_target_xy = None if np.isnan(target_xy[0]) else target_xy
        task_name = format_task_name(
            self._conditioning.prompt, optional_target_xy, self._language_mode
        )

        return {"oracle_task_name": [task_name], "target_heatmap": batch}

    def rollout(self, num_steps: int, stop_rule: StopRule | None = _stop_rule) -> None:
        """Rollout the current policy for the given number of steps."""
        self._policy_adapter.policy.reset()  # Refresh the policy so its prompt updates
        self._obs_transform.set_pd_mode(self._pd_controller.current_pd_mode())

        obs = self._toyblocks_env.current_obs
        conditioning = self._build_conditioning()

        latched = False
        for step in range(num_steps):
            action = self._policy_adapter.act({**obs, **conditioning})
            if stop_rule is not None and stop_rule(action):
                logger.info(f"Skill terminated early by its stop rule on step {step}.")
                break

            arm_gripper_action = action[..., :8]
            mode = action[..., 8]

            newly_latched = (not latched) and bool((mode <= 0).all())
            if newly_latched:
                latched = True  # Latch once mode output has been 0
                logger.info(f"Skill mode latched to 0 on step {step}.")

            self._obs_transform.set_pd_mode(mode.clamp(min=0, max=0 if latched else 1))
            obs, *_ = self._toyblocks_env.step(arm_gripper_action)

            self.record_current_frame()
        else:
            logger.info(f"Skill terminated after reaching maximum number of steps ({num_steps}).")
