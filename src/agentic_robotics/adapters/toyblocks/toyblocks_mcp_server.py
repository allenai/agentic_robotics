"""Define an abstract base class for an MCP server exposing skills for a ToyBlocks environment."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from inspect import cleandoc, ismethod, signature
from typing import TYPE_CHECKING, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver import Image as MCPImage  # noqa: TC002 (used for MCP tool registration)

from agentic_robotics.adapters.toyblocks.config import ToyBlocksConfig
from agentic_robotics.adapters.toyblocks.errors import InvalidPixelError
from agentic_robotics.core.filesystem import IncrementalVideoWriter, create_unique_dir
from agentic_robotics.core.images import PixelCoord, Resolution, array_to_mcp_image, resize_to_width

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from agentic_robotics.adapters.toyblocks.toyblocks_env import ToyBlocksEnv

logger = logging.getLogger(__name__)

RotateDirection = Literal["cw", "ccw"]
"""Yaw-rotation direction (either clockwise or counterclockwise)."""

FlipDirection = Literal["front", "back", "left", "right"]
"""Where does the top face of the block end up after the flip?"""


def _model_facing_signature(fn: Callable) -> tuple[list[str], str]:
    """Render a callable's signature as the MCP schema sees it.

    :param fn: A plain function, bound method, or a `functools.wraps` wrapper around either
    :return: Tuple of (printed parameters, printed return annotation)
    """
    fn_signature = signature(fn, eval_str=True)
    parameters = list(fn_signature.parameters.values())
    if not ismethod(fn) and parameters and parameters[0].name == "self":
        parameters = parameters[1:]

    return [repr(p) for p in parameters], repr(fn_signature.return_annotation)


def assert_matches_declaration(
    implementation: Callable, declaration: Callable, *, owner: str = ""
) -> None:
    """Raise if `implementation` has a different signature than `declaration`.

    Parameter names, kinds, annotations (pydantic `Field` metadata included), defaults,
    and return annotations are compared in their printed form.

    :raises TypeError: If the two signatures differ in any way
    """
    actual_params, actual_return = _model_facing_signature(implementation)
    declared_params, declared_return = _model_facing_signature(declaration)
    if (actual_params, actual_return) == (declared_params, declared_return):
        return

    where = f"{owner}." if owner else ""
    declared_by = getattr(declaration, "__qualname__", declaration.__name__)
    raise TypeError(
        f"{where}{implementation.__name__} does not match its declaration {declared_by}:\n"
        f"  declared: ({', '.join(declared_params)}) -> {declared_return}\n"
        f"  actual:   ({', '.join(actual_params)}) -> {actual_return}"
    )


class ToyBlocksMCPServer(ABC):
    """An MCP server providing MCP tools to call skills for a ToyBlocks environment."""

    def __init__(self, config: ToyBlocksConfig | None = None) -> None:
        """Initialize the MCP server with a ToyBlocks environment and skill backend."""
        if config is None:
            _config = ToyBlocksConfig()
            output_dir = create_unique_dir(parent_dir=_config.output_root_dir)
            config = ToyBlocksConfig(output_dir=output_dir)

        self.config = config
        self._toyblocks_env = self._build_env(config)

        self._video_writer = (
            IncrementalVideoWriter(config.output_dir / "video.mp4") if config.output_dir else None
        )
        self.last_frame: np.ndarray | None = None
        self.frame_count = 0

        self._build_skill_backend()

        self.record_current_frame()

        assert self.last_frame is not None
        self.camera_resolution = Resolution.get_resolution(self.last_frame)
        self.observation_resolution = self.config.observation_resolution

        # Verify that the camera and observations use the same aspect ratio
        camera_aspect = self.camera_resolution.width / self.camera_resolution.height
        obs_aspect = self.observation_resolution.width / self.observation_resolution.height
        if abs(camera_aspect - obs_aspect) > 1e-6:
            raise RuntimeError(
                "The camera resolution must have the same aspect ratio as agent observations.\n"
                f"Camera resolution: {self.camera_resolution}\n"
                f"Agent observation resolution: {self.observation_resolution}."
            )

        logger.info(f"Camera resolution: {self.camera_resolution}")
        logger.info(f"Resolution of agent observations: {self.observation_resolution}")

        self.has_shutdown = False

        self.mcp = MCPServer("ToyBlocks")
        self._register_tool(title="Get Image Observation", skill=self.get_image_observation)
        self._register_skills()

    @abstractmethod
    def _build_skill_backend(self) -> None:
        """Initialize the ToyBlocks skill backend for the MCP server."""

    @abstractmethod
    def _register_skills(self) -> None:
        """Register backend-specific skills as MCP tools provided by the server."""

    @property
    def current_image_obs(self) -> MCPImage:
        """Retrieve the current image observation as an MCP image."""
        frame = self._current_frame(self.config.camera_name)
        obs_arr = resize_to_width(image=frame, width=self.observation_resolution.width)
        return array_to_mcp_image(obs_arr)

    def _build_env(self, config: ToyBlocksConfig) -> ToyBlocksEnv:
        """Build the ToyBlocks environment interface used by the server."""
        from agentic_robotics.adapters.toyblocks.toyblocks_env import ToyBlocksEnv

        return ToyBlocksEnv(config.task_config, viewer=config.viewer)

    def _register_tool(self, title: str, skill: Callable) -> None:
        """Register a skill as an MCP tool, as described by the class that declares it.

        Rather than allowing MCP to fall back to `skill.__doc__`, which is defined
        based on the docstring where the skill is implemented, we resolve the
        description based on the furthest ancestor class. This deliberately
        prevents an overriding implementation from replacing the docstring.
        """
        declaration = self._declaring_member(skill)
        if declaration is not None:
            assert_matches_declaration(
                implementation=skill, declaration=declaration, owner=type(self).__name__
            )

        description = self._tool_description(skill)
        self.mcp.tool(title=title, description=description)(skill)

    def _declaring_member(self, tool: Callable) -> Callable | None:
        """Retrieve the furthest ancestor's definition of `tool`."""
        for klass in reversed(type(self).__mro__):
            member = klass.__dict__.get(tool.__name__)
            if member is not None:
                return member

        return None

    def _tool_description(self, skill: Callable) -> str | None:
        """Retrieve the model-facing description for a skill: the docstring from its declaration.

        :param skill: Skill method being registered (possibly wrapped by MCP server subclasses)
        :return: Docstring of the skill, according to the class that originally declared it
        """
        for klass in reversed(type(self).__mro__):
            member = klass.__dict__.get(skill.__name__)
            if member is not None and member.__doc__:
                return cleandoc(member.__doc__)

        return None

    def _raise_if_shutdown(self, skill_name: str) -> None:
        """Ensure that the MCP server has not been shut down before running a skill."""
        if self.has_shutdown:
            raise RuntimeError(f"Cannot run {skill_name}; the MCP server has been shut down.")

        logger.info(f"Running '{skill_name}' skill...")

    def _current_frame(self, camera_name: str) -> np.ndarray:
        """Retrieve the current image from the named camera."""
        return self._toyblocks_env.get_current_image(camera_name=camera_name)

    async def get_image_observation(self) -> MCPImage:
        """Retrieve the current image observation from the robot."""
        self._raise_if_shutdown(skill_name="GetCurrentImage")
        return self.current_image_obs

    def record_current_frame(self, camera_name: str | None = None) -> None:
        """Record the current image from the named camera."""
        if camera_name is None:
            camera_name = self.config.camera_name

        frame = self._current_frame(camera_name)
        self.last_frame = frame
        self.frame_count += 1
        if self._video_writer is not None:
            self._video_writer.append(frame)

    def convert_obs_pixel_to_camera(self, pixel: PixelCoord) -> PixelCoord:
        """Convert a pixel based on the observation's resolution to use the camera's resolution.

        :param pixel: Pixel specified relative to an image observation's resolution
        :return: Equivalent pixel coordinate, based instead on the camera's resolution
        """
        if not (
            (0 <= pixel.u < self.observation_resolution.width)
            and (0 <= pixel.v < self.observation_resolution.height)
        ):
            raise InvalidPixelError(obs_resolution=self.observation_resolution, pixel=pixel)

        new_u = round(pixel.u / self.observation_resolution.width * self.camera_resolution.width)
        new_v = round(pixel.v / self.observation_resolution.height * self.camera_resolution.height)
        return PixelCoord(u=new_u, v=new_v)

    def shutdown(self) -> None:
        """Shut down the MCP server and end the episode."""
        if self.has_shutdown:
            return

        self.shutdown_video()

        self._toyblocks_env.env.close()
        self.has_shutdown = True

    def shutdown_video(self) -> None:
        """Shut down the video writer of the MCP server."""
        if self._video_writer is None:
            logger.warning("Output directory of ToyBlocksConfig was None; no video was recorded.")
        else:
            self._video_writer.close()
