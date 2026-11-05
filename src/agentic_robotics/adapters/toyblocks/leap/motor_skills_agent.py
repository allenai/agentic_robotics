"""Define an abstract base class for an agent with motor skills, with standard docstrings."""

from abc import ABC, abstractmethod

from mcp.server.mcpserver import Image as MCPImage

from agentic_robotics.adapters.toyblocks.toyblocks_mcp_server import (
    FlipDirection,
    RotateDirection,
)
from agentic_robotics.core.images import PixelCoord


class MotorSkillsAgent(ABC):
    """An agent with motor skills, with standard (i.e., control) docstrings."""

    @abstractmethod
    async def pick(self, pixel: PixelCoord) -> MCPImage:
        """Pick up the block at the given pixel in the current image.

        :param pixel: Raw (u,v) pixel coordinate of the block to pick
        :return: Updated image observation after the skill has finished
        """
        ...

    @abstractmethod
    async def stack(self, pixel: PixelCoord) -> MCPImage:
        """Stack the held block onto the block at the given pixel in the current image.

        :param pixel: Raw (u,v) pixel coordinate of the block to be stacked on
        :return: Updated image observation after the skill has finished
        """
        ...

    @abstractmethod
    async def place_on_table(self, pixel: PixelCoord) -> MCPImage:
        """Place the held block onto the table, at the given pixel in the current image.

        :param pixel: Raw (u,v) pixel coordinate indicating where to place the block
        :return: Updated image observation after the skill has finished
        """
        ...

    @abstractmethod
    async def rotate(self, pixel: PixelCoord, direction: RotateDirection) -> MCPImage:
        """Rotate the block at the given pixel in the current image.

        :param pixel: Raw (u,v) pixel coordinate of the block to rotate
        :param direction: Direction to rotate the block, as viewed from above ("cw" or "ccw")
        :return: Updated image observation after the skill has finished
        """
        ...

    @abstractmethod
    async def flip(self, pixel: PixelCoord, direction: FlipDirection) -> MCPImage:
        """Flip the block at the given pixel in the current image.

        :param pixel: Raw (u,v) pixel coordinate of the block to flip
        :param direction: Specifies where the block's top face ends up ("front", "back", "left", "right")
        :return: Updated image observation after the skill has finished
        """
        ...

    @abstractmethod
    async def return_home(self) -> MCPImage:
        """Move the arm back to a neutral resting configuration.

        The state of the robot's gripper is unchanged.

        :return: Updated image observation after the skill has finished
        """
        ...

    @abstractmethod
    async def open_gripper(self) -> MCPImage:
        """Open the robot's gripper."""
        ...

    @abstractmethod
    async def close_gripper(self) -> MCPImage:
        """Close the robot's gripper."""
        ...
