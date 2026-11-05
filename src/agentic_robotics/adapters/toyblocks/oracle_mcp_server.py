"""Define an abstract interface for an MCP server providing oracle skills as MCP tools."""

from abc import ABC, abstractmethod
from typing import Annotated

from mcp.server.mcpserver import Image as MCPImage
from pydantic import Field

from agentic_robotics.adapters.toyblocks.toyblocks_mcp_server import (
    FlipDirection,
    RotateDirection,
    ToyBlocksMCPServer,
)
from agentic_robotics.core.images import PixelCoord


class OracleMCPServer(ToyBlocksMCPServer, ABC):
    """The interface for an MCP server providing MCP tools to call TAMP and RL oracle skills."""

    def _register_skills(self) -> None:
        """Register oracle-specific skills as MCP tools provided by the server."""
        self._register_tool(title="Get Zoomed-In Image Observation", skill=self.get_zoomed_image)
        self._register_tool(title="Pick Up Block", skill=self.pick)
        self._register_tool(title="Stack on Block", skill=self.stack)
        self._register_tool(title="Place on Table", skill=self.place_on_table)
        self._register_tool(title="Rotate Block", skill=self.rotate)
        self._register_tool(title="Flip Block", skill=self.flip)
        self._register_tool(title="Return Arm to Home Configuration", skill=self.return_home)
        self._register_tool(title="Open Gripper", skill=self.open_gripper)
        self._register_tool(title="Close Gripper", skill=self.close_gripper)

    @abstractmethod
    async def get_zoomed_image(
        self, pixel: PixelCoord, zoom: Annotated[float, Field(ge=1.0, le=20.0)]
    ) -> MCPImage:
        """Retrieve a magnified view of a cropped area of the current image observation.

        A zoom factor of 4 produces a zoomed image covering a quarter of the original
        image's height and width, magnified to be four times its original size. Thus,
        the zoomed image has the same resolution as the original image observation.

        :param pixel: Raw pixel coordinate defining the center of the cropped region
        :param zoom: Zoom factor for the magnified view (must be between 1.0 and 20.0)
        :return: Zoomed crop of the current image observation
        """

    @abstractmethod
    async def pick(self, pixel: PixelCoord) -> MCPImage:
        """Pick up the block at the given pixel in the current image.

        :param pixel: Raw pixel coordinate in the current image observation
        :return: Updated image observation after the skill has finished
        """

    @abstractmethod
    async def stack(self, pixel: PixelCoord) -> MCPImage:
        """Stack the held block onto the block at the given pixel in the current image.

        Precondition: The robot's gripper must be holding a block.

        :param pixel: Raw pixel coordinate in the current image observation
        :return: Updated image observation after the skill has finished
        """

    @abstractmethod
    async def place_on_table(self, pixel: PixelCoord) -> MCPImage:
        """Place the held block on the table, at the given pixel in the current image.

        Precondition: The robot's gripper must be holding a block.

        :param pixel: Raw pixel coordinate in the current image observation
        :return: Updated image observation after the skill has finished
        """

    @abstractmethod
    async def rotate(self, pixel: PixelCoord, direction: RotateDirection) -> MCPImage:
        """Rotate the block at the given pixel in the current image.

        Preconditions:
            - The robot's gripper must be empty.
            - The block to be rotated must be laying directly on the table's surface.

        :param pixel: Raw pixel coordinate in the current image observation
        :param direction: Direction to rotate the block, as viewed from above ("cw" or "ccw")
        :return: Updated image observation after the skill has finished
        """

    @abstractmethod
    async def flip(self, pixel: PixelCoord, direction: FlipDirection) -> MCPImage:
        """Flip the block at the given pixel in the current image.

        Preconditions:
            - The robot's gripper must be empty.
            - The block to be flipped must be laying directly on the table's surface.

        :param pixel: Raw pixel coordinate in the current image observation
        :param direction: Specifies where the top face of the block ends up after the flip
        :return: Updated image observation after the skill has finished
        """

    @abstractmethod
    async def return_home(self) -> MCPImage:
        """Open the gripper, then move the arm back to a neutral resting configuration.

        Use this when another skill reports that the arm cannot plan from its current
        configuration. All motion-planning-based skills will continue to fail until
        the arm is moved. To get the arm unstuck, this skill drives the joints directly.

        :return: Updated image observation after the skill has finished
        """

    @abstractmethod
    async def open_gripper(self) -> MCPImage:
        """Open the robot gripper, releasing anything it holds.

        :return: Updated image observation after the skill has finished
        """

    @abstractmethod
    async def close_gripper(self) -> MCPImage:
        """Close the robot gripper, grasping anything between its fingers.

        :return: Updated image observation after the skill has finished
        """
