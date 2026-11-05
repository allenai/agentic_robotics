"""Define an abstract base class for an agent with perception tools."""

from abc import ABC, abstractmethod
from typing import Annotated

from mcp.server.mcpserver import Image as MCPImage
from pydantic import Field

from agentic_robotics.core.images import PixelCoord


class PerceptionToolsAgent(ABC):
    """An agent with image-based perception tools."""

    @abstractmethod
    async def get_image_observation(self) -> MCPImage:
        """Retrieve the current image observation from the robot."""
        ...

    @abstractmethod
    async def get_zoomed_image(
        self, pixel: PixelCoord, zoom: Annotated[float, Field(ge=1.0, le=20.0)]
    ) -> MCPImage:
        """Retrieve a magnified view of a cropped area of the current image observation.

        A zoom factor of 4.0 produces a zoomed image covering a quarter of the original
        image's height and width, magnified to be four times its original size. Thus,
        the zoomed image has the same resolution as the original image observation.

        :param pixel: Raw (u,v) pixel coordinate of the center of the cropped area
        :param zoom: Zoom factor for the magnified view (must be between 1.0 and 20.0)
        :return: Zoomed crop of the current image observation
        """
        ...
