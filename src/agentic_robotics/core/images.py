"""Utility definitions and functions for representing and working with images."""

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from mcp.server.mcpserver import Image as MCPImage
from PIL import Image
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class Resolution:
    """Resolution of an image in pixels (H x W)."""

    height: int
    width: int

    @classmethod
    def get_resolution(cls, image: np.ndarray) -> "Resolution":
        """Construct a resolution based on the given image."""
        if len(image.shape) not in {2, 3}:
            raise ValueError(f"Cannot find the resolution of an array with shape: {image.shape}.")
        return Resolution(height=image.shape[0], width=image.shape[1])


class PixelCoord(BaseModel):
    """A raw (u,v) pixel coordinate within an image."""

    u: int = Field(description="Horizontal pixel coordinate index in the image.")
    v: int = Field(description="Vertical pixel coordinate index in the image.")

    def normalize(self, res: Resolution) -> "ImagePoint":
        """Convert the pixel coordinate into a normalized image point."""
        return ImagePoint(x=int(self.u / res.width * 1000), y=int(self.v / res.height * 1000))

    def as_array(self) -> np.ndarray:
        """Convert the pixel coordinate into a float32 NumPy array."""
        return np.array([self.u, self.v], dtype=np.float32)


class ImagePoint(BaseModel):
    """An (x,y) point in an image, normalized into the range [0,1000)."""

    x: int = Field(description="Horizontal coordinate, normalized into [0,1000).", ge=0, lt=1000)
    y: int = Field(description="Vertical coordinate, normalized into [0,1000).", ge=0, lt=1000)

    def to_pixel(self, res: Resolution) -> PixelCoord:
        """Convert the normalized image point into a raw pixel coordinate within an image."""
        return PixelCoord(u=int(self.x / 1000 * res.width), v=int(self.y / 1000 * res.height))


def array_to_mcp_image(array: np.ndarray, image_format: str = "JPEG") -> MCPImage:
    """Convert a NumPy array containing image data into an MCP-compatible image."""
    assert image_format in {"JPEG", "PNG"}, f"Invalid image format: '{image_format}'."

    pil_image = Image.fromarray(array)
    byte_stream = BytesIO()
    pil_image.save(byte_stream, format=image_format)

    return MCPImage(data=byte_stream.getvalue(), format=image_format)


def crop_around(image: np.ndarray, center: PixelCoord, crop_factor: float) -> np.ndarray:
    """Crop a window of an image centered around a pixel, as a fraction of the image's size.

    A window that would run off the edge of the image is shifted inside, rather than clipped.

    :param image: NumPy array containing image data, shape (H, W, C)
    :param center: Pixel the crop is centered on
    :param crop_factor: Fraction of the image's width and height to keep, in the range (0, 1]
    :return: Image data of the cropped window
    """
    assert 0 < crop_factor <= 1.0, f"Crop factor must be in range (0, 1]; was {crop_factor}."

    height, width = image.shape[:2]
    window_w = np.clip(round(width * crop_factor), a_min=1, a_max=width)
    window_h = np.clip(round(height * crop_factor), a_min=1, a_max=height)

    left_c = np.clip(center.u - window_w // 2, a_min=0, a_max=width - window_w)
    top_r = np.clip(center.v - window_h // 2, a_min=0, a_max=height - window_h)

    return image[top_r : top_r + window_h, left_c : left_c + window_w]


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    """Resize an image to the specified width, while preserving its aspect ratio.

    :param image: NumPy array containing image data, shape (H, W, C)
    :param width: Target width (in pixels) of the resulting image
    :return: Resized image with the requested width
    """
    original_h, original_w = image.shape[:2]
    if original_w == width:
        return image

    scale_by = width / original_w
    height = max(1, round(original_h * scale_by))

    return np.asarray(Image.fromarray(image).resize((width, height), Image.LANCZOS))
