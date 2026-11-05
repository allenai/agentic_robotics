"""Define utilities to represent and interface with cameras."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from agentic_robotics.core.images import PixelCoord
from agentic_robotics.core.transforms import transform_from


@dataclass(frozen=True)
class CameraModel:
    """A model of a pinhole camera with intrinsic and extrinsic parameters.

    The camera's "principal axis" is the line perpendicular to the image plane that
    passes through the pinhole. The "principal point" refers to the point where
    the principal axis intersects with the image plane.

    Reference: https://ksimek.github.io/2013/08/13/intrinsic/
    """

    fx: float
    """Focal length of the camera in pixels along the horizontal (x) axis."""

    fy: float
    """Focal length of the camera in pixels along the vertical (y) axis."""

    x0: float
    """The x-coordinate of the principal point in pixels, relative to the sensor origin."""

    y0: float
    """The y-coordinate of the principal point in pixels, relative to the sensor origin."""

    position_w_c: np.ndarray
    """Camera position in the world frame; shape (3,)."""

    rotation_w_c: np.ndarray
    """Camera rotation matrix in the world frame; shape (3, 3)."""

    @property
    def transform_w_c(self) -> np.ndarray:
        """Retrieve the transformation matrix for the camera's pose w.r.t. the world frame."""
        return transform_from(position=self.position_w_c, rotation=self.rotation_w_c)

    @property
    def transform_c_w(self) -> np.ndarray:
        """Retrieve the camera-to-world transformation matrix."""
        rotation_c_w = self.rotation_w_c.T
        position_c_w = -rotation_c_w @ self.position_w_c
        return transform_from(position=position_c_w, rotation=rotation_c_w)

    def points_to_pixels(self, points_w: npt.ArrayLike) -> np.ndarray:
        """Project world-frame points to the camera's pixel space.

        :param points_w: Positions in the world frame; shape (n, 3)
        :return: Corresponding camera pixels, or NaN when behind the camera; shape (n, 2)
        """
        points_w = np.atleast_2d(np.asarray(points_w, dtype=np.float64))  # (n, 3)
        ones = np.ones((len(points_w), 1))  # (n, 1)
        homogeneous_points_w = np.hstack([points_w, ones]).T  # (n, 4) -> (4, n)

        homogeneous_points_c = self.transform_c_w @ homogeneous_points_w  # (4, n)
        points_c = homogeneous_points_c[:3, :].T  # (3, n) -> (n, 3)

        depth_m = points_c[:, 2]  # (n, 1)
        ahead = depth_m > 1e-6  # (n, 1)

        pixels = np.full(shape=(len(points_w), 2), fill_value=np.nan)  # (n, 2)

        # Reference: Equation 39.6 in "Foundations of Computer Vision" (Torralba et al., 2024)
        #   https://visionbook.mit.edu/imaging_geometry.html#from-meters-to-pixels
        pixels[ahead, 0] = self.fx * points_c[ahead, 0] / depth_m[ahead] + self.x0
        pixels[ahead, 1] = self.fy * points_c[ahead, 1] / depth_m[ahead] + self.y0
        return pixels

    def pixel_to_ray(self, pixel: PixelCoord) -> tuple[np.ndarray, np.ndarray]:
        """Return the world-frame (origin, unit direction) of the ray through a pixel."""
        direction_c = np.array([(pixel.u - self.x0) / self.fx, (pixel.v - self.y0) / self.fy, 1.0])
        direction_w = self.rotation_w_c @ direction_c
        return self.position_w_c, (direction_w / np.linalg.norm(direction_w))

    def projected_ray_length_px(self, ray_origin_w: np.ndarray, length_m: float) -> float | None:
        """Compute the camera-projected length (in pixels) of the specified ray.

        :param ray_origin_w: World-frame position of the ray's origin
        :param length_m: Real-world length of the ray (meters)
        :return: Projected length of the ray (in pixels), or None if it cannot be computed
        """
        x_axis_w_c = self.rotation_w_c[:, 0]  # Camera's unit x-axis w.r.t. world frame
        ray_endpoint_w = ray_origin_w + length_m * x_axis_w_c
        pixels = self.points_to_pixels(np.stack([ray_origin_w, ray_endpoint_w], axis=0))
        if not np.isfinite(pixels).all():
            return None

        return float(np.linalg.norm(pixels[1] - pixels[0]))

    def asdict(self) -> dict[str, Any]:
        """Convert the camera model into a JSON-serializable dictionary of data."""
        return {
            "intrinsics": {"fx": self.fx, "fy": self.fy, "x0": self.x0, "y0": self.y0},
            "position_w_c": self.position_w_c.tolist(),
            "rotation_w_c": self.rotation_w_c.tolist(),
        }
