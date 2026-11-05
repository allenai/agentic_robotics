"""Define ToyBlocks-related exception classes."""

from agentic_robotics.core.images import PixelCoord, Resolution


class InvalidPixelError(Exception):
    """Raised when an agent-selected pixel falls outside the image observation's resolution."""

    def __init__(self, obs_resolution: Resolution, pixel: PixelCoord) -> None:
        """Define the error message and pass it to the base Exception class."""
        message = (
            f"Pixel ({pixel.u},{pixel.v}) falls outside the image observation, "
            f"which has a resolution of {obs_resolution.width}x{obs_resolution.height} (WxH)."
        )

        super().__init__(message)
