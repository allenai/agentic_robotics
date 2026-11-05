"""Data structures for reading and writing an object-centric simulator state.

We use pydantic.BaseModel to ensure that these representations are MCP-compatible.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

DECIMALS = 4
"""Number of decimal places kept when rounding floats."""


class Position(BaseModel):
    """A 3D position expressed in meters."""

    x: float
    y: float
    z: float


class Quaternion(BaseModel):
    """A unit quaternion (w, x, y, z) representing a 3D orientation."""

    w: float
    x: float
    y: float
    z: float


class Pose(BaseModel):
    """A 3D position and orientation relative to a reference frame."""

    position: Position = Field(description="(x, y, z) position relative to the parent frame.")
    orientation: Quaternion = Field(
        description="Unit quaternion representing an orientation relative to the parent frame."
    )
    parent_frame: str = "world"


class ObjectPose(BaseModel):
    """Specifies the name and pose of an object."""

    name: str
    pose: Pose


JointType = Literal["revolute", "prismatic"]


class Configuration(BaseModel):
    """Specifies the configuration of a robot's joints."""

    joint_names: list[str] = Field(
        description="Names of the robot's joints, in the same order as `joint_values`."
    )
    joint_types: list[JointType] = Field(
        description="Types of the robot's joints, in the same order as `joint_values`."
    )
    joint_values: list[float] = Field(
        description="Joint values in radians (revolute) or meters (prismatic)."
    )


class SimState(BaseModel):
    """Specifies a complete state of the simulation environment."""

    robot_base_pose: Pose = Field(description="Pose of the robot's base link.")
    robot_configuration: Configuration = Field(description="Positions of the robot's joints.")
    objects: list[ObjectPose] = Field(description="Name and pose of each object in the scene.")


def _rounded(values: Sequence, decimals: int = DECIMALS) -> list[float]:
    """Round the given values to the specified number of decimal places."""
    return [round(float(v), ndigits=decimals) for v in values]
