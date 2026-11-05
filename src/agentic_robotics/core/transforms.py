"""Define utility functions to process 3D transformations."""

import numpy as np


def transform_from(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Construct a homogeneous transformation matrix from a position and a rotation.

    :param position: (x,y,z) position; shape (3,)
    :param rotation: Rotation matrix; shape (3, 3)
    :return: Homogeneous transformation matrix; shape (4, 4)
    """
    matrix = np.eye(4)
    matrix[:3, 3] = position
    matrix[:3, :3] = rotation
    return matrix
