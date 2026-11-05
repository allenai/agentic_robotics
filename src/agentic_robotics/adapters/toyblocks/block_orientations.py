"""The 24 axis-aligned block orientations, which body face each one shows the front camera, and
the test for a block being in one of them.

`ORI_QUATERNIONS` is TAMP's table (`tmp3.Utils.orientations`, w, x, y, z), copied so that task
generators and scorers run without the heavy venv. A PDDL `(ori_of block oN)` puts the block at
this quaternion plus a world-yaw jitter of up to 30 degrees (`ori_of.sample`), so the body face
that lands on world +x and the one that lands on +z are what the orientation decides.

Faces are named by body axis, `+x` .. `-z`. World +x is toward the front camera (it sits at
x = 0.94 looking back along -x at blocks near x = 0.5), so the body face that `oN` turns onto
world +x is the one the camera sees square-on, and the face on world +z is the top.

`orientation_ok` is the classifier's test (`_quat_close_batched` in ToyBlocks's
`pddl_goal_tensor_conversions.py`): the block's z axis within `TILT_TOL_DEG` of the target's and
its yaw within a tolerance, against any accepted target. The classifier measures yaw from the
world x axis, but the camera sees a tower away from the table's center line at an angle (34
degrees for a tower at |y| = 0.35 m), so a letter turned toward the camera can fail and read
perfectly. A block therefore also passes when its yaw is within the tolerance of the line from the
block to the front camera (`camera_azimuth_deg`). Either test is accepted, not the camera-relative
one alone: on the final-state grids of the Oracle suite, the camera-relative test alone rejects 13
towers the developer judged readable, while accepting either test changes exactly the two judged
readable and nothing else.
"""

from __future__ import annotations

import math

Quaternion = tuple[float, float, float, float]
"""(w, x, y, z)."""

ORI_QUATERNIONS: dict[str, Quaternion] = {
    "o1": (1.0, 0.0, 0.0, 0.0),
    "o2": (0.7071, 0.0, 0.0, -0.7071),
    "o3": (0.0, 0.0, 0.0, -1.0),
    "o4": (0.7071, 0.0, 0.0, 0.7071),
    "o5": (0.7071, 0.0, -0.7071, 0.0),
    "o6": (0.5, -0.5, -0.5, -0.5),
    "o7": (0.0, 0.7071, 0.0, 0.7071),
    "o8": (0.5, 0.5, -0.5, 0.5),
    "o9": (0.0, 0.0, 1.0, 0.0),
    "o10": (0.0, 0.7071, 0.7071, 0.0),
    "o11": (0.0, -1.0, 0.0, 0.0),
    "o12": (0.0, -0.7071, 0.7071, 0.0),
    "o13": (0.7071, 0.0, 0.7071, 0.0),
    "o14": (0.5, 0.5, 0.5, -0.5),
    "o15": (0.0, -0.7071, 0.0, 0.7071),
    "o16": (0.5, -0.5, 0.5, 0.5),
    "o17": (0.5, 0.5, -0.5, -0.5),
    "o18": (0.0, 0.0, 0.7071, 0.7071),
    "o19": (0.5, 0.5, 0.5, 0.5),
    "o20": (0.7071, 0.7071, 0.0, 0.0),
    "o21": (0.5, -0.5, 0.5, -0.5),
    "o22": (0.0, 0.0, 0.7071, -0.7071),
    "o23": (0.5, -0.5, -0.5, 0.5),
    "o24": (0.7071, -0.7071, 0.0, 0.0),
}

ORIENTATIONS: tuple[str, ...] = tuple(ORI_QUATERNIONS)

FACES: tuple[str, ...] = ("+x", "-x", "+y", "-y", "+z", "-z")

FRONT_AXIS = "+x"
"""World axis toward the front camera."""

TOP_AXIS = "+z"

_AXIS_VECTORS: dict[str, tuple[float, float, float]] = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}


def qmul(a: Quaternion, b: Quaternion) -> Quaternion:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def rotate(q: Quaternion, v: tuple[float, float, float]) -> tuple[float, float, float]:
    """`v` (a body-frame vector) expressed in the world frame under orientation `q`."""
    w, x, y, z = q
    return qmul(qmul(q, (0.0, *v)), (w, -x, -y, -z))[1:]


def axis_name(v: tuple[float, float, float]) -> str:
    """The signed world axis a vector lies closest to."""
    index = max(range(3), key=lambda i: abs(v[i]))
    return ("+" if v[index] > 0 else "-") + "xyz"[index]


def world_axis_of(face: str, ori: str) -> str:
    """Which world axis a body face points along under an orientation."""
    return axis_name(rotate(ORI_QUATERNIONS[ori], _AXIS_VECTORS[face]))


def front_face(ori: str) -> str:
    """The body face turned toward the front camera under `ori`."""
    return next(face for face in FACES if world_axis_of(face, ori) == FRONT_AXIS)


def top_face(ori: str) -> str:
    """The body face on top under `ori`."""
    return next(face for face in FACES if world_axis_of(face, ori) == TOP_AXIS)


def nominal_orientation(face: str, up: str) -> str:
    """The one orientation that turns `face` toward the camera with the body axis `up` upward.

    :raises ValueError: If `up` is not perpendicular to `face`
    """
    matches = [
        ori
        for ori in ORIENTATIONS
        if world_axis_of(face, ori) == FRONT_AXIS and world_axis_of(up, ori) == TOP_AXIS
    ]
    if len(matches) != 1:
        raise ValueError(f"No single orientation shows face {face} with {up} up: {matches}.")
    return matches[0]


def orientation_errors(quat_wxyz: Quaternion, target_wxyz: Quaternion) -> tuple[float, float]:
    """(tilt, |yaw|) in degrees of `quat` relative to `target`, as the classifier measures them
    (`_quat_close_batched` in ToyBlocks's `pddl_goal_tensor_conversions.py`)."""
    tw, tx, ty, tz = target_wxyz
    err = qmul(quat_wxyz, (tw, -tx, -ty, -tz))
    norm = math.sqrt(sum(c * c for c in err)) or 1.0
    err = (err[0] / norm, err[1] / norm, err[2] / norm, err[3] / norm)
    z_axis = rotate(err, (0.0, 0.0, 1.0))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, z_axis[2]))))
    x_axis = rotate(err, (1.0, 0.0, 0.0))
    yaw = math.degrees(math.atan2(x_axis[1], x_axis[0]))
    return tilt, abs(yaw)


def yawed(target_wxyz: Quaternion, yaw_deg: float) -> Quaternion:
    """`target` turned about world z by `yaw_deg`."""
    half = math.radians(yaw_deg) / 2.0
    return qmul((math.cos(half), 0.0, 0.0, math.sin(half)), target_wxyz)


TILT_TOL_DEG = 5.0
"""The classifier's tolerance on the block's z axis."""

YAW_TOL_DEG = 40.0
"""The classifier's yaw tolerance; a scorer may pass its own."""

FRONT_CAMERA_XY_M: tuple[float, float] = (0.944958, 0.000714)
"""World x, y of the front camera the agent sees through, from a run manifest's
`observation.camera_model.position_w_c`."""


def camera_azimuth_deg(
    position_xy: tuple[float, float], camera_xy: tuple[float, float] = FRONT_CAMERA_XY_M
) -> float:
    """Heading, in degrees about world z from the x axis, of the line from a block to the camera."""
    return math.degrees(math.atan2(camera_xy[1] - position_xy[1], camera_xy[0] - position_xy[0]))


def orientation_ok(
    quat_wxyz: Quaternion,
    accepted: tuple[str, ...],
    camera_azimuth: float | None = None,
    yaw_tol_deg: float = YAW_TOL_DEG,
) -> bool:
    """Whether the block is in an accepted orientation, its yaw measured from the world x axis or,
    when `camera_azimuth` is given, also from the line to the camera."""
    targets = [ORI_QUATERNIONS[name] for name in accepted]
    if camera_azimuth is not None:
        targets += [yawed(target, camera_azimuth) for target in targets]
    return any(
        tilt <= TILT_TOL_DEG and yaw <= yaw_tol_deg
        for tilt, yaw in (orientation_errors(quat_wxyz, target) for target in targets)
    )
