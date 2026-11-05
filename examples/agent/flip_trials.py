"""Model-free `flip` trials on a pool simulator, and contact sheets cut from its video.

A probe in the sense of CLAUDE.md: it drives a served `agentic-robotics pool` simulator over MCP and runs no
agent, so it carries none of `cli.py`'s checks beyond what the pool did at boot (block library,
policy server). Unlike the driver of the 2026-09-29 gripper check
(`runs/gripper_check_20260929/experiment.py`), it takes a chosen start orientation for the target
block.

    # One pool simulator, freshly started (the frame accounting below assumes nothing else used it)
    agentic-robotics pool --task ai2_easy --pool-port 8101

    # Five sideways flips per direction, then five upright controls
    python examples/agent/flip_trials.py run --out-dir RUN --pool-run-dir POOL \
        --base-state runs/gripper_check_20260929/base_state.json \
        --orientations o6,o14 --directions front,back,left,right --trials 5
    python examples/agent/flip_trials.py run ... --orientations base --directions front --trials 5

    # After the pool is stopped and its video.mp4 is finalized
    python examples/agent/flip_trials.py sheet --out-dir RUN --pool-run-dir POOL

**Start state.** Each trial writes the base state with the target block moved to a fresh spot
(x 0.42-0.58 m, y -0.18-0.18 m, clear of the other blocks, as on 2026-09-29) and its orientation
drawn the way `ori_of` samples one for a generated task: the face's canonical quaternion from
the adapter's `ORI_QUATERNIONS` table, turned about world z by up to 30 degrees either way. The label `base`
keeps the base state's own quaternion instead, which is what every 2026-09-29 flip started from.
The arm and gripper come from the base state (home pose, gripper open). A failed state write, or
one that settles with a different face up than was asked for, stops the run.

**Direction.** Passed to the tool as given: `front`, `back`, `left` or `right`, the tool's
`FlipDirection`. The server turns `front`/`back` into the trained "flip forward"/"flip backward".

**Outcome.** `success` is the up face changing through a rotation of more than 60 degrees.
`strict_success` is the 2026-09-29 driver's test (65-115 degrees about a non-vertical axis, face
changed, moved at most 5 cm, still on the table), recorded alongside for comparison. The finger
joint is read after every flip: about 0 open, 0.4-0.55 on a block, 0.72-0.84 closed on nothing.

**Frames.** The pool's `video.mp4` gains one frame at boot, one per `set_sim_state` and one per
step of a skill, and the client cannot ask for the count. So each trial records the frame range
its flip should occupy, counted from a fresh pool, and `sheet` refuses to cut unless the video's
length equals the count after the last trial. Every call that adds frames is made once, never
retried, and any tool error stops the run, so the count cannot silently drift.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mcp import Client as MCPClient
from PIL import Image
from scipy.spatial.transform import Rotation

from agentic_robotics.adapters.toyblocks.block_orientations import ORI_QUATERNIONS

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from video_render import open_reader

from agentic_robotics.core.cameras import CameraModel
from agentic_robotics.core.states import Pose, Position, Quaternion, SimState

DEFAULT_URL = "http://127.0.0.1:8101/mcp"
FINGER_JOINT = "finger_joint"
BASE_LABEL = "base"
YAW_RANGE_DEG = 30.0
"""The world-yaw spread `ori_of.sample_vectorized` draws a face's orientation with."""

CAMERA_ROTATION_WXYZ = (-0.3265056, 0.6272114, 0.6272114, -0.3265056)
"""`front_camera`'s rotation in ToyBlocks's `scenes/sim2real.py` (ROS convention). Copied because
that module imports Isaac Lab, and manifests before schema 3.1 record the camera's position but
not its rotation. Used only for those; later manifests carry the whole camera model."""

SHEET_COLUMNS = 5
SHEET_ROWS = 3
SHEET_TILE = (320, 180)


# --- Geometry --------------------------------------------------------------------------------


def load_camera(pool_run_dir: Path) -> tuple[CameraModel, float]:
    """The pool's camera model, and the scale from camera pixels to observation pixels."""
    observation = json.loads((pool_run_dir / "run.json").read_text())["observation"]
    if "camera_model" in observation:  # Manifest schema 3.1 and later
        model = observation["camera_model"]
        k = model["intrinsics"]
        position_w_c = np.array(model["position_w_c"])
        rotation_w_c = np.array(model["rotation_w_c"])
    else:
        k = observation["intrinsics"]
        position_w_c = np.array(observation["camera_position_w"])
        rotation_w_c = Rotation.from_quat(CAMERA_ROTATION_WXYZ, scalar_first=True).as_matrix()
    camera = CameraModel(
        fx=k["fx"],
        fy=k["fy"],
        x0=k["x0"],
        y0=k["y0"],
        position_w_c=position_w_c,
        rotation_w_c=rotation_w_c,
    )
    scale = (
        observation["observation_resolution"]["width"] / observation["camera_resolution"]["width"]
    )
    return camera, scale


DEFAULT_FLIP_STEPS = 450
"""`FLIP_STEPS` in `adapters/toyblocks/leap/constants.py`, the tool's own default (set from the
cutoff sweep in `runs/flip_cutoff_20261005/`); kept in step by hand because
importing the server module pulls in the policy client. Passed explicitly on every call so
the frame accounting below and the tool agree on the count."""


def block_pixel(camera: CameraModel, scale: float, position_xyz_m: list[float]) -> dict[str, int]:
    """The observation pixel a block's center projects to, in the form the skills take."""
    u, v = camera.points_to_pixels(np.array([position_xyz_m]))[0] * scale
    return {"u": int(round(u)), "v": int(round(v))}


def rotation_of(quaternion_wxyz: list[float]) -> Rotation:
    return Rotation.from_quat(quaternion_wxyz, scalar_first=True)


def up_face(quaternion_wxyz: list[float]) -> str:
    """Which body axis points up in the world: '+x', '-y', ..."""
    body_up = rotation_of(quaternion_wxyz).inv().apply([0.0, 0.0, 1.0])
    i = int(np.argmax(np.abs(body_up)))
    return ("+" if body_up[i] > 0 else "-") + "xyz"[i]


def sample_position(rng: np.random.Generator, others: list[list[float]]) -> list[float]:
    """Uniform in x [0.42, 0.58], y [-0.18, 0.18], at least 9 cm (Chebyshev) and 11 cm from every other block."""
    for _ in range(10000):
        x, y = rng.uniform(0.42, 0.58), rng.uniform(-0.18, 0.18)
        if all(
            max(abs(x - o[0]), abs(y - o[1])) >= 0.09 and math.hypot(x - o[0], y - o[1]) >= 0.11
            for o in others
        ):
            return [round(x, 4), round(y, 4)]
    raise RuntimeError("Could not sample a free position for the target block.")


def sample_world_yaw_perturb(
    q_base: torch.Tensor,
    yaw_range_deg: float,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """`q_base` (w, x, y, z) turned about world z by a yaw drawn uniformly from the range.

    A world-frame yaw is a left multiplication by the yaw quaternion. The result is unit-normalized
    with a nonnegative w, so equal rotations compare equal.

    :param q_base: Base orientation, shape (4,)
    :param yaw_range_deg: The draw is uniform in [-yaw_range_deg, +yaw_range_deg]
    :param device: Device of the draw and the result; the generator must be on it too
    :param generator: Generator for a reproducible draw
    :return: The perturbed orientation, shape (4,)
    """
    q_base = q_base.to(device=device, dtype=torch.float32)
    q_base = q_base / (q_base.norm() + 1e-8)
    yaw_range = math.radians(yaw_range_deg)
    half = 0.5 * (2.0 * torch.rand((), device=device, generator=generator) - 1.0) * yaw_range
    w1, z1 = torch.cos(half), torch.sin(half)
    w2, x2, y2, z2 = q_base
    q = torch.stack([w1 * w2 - z1 * z2, w1 * x2 - z1 * y2, w1 * y2 + z1 * x2, w1 * z2 + z1 * w2])
    q = q / (q.norm() + 1e-8)
    return -q if q[0] < 0 else q


def sample_orientation(
    label: str, base_quaternion: list[float], generator: torch.Generator
) -> list[float]:
    """A start quaternion for the target: the base state's for `base`, else as `ori_of` samples the face."""
    if label == BASE_LABEL:
        return list(base_quaternion)
    quaternion = sample_world_yaw_perturb(
        torch.tensor(ORI_QUATERNIONS[label], dtype=torch.float32),
        yaw_range_deg=YAW_RANGE_DEG,
        device="cpu",
        generator=generator,
    )
    return [round(float(c), 4) for c in quaternion.tolist()]


def flip_outcome(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Score one flip from the target block's pose before and after it."""
    q0, q1 = before["quaternion_wxyz"], after["quaternion_wxyz"]
    p0, p1 = np.array(before["position_xyz_m"]), np.array(after["position_xyz_m"])
    rotvec = (rotation_of(q1) * rotation_of(q0).inv()).as_rotvec()
    angle_deg = float(np.degrees(np.linalg.norm(rotvec)))
    axis_z_abs = float(abs(rotvec[2]) / np.linalg.norm(rotvec)) if angle_deg > 1e-6 else 0.0
    face_changed = up_face(q0) != up_face(q1)
    xy_displacement_m = float(np.linalg.norm(p1[:2] - p0[:2]))
    on_table = bool(abs(p1[2] - p0[2]) < 0.01)
    return {
        "angle_deg": angle_deg,
        "axis_z_abs": axis_z_abs,
        "up_before": up_face(q0),
        "up_after": up_face(q1),
        "face_changed": face_changed,
        "xy_displacement_m": xy_displacement_m,
        "on_table": on_table,
        "success": face_changed and angle_deg > 60.0,
        "strict_success": (
            face_changed
            and 65.0 <= angle_deg <= 115.0
            and axis_z_abs < 0.5
            and xy_displacement_m <= 0.05
            and on_table
        ),
    }


# --- Pool calls ------------------------------------------------------------------------------


class ToolCallError(RuntimeError):
    """A pool tool reported an error; the run stops rather than working around it."""


async def call_tool(
    url: str, name: str, arguments: dict[str, Any]
) -> tuple[str | None, bytes | None]:
    """Call one pool tool once; return its text and image, raising `ToolCallError` on a tool error."""
    async with MCPClient(url, read_timeout_seconds=600) as client:
        result = await client.call_tool(name, arguments)
    text = next((block.text for block in result.content if getattr(block, "text", None)), None)
    image_block = next((block for block in result.content if getattr(block, "data", None)), None)
    if result.is_error:
        raise ToolCallError(f"{name} failed: {text}")
    import base64

    image = base64.b64decode(image_block.data) if image_block is not None else None
    return text, image


async def read_state(url: str) -> SimState:
    text, _ = await call_tool(url, "get_sim_state", {})
    if text is None:
        raise ToolCallError("get_sim_state returned no text.")
    return SimState.model_validate_json(text)


def pose_lists(pose: Pose) -> dict[str, Any]:
    """A pose as the lists this probe's records use: position_xyz_m and quaternion_wxyz."""
    p, q = pose.position, pose.orientation
    return {"position_xyz_m": [p.x, p.y, p.z], "quaternion_wxyz": [q.w, q.x, q.y, q.z]}


def block_pose(state: SimState, name: str) -> dict[str, Any]:
    return pose_lists(next(obj.pose for obj in state.objects if obj.name == name))


def finger(state: SimState) -> float:
    configuration = state.robot_configuration
    return configuration.joint_values[configuration.joint_names.index(FINGER_JOINT)]


# --- run -------------------------------------------------------------------------------------


def next_frame(trials_path: Path) -> int:
    """The video frame the next trial's state write lands on: after the last recorded flip, or after boot."""
    if not trials_path.exists():
        return 1
    rows = [json.loads(line) for line in trials_path.read_text().splitlines() if line.strip()]
    return rows[-1]["frames"]["flip"][1] if rows else 1


async def run_trials(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = args.out_dir / "trials.jsonl"
    images_dir = args.out_dir / "images"
    images_dir.mkdir(exist_ok=True)

    camera, scale = load_camera(args.pool_run_dir)
    flip_steps = args.flip_steps
    base = SimState.model_validate_json(args.base_state.read_text())
    base_target = pose_lists(next(obj.pose for obj in base.objects if obj.name == args.target))
    others = [
        pose_lists(obj.pose)["position_xyz_m"] for obj in base.objects if obj.name != args.target
    ]
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    frame = next_frame(trials_path)

    for direction in args.directions:
        for index in range(args.trials):
            label = args.orientations[int(rng.integers(len(args.orientations)))]
            xy = sample_position(rng, others)
            quaternion = sample_orientation(label, base_target["quaternion_wxyz"], generator)
            state = base.model_copy(deep=True)
            for obj in state.objects:
                if obj.name == args.target:
                    obj.pose = Pose(
                        position=Position(x=xy[0], y=xy[1], z=base_target["position_xyz_m"][2]),
                        orientation=Quaternion(
                            w=quaternion[0], x=quaternion[1], y=quaternion[2], z=quaternion[3]
                        ),
                    )

            await call_tool(args.url, "set_sim_state", {"state": state.model_dump()})
            set_state_frame = frame
            frame += 1
            start = await read_state(args.url)
            start_pose = block_pose(start, args.target)
            if up_face(start_pose["quaternion_wxyz"]) != up_face(quaternion):
                raise ToolCallError(
                    f"The state write settled with {up_face(start_pose['quaternion_wxyz'])} up, "
                    f"not the requested {up_face(quaternion)}: {start_pose}"
                )

            pixel = block_pixel(camera, scale, start_pose["position_xyz_m"])
            began = time.time()
            _, image = await call_tool(args.url, "flip", {"pixel": pixel, "direction": direction})
            seconds = time.time() - began
            flip_frames = [frame, frame + flip_steps]
            frame += flip_steps
            end = await read_state(args.url)

            name = f"{direction}_{label}_{index}"
            image_path = None
            if image is not None:
                image_path = images_dir / f"{name}.png"
                image_path.write_bytes(image)
            outcome = flip_outcome(start_pose, block_pose(end, args.target))
            row = {
                "name": name,
                "direction": direction,
                "orientation": label,
                "trial": index,
                "target": args.target,
                "requested_quaternion_wxyz": quaternion,
                "pixel": pixel,
                "start_pose": start_pose,
                "end_pose": block_pose(end, args.target),
                "finger_before": finger(start),
                "finger_after": finger(end),
                "seconds": round(seconds, 1),
                "flip_steps": flip_steps,
                "frames": {"set_state": set_state_frame, "flip": flip_frames},
                "image": str(image_path) if image_path else None,
                "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(began)),
                **outcome,
            }
            with trials_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(
                f"{name}: {outcome['up_before']}->{outcome['up_after']} {outcome['angle_deg']:.0f} deg, "
                f"success={outcome['success']} strict={outcome['strict_success']}, "
                f"finger {finger(start):.2f}->{finger(end):.2f}, {seconds:.0f} s",
                flush=True,
            )


# --- sheet -----------------------------------------------------------------------------------


def make_sheets(args: argparse.Namespace) -> None:
    rows = [
        json.loads(line)
        for line in (args.out_dir / "trials.jsonl").read_text().splitlines()
        if line.strip()
    ]
    reader = open_reader(args.pool_run_dir / "video.mp4")
    frame_count = reader.count_frames()
    expected = rows[-1]["frames"]["flip"][1]
    if frame_count != expected:
        raise SystemExit(
            f"video.mp4 has {frame_count} frames but the trials account for {expected}; something "
            "else added frames to this pool, so the recorded ranges cannot be trusted."
        )

    chosen: dict[Any, dict[str, Any]] = {}
    for row in rows:
        if args.names:
            if row["name"] in args.names:
                chosen[row["name"]] = row
        else:
            kind = "upright" if row["orientation"] == BASE_LABEL else "sideways"
            chosen.setdefault((row["direction"], kind), row)

    sheets_dir = args.out_dir / "flip_video"
    sheets_dir.mkdir(exist_ok=True)
    for row in chosen.values():
        start, stop = row["frames"]["flip"]
        indices = np.linspace(start, stop - 1, SHEET_COLUMNS * SHEET_ROWS).round().astype(int)
        sheet = Image.new("RGB", (SHEET_TILE[0] * SHEET_COLUMNS, SHEET_TILE[1] * SHEET_ROWS))
        for i, index in enumerate(indices):
            tile = Image.fromarray(reader.get_data(int(index))).resize(SHEET_TILE)
            sheet.paste(
                tile, ((i % SHEET_COLUMNS) * SHEET_TILE[0], (i // SHEET_COLUMNS) * SHEET_TILE[1])
            )
        path = sheets_dir / f"{row['name']}_sheet.png"
        sheet.save(path)
        print(f"{path}: frames {start}-{stop - 1}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run flip trials against a pool simulator")
    run.add_argument("--url", default=DEFAULT_URL, help="The pool simulator's MCP URL")
    run.add_argument(
        "--base-state", type=Path, required=True, help="JSON state as get_sim_state returns it"
    )
    run.add_argument("--target", default="block_a", help="Block to move and flip")
    run.add_argument(
        "--orientations",
        type=lambda s: s.split(","),
        required=True,
        help=f"Comma-separated faces to draw from (o1-o24), or '{BASE_LABEL}' for the base state's",
    )
    run.add_argument(
        "--directions",
        type=lambda s: s.split(","),
        required=True,
        help="Comma-separated: front,back,left,right",
    )
    run.add_argument("--trials", type=int, default=5, help="Trials per direction")
    run.add_argument(
        "--flip-steps",
        type=int,
        default=DEFAULT_FLIP_STEPS,
        help="Control steps passed to the flip tool and assumed by the frame accounting",
    )
    run.add_argument("--seed", type=int, default=0)
    sheet = commands.add_parser(
        "sheet", help="Cut one contact sheet per direction from the pool video"
    )
    sheet.add_argument(
        "--names",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated trial names to cut instead of the first per direction",
    )
    for sub in (run, sheet):
        sub.add_argument(
            "--out-dir",
            type=Path,
            required=True,
            help="Where trials.jsonl, images/ and flip_video/ go",
        )
        sub.add_argument(
            "--pool-run-dir", type=Path, required=True, help="The pool simulator's run directory"
        )
    args = parser.parse_args()
    if args.command == "run":
        for label in args.orientations:
            if label != BASE_LABEL and label not in ORI_QUATERNIONS:
                parser.error(f"Unknown orientation {label!r}")
        for direction in args.directions:
            if direction not in ("front", "back", "left", "right"):
                parser.error(f"Unknown direction {direction!r}")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "run":
        asyncio.run(run_trials(args))
    else:
        make_sheets(args)


if __name__ == "__main__":
    main()
