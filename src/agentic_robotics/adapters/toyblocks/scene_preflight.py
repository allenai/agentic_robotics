"""Refuse to start an episode the agent cannot possibly act in, and check the camera behind it.

Every skill resolves an `ImagePoint` to a block, so a block no pixel can name is a block the
agent cannot touch -- and nothing downstream says so. `pick` just answers "No block is near"
for every point tried, which reads as the agent aiming badly rather than as a scene that was
never actionable.

`assert_all_blocks_visible` only checks `resolve_block` against itself, so a camera model that is
*consistently* wrong passes it. `report_intrinsics_scale` is the other half and only logs: it
measures the reported intrinsics against the rendered depth buffer, which is the one thing that
can see that class of failure. `preflight_scene` runs both.

`runner_bootstrap.run_episode` runs it before the agent's first turn, so every `agentic-robotics`
episode passes through it; a guard nothing calls is indistinguishable from no guard.

It pairs with `reset_state_check.py` and the two do not overlap: that one reads
`reset_states.pkl` off disk and answers "could any of these 128 states be acted in", with no
GPU and before an episode exists; this one reads the live scene through the same camera model
and the same `resolve_block` the skills use, and answers "can this episode, the one about to
start, be acted in". A package can pass the first and an episode still fail the second, because
the state drawn is one of the bad ones.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentic_robotics.adapters.toyblocks.config import OracleConfig
    from agentic_robotics.adapters.toyblocks.toyblocks_env import ToyBlocksEnv
    from agentic_robotics.adapters.toyblocks.toyblocks_mcp_server import ToyBlocksMCPServer
    from agentic_robotics.core.cameras import CameraModel
    from agentic_robotics.core.images import Resolution

logger = logging.getLogger("agentic_robotics.examples.scene_preflight")


class BlocksNotVisibleError(RuntimeError):
    """Raised when a block in the scene cannot be named by any pixel of the agent's image."""


def _visibility_rows(
    env: ToyBlocksEnv,
    *,
    camera_name: str,
    oracle_config: OracleConfig,
    resolution: Resolution,
) -> tuple[CameraModel, list[dict[str, Any]]]:
    """Check every block round-trips: world pose -> centre pixel -> back to the same block.

    Projection alone is not the contract the skills honour. `pick` resolves a pixel through
    `resolve_block`, which skips blocks behind the camera, quantizes the agent's normalized
    [0,1000) point onto the render grid, and accepts a block only within
    `pointing_error_ratio` times its projected radius. So the check that actually matters is the
    round trip through that same function: a block is reachable iff its own centre pixel resolves
    back to it. That also catches two blocks overlapping in image space, which a bounds test would
    pass.

    :param env: Interface to access the ToyBlocks environment
    :param camera_name: Camera the agent sees, i.e. the run configuration's `camera_name`
    :param oracle_config: The `OracleConfig` the skills resolve pixels with
    :param resolution: Resolution of the rendered observation, i.e. the server's `camera_resolution`
    :return: The camera model that was read, and one row per block describing its reachability
    """
    import numpy as np

    from agentic_robotics.adapters.toyblocks.block_pointing import resolve_block
    from agentic_robotics.core.images import PixelCoord

    camera = env.read_camera_model(camera_name)
    rows: list[dict[str, Any]] = []

    for name in env.block_names:
        center = env.get_block_center(name)
        depth = float((center - camera.position_w_c) @ camera.rotation_w_c[:, 2])
        projected = camera.points_to_pixels(center)[0]
        row: dict[str, Any] = {
            "name": name,
            "world": center,
            "depth": depth,
            "pixel": projected,
            "resolved": None,
            "problem": None,
        }
        rows.append(row)

        if not np.isfinite(projected).all():
            row["problem"] = f"behind the camera (depth {depth:+.3f} m)"
            continue

        u, v = round(float(projected[0])), round(float(projected[1]))
        if not (0 <= u < resolution.width and 0 <= v < resolution.height):
            off_u = max(0, -u, u - (resolution.width - 1))
            off_v = max(0, -v, v - (resolution.height - 1))
            row["problem"] = f"projects off-frame by ({off_u},{off_v}) px"
            continue

        # Quantize the way the tools do: the agent can only name normalized [0,1000) integers,
        # so a block reachable at a sub-pixel centre may not be reachable at any nameable one.
        nameable = PixelCoord(u=u, v=v).normalize(resolution).to_pixel(resolution)
        hit = resolve_block(env, nameable, camera_name=camera_name, config=oracle_config)
        row["resolved"] = hit

        if hit is None:
            row["problem"] = "its own centre pixel resolves to no block"
        elif hit != name:
            row["problem"] = f"its own centre pixel resolves to '{hit}' instead"

    return camera, rows


def _format_visibility_table(
    camera: CameraModel, rows: list[dict[str, Any]], resolution: Resolution
) -> str:
    """Render the visibility rows as a table, for both the happy path and the error."""
    columns = (14, 26, 8, 13, 44)
    lines = [
        "{:<{}}{:<{}}{:>{}} {:<{}}{}".format(
            "block",
            columns[0],
            "world (x, y, z)",
            columns[1],
            "depth",
            columns[2],
            " pixel",
            columns[3],
            "verdict",
        ),
        " ".join("-" * width for width in columns),
    ]
    for row in rows:
        x, y, z = row["world"]
        pixel = row["pixel"]
        # NaN pixel means the block is behind the camera, so there is nothing to print.
        pixel_text = "--" if pixel[0] != pixel[0] else f" ({pixel[0]:.0f}, {pixel[1]:.0f})"
        lines.append(
            f"{row['name']:<{columns[0]}}"
            f"{f'({x:+.3f}, {y:+.3f}, {z:+.3f})':<{columns[1]}}"
            f"{row['depth']:>+{columns[2]}.3f} "
            f"{pixel_text:<{columns[3]}}"
            f"{'reachable' if row['problem'] is None else row['problem']}"
        )
    lines.append(
        f"\ncamera at ({camera.position_w_c[0]:+.3f}, {camera.position_w_c[1]:+.3f}, "
        f"{camera.position_w_c[2]:+.3f}), optical axis "
        f"({camera.rotation_w_c[0, 2]:+.3f}, {camera.rotation_w_c[1, 2]:+.3f}, "
        f"{camera.rotation_w_c[2, 2]:+.3f}), image {resolution.width}x{resolution.height}, "
        f"fx={camera.fx:.1f} fy={camera.fy:.1f} x0={camera.x0:.1f} y0={camera.y0:.1f}"
    )
    return "\n".join(lines)


def assert_all_blocks_visible(
    env: ToyBlocksEnv,
    *,
    camera_name: str,
    oracle_config: OracleConfig,
    resolution: Resolution,
) -> None:
    """Refuse to start the episode unless every block is reachable by some pixel.

    Guards the failure mode from run `r-on-red-oracle-20260826-145057`: the reset states placed
    all five blocks in a line on the floor, so four were behind the camera and the fifth projected
    135 px below the bottom row. Every `pick` was rejected with "No block is near", the agent spent
    its whole budget probing a grid, and nothing in the stack complained -- the run looked healthy
    from the outside and cost a full episode to discover.

    :param env: Interface to access the ToyBlocks environment
    :param camera_name: Camera the agent sees, i.e. the run configuration's `camera_name`
    :param oracle_config: The `OracleConfig` the skills resolve pixels with
    :param resolution: Resolution of the rendered observation, i.e. the server's `camera_resolution`
    :raises BlocksNotVisibleError: If any block in the scene cannot be named by any pixel
    """
    camera, rows = _visibility_rows(
        env,
        camera_name=camera_name,
        oracle_config=oracle_config,
        resolution=resolution,
    )
    table = _format_visibility_table(camera, rows, resolution)

    if not rows:
        raise BlocksNotVisibleError(
            f"No blocks in the scene at all; '{camera_name}' has nothing to point at.\n\n{table}"
        )

    unreachable = [row for row in rows if row["problem"] is not None]
    if not unreachable:
        logger.info(f"Preflight: all {len(rows)} blocks reachable via '{camera_name}'.\n{table}")
        return

    names = ", ".join(row["name"] for row in unreachable)
    raise BlocksNotVisibleError(
        f"{len(unreachable)} of {len(rows)} blocks cannot be named by any pixel of "
        f"'{camera_name}': {names}. The agent's pick/place/rotate/flip skills all resolve a "
        f"pixel to a block, so this episode cannot be acted in and is not worth running.\n\n"
        f"{table}\n\n"
        f"Two causes are worth separating, and `reset_state_check.py` on this task's autogen "
        f"package tells you which: reset states that never placed the blocks at all (the "
        f"codegen spawn layout, blocks lined up at x = 1, 2, 3... m), or states whose sampler "
        f"fell through to an unvalidated pose and put a block outside the arm's reach. The "
        f"first is a package to regenerate; the second is one bad state out of many, and "
        f"pinning `instance_ids` past it is cheaper. Pass --no-preflight to run anyway."
    )


def report_intrinsics_scale(env: ToyBlocksEnv, *, camera_name: str) -> None:
    """Measure the reported intrinsics against the rendered depth buffer, and log the result.

    A regression guard for a bug that is fixed, kept because nothing else can see it.
    `assert_all_blocks_visible` only checks `resolve_block` against itself, so a camera model that
    is *consistently* wrong passes it. On 2026-08-26 `read_camera_model` reported Isaac Lab's
    intrinsics (fx=fy=263.68 at 640x360) while the image the blocks were drawn into had an
    effective focal length ~1.20x longer. Projected centres sat 12-20 px from where a block
    appeared against a match tolerance of ~9-13 px, so visually dead-on points were rejected and
    the stack blamed the agent's aim.

    The cause was `PinholeCameraCfg.from_intrinsic_matrix` not inheriting the class default lens
    length: left at None it derived a 0.412 cm lens behind a 1.0 cm aperture, and Isaac Lab reads
    fx back as `width * focalLength / horizontalAperture` either way, so the *reported* intrinsics
    looked right while the render did not match them. Fixed in ToyBlocks by passing
    `focal_length=24.0` in `src/toyblocks/scenes/sim2real.py`.

    The check: the camera also renders `distance_to_image_plane` at the same resolution. Take a
    grid of pixels over the table, turn each into a world point using the *reported* intrinsics
    plus that rendered depth, and ask what focal scale puts those points on the table plane. A
    correct camera model gives ~1.0; the radial error of a wrong focal length grows with distance
    from the principal point, so a grid spanning the frame measures it directly. Expect ~1.0,
    and treat anything else as the scene preset having regressed.

    This only reports. The scale is not applied: correcting it belongs in the camera model, and a
    measured fudge factor is not a substitute for knowing why the two disagree.
    """
    import numpy as np

    from agentic_robotics.adapters.toyblocks.block_pointing import find_table_z

    # `_get_camera` is private, but it is the same lookup `read_camera_model` uses, and the two
    # must not diverge -- the depth buffer has to come from the camera the model describes.
    camera = env.read_camera_model(camera_name)
    depth_output = env._get_camera(camera_name).data.output.get("distance_to_image_plane")
    if depth_output is None:
        logger.warning(f"Camera '{camera_name}' renders no depth; skipping the intrinsics check.")
        return

    depth = depth_output[0].detach().cpu().numpy().squeeze(-1)  # (H, W), metres along z_cam
    height, width = depth.shape
    table_z = find_table_z(env)

    # Grid over the lower frame, where the table fills the image.
    us = np.linspace(width * 0.15, width * 0.85, 12)
    vs = np.linspace(height * 0.55, height * 0.95, 8)
    grid_u, grid_v = (a.ravel() for a in np.meshgrid(us, vs))
    sampled = depth[np.rint(grid_v).astype(int), np.rint(grid_u).astype(int)]
    finite = np.isfinite(sampled) & (sampled > 0)
    if finite.sum() < 8:
        logger.warning("Too few valid depth samples; skipping the intrinsics check.")
        return

    def plane_error(scale: float) -> float:
        """RMS deviation from the table plane when the focal lengths are scaled by `scale`."""
        x = (grid_u[finite] - camera.x0) / (camera.fx * scale)
        y = (grid_v[finite] - camera.y0) / (camera.fy * scale)
        points_camera = np.stack([x, y, np.ones(finite.sum())], axis=1) * sampled[finite, None]
        world_z = camera.position_w_c[2] + points_camera @ camera.rotation_w_c[2, :]
        return float(np.sqrt(np.mean((world_z - table_z) ** 2)))

    scales = np.linspace(0.7, 1.6, 181)
    errors = [plane_error(s) for s in scales]
    best = float(scales[int(np.argmin(errors))])

    message = (
        f"Intrinsics vs render: reported fx={camera.fx:.2f} fy={camera.fy:.2f} at "
        f"{width}x{height}; the rendered depth is best explained by scaling them x{best:.3f} "
        f"(plane RMS {min(errors) * 1000:.0f} mm, vs {plane_error(1.0) * 1000:.0f} mm unscaled)."
    )
    if abs(best - 1.0) <= 0.05:
        logger.info(message)
    else:
        logger.warning(
            f"{message} A scale far from 1.0 means projected block centres do not land where "
            f"blocks are drawn, so correct-looking points will be rejected by pick/rotate/flip. "
            f"Implied true focal: fx={camera.fx * best:.1f} fy={camera.fy * best:.1f}."
        )


def preflight_scene(mcp_server: ToyBlocksMCPServer, *, require: bool = True) -> None:
    """Run the visibility check against a built server's scene.

    Everything the check needs is already on the server, so callers do not assemble it: the
    camera the agent sees, the `OracleConfig` its skills resolve pixels with, and the rendered
    resolution are all decided by the time a server exists.

    Must run on the simulator's thread -- it reads block poses and a camera model out of the
    live scene.

    :param mcp_server: A built `ToyBlocksMCPServer`, whose scene is about to be acted in
    :param require: Whether an unactionable scene ends the run; False logs and continues
    :raises BlocksNotVisibleError: If `require` and any block cannot be named by any pixel
    """
    report_intrinsics_scale(mcp_server._toyblocks_env, camera_name=mcp_server.config.camera_name)

    try:
        assert_all_blocks_visible(
            mcp_server._toyblocks_env,
            camera_name=mcp_server.config.camera_name,
            oracle_config=mcp_server.config.oracle,
            resolution=mcp_server.camera_resolution,
        )
    except BlocksNotVisibleError:
        if require:
            raise
        logger.warning("Preflight failed, continuing because it was not required:", exc_info=True)
