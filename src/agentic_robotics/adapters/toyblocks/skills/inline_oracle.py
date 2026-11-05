"""The oracle skill surface, on ToyBlocks's vecgen skills.

The scripted ("oracle") skills: pick, stack, place on the table, rotate, flip, return home and
the gripper, registered as exactly the nine tools ``OracleMCPServer`` declares. Each motion is a
ToyBlocks vecgen skill planned with cuRobo v2 and run through
:class:`skills.vecgen_motion.VecgenMotion` -- the skills that record ToyBlocks's training
demonstrations, wired the way ToyBlocks's evaluation low-level oracle
(``blocksuite.eval.ll_oracle``) wires them.

Skills executed by scripted motion rather than by a learned policy server, so a failure can be
attributed to the *agent's* planning rather than to an undertrained checkpoint. That is the whole
of what the oracle is for; it cannot appear in a demo, because it reads simulator state the
physical robot does not have.

**Importing this module boots Isaac Sim.** ``cli.py`` is the caller, and passes the run's
`ToyBlocksConfig` to the server it builds from here. The imports below are ordered against the
boot and must not be hoisted.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

import torch
from mcp.server.mcpserver import Image as MCPImage  # noqa: TC002 (used for MCP tool registration)
from pydantic import Field

# This import boots Isaac Sim, on whichever thread imports this module -- which is the only
# thread allowed to step it. Everything Isaac-dependent must follow it.
from agentic_robotics.adapters.toyblocks.toyblocks_env import ToyBlocksEnv

# isort: split

from agentic_robotics.adapters.toyblocks.block_pointing import find_table_z, resolve_block
from agentic_robotics.adapters.toyblocks.oracle_mcp_server import (
    FlipDirection as FlipDirectionArg,
)
from agentic_robotics.adapters.toyblocks.oracle_mcp_server import (
    OracleMCPServer,
)
from agentic_robotics.adapters.toyblocks.oracle_mcp_server import (
    RotateDirection as RotateDirectionArg,
)
from agentic_robotics.adapters.toyblocks.skills.vecgen_motion import (
    FlipDirection,
    OracleSkillError,
    RotateDirection,
    SkillStopped,
    VecgenMotion,
)
from agentic_robotics.core.images import (
    PixelCoord,
    array_to_mcp_image,
    crop_around,
    resize_to_width,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from agentic_robotics.adapters.toyblocks.config import TaskConfig, ToyBlocksConfig
    from agentic_robotics.adapters.toyblocks.toyblocks_runner import SimulatorDispatcher

__all__ = [
    "FlipDirection",
    "InlineOracleMCPServer",
    "OracleSkillError",
    "OracleToyBlocksEnv",
    "RotateDirection",
    "SimThreadOracleServer",
    "SkillStopped",
]

logger = logging.getLogger("agentic_robotics.examples.inline_oracle")

RESET_RERENDERS = 30
"""Renders after a reset or scene build before the first image is read; see
`InlineOracleMCPServer.settle_and_rerender`."""


class OracleToyBlocksEnv(ToyBlocksEnv):
    """A `ToyBlocksEnv` prepared for the oracle: viewer trajectory open, denoiser settled.

    Reached through `InlineOracleMCPServer._build_env` rather than by rebinding the name the base
    server constructs.
    """

    def __init__(self, task_config: TaskConfig, *, viewer: bool = False) -> None:
        """Open the viewer's first trajectory, which nothing else in this run will open.

        `Viewer._frames` is a deque *of trajectories*, and the only thing that appends one is the
        `_after_reset_idx` hook `Viewer.hook_to_env` installs. `ToyBlocksEnv.__init__` resets
        before it hooks the viewer, so that hook never fires here, and the motion layer
        deliberately does not reset the scene a second time, so `_frames` stays empty and the first
        `_refresh_viewer` -- reached from this server's `record_current_frame`, which every
        simulator step fires -- raises `IndexError: deque index out of range`.

        :param task_config: Configuration specifying which task to load
        :param viewer: Whether to launch a viewer for the simulation environment
        """
        super().__init__(task_config, viewer=viewer)
        if self._viewer is not None and len(self._viewer._frames) == 0:
            self._viewer._frames.append(deque(maxlen=self._viewer.cfg.max_frames_per_trajectory))

        # ToyBlocks re-renders once after a reset. The RTX denoiser keeps several frames of
        # history, so one render leaves ghosts of the blocks' old positions in the first image.
        self.base_env.cfg.num_renders_on_reset = max(
            self.base_env.cfg.num_renders_on_reset, RESET_RERENDERS
        )


class InlineOracleMCPServer(OracleMCPServer):
    """The oracle tool surface, over ToyBlocks's vecgen skills.

    Three things about the surface are decisions rather than mechanism, and are recorded here
    because the surface *is* the abstraction handed to the model.

    - **Placement is two tools, not one.** `stack` stacks on the block nearest the pixel;
      `place_on_table` deprojects the exact pixel onto the table plane and places there. The agent
      states its intent instead of the server inferring block-vs-table from a distance threshold,
      and `place_on_table` honours the point given rather than resampling a nearby free spot.
    - **Pick and place actuate the gripper themselves.** A `pick` approaches, closes and lifts; a
      place positions and opens. `open_gripper`/`close_gripper` are kept for skills that separate
      motion from grasping, and are rarely needed in this loop.
    - **Skills fail loudly.** The oracle can know a plan failed, so it raises `OracleSkillError`
      instead of returning a silently-unchanged frame. `pick` only raises once *every* ranked
      grasp candidate fails to plan, or once the arm has lifted and the block has not.

    The surface is wider than ManiFlow's, which is `pick` and `place`. The two are not reconciled,
    so a cross-condition comparison has to account for it.
    """

    VIEWER_STRIDE = 4
    """Sim steps between viewer pushes: about 6 Hz against the 25 Hz simulation.

    Each push copies every link, block and marker pose off the GPU, stores a frame in the viewer's
    history and sets position and orientation on every viser handle, and the browser applies all of
    it and re-renders. At 1 that runs on every simulated step and the view drags; the recorded video
    is unaffected, since `record_current_frame` still captures every step.
    """

    RESET_SETTLE_STEPS = 200
    """Sim steps run after a scene is built or reset, before it is observed, so the blocks have
    come to rest where the reset put them."""

    RESET_RERENDERS = RESET_RERENDERS
    """Extra renders after those steps, so the RTX denoiser's temporal history holds only the new
    scene; ghosts of the blocks' old positions otherwise survive into the first image."""

    def set_should_stop(self, should_stop: Callable[[], bool]) -> None:
        """Give the motion layer something to ask before every simulated step.

        The page's stop control lives on the other side of the MCP boundary from the arm, and this
        is the one wire between them. Passed as a callable rather than a flag so the caller keeps
        ownership of what "stopped" means; `web_ui/server.py` hands it the channel's property.

        :param should_stop: Called with no arguments, truthy when the robot should halt
        """
        self._motion.should_stop = should_stop

    def settle_and_rerender(self) -> None:
        """Let physics settle and flush the renderer's history. Runs on the simulator's thread."""
        self._motion.step(self.RESET_SETTLE_STEPS)
        sim = self._toyblocks_env.base_env.sim
        for _ in range(self.RESET_RERENDERS):
            sim.render()

    def _build_env(self, config: ToyBlocksConfig) -> ToyBlocksEnv:
        """Build the oracle's environment: viewer trajectory opened, denoiser settled.

        The base names `ToyBlocksEnv` inside `_build_env`, so overriding the hook is how the oracle
        substitutes its environment. Rebinding the module-level name instead would have to cover
        both the base server's namespace and `toyblocks_env`'s, since which one the base reads
        depends on how it imports the name, and would reach every server in the process. A hook
        cannot be missed that way, and it cannot reach a server that did not ask for it.

        :param config: The `ToyBlocksConfig` this server was built with
        :return: The environment wrapper the oracle skills drive
        """
        return OracleToyBlocksEnv(config.task_config, viewer=config.viewer)

    def _build_skill_backend(self) -> None:
        """Stand up the motion backend, then settle the scene it will plan against.

        Settling comes before the base records the first image, which happens as soon as this
        returns: without it the first plan would see blocks mid-fall, and the first image would
        carry ghosts of their spawn poses.

        Nothing here prepares the base's frame state. Settling steps the simulator, and every step
        fires ``on_step=self.record_current_frame``, so that state has to exist before this hook
        runs; `ToyBlocksMCPServer.__init__` opens the video writer and zeroes the counter before
        calling it.
        """
        self._motion = VecgenMotion(
            self._toyblocks_env.env, self.config.oracle, on_step=self.record_current_frame
        )
        self.settle_and_rerender()
        self._toyblocks_env.refresh_observation()

    @property
    def _held_block(self) -> str | None:
        """The block the gripper holds, as the motion layer's engine tracks it."""
        return self._motion.held_block

    @property
    def toyblocks_env(self) -> ToyBlocksEnv:
        """The adapter's environment wrapper, which the base class holds privately."""
        return self._toyblocks_env

    @property
    def table_z(self) -> float:
        """World-frame height of the table's top surface, derived on each use."""
        table_z = find_table_z(self._toyblocks_env)
        logger.debug(f"Placing blocks on a table at z={table_z:.4f} m.")
        return table_z

    def record_current_frame(self, camera_name: str | None = None) -> None:
        """Record a video frame, and push the current state to the viewer.

        Viser only advances what the browser shows when something calls `view_frame`, and these
        skills step PhysX directly rather than going through `ToyBlocksEnv.step`. Without this the
        viewer stays frozen on the boot state for the whole episode.
        """
        super().record_current_frame(camera_name)
        if self.frame_count % self.VIEWER_STRIDE == 0:
            self._toyblocks_env._refresh_viewer()

    def viewer_url(self) -> str | None:
        """Where this scene's viser server is listening, or None without a viewer.

        Asked for rather than assumed, because viser silently takes the next free port when its
        default is busy -- which it can be for a moment after the previous scene's server stopped.
        """
        viewer = self._toyblocks_env._viewer
        if viewer is None:
            return None
        # viser reports its bind address; 0.0.0.0 is not something a browser can dial, and the
        # page knows to swap "localhost" for the host it reached this machine by.
        return viewer.get_server_url().replace("//0.0.0.0:", "//localhost:")

    def dispose(self) -> None:
        """Tear this scene down so another can be built in the same process.

        Stops the viser server first, so the next scene's viewer can take its port; then the base
        `shutdown` exports the video and closes the gym environment. Runs on the simulator's thread.
        """
        viewer = self._toyblocks_env._viewer
        if viewer is not None:
            try:
                viewer._server.stop()
            except Exception:
                logger.exception(
                    "Could not stop the viser server; the next one may take another port."
                )
        self.shutdown()
        logger.info(
            f"Scene '{self.config.task_config.name}' disposed; "
            f"video and transcript in {self.config.output_dir}"
        )

    def seed_for_episode(self, episode_index: int) -> None:
        """Make the next reset's layout a function of the episode index rather than of history.

        The autogen env picks its reset state with `torch.randint` off the ambient device RNG, and
        the oracle's pose generators draw from the same stream, so without this an episode's layout
        depends on how many draws the previous episode happened to make. Two runs of the same task
        at the same seed then diverge from the second episode onward, which is exactly the
        comparison the memory metric needs to hold fixed.

        Seeding here rather than passing `instance_ids` keeps the fix on this side: `env.reset()`
        reaches `_reset_idx` with `instance_ids=None` and there is no argument to thread through it.

        :param episode_index: Which episode of this run is about to start, counting from zero
        """
        import random

        import numpy as np

        seed = self.config.rng_seed * 1_000_003 + episode_index
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        logger.info(f"Seeded episode {episode_index} with {seed}.")

    async def reset_scene_on_sim_thread(self, episode_index: int = 0) -> None:
        """Put the scene back to a fresh episode. Runs on the simulator's thread and never awaits.

        Reached from the page's "reset episode" control through `SimThreadOracleServer.reset_scene`.
        The gym reset draws a fresh reset state from the autogen package, so the blocks come back
        in a new random layout; the motion layer then forgets any held block and its in-flight
        skill, and re-applies the arm gains its skills are tuned for.

        The gripper is *not* opened first. `set_state_dict` writes the whole articulation --
        `write_joint_state_to_sim` takes no joint filter, and both position and velocity targets
        are set -- so the reset restores the fingers along with the arm. Actuating them beforehand
        would only play out a held block falling to the table, several seconds before the arm it
        fell from is reset out from under it.

        Then :meth:`settle_and_rerender` runs before anything reads the scene: right after a reset
        the RTX renderer still shows the blocks where they were, as ghosts, until enough frames
        have gone by, and a first image with ghost blocks in it would mislead the model.
        """
        self.seed_for_episode(episode_index)

        def block_positions() -> dict[str, list[float]]:
            return {
                name: [round(v, 3) for v in self._toyblocks_env.get_block_center(name).tolist()]
                for name in self._toyblocks_env.block_names
            }

        logger.info(f"Block positions before reset: {block_positions()}")
        self._toyblocks_env.env.reset()
        logger.info(f"Block positions after reset: {block_positions()}")
        self._motion.reset()
        self.settle_and_rerender()
        self._toyblocks_env.refresh_observation()

        self._show_fresh_episode_in_viewer()
        self.record_current_frame()
        logger.info("Scene reset to a fresh episode.")

    def _show_fresh_episode_in_viewer(self) -> None:
        """Point viser at the trajectory the reset just opened, so the page shows the new scene.

        `Viewer.hook_to_env` appends a new trajectory on every env reset, but only views the very
        first one; later resets leave the browser on the old trajectory's last frame.
        """
        viewer = self._toyblocks_env._viewer
        if viewer is None:
            return
        slider = viewer._trajectory_id_slider
        before = (slider.value, slider.max, [len(t) for t in viewer._frames][-4:])
        try:
            slider.value = len(viewer._frames) - 1
        except Exception:
            logger.exception("Could not switch the viewer to the new trajectory.")
        self._toyblocks_env._refresh_viewer()
        logger.info(
            f"Viewer after reset: trajectories={len(viewer._frames)} slider(value,max,lens) "
            f"before={before} after=({slider.value}, {slider.max}) "
            f"viewed={viewer.viewed_trajectory_id} "
            f"last_viewed_set={viewer._last_viewed_set}"
        )

    # -- tools ---------------------------------------------------------------------------------

    async def get_detailed_image(self) -> MCPImage:
        """Retrieve the whole scene at the camera's full resolution.

        The image you normally receive is reduced in size; this is the same view at full size, and
        costs several times as much to look at. It is worth that when you want a more faithful
        view of the entire scene at once.

        To identify something small -- a letter or a marking on a block, or which face of a block
        points towards you -- use the zoomed view instead. Even at full resolution a block is a
        small part of this image, so this is usually still too coarse to read such things.

        Locations you identify in this image can be given to any other tool unchanged.
        """
        self._raise_if_shutdown(skill_name="GetDetailedImage")
        return array_to_mcp_image(self._current_frame(self.config.camera_name))

    async def get_zoomed_image(
        self,
        pixel: PixelCoord,
        zoom: Annotated[float, Field(ge=1.0, le=20.0)],
    ) -> MCPImage:
        """Retrieve a magnified view of one part of the current image observation.

        Use this whenever something is too small to identify: reading a letter or a marking on a
        block, or telling which face of a block is pointing towards you. This is the reliable way
        to see fine detail. Moving a block closer to the camera is not: the camera is low and
        distant, so carrying a block across the table barely changes how large it appears.

        The view returned is centred on the point you give.

        Coordinates read off this magnified view are meaningless to other tools. Points for
        picking, placing, rotating and flipping must be read off the standard image observation.

        :param point: Point in the image to centre the magnified view on, in the same coordinates
            as every other tool
        :param zoom: How much larger things should appear than in the standard image. 4 shows a
            quarter of the image's width and height around your point, at four times the size;
            10 magnifies ten times. Higher values magnify further; 1 returns the whole image.
        """
        self._raise_if_shutdown(skill_name="GetZoomedImage")

        frame = self._current_frame(self.config.camera_name)
        window = crop_around(frame, self.convert_obs_pixel_to_camera(pixel), 1.0 / zoom)

        return array_to_mcp_image(resize_to_width(window, self.observation_resolution.width))

    async def pick(self, pixel: PixelCoord) -> MCPImage:
        """Pick up the block at the given location in the current image observation.

        The gripper closes on the block and lifts it, so no separate gripper call is needed.

        Precondition: The gripper must be empty.
        """
        self._raise_if_shutdown(skill_name="Pick")
        if self._held_block is not None:
            raise OracleSkillError(
                f"The gripper is already holding '{self._held_block}'; place that block before "
                "picking up another."
            )

        block_name = self._resolve_block(pixel)
        self._motion.grasp(block_name)

        return self.current_image_obs

    async def stack(self, pixel: PixelCoord) -> MCPImage:
        """Stack the held block onto the block at the given location in the current image.

        The held block is centered on top of the one pointed at, then released.

        Precondition: The gripper must be holding a block.
        """
        self._raise_if_shutdown(skill_name="PlaceOnBlock")
        held_block = self._require_held_block()

        support_name = self._resolve_block(pixel)
        self._motion.place_on_block(held_block, support_name)

        return self.current_image_obs

    async def place_on_table(self, pixel: PixelCoord) -> MCPImage:
        """Place the held block on the table, at the given location in the current image.

        The block is placed at exactly the point given, not at a nearby free spot.

        Precondition: The gripper must be holding a block, and the point must be on the table.
        """
        self._raise_if_shutdown(skill_name="PlaceOnTable")
        held_block = self._require_held_block()

        world_xyz = self._deproject_to_table(pixel)
        self._motion.place_on_table(held_block, world_xyz)

        return self.current_image_obs

    async def rotate(self, pixel: PixelCoord, direction: RotateDirectionArg) -> MCPImage:
        """Rotate the block at the given location in the current image observation.

        The block is turned 90 degrees about the vertical axis, in place, and stays on the table.

        Precondition: The gripper must be empty.
        """
        self._raise_if_shutdown(skill_name="Rotate")

        block_name = self._resolve_block(pixel)
        self._motion.rotate(block_name, direction)

        return self.current_image_obs

    TOP_FACE_ENDS_UP: ClassVar[dict[str, FlipDirection]] = {
        "back": FlipDirection.FORWARD,
        "front": FlipDirection.BACKWARD,
        "left": FlipDirection.LEFT_FLIP,
        "right": FlipDirection.RIGHT_FLIP,
    }
    """The interface's flip names, mapped onto the domain's `flip_dir` values.

    The interface asks where the *top face* ends up; `domain.pddl` names how the block *tips*, so
    the two are inverted for the fore-aft pair. `back` and `front` are settled empirically: in run
    `ai2-colors20260908-152044` a `forward` flip moved front->top and bottom->front, leaving the
    top face at the back. `left` and `right` are mapped by name and are **unverified** -- the
    in-plane experiment compared an observed state against a remembered one twenty calls older.
    """

    async def flip(self, pixel: PixelCoord, direction: FlipDirectionArg) -> MCPImage:
        """Flip the block at the given location in the current image observation.

        The block is tipped onto the adjacent face in the given direction, so the face that was
        pointing up ends up pointing sideways.

        Precondition: The gripper must be empty.
        """
        self._raise_if_shutdown(skill_name="Flip")

        block_name = self._resolve_block(pixel)
        self._motion.flip(block_name, self.TOP_FACE_ENDS_UP[direction])

        return self.current_image_obs

    async def return_home(self) -> MCPImage:
        """Open the gripper, drop anything held, and move the arm back to its resting pose.

        Use this when a skill reports that the arm cannot plan a move from where it is: every
        pick, place, rotate and flip will keep failing until the arm is moved, and this is the
        only tool that can move it, because it drives the joints directly instead of planning.
        It is also worth using to clear the arm out of the camera's view.

        Anything currently held is dropped where it is, so it lands on the table rather than
        being carried; pick it up again afterwards if it is still needed.
        """
        self._raise_if_shutdown(skill_name="ReturnHome")
        self._motion.return_home()
        return self.current_image_obs

    async def toggle_gripper(self) -> MCPImage:
        """Toggle the gripper between an 'open' and a 'closed' state.

        The picking and placing skills operate the gripper themselves, so this is rarely needed.
        """
        self._raise_if_shutdown(skill_name="ToggleGripper")

        if self._motion.gripper_state == "open":
            self._motion.close_gripper()
        else:
            self._motion.open_gripper()

        return self.current_image_obs

    async def open_gripper(self) -> MCPImage:
        """Open the gripper, releasing anything it holds."""
        self._raise_if_shutdown(skill_name="OpenGripper")
        self._motion.open_gripper()
        return self.current_image_obs

    async def close_gripper(self) -> MCPImage:
        """Close the gripper, grasping whatever lies between its fingers."""
        self._raise_if_shutdown(skill_name="CloseGripper")
        self._motion.close_gripper()
        return self.current_image_obs

    # -- internals -----------------------------------------------------------------------------

    def _current_frame(self, camera_name: str) -> np.ndarray:
        """Retrieve the current image, re-reading the observation from the simulator first.

        These skills step the simulator directly rather than through `ToyBlocksEnv.step`, so the
        environment's cached observation is otherwise stale.
        """
        self._toyblocks_env.refresh_observation()
        return super()._current_frame(camera_name)

    def _deproject_to_table(self, pixel: PixelCoord) -> np.ndarray:
        """Intersect the camera ray through a pointed pixel with the table plane.

        :param pixel: Pixel in the observation the agent asked to place at
        :return: The world-frame point on the table plane, shape (3,)
        :raises OracleSkillError: If the ray is parallel to, or points away from, the plane
        """
        camera_model = self._toyblocks_env.read_camera_model(camera_name=self.config.camera_name)
        # The camera model carries the render's intrinsics, so the pixel has to be in the
        # render's raster, not the downscaled one the agent read it off.
        origin_w, direction_w = camera_model.pixel_to_ray(self.convert_obs_pixel_to_camera(pixel))

        if abs(direction_w[2]) < 1e-6:
            raise OracleSkillError("That point does not lie on the table; try another.")

        distance_m = (self.table_z - origin_w[2]) / direction_w[2]
        if distance_m <= 0:
            raise OracleSkillError("That point does not lie on the table; try another.")

        return origin_w + distance_m * direction_w

    def _resolve_block(self, pixel: PixelCoord) -> str:
        """Identify which block the agent pointed at.

        :param pixel: Raw pixel in the observation the model was shown
        :return: Name of the block whose projected center is nearest to that point
        :raises OracleSkillError: If no block projects close enough to the given point
        """
        held = frozenset() if self._held_block is None else frozenset({self._held_block})

        block_name = resolve_block(
            self._toyblocks_env,
            self.convert_obs_pixel_to_camera(pixel),
            camera_name=self.config.camera_name,
            config=self.config.oracle,
            exclude=held,
        )
        if block_name is None:
            # Deliberately says nothing more. Reporting the miss distance and the tracked block
            # positions hands the agent privileged state it cannot read out of the image, and
            # paying that out only on a miss makes "probe anywhere, read the answer off the error"
            # a cheaper strategy than looking. The pixel echoed back is the caller's own, in the
            # raster it named it in -- not the render-space pixel `convert_obs_pixel_to_camera`
            # produced, which is twice the size and would read as the arm having misheard the
            # request.
            raise OracleSkillError(
                f"No block found at ({pixel.u},{pixel.v}). Point at a block visible in the image."
            )

        return block_name

    def _require_held_block(self) -> str:
        """Return the name of the held block.

        :raises OracleSkillError: If the gripper is not currently holding a block
        """
        if self._held_block is None:
            raise OracleSkillError("The gripper is not holding a block; pick one up first.")
        return self._held_block


class SimThreadOracleServer(InlineOracleMCPServer):
    """Runs every registered skill on the simulator's thread.

    Isaac must be stepped from the thread that created it, so the simulator runs on the main thread
    and the agentic loop on a second one. This mirrors `SimThreadMCPServer` in
    `toyblocks_runner._boot_mcp_server`, which cannot be imported: it is defined inside that
    function and closed over a `SkillBackend` match that resolves to the `src` stub server. Stepping
    PhysX from the wrong thread crashes rather than raising, so the body is kept identical
    rather than restructured.
    """

    def __init__(self, dispatcher: SimulatorDispatcher, *args: Any, **kwargs: Any) -> None:
        """Initialize the simulator-thread MCP server."""
        self._dispatcher = dispatcher  # Set before super().__init__() registers the MCP tools
        self.registered_tools: list[str] = []  # Likewise; the tools append to it
        super().__init__(*args, **kwargs)

    def _register_tool(self, title: str, skill: Callable) -> None:
        """Register an MCP tool whose body will be executed on the simulator's thread.

        A stop raised inside the skill is caught here rather than escaping as a tool error,
        because a stopped skill is not a failed one: the arm is somewhere legitimate, and what the
        model needs is the scene as it now stands. The next skill plans from the live scene, so
        nothing has to be re-read first.
        """
        import asyncio
        import functools

        dispatcher = self._dispatcher

        @functools.wraps(skill)
        async def dispatched(*args: Any, **kwargs: Any) -> object:
            async def body() -> object:
                try:
                    return await skill(*args, **kwargs)
                except SkillStopped:
                    logger.info(f"'{title}' stopped by a human.")
                    self.record_current_frame()
                    return self.current_image_obs

            return await asyncio.wrap_future(dispatcher.submit(body()))

        self.registered_tools.append(title)
        super()._register_tool(title, dispatched)

    async def reset_scene(self, episode_index: int = 0) -> None:
        """Reset the scene on the simulator's thread. `web_ui/server.py` finds this by name.

        :param episode_index: Which episode of this run is about to start, counting from zero
        """
        import asyncio

        await asyncio.wrap_future(
            self._dispatcher.submit(self.reset_scene_on_sim_thread(episode_index))
        )
