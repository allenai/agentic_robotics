"""The ManiFlow skill surface, as a skill module `agentic-robotics` can boot.

`inline_oracle.py` is the other one. Both expose a module-level simulator-thread server, which is
what `runner_bootstrap.sim_thread_server_class` looks up: `config.skill_backend` picks the name,
`MANIFLOW` resolves to `SimThreadManiFlowServer` here and `ORACLE` to `SimThreadOracleServer`
there.

**Importing this module boots Isaac Sim**, because `ManiFlowMCPServer` imports the adapter. As with
`inline_oracle`, whatever imports it passes the run's `ToyBlocksConfig` to the server it builds.

**It needs a policy server already listening** on `config.maniflow.policy_host:policy_port`; start
it with `agentic-robotics serve`. Constructing the server connects, reads the language mode out of
the server's metadata, and sets the arm's PD gains to the ones `config.maniflow` carries (the gains
the checkpoint was trained under), so a missing server fails at construction rather than at the
first skill.

The tool surface is what `MotorSkillsAgent` and `PerceptionToolsAgent` declare, which
`ManiFlowMCPServer` inherits: `pick`, `stack`, `place_on_table`, `rotate` and `flip`, the five
policy skills it registers, plus `get_image_observation` from the base, plus the four registered
here, `open_gripper`, `close_gripper` (`pick` has a behavioral open-gripper precondition and a
`flip` leaves the gripper closed, so without them a closed gripper is unrecoverable),
`return_home` (every policy skill leaves the arm over the block it acted on, in the camera's
view; the skills themselves do not need the arm at home) and `get_zoomed_image` (the oracle's
body; a crop of the current frame). The server's
`_register_tool` serves the ABC's docstrings and refuses a signature that drifts from them. The
policy skills are the ones there are checkpoints for, and the step budget each one rolls out for
is a percentile of that skill's segments in the training dataset rather than a number that can be
read off the scene; the other four run no policy.

What this does not carry, because `agentic-robotics episode` does not ask for it: `reset_scene`,
which the menu and page modes want; bring it up on `episode` first.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections import deque
from typing import TYPE_CHECKING, Annotated, Any

import torch
from mcp.server.mcpserver import Image as MCPImage  # noqa: TC002 (used for MCP tool registration)
from pydantic import Field

from agentic_robotics.adapters.toyblocks.maniflow_mcp_server import ManiFlowMCPServer
from agentic_robotics.adapters.toyblocks.toyblocks_env import ToyBlocksEnv
from agentic_robotics.core.images import (
    PixelCoord,
    array_to_mcp_image,
    crop_around,
    resize_to_width,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from agentic_robotics.adapters.toyblocks.config import TaskConfig, ToyBlocksConfig
    from agentic_robotics.adapters.toyblocks.toyblocks_runner import SimulatorDispatcher

logger = logging.getLogger(__name__)

RETURN_HOME_STEPS = 60
"""Control steps the return-home ramp is spread over (the oracle's joint ramp is 60 steps)."""

SETTLE_MAX_STEPS = 120
"""Cap on the extra steps a tool holds its final target while waiting for the joints to settle."""

SETTLE_JOINT_TOLERANCE = 2e-3
"""Joints that move less than this (radians) in one control step count as settled.

Both tools hold their final target until nothing moves, because their fixed step counts stop
short: the ramp alone leaves the arm short of home when it starts far away (the PD-controlled
joints lag their targets, and the next tool then holds the lagging pose), and the gripper's six
commanded steps moved a closed finger joint only from 0.836 to 0.831 when measured on
2026-10-04, while a state write opens it at once."""

GRIPPER_CLOSED_THRESHOLD = 0.2
"""A `finger_joint` reading above this is a closed gripper: open reads about 0, closed on a block
0.4 to 0.55, closed on nothing 0.72 or more (the 2026-09-29 gripper check)."""

GRIPPER_JOINT = "finger_joint"


class ManiFlowToyBlocksEnv(ToyBlocksEnv):
    """A `ToyBlocksEnv` with the viewer's first trajectory opened.

    `Viewer._frames` is a deque *of trajectories*, and the only thing that appends one is the
    `_after_reset_idx` hook `Viewer.hook_to_env` installs. `ToyBlocksEnv.__init__` resets before
    it hooks the viewer, so that hook never fires, `_frames` stays empty, and the first
    `_refresh_viewer` raises `IndexError: deque index out of range`. The ManiFlow backend reaches
    it through `ToyBlocksEnv.step`, which indexes the deque to show the frame it just recorded, so
    the crash lands on the first step of the first skill rather than anywhere near the cause.

    `inline_oracle.OracleToyBlocksEnv` opens the deque for the same reason. The fix belongs in
    `ToyBlocksEnv` itself, which would remove both copies.
    """

    def __init__(self, task_config: TaskConfig, *, viewer: bool = False) -> None:
        """Open the viewer's first trajectory, which nothing else in this run will open.

        :param task_config: Configuration specifying which task to load
        :param viewer: Whether to launch a viewer for the simulation environment
        """
        super().__init__(task_config, viewer=viewer)
        if self._viewer is not None and len(self._viewer._frames) == 0:
            self._viewer._frames.append(deque(maxlen=self._viewer.cfg.max_frames_per_trajectory))


class SimThreadManiFlowServer(ManiFlowMCPServer):
    """Runs every registered skill on the simulator's thread.

    Isaac must be stepped from the thread that created it, so the simulator runs on the main thread
    and the agentic loop on a second one. Every tool is wrapped at registration rather than
    overridden one at a time: an override that is forgotten still works, but runs its body on the
    MCP client's thread, and stepping PhysX from the wrong thread crashes rather than raising.
    """

    def _build_env(self, config: ToyBlocksConfig) -> ToyBlocksEnv:
        """Build the environment with the viewer's first trajectory already open.

        :param config: Run-specific configuration parameters
        :return: The environment this server's skills act in
        """
        return ManiFlowToyBlocksEnv(config.task_config, viewer=config.viewer)

    def __init__(self, dispatcher: SimulatorDispatcher, *args: Any, **kwargs: Any) -> None:
        """Initialize the simulator-thread ManiFlow server.

        :param dispatcher: Dispatcher owning the thread holding the simulator
        """
        self._dispatcher = dispatcher  # Set before super().__init__() registers the MCP tools
        self.registered_tools: list[str] = []  # Likewise; the tools append to it
        super().__init__(*args, **kwargs)

    def record_current_frame(self, camera_name: str | None = None) -> None:
        """Record a video frame, and keep the viewer on the newest trajectory while a browser
        watches.

        Every step here goes through `ToyBlocksEnv.step`, whose hooks record a viewer frame into
        the newest trajectory and then show the latest frame of the trajectory viser is viewing.
        Viser stays on the first, empty one: scene setup's state write opens a new trajectory, and
        nothing moves the view to it, so without this the page shows the robot's default joint
        angles for the whole run. The switch waits for a connected browser, so an episode nobody
        watches steps exactly as it would without a viewer push.
        """
        super().record_current_frame(camera_name)
        viewer = self._toyblocks_env._viewer
        if viewer is None or not viewer._server.get_clients():
            return
        newest = len(viewer._frames) - 1
        if viewer.viewed_trajectory_id != newest:
            viewer._trajectory_id_slider.value = newest
            logger.info(f"Viewer switched to trajectory {newest}, the scene being stepped.")

    def _register_tool(self, title: str, skill: Callable) -> None:
        """Register an MCP tool whose body will be executed on the simulator's thread."""
        dispatcher = self._dispatcher

        @functools.wraps(skill)
        async def dispatched(*args: Any, **kwargs: Any) -> object:
            try:
                return await asyncio.wrap_future(dispatcher.submit(skill(*args, **kwargs)))
            except Exception:
                # The MCP SDK turns this into a one-line tool error with no traceback; a
                # served simulator's log is otherwise the only place the cause can be read.
                logger.exception(f"Skill '{title}' raised on the simulator's thread.")
                raise

        self.registered_tools.append(title)
        super()._register_tool(title, dispatched)

    def _register_skills(self) -> None:
        """Register the ManiFlow skills, then the four tools declared by the ABCs but not
        implemented by `ManiFlowMCPServer`."""
        super()._register_skills()
        self._register_tool(title="Get Zoomed-In Image Observation", skill=self.get_zoomed_image)
        self._register_tool(title="Return Home", skill=self.return_home)
        self._register_tool(title="Open Gripper", skill=self.open_gripper)
        self._register_tool(title="Close Gripper", skill=self.close_gripper)

    async def get_zoomed_image(
        self, pixel: PixelCoord, zoom: Annotated[float, Field(ge=1.0, le=20.0)]
    ) -> MCPImage:
        """A crop of the current frame around `pixel`, resized to the observation width.

        The same body as the oracle's: no backend dependency, just the camera frame.
        """
        self._raise_if_shutdown(skill_name="Get Zoomed-In Image Observation")
        frame = self._current_frame(self.config.camera_name)
        window = crop_around(frame, self.convert_obs_pixel_to_camera(pixel), 1.0 / zoom)
        return array_to_mcp_image(resize_to_width(window, self.observation_resolution.width))

    async def return_home(self) -> MCPImage:
        """Ramp the arm's joint targets to the environment's reset pose; gripper unchanged."""
        self._raise_if_shutdown(skill_name="Return Home")
        self._move_arm_home()
        return self.current_image_obs

    def _move_arm_home(self) -> None:
        """Drive the seven arm joints to the articulation's default (reset) positions.

        Goes through `ToyBlocksEnv.step` the way `_command_gripper` does: a linear ramp of
        absolute arm joint targets from where the arm is to the default joint positions Isaac Lab
        holds for the articulation (the autogen env's reset pose), with the binary gripper
        command chosen to keep the gripper as it is, closed when `finger_joint` reads closed. No
        planner and no policy, so this works from any configuration a policy left the arm in.
        """
        robot = self._toyblocks_env.base_env.scene.articulations["robot"]
        joint_pos = robot.data.joint_pos
        start = joint_pos[:, :7].clone()  # The seven arm joints come first, then the gripper's six
        home = robot.data.default_joint_pos[:, :7]
        finger = joint_pos[:, robot.joint_names.index(GRIPPER_JOINT)]
        gripper = torch.where(
            finger > GRIPPER_CLOSED_THRESHOLD,
            torch.tensor(-1.0, device=start.device, dtype=start.dtype),
            torch.tensor(1.0, device=start.device, dtype=start.dtype),
        ).unsqueeze(-1)
        for step in range(1, RETURN_HOME_STEPS + 1):
            arm = start + (home - start) * (step / RETURN_HOME_STEPS)
            self._toyblocks_env.step(torch.cat([arm, gripper], dim=-1))
            self.record_current_frame()
        steps = self._hold_until_settled(torch.cat([home, gripper], dim=-1))
        logger.info(f"Return home: settled after {steps} extra step(s) at the home target.")

    async def open_gripper(self) -> MCPImage:
        """Open the robot gripper, releasing anything it holds.

        :return: Updated image observation after the skill has finished
        """
        self._raise_if_shutdown(skill_name="Open Gripper")
        self._command_gripper(close=False)
        return self.current_image_obs

    async def close_gripper(self) -> MCPImage:
        """Close the robot gripper, grasping anything between its fingers.

        :return: Updated image observation after the skill has finished
        """
        self._raise_if_shutdown(skill_name="Close Gripper")
        self._command_gripper(close=True)
        return self.current_image_obs

    def _command_gripper(self, *, close: bool) -> None:
        """Hold the arm where it is and drive the gripper for `config.maniflow.gripper_steps` steps.

        Goes through `ToyBlocksEnv.step` the way a policy action does: seven absolute arm joint
        targets, here the current positions, and the binary gripper command, negative to close
        and positive to open (Isaac Lab's `BinaryJointPositionAction`, as
        `toyblocks.utils.tamp_dataset.action_tensor` documents). No policy is called, so the
        ManiFlow conditioning is left as the last skill set it. The configured step count does not
        finish the motion on its own, so the command is then held until the joints settle
        (`SETTLE_JOINT_TOLERANCE`); the oracle settles its gripper over 24 simulation steps.
        """
        joint_pos = self._toyblocks_env.base_env.scene.articulations["robot"].data.joint_pos
        arm = joint_pos[:, :7]  # The seven arm joints come first, then the gripper's six
        gripper = torch.full(
            (arm.shape[0], 1), -1.0 if close else 1.0, device=arm.device, dtype=arm.dtype
        )
        action = torch.cat([arm, gripper], dim=-1)
        for _ in range(self.config.maniflow.gripper_steps):
            self._toyblocks_env.step(action)
            self.record_current_frame()
        steps = self._hold_until_settled(action)
        logger.info(
            f"Gripper {'closed' if close else 'opened'}: settled after {steps} extra step(s)."
        )

    def _hold_until_settled(self, action: torch.Tensor) -> int:
        """Keep stepping with `action` until no joint moves more than the tolerance, or the cap.

        :return: How many extra steps were taken
        """
        robot = self._toyblocks_env.base_env.scene.articulations["robot"]
        previous = robot.data.joint_pos.clone()
        for step in range(1, SETTLE_MAX_STEPS + 1):
            self._toyblocks_env.step(action)
            self.record_current_frame()
            current = robot.data.joint_pos
            if float((current - previous).abs().max()) < SETTLE_JOINT_TOLERANCE:
                return step
            previous = current.clone()
        logger.warning(
            f"Joints still moving after {SETTLE_MAX_STEPS} extra steps; holding target no longer."
        )
        return SETTLE_MAX_STEPS

    def viewer_url(self) -> str | None:
        """Where this scene's viser server is listening, or None without a viewer.

        Asked for rather than assumed, because viser silently takes the next free port when its
        default is busy.
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
        `shutdown` exports the video and closes the gym environment. Runs on the simulator's
        thread. Same body as `inline_oracle.InlineOracleMCPServer.dispose`, which touches nothing
        backend-specific -- both belong in `ToyBlocksMCPServer`.
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
            f"Scene '{self.config.task_config.name}' disposed; video and transcript in "
            f"{self.config.output_dir}"
        )
