"""The configured skill server with the two simulator-state tools added.

`build_server_class` resolves the simulator-thread server the skill module exposes for the
configured backend (the way `runner_bootstrap.run_episode` does) and derives a class from it with
`SimStateToolsMixin` in front, so the state tools work on either backend. Importing the skill
module boots Isaac, so call this only once the run's configuration is built.

The mixin inherits `SimulatorToolsAgent`, so the server's `_register_tool` finds that declaration
as the furthest ancestor: the model sees the ABC's docstrings, and registration fails if these
signatures drift from it. Docstrings here are developer notes only.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from mcp.server.mcpserver import Image as MCPImage  # noqa: TC002 (used for MCP tool registration)

from agentic_robotics.adapters.toyblocks import runner_bootstrap as boot
from agentic_robotics.adapters.toyblocks.leap.sim_state import read_sim_state, write_sim_state
from agentic_robotics.adapters.toyblocks.leap.simulator_tools_agent import SimulatorToolsAgent
from agentic_robotics.core.states import SimState  # noqa: TC001 (used for MCP tool registration)

if TYPE_CHECKING:
    from types import ModuleType

    from agentic_robotics.adapters.toyblocks.config import ToyBlocksConfig


class SimStateToolsMixin(SimulatorToolsAgent):
    """Adds get_sim_state and set_sim_state to a simulator-thread `ToyBlocksMCPServer`."""

    # Provided by the server class this mixes into
    _dispatcher: Any
    registered_tools: list[str]
    mcp: Any
    config: Any
    _toyblocks_env: Any

    def _register_skills(self) -> None:
        super()._register_skills()  # type: ignore[misc]
        self._register_tool(title="Get Simulator State", skill=self.get_sim_state)  # type: ignore[attr-defined]
        self._register_tool(title="Set Simulator State", skill=self.set_sim_state)  # type: ignore[attr-defined]

    async def get_sim_state(self) -> tuple:
        """The state as JSON text, then the current image observation (see the ABC)."""
        self._raise_if_shutdown(skill_name="Get Simulator State")  # type: ignore[attr-defined]
        state = read_sim_state(self._toyblocks_env)
        return (state.model_dump_json(indent=1), self.current_image_obs)  # type: ignore[attr-defined]

    async def set_sim_state(self, state: SimState) -> MCPImage:
        """Write the state, let physics settle, and return the new image (see the ABC)."""
        self._raise_if_shutdown(skill_name="Set Simulator State")  # type: ignore[attr-defined]
        write_sim_state(self._toyblocks_env, state)
        self.record_current_frame()  # type: ignore[attr-defined]
        return self.current_image_obs  # type: ignore[attr-defined]


ALLOW_MISMATCH_ENV_VAR = "AGENTIC_POOL_ALLOW_DECLARATION_MISMATCH"
"""Set to 1 to let a pool register a tool whose signature differs from its declaration, logging
the difference. The LEAP declarations have no `steps` argument, which the ManiFlow tools take, so
a ManiFlow pool fails the declaration check without this, and the ManiFlow skill probe
(`maniflow_skill_probe.py`) needs `steps` to read a skill every 25 steps. Off by default, so an
agent-facing pool still fails on drift."""


def tolerate_declaration_mismatch() -> None:
    """Make the servers' declaration check log a mismatch instead of raising.

    Replaces `assert_matches_declaration` in the module whose `_register_tool` calls it, so every
    wrapper the server class adds on the way down (the simulator-thread dispatch above all) still
    applies. Re-registering the bare skill after a failed check would skip that dispatch, and the
    tool would step Isaac from the MCP thread and hang.
    """
    from agentic_robotics.adapters.toyblocks import toyblocks_mcp_server as module

    strict = module.assert_matches_declaration

    def lenient(*args: Any, **kwargs: Any) -> None:
        try:
            strict(*args, **kwargs)
        except TypeError as exc:
            logging.getLogger(__name__).warning(
                f"Registering despite a declaration mismatch: {exc}"
            )

    module.assert_matches_declaration = lenient


def build_server_class(skills: ModuleType, config: ToyBlocksConfig) -> type:
    """The configured backend's simulator-thread server, with the state tools mixed in."""
    base = boot.sim_thread_server_class(skills, config)
    if os.environ.get(ALLOW_MISMATCH_ENV_VAR) == "1":
        tolerate_declaration_mismatch()
    return type(f"Leap{base.__name__}", (SimStateToolsMixin, base), {})


class FrameCountToolMixin:
    """Adds get_frame_count to a `ToyBlocksMCPServer`: the pool's video frame counter.

    The server appends one frame to the run directory's `video.mp4` per policy step and per
    state write and keeps the count in `frame_count`; the agent runners sample it around each
    tool call through `metadata_callback`, which a client driving the pool over MCP cannot do.
    Reading the counter before and after a call gives the frames that call wrote, so a clip can
    be cut per call. Served only by `agentic-robotics pool --frame-count-tool`: the LEAP baselines'
    declaration check refuses a pool that serves a tool they do not declare.
    """

    frame_count: int

    def _register_skills(self) -> None:
        super()._register_skills()  # type: ignore[misc]
        self._register_tool(title="Get Frame Count", skill=self.get_frame_count)  # type: ignore[attr-defined]

    async def get_frame_count(self) -> str:
        """How many frames this simulator's video holds so far, as an integer in text.

        One frame is written at boot, one per state write, and one per policy step a skill runs.
        Read it before and after a call: the frames the call wrote are [before, after).
        """
        return str(self.frame_count)


def build_frame_counted_server_class(skills: ModuleType, config: ToyBlocksConfig) -> type:
    """`build_server_class`'s result with `FrameCountToolMixin` in front."""
    base = build_server_class(skills, config)
    return type(f"FrameCounted{base.__name__}", (FrameCountToolMixin, base), {})
