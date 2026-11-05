"""Define an abstract base class for an agent that can get and set the simulator state."""

from abc import ABC, abstractmethod

from mcp.server.mcpserver import Image as MCPImage

from agentic_robotics.core.states import SimState


class SimulatorToolsAgent(ABC):
    """An agent with tools to get and set the simulator state."""

    @abstractmethod
    async def get_sim_state(self) -> tuple:
        """Retrieve the current ground-truth simulator state and image observation.

        Returns the robot's base pose, joint values, and the pose of every block, as JSON
        in exactly the form `set_sim_state` accepts. Positions are in meters relative to the
        world frame; orientations are unit quaternions as (w, x, y, z).

        :return: Tuple of (JSON-serialized simulator state, image observation)
        """
        ...

    @abstractmethod
    async def set_sim_state(self, state: SimState) -> MCPImage:
        """Set the simulator to the given state, let physics settle, and return the new image.

        Expects the robot's base pose, joint values, and the pose of every block. Positions
        are in meters relative to the world frame; orientations are unit quaternions as
        (w, x, y, z). Blocks placed at overlapping poses are pushed apart as the scene settles,
        so the settled state can differ from the given state.

        :param state: State to which the simulator is set
        :return: Updated image observation after simulator physics have settled
        """
        ...
