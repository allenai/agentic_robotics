"""Define an abstract base class for an agent that can spawn subagents."""

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field

from agentic_robotics.core.states import SimState

MAX_EPISODE_TOOL_CALLS = 100
"""Maximum number of tool calls that can be budgeted for a subagent episode."""


class Episode(BaseModel):
    """One episode, to be executed by an agent."""

    prompt: str = Field(
        description=(
            "Instructions for the episode's agent: what to do and what to report back. "
            "It starts with an empty conversation history, so include everything it needs to know."
        )
    )

    tools: list[str] = Field(
        min_length=1, description="Names of the tools the episode's agent may use."
    )

    max_tool_calls: int = Field(
        ge=1,
        le=MAX_EPISODE_TOOL_CALLS,
        description=f"Tool call budget for the episode's agent, at most {MAX_EPISODE_TOOL_CALLS}.",
    )

    initial_state: SimState | str = Field(
        description=(
            "Initial simulator state of the episode, specified as either a structured "
            "SimState or as a workspace path to a file holding the JSON for a state."
        )
    )


EpisodeOutcomeStatus = Literal["finished", "budget_exceeded", "error"]


class EpisodeResult(BaseModel):
    """Output from a single episode run by a subagent."""

    trial: int
    status: EpisodeOutcomeStatus

    final_message: str
    """Final message from the subagent at the end of the episode."""

    log_path: str
    """Relative path to the episode log."""

    tool_calls_used: int
    cost_usd: float
    model: str
    """Name of the model used for the subagent."""


class RecursiveAgent(ABC):
    """An agent that can spawn subagents."""

    @abstractmethod
    async def spawn_subagents(self, episodes: list[Episode]) -> str:
        """Run one or more episodes using subagents at the same time, each in its own simulator.

        This tool will wait for all subagents to finish before returning.

        :param episodes: List of episodes to run (prompt, tools, max_tool_calls, initial_state)
        :return: String of a JSON dump summarizing the result of each subagent episode
        """
        ...
