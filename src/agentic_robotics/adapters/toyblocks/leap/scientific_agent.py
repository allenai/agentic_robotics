"""Define an abstract base class for an agent with tools based on the scientific method."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from .recursive_agent import Episode

MAX_TRIALS = 8
"""Maximum number of trials per experiment."""


class PlannedTrial(Episode):
    """A labeled subagent episode, functioning as a trial within an experiment."""

    label: str = Field(description="Short name for this trial, used in the results table.")


class ExperimentPlan(BaseModel):
    """A plan for an experiment based on the scientific method."""

    hypothesis: str = Field(
        min_length=1, description="A testable statement or prediction motivating the experiment."
    )
    success_criteria: str = Field(
        min_length=1,
        description=(
            "Condition(s) necessary to provide evidence that the hypothesis holds, stated "
            "so as to be classifiable based on observed experiment data (states, images)."
        ),
    )
    trials: list[PlannedTrial] = Field(
        min_length=1, max_length=MAX_TRIALS, description="Trials to be run, concurrently."
    )


class ExperimentRecord(BaseModel):
    """The output from each `plan_experiment` call."""

    id: int
    planned_at: str
    plan: ExperimentPlan
    run_at: str | None = None


class ScientificAgent(ABC):
    """An agent with tools based on the scientific method."""

    @abstractmethod
    async def plan_experiment(self, plan: ExperimentPlan) -> str:
        """Record an experiment plan before running it.

        :param plan: Planned experiment, including a hypothesis, success criteria, and planned
            trials, each to be executed by a subagent in an independent simulator environment.
        :return: String of a JSON dump recording the plan, with a system-assigned experiment ID
        """
        ...

    @abstractmethod
    async def run_experiment(self, id: int) -> str:
        """Run a planned experiment's trials, all at once, and wait for them.

        Use this tool to execute motor skills on the robot; do not call them directly.
        Each trial's agent is told the experiment's hypothesis and success criteria.

        :param id: System-assigned experiment ID, from the output of `plan_experiment`
        :return: Each trial's final message, log path, tool calls used, and dollars spent
        """
        ...
