"""The Harness baseline's experiment tools: plan_experiment and run_experiment.

The tools are `ScientificAgent.plan_experiment` and `run_experiment` from the ABCs, which fix
their names, docstrings and the `ExperimentPlan` schema: a hypothesis, success criteria, and the
trials (each a subagent `Episode` with a label). The Harness agent keeps the file, docstring,
state and perception tools and context clearing; its motor skills are registered with their
docstrings but refuse every call (the composed server does that, with the refusal text from
`NoOpMotorSkillsAgent`), and it has no `spawn_subagents`. What replaces it is the same machinery,
the subagent spawner and the pool, behind a structure the agent has to fill in before anything
runs:

- **plan_experiment** validates the plan (every trial's tools from the permitted set, exactly one
  readable starting scene per trial, distinct labels) and writes it as
  `logs/experiments/<id>/plan.json`. It runs nothing.
- **run_experiment** runs a planned experiment's trials once, as the Naive agent's
  `spawn_subagents` would, with each trial's prompt wrapped in a preamble that names the
  hypothesis and the criteria and asks the trial to report against them. Results go to
  `logs/experiments/<id>/results.json` and each trial's rendered log beside it, all readable
  with `read_file`, which is why there is no separate log-review tool.

What this encodes is a research decision, not mechanism: the harness forces the
hypothesis-criteria-trials shape and records it, and a trial is still an agent. The paper's own
documents win where they specify the tools differently.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from mcp.server.mcpserver.exceptions import ToolError

from agentic_robotics.adapters.toyblocks.leap import messages
from agentic_robotics.adapters.toyblocks.leap.declarations import register_declared_tool
from agentic_robotics.adapters.toyblocks.leap.recursive_agent import Episode
from agentic_robotics.adapters.toyblocks.leap.scientific_agent import (
    ExperimentPlan,
    ExperimentRecord,
    PlannedTrial,
    ScientificAgent,
)

if TYPE_CHECKING:
    from pathlib import Path

    from mcp.server.mcpserver import MCPServer

    from agentic_robotics.adapters.toyblocks.leap.subagents import SubagentSpawner
    from agentic_robotics.adapters.toyblocks.leap.workspace import Workspace

logger = logging.getLogger("agentic_robotics.examples.leap.harness")

EXPERIMENTS_DIR = "experiments"
"""Under the workspace's logs directory: one directory per experiment."""


def trial_episode(plan: ExperimentPlan, trial: PlannedTrial) -> Episode:
    """The episode a trial runs as: its prompt behind the experiment's preamble."""
    return Episode(
        prompt=messages.harness_trial_preamble(trial.label, plan.hypothesis, plan.success_criteria)
        + trial.prompt,
        tools=trial.tools,
        max_tool_calls=trial.max_tool_calls,
        initial_state=trial.initial_state,
    )


class ExperimentHarness:
    """Plans and runs experiments for one parent agent."""

    def __init__(self, *, spawner: SubagentSpawner, workspace: Workspace) -> None:
        """Initialize the harness.

        :param spawner: The subagent spawner the experiments' trials run through
        :param workspace: The parent's workspace; experiments live under its logs
        """
        self._spawner = spawner
        self._workspace = workspace
        self._dir = workspace.logs_dir / EXPERIMENTS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self.experiments_run = 0

    def _path(self, experiment_id: int) -> Path:
        return self._dir / f"{experiment_id:03d}"

    def _relative(self, experiment_id: int) -> str:
        return self._workspace.relative(self._path(experiment_id))

    def _load(self, experiment_id: int) -> ExperimentRecord:
        path = self._path(experiment_id) / "plan.json"
        if not path.is_file():
            raise ToolError(
                f"No experiment {experiment_id}. Experiments are recorded under "
                f"logs/{EXPERIMENTS_DIR}/; call list_files to see them."
            )
        return ExperimentRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def _save(self, record: ExperimentRecord) -> None:
        path = self._path(record.id)
        path.mkdir(parents=True, exist_ok=True)
        (path / "plan.json").write_text(record.model_dump_json(indent=1), encoding="utf-8")

    def _ids(self) -> list[int]:
        return sorted(int(p.name) for p in self._dir.iterdir() if p.is_dir() and p.name.isdigit())

    def plan(self, plan: ExperimentPlan) -> ExperimentRecord:
        """Validate and record a plan; nothing runs.

        :raises ToolError: If a trial's tools or starting scene are invalid, or labels repeat
        """
        self._spawner.validate(list(plan.trials))
        labels = [trial.label for trial in plan.trials]
        if len(set(labels)) != len(labels):
            raise ToolError("Trial labels must be distinct.")
        ids = self._ids()
        record = ExperimentRecord(
            id=(ids[-1] + 1) if ids else 0,
            planned_at=datetime.now().astimezone().isoformat(),
            plan=plan,
        )
        self._save(record)
        return record

    async def run(self, experiment_id: int) -> list[dict[str, Any]]:
        """Run a planned experiment's trials once and record the results.

        :return: One row per trial: its label and its `EpisodeResult` fields
        :raises ToolError: If the experiment is unknown or has already run
        """
        record = self._load(experiment_id)
        if record.run_at is not None:
            raise ToolError(
                f"Experiment {experiment_id} already ran at {record.run_at}; plan a new one to "
                "repeat it, so every run is recorded against its own plan."
            )
        episodes = [trial_episode(record.plan, trial) for trial in record.plan.trials]
        results = await self._spawner.run_batch(episodes, batch_dir=self._path(experiment_id))
        record.run_at = datetime.now().astimezone().isoformat()
        self._save(record)
        self.experiments_run += 1
        table = [
            {"label": trial.label, **result.model_dump()}
            for trial, result in zip(record.plan.trials, results, strict=True)
        ]
        (self._path(experiment_id) / "results.json").write_text(
            json.dumps(table, indent=1), encoding="utf-8"
        )
        return table


def register_harness_tools(server: MCPServer, harness: ExperimentHarness) -> list[str]:
    """Register the two experiment tools on `server` as `ScientificAgent` declares them.

    :return: The tools' names, for allowlists
    """

    async def plan_experiment(plan: ExperimentPlan) -> str:
        record = harness.plan(plan)
        return record.model_dump_json(indent=1)

    async def run_experiment(id: int) -> str:  # The ABC names it `id`
        return json.dumps(await harness.run(id), indent=1)

    return [
        register_declared_tool(server, ScientificAgent, "plan_experiment", plan_experiment),
        register_declared_tool(server, ScientificAgent, "run_experiment", run_experiment),
    ]
