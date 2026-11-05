"""Experiment settings: what an experiment fixes, loaded from a TOML file, and how a run's
`ToyBlocksConfig` is built from them.

Two kinds of values go into a run. What varies between invocations -- the task, the instance, the
model, the output directory, the budget -- is typed on the command line. What an experiment holds
fixed across all of its runs -- the skill backend, the model, the checkpoint, the viewer, the
camera, the seed, the scripted skills' parameters -- is an :class:`ExperimentSettings`, read from
``experiments/<name>.toml`` beside this module (``default.toml`` is what runs with no
``--experiment``). A flag overrides the file's value for one run. The result of combining them is
one immutable `ToyBlocksConfig`, passed to the code that needs it, and written as
``config.json`` into the run directory so a result is reproducible from the artifact alone.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import logging
import tomllib
from pathlib import Path
from typing import Any

from agentic_robotics.adapters.toyblocks.config import (
    IMAGE_OBS_HEIGHT,
    IMAGE_OBS_WIDTH,
    TOYBLOCKS_TASKS,
    OracleConfig,
    SkillBackend,
    TaskConfig,
    ToyBlocksConfig,
)
from agentic_robotics.adapters.toyblocks.maniflow_config import MANIFLOW_CONFIGS
from agentic_robotics.core.foundation_models import OPENAI_API_MODEL_CONFIGS

logger = logging.getLogger("agentic_robotics.examples.experiment")

EXPERIMENTS_DIR = Path(__file__).resolve().parent / "experiments"
"""Where the named experiment files live."""

DEFAULT_EXPERIMENT = "default"


@dataclasses.dataclass(frozen=True)
class ExperimentSettings:
    """What an experiment holds fixed across its runs. One TOML table, keys as the fields."""

    name: str
    """The experiment's name: the file's stem, or the path given."""

    task: str
    """Key into the task table, run when no ``--task`` is given."""

    model: str
    """Key into `OPENAI_API_MODEL_CONFIGS`."""

    skill_backend: SkillBackend
    """Which skills execute: the scripted (oracle) skills or the ManiFlow policies."""

    maniflow: str
    """Key into `MANIFLOW_CONFIGS`: the checkpoint the ManiFlow skills and `serve` use."""

    viewer: bool = True
    """Whether a run opens the simulator's viewer."""

    camera_name: str = "front_camera"
    """Camera the agent sees and the video records."""

    rng_seed: int = 1612
    """Seed for the scene draws and the quirk skills."""

    action_horizon: int = 25
    """ManiFlow action chunk length."""

    image_obs_width: int = IMAGE_OBS_WIDTH
    image_obs_height: int = IMAGE_OBS_HEIGHT

    oracle: OracleConfig = dataclasses.field(default_factory=OracleConfig)
    """The scripted skills' parameters."""


def experiment_path(name_or_path: str) -> Path:
    """The TOML file for an experiment: a path to one, or the name of one under
    :data:`EXPERIMENTS_DIR`.

    :raises SystemExit: If neither resolves to a file
    """
    candidate = Path(name_or_path)
    if candidate.suffix == ".toml" and candidate.is_file():
        return candidate
    named = EXPERIMENTS_DIR / f"{name_or_path}.toml"
    if named.is_file():
        return named
    have = ", ".join(sorted(p.stem for p in EXPERIMENTS_DIR.glob("*.toml")))
    raise SystemExit(
        f"No experiment file {candidate} and no experiment named {name_or_path!r}; have: {have}."
    )


def load_experiment(name_or_path: str = DEFAULT_EXPERIMENT) -> ExperimentSettings:
    """Read an experiment's settings from its TOML file.

    Every key is a field of :class:`ExperimentSettings`; ``oracle`` is a table of
    `OracleConfig` fields. Unknown keys and unknown table names are errors, since a misspelled
    key would otherwise silently fall back to the default.

    :param name_or_path: Experiment name or path to a TOML file
    :return: The settings, validated against the model, checkpoint and task tables
    :raises SystemExit: On an unknown key or an unknown table name
    """
    path = experiment_path(name_or_path)
    with path.open("rb") as handle:
        raw: dict[str, Any] = tomllib.load(handle)
    raw.setdefault("name", path.stem)

    fields = {f.name for f in dataclasses.fields(ExperimentSettings)}
    unknown = set(raw) - fields
    if unknown:
        raise SystemExit(f"{path}: unknown keys {sorted(unknown)}; known: {sorted(fields)}.")
    oracle_fields = {f.name for f in dataclasses.fields(OracleConfig)}
    unknown_oracle = set(raw.get("oracle", {})) - oracle_fields
    if unknown_oracle:
        raise SystemExit(
            f"{path}: unknown [oracle] keys {sorted(unknown_oracle)}; "
            f"known: {sorted(oracle_fields)}."
        )
    raw["oracle"] = OracleConfig(**raw.get("oracle", {}))
    raw["skill_backend"] = SkillBackend(raw["skill_backend"])
    settings = ExperimentSettings(**raw)
    validate_names(settings)
    return settings


def validate_names(settings: ExperimentSettings) -> None:
    """Check that the settings' model, checkpoint and task keys exist in their tables.

    :raises SystemExit: Naming the key and the table it is missing from
    """
    if settings.model not in OPENAI_API_MODEL_CONFIGS:
        raise SystemExit(
            f"Unknown model {settings.model!r}; known: {', '.join(OPENAI_API_MODEL_CONFIGS)}."
        )
    if settings.maniflow not in MANIFLOW_CONFIGS:
        raise SystemExit(
            f"Unknown ManiFlow checkpoint {settings.maniflow!r}; known: {sorted(MANIFLOW_CONFIGS)}."
        )


def with_overrides(
    settings: ExperimentSettings,
    *,
    model: str | None = None,
    skill_backend: str | None = None,
    maniflow: str | None = None,
    viewer: bool | None = None,
) -> ExperimentSettings:
    """The settings with the given command-line overrides applied; validated like a file."""
    changes: dict[str, Any] = {}
    if model is not None:
        changes["model"] = model
    if skill_backend is not None:
        changes["skill_backend"] = SkillBackend(skill_backend)
    if maniflow is not None:
        changes["maniflow"] = maniflow
    if viewer is not None:
        changes["viewer"] = viewer
    if not changes:
        return settings
    updated = dataclasses.replace(settings, **changes)
    validate_names(updated)
    return updated


def build_config(
    settings: ExperimentSettings,
    *,
    task_config: TaskConfig,
    output_dir: Path,
    policy_port: int | None = None,
) -> ToyBlocksConfig:
    """One run's configuration: the experiment's settings, this task, this output directory.

    :param settings: The experiment
    :param task_config: The task (or variant) this run drives
    :param output_dir: The run's directory, already created
    :param policy_port: Port of the ManiFlow policy server, instead of the checkpoint entry's
    :return: The immutable configuration every component of the run reads
    """
    maniflow = MANIFLOW_CONFIGS[settings.maniflow]
    if policy_port is not None:
        maniflow = dataclasses.replace(maniflow, policy_port=policy_port)
    return ToyBlocksConfig(
        skill_backend=settings.skill_backend,
        model_config=OPENAI_API_MODEL_CONFIGS[settings.model],
        task_config=task_config,
        image_obs_width=settings.image_obs_width,
        image_obs_height=settings.image_obs_height,
        viewer=settings.viewer,
        action_horizon=settings.action_horizon,
        oracle=settings.oracle,
        maniflow=maniflow,
        camera_name=settings.camera_name,
        rng_seed=settings.rng_seed,
        output_root_dir=output_dir.parent,
        output_dir=output_dir,
        experiment=settings.name,
    )


def retarget(config: ToyBlocksConfig, task_config: TaskConfig, output_dir: Path) -> ToyBlocksConfig:
    """The same configuration pointed at another task and run directory, for a process that
    builds scenes one after another (the menu page)."""
    return dataclasses.replace(
        config, task_config=task_config, output_root_dir=output_dir.parent, output_dir=output_dir
    )


def default_task(settings: ExperimentSettings) -> str:
    """The task key the experiment runs when none is named; it must exist in the task table.

    :raises SystemExit: If it does not
    """
    if settings.task not in TOYBLOCKS_TASKS:
        raise SystemExit(
            f"Experiment {settings.name!r} names task {settings.task!r}, which does not exist."
        )
    return settings.task


def _jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def write_resolved_config(config: ToyBlocksConfig) -> Path:
    """Write the run's fully resolved configuration as ``config.json`` in its run directory.

    Everything the run was built from, after every override, so the record needs no other file
    to be read. The task's prompt is included in full.

    :return: The file written
    :raises ValueError: If the configuration names no run directory
    """
    if config.output_dir is None:
        raise ValueError("The configuration names no run directory to write into.")
    path = config.output_dir / "config.json"
    path.write_text(json.dumps(_jsonable(config), indent=2), encoding="utf-8")
    return path
