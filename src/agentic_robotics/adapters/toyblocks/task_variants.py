"""Per-task PDDL variants: discover the problem files a task owns, pick one per load.

Motivation: a task should not always spawn the same scene. Instead of one
`data/problems/<slug>.pddl` per task, a task may own a *directory* of variant problems --
`data/problems/<slug>/*.pddl` -- and a run loads one of them at random. Each variant PDDL becomes
its own autogen env package (Stage-1 names the package after the PDDL basename, via
`task_slug_from_pddl_path` in toyblocks's `generate_task_from_pddl.py`), so a variant carries its
own `reset_states.pkl` with no further bookkeeping here.

This module is deliberately free of `config`, `runner_bootstrap`, and any adapter import: it reads
the filesystem and reshapes a `TaskConfig` into a `ResolvedTask` pointing at one chosen variant.
`agentic-robotics` owns *which* variant is loadable on this machine (built or saved), because that
question needs `runner_bootstrap`; this module only enumerates and reshapes.

Layout, keyed on a task's base slug (its `TOYBLOCKS_TASKS` key / `TaskConfig.task_slug`):

    data/problems/<slug>/*.pddl   the variants, when the directory exists
    data/problems/<slug>.pddl     a single legacy problem, used when the directory does not

A variant's slug is its PDDL stem with hyphens turned to underscores, matching what Stage-1 names
the env package and gym id after. Variant PDDLs therefore need distinct stems (e.g. `r_on_red_00`,
`r_on_red_01`), which is what the parent repo's `scripts/task_gen/*.py --count N` already emits.

What `config.py` might absorb later: a first-class notion of variants on `TaskConfig`
(an explicit `problems_dir`, or a `variants` list), instead of this module inferring them from the
slug. Nothing here requires that change -- it is flagged, not made.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_robotics.adapters.toyblocks.config import TaskConfig

logger = logging.getLogger("agentic_robotics.examples.task_variants")

AGENTIC_ROBOTICS_ROOT = Path(__file__).resolve().parents[4]
PROBLEMS_DIR = AGENTIC_ROBOTICS_ROOT / "data/problems"


@dataclass(frozen=True)
class ResolvedTask:
    """One task narrowed to a single PDDL variant, shaped like a `TaskConfig`.

    Carries exactly the attributes the loaders and adapter read off a `TaskConfig`
    (`prompt`, `max_tool_calls`, `task_slug`, `env_name`, `pddl_file`, `name`), so it stands in
    wherever a `TaskConfig` was passed. `base_slug` is the task it was drawn from, kept for
    grouping this variant's package with its siblings under one task.
    """

    name: str
    base_slug: str
    task_slug: str
    prompt: str
    max_tool_calls: int
    pddl_file: Path
    glyph_library: str | None = None
    """The glyph asset library the task declares for its packages, when its config names one
    (`cli.py`'s `glyph_library_for` reads it); None leaves `agentic-robotics` on its default."""

    @property
    def env_name(self) -> str:
        """Gymnasium environment ID registered for this variant."""
        return f"{self.task_slug}-v0"


def variant_slug(pddl_path: Path) -> str:
    """Slug Stage-1 names a variant's env package after: its PDDL stem, hyphens to underscores.

    Kept identical to `task_slug_from_pddl_path` in toyblocks's `generate_task_from_pddl.py`,
    which is the code that actually names the package; a mismatch here would look up a package
    under a name Stage-1 never wrote.

    :param pddl_path: A variant's PDDL problem file
    :return: The env-package slug for that variant
    """
    return pddl_path.stem.replace("-", "_")


def discover_variants(base_slug: str, problems_dir: Path = PROBLEMS_DIR) -> list[Path]:
    """PDDL problem files a task owns: the directory's contents, or the one legacy file.

    :param base_slug: A task's base slug (`TaskConfig.task_slug`)
    :param problems_dir: Directory holding the problems (default: repo `data/problems`)
    :return: Sorted variant PDDL paths; empty when the task has neither a directory nor a file
    """
    variant_dir = problems_dir / base_slug
    if variant_dir.is_dir():
        return sorted(variant_dir.glob("*.pddl"))
    single = problems_dir / f"{base_slug}.pddl"
    return [single] if single.is_file() else []


def _to_resolved(task_config: TaskConfig, pddl_path: Path) -> ResolvedTask:
    """Reshape a `TaskConfig` onto one variant PDDL."""
    return ResolvedTask(
        name=task_config.name,
        base_slug=task_config.task_slug,
        task_slug=variant_slug(pddl_path),
        prompt=task_config.prompt,
        max_tool_calls=task_config.max_tool_calls,
        pddl_file=pddl_path,
        glyph_library=getattr(task_config, "glyph_library", None),
    )


def all_variants(task_config: TaskConfig) -> list[ResolvedTask]:
    """Every variant of a task, as `ResolvedTask`s. Used to build and to enumerate.

    :param task_config: A `TaskConfig` from `TOYBLOCKS_TASKS`
    :return: One `ResolvedTask` per PDDL the task owns, in sorted order
    """
    return [_to_resolved(task_config, pddl) for pddl in discover_variants(task_config.task_slug)]


def choose_variant(
    task_config: TaskConfig, *, among: list[ResolvedTask] | None = None
) -> ResolvedTask:
    """Pick one variant of a task at random, logging which of how many was chosen.

    :param task_config: A `TaskConfig` from `TOYBLOCKS_TASKS`
    :param among: Restrict the draw to these variants (e.g. only ones built on this machine);
        defaults to every variant the task owns
    :return: The chosen `ResolvedTask`
    :raises FileNotFoundError: If the task owns no PDDL variant to choose from
    """
    variants = all_variants(task_config)
    if not variants:
        raise FileNotFoundError(
            f"Task '{task_config.task_slug}' has no PDDL variant: neither "
            f"{PROBLEMS_DIR / task_config.task_slug}/ nor "
            f"{PROBLEMS_DIR / f'{task_config.task_slug}.pddl'} exists."
        )
    pool = among if among else variants
    chosen = random.choice(pool)
    logger.info(
        f"Task '{task_config.task_slug}': {len(variants)} PDDL variant(s), "
        f"drew '{chosen.task_slug}' ({chosen.pddl_file.name})."
    )
    return chosen
