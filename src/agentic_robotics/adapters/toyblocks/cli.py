"""The one entry point: every supported path through this repo, in a job or on a workstation.

    agentic-robotics episode --task r_on_red        one headless episode, no page
    agentic-robotics episode --task <slug> --baseline oracle --model opus-5.5 \\
                             --model-host claude-code
                                                    the same evaluation episode, hosted by
                                                    Claude Code
    agentic-robotics page    --task r_on_red        the same episode, with the interjection
                                                    page around it
    agentic-robotics human   --task r_on_red        the same episode, with a person in the
                                                    model's chair
    agentic-robotics menu                           Isaac with no scene; the page picks the task
    agentic-robotics mock                           the page alone, no Isaac and no GPU
    agentic-robotics portal                         browse finished runs; no Isaac, no model
    agentic-robotics generate-env --task r_on_red   Stage-1 codegen for a task, then exit
    agentic-robotics serve                          the ManiFlow policy server the experiment names
    agentic-robotics tasks                          what is defined, and what is built on this
                                                    machine
    agentic-robotics leap --task ai2_easy           the LEAP experimentation phase (naive agent)
    agentic-robotics pool --task ai2_easy --pool-port 8101
                                                    one pre-warmed simulator for that phase's
                                                    subagents

`agentic-robotics` is this module's console script; ``python -m
agentic_robotics.adapters.toyblocks.cli`` is the same command, and is how a job script
names it alongside an explicit interpreter.

Runs differ in their task and prompt, and both are carried by a `TaskConfig`: an entry of
`config.TOYBLOCKS_TASKS`, a LEAP family, or a table under `tasks/`. Adding a task is an entry
rather than a mode, and the ordering that is easy to get wrong lives once in `runner_bootstrap`.

**Which interpreter to use: whichever you have.** A mode that needs Isaac re-execs itself under an
interpreter that can import ToyBlocks, so both of these do the same thing::

    python -m agentic_robotics.adapters.toyblocks.cli tasks
    agentic-robotics tasks

The second re-execs into the first and says so in the log. Naming the heavy interpreter directly
skips that hop; ``--python`` or ``$AGENTIC_ROBOTICS_PYTHON`` overrides the choice on a machine
where neither guess is right. Isaac also stops at a licence prompt unless ``OMNI_KIT_ACCEPT_EULA``
is set, which this refuses to set for you -- ``export OMNI_KIT_ACCEPT_EULA=YES`` accepts it.

**Nothing about the machine is guessed.** The interpreter codegen runs under, the directory runs
land in, and whether Isaac's licence has been accepted are each resolved explicitly and each say
what to do when there is no answer; see `runner_bootstrap`. The page binds loopback unless --lan is
typed, because the alternative is an unauthenticated arm-control surface on every interface a
machine has.

`menu` is what a demo sitting uses: one Isaac boot, and the page loads a scene per task, since
Isaac hangs rather than failing when a second environment is built over a closed one's prims.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentic_robotics.adapters.toyblocks import experiment, task_variants
from agentic_robotics.adapters.toyblocks import runner_bootstrap as boot
from agentic_robotics.adapters.toyblocks.config import TOYBLOCKS_TASKS, SkillBackend
from agentic_robotics.adapters.toyblocks.maniflow_config import MANIFLOW_CONFIGS
from agentic_robotics.adapters.toyblocks.tasks.le import le_tasks, letter_probe_tasks
from agentic_robotics.adapters.toyblocks.tasks.library_swaps import library_swap_tasks
from agentic_robotics.adapters.toyblocks.tasks.rao import rao_tasks
from agentic_robotics.adapters.toyblocks.tasks.spell import spell_tasks

if TYPE_CHECKING:
    import threading
    from types import ModuleType

    from agentic_robotics.adapters.toyblocks.config import TaskConfig, ToyBlocksConfig
    from agentic_robotics.adapters.toyblocks.experiment import ExperimentSettings
    from agentic_robotics.adapters.toyblocks.skills.inline_oracle import InlineOracleMCPServer
    from agentic_robotics.adapters.toyblocks.skills.maniflow_skills import (
        SimThreadManiFlowServer,
    )
    from agentic_robotics.adapters.toyblocks.task_variants import ResolvedTask
    from agentic_robotics.adapters.toyblocks.toyblocks_runner import SimulatorDispatcher

logger = logging.getLogger("agentic_robotics.examples.run")

CHAT_PORT = 8083
VISER_PORT = 8080
PORTAL_PORT = 8082
"""Where `portal` binds when --port is left at its default."""


def load_tasks() -> dict[str, Any]:
    """Read `TOYBLOCKS_TASKS` without importing the adapter, which would boot Isaac, plus the
    LEAP family tasks from `leap/prompts/evaluation.py`, which `config.py` does not list yet.

    :return: Task key to `TaskConfig`
    """
    tasks = dict(TOYBLOCKS_TASKS)
    tasks.update(boot.load_leap_tasks())
    tasks.update(rao_tasks())  # The Rao release instances, defined in `tasks/rao.py`
    tasks.update(le_tasks())  # One LE training scene, for model-free skill trials
    # Untrained letters beside trained ones, for the per-letter probe
    tasks.update(letter_probe_tasks())
    tasks.update(spell_tasks())  # Search & Spell, five words x ten problems
    tasks.update(library_swap_tasks())  # Fixed-start scenes rebuilt with another block library
    return tasks


def default_task_key(settings: ExperimentSettings) -> str:
    """The task a mode runs when none is named: the experiment's.

    :return: Key into the task table
    """
    return experiment.default_task(settings)


def chosen_task(args: argparse.Namespace, settings: ExperimentSettings) -> ResolvedTask:
    """The variant a single-task mode should run: the one `--task` names by slug, one drawn from
    the task it names by key, or one drawn from the experiment's task."""
    return resolve_task(args.task or default_task_key(settings))


def task_config_for_key(key: str) -> TaskConfig:
    """The base `TaskConfig` for a key, or exit listing what exists.

    :param key: Key into `config.TOYBLOCKS_TASKS`
    :return: The `TaskConfig`
    :raises SystemExit: If the key names no task
    """
    tasks = load_tasks()
    if key not in tasks:
        raise SystemExit(f"No task '{key}'. Have: {', '.join(sorted(tasks))}")
    return tasks[key]


def chosen_task_config(args: argparse.Namespace, settings: ExperimentSettings) -> TaskConfig:
    """The base `TaskConfig` a single-task mode names, before a variant is drawn.

    `--task` may be a task key or one variant's slug (how `leap` and `pool` are pinned to one
    scene); a variant slug resolves to its task's config, so the checks that run before the boot
    (policy server, block library) see the same task the boot will.
    """
    name = args.task or default_task_key(settings)
    if name not in load_tasks():
        variant = variant_by_slug(name)
        if variant is not None:
            return task_config_for_key(variant.base_slug)
    return task_config_for_key(name)


def variant_available(variant: ResolvedTask) -> bool:
    """Whether a variant's env package is built."""
    return boot.task_package_is_built(variant.task_slug)


DEFAULT_GLYPH_LIBRARY = "train_blocks_large"
"""Glyph asset library a task is built against when nothing names one.

Large rather than ToyBlocks's own `train_blocks` default, because the checkpoints being served
are trained on the large blocks and a package built against the other set produces a run that
executes every skill and grasps nothing. `block_asset_check` is what catches that, after the fact;
this is what stops producing it.

A task whose config carries a `glyph_library` (the Search & Spell tasks) names its own;
`--usd-library` overrides either per invocation.
"""


def glyph_library_for(task: TaskConfig | ResolvedTask) -> str:
    """The glyph asset library a task's packages are built against.

    Reads `glyph_library` off the task when its config carries one, else the default above.

    :param task: A `TaskConfig` or `ResolvedTask`
    :return: A library name from `toyblocks.tamp.blocks.glyph_blocks.GLYPH_LIBRARIES`
    """
    return getattr(task, "glyph_library", None) or DEFAULT_GLYPH_LIBRARY


def variant_label(variant: ResolvedTask, position: int) -> str:
    """The number the page's variant button shows for a variant.

    A seeded generator names variants `<task>_00`, `<task>_01`, so the slug already carries the
    number to show and the button matches what is on disk. Nothing enforces that shape, so a slug
    that is not the base slug plus `_<digits>` falls back to its 1-based position in the sorted
    list. Matching the whole suffix rather than trailing digits is what keeps a name ending in a
    timestamp from being labelled `8086`.

    :param variant: The `ResolvedTask` its button loads
    :param position: Zero-based index among the task's sorted variants
    :return: Short label for the button
    """
    match = re.fullmatch(rf"{re.escape(variant.base_slug)}_(\d+)", variant.task_slug)
    return str(int(match.group(1))) if match else str(position + 1)


def select_task(key: str) -> ResolvedTask:
    """Draw one PDDL variant of a task, preferring one already built or saved here.

    Selection is per load: each call re-draws, so loading a task twice can spawn different scenes.
    A built or saved variant is preferred over one that would need Stage-1 codegen, so a run uses a
    ready scene when the task has one; when none is ready, any variant is drawn and the
    build/restore/error downstream applies to it.

    :param key: Key into `config.TOYBLOCKS_TASKS`
    :return: A `ResolvedTask` for the chosen variant
    :raises SystemExit: If the key names no task, or the task owns no PDDL variant
    """
    task = task_config_for_key(key)
    variants = task_variants.all_variants(task)
    if not variants:
        raise SystemExit(
            f"Task '{key}' has no PDDL variant: neither {task_variants.PROBLEMS_DIR / key}/ nor "
            f"{task_variants.PROBLEMS_DIR / f'{key}.pddl'} exists. Drop problem files there."
        )
    ready = [v for v in variants if variant_available(v)]
    return task_variants.choose_variant(task, among=ready or None)


def variant_by_slug(slug: str) -> ResolvedTask | None:
    """The `ResolvedTask` a variant's own slug names, or None when no task owns one.

    :param slug: A variant's `task_slug`, as the page's menu sends it
    :return: That variant, or None
    """
    for task in load_tasks().values():
        for variant in task_variants.all_variants(task):
            if variant.task_slug == slug:
                return variant
    return None


def resolve_task(name: str) -> ResolvedTask:
    """The variant a menu selection names: a task key draws one, a variant slug loads that one.

    The page's menu sends both through the same field, because that is the whole of what the
    interjection channel carries -- a task request is a string. The two namespaces do not collide:
    a variant slug is a PDDL stem, and Stage-1 names one env package after each, so a slug that
    matches a task key is that task's only variant and both readings give the same scene.

    :param name: A key into `config.TOYBLOCKS_TASKS`, or a variant's `task_slug`
    :return: A `ResolvedTask` for the chosen variant
    :raises SystemExit: If the name is neither
    """
    tasks = load_tasks()
    if name in tasks:
        return select_task(name)
    variant = variant_by_slug(name)
    if variant is None:
        raise SystemExit(f"No task or variant '{name}'. Have: {', '.join(sorted(tasks))}")
    logger.info(f"Loading variant '{variant.task_slug}' of task '{variant.base_slug}' by name.")
    return variant


class OracleSceneLoader:
    """Builds and disposes scenes on the simulator's thread, for the page's task menu.

    What `server.py` asks of a task loader: `tasks`, `describe(name)`, `load(name)` and
    `viser_url()`. A scene's whole construction -- gym environment, skill engine, viser server --
    runs inside one dispatched job, so it happens on the thread that owns Isaac. No episode can be
    running while a load is applied, so nothing else is waiting on that thread meanwhile.
    """

    def __init__(
        self,
        dispatcher: SimulatorDispatcher,
        skills: ModuleType,
        config: ToyBlocksConfig,
        *,
        require_preflight: bool = True,
        require_block_check: bool = True,
    ) -> None:
        """
        :param dispatcher: The `SimulatorDispatcher` whose thread owns Isaac
        :param skills: The imported skill module, providing a server for the configured backend
        :param config: The run's configuration; each scene loaded gets a copy pointed at its task
            and its own run directory
        :param require_preflight: Whether an unactionable scene fails the load
        :param require_block_check: Whether a ManiFlow scene whose blocks the checkpoint never
            trained on fails the load (the menu's counterpart of `require_matching_blocks`)
        """
        self._dispatcher = dispatcher
        self._skills = skills
        self._config = config
        self._current: InlineOracleMCPServer | SimThreadManiFlowServer | None = None
        self._require_preflight = require_preflight
        self._require_block_check = require_block_check

    @property
    def tasks(self) -> list[dict[str, Any]]:
        """Menu entries: key, description, whether this machine can build the scene, and variants.

        Availability is built *or* saved, not built alone. A saved copy is restored by `load`
        for the price of a file copy, so a menu that offered only built packages disabled
        three of the four demos on a fresh session and pointed at an Isaac boot to fix it.

        Each variant is listed with its own availability, so the page can offer the ready ones and
        grey out the rest rather than sending a load that fails on the simulator's thread. A
        variant's `slug` goes back in the same field a task key does; see `resolve_task`.
        """
        entries = []
        for key, task in sorted(load_tasks().items()):
            variants = [
                {
                    "slug": variant.task_slug,
                    "label": variant_label(variant, position),
                    "reason": self._unavailable_reason(variant),
                }
                for position, variant in enumerate(task_variants.all_variants(task))
            ]
            for entry in variants:
                entry["available"] = entry["reason"] is None
            available = any(entry["available"] for entry in variants)
            entries.append(
                {
                    "name": key,
                    "description": task.prompt,
                    # Available when *any* variant is; `load` draws among the ready ones.
                    "available": available,
                    "reason": None
                    if available
                    else (variants[0]["reason"] if variants else NO_PACKAGE_REASON),
                    "variants": variants,
                }
            )
        return entries

    def _unavailable_reason(self, variant: ResolvedTask) -> str | None:
        """Why the page should grey a variant out, or None when it can be loaded here.

        Two reasons exist: nothing built or saved for it, and blocks the served checkpoint never
        trained on. The second is skipped when the block check is switched off.
        """
        if not variant_available(variant):
            return NO_PACKAGE_REASON
        if self._require_block_check:
            return block_mismatch_reason(variant.task_slug, self._config)
        return None

    def rejection(self, name: str) -> str | None:
        """Why a load of `name` would be refused, or None when it would go ahead.

        Asked by the page's server before it ends a running episode for a task switch, so that a
        switch to a task this scene cannot run refuses without costing the episode. `load` still
        checks the variant it draws.

        :param name: A task key or a variant slug, as the menu sends it
        :return: The reason, or None
        """
        tasks = load_tasks()
        if name in tasks:
            variants = task_variants.all_variants(tasks[name])
        else:
            variant = variant_by_slug(name)
            if variant is None:
                return f"no task or variant named {name!r}"
            variants = [variant]
        reasons = [self._unavailable_reason(v) for v in variants]
        if any(reason is None for reason in reasons):
            return None
        return reasons[0] if reasons else NO_PACKAGE_REASON

    @property
    def current(self) -> InlineOracleMCPServer | SimThreadManiFlowServer | None:
        """The loaded scene's server, or None."""
        return self._current

    def describe(self, name: str) -> str:
        """The prompt behind a menu name, which a variant shares with the task it was drawn from.

        Resolved without `resolve_task`, which would draw a second variant at random just to read
        a prompt off it and log a draw the page never made.
        """
        tasks = load_tasks()
        if name in tasks:
            return tasks[name].prompt
        variant = variant_by_slug(name)
        return variant.prompt if variant is not None else ""

    def viser_url(self) -> str | None:
        return self._current.viewer_url() if self._current is not None else None

    async def load(self, name: str) -> InlineOracleMCPServer | SimThreadManiFlowServer:
        """Dispose the loaded scene, if any, and build the named one, returning its server."""
        import asyncio

        return await asyncio.wrap_future(self._dispatcher.submit(self._load_on_sim_thread(name)))

    async def _load_on_sim_thread(
        self, name: str
    ) -> InlineOracleMCPServer | SimThreadManiFlowServer:
        """Body never awaits; driven on the simulator's thread like every skill."""
        task = resolve_task(name)
        if not boot.task_package_is_built(task.task_slug):
            raise RuntimeError(
                f"The task package for '{task.task_slug}' is not built at "
                f"{boot.task_package_dir(task.task_slug)}. Build it with "
                f"`agentic-robotics generate-env --task {task.base_slug}`, "
                "which builds every variant."
            )

        # Before the current scene is disposed, so a refused load leaves it standing. The startup
        # check only saw the task named on the command line; the page can name any other.
        if self._require_block_check:
            require_blocks_match(task.task_slug, task.glyph_library)

        self.dispose_current()

        # A closed environment leaves its prims on the stage, and a scene built on top of them
        # hangs in Isaac's asset-loading wait instead of failing. Isaac Lab's own test harness
        # opens a new stage before every environment it makes in one process; so does this.
        import omni.usd

        omni.usd.get_context().new_stage()

        config = experiment.retarget(
            self._config,
            task,
            boot.create_run_dir(
                f"menu-{task.task_slug.replace('_', '-')}-", self._config.output_root_dir
            ),
        )
        logger.info(
            f"Wrote the resolved configuration to {experiment.write_resolved_config(config)}."
        )
        logger.info(f"Building scene '{name}' (gym env, skills, viser)...")
        server_class = boot.sim_thread_server_class(self._skills, config)
        self._current = server_class(self._dispatcher, config)

        # Already on the simulator's thread. A scene the agent cannot point at is worth
        # refusing here rather than after someone has typed a task at it.
        from agentic_robotics.adapters.toyblocks.scene_preflight import preflight_scene

        preflight_scene(self._current, require=self._require_preflight)

        logger.info(f"Scene '{name}' ready; viser at {self.viser_url()}")

        # Guarded like `runner_bootstrap.run_episode`'s: the manifest module asserts a
        # checkout layout at import time, and telemetry is never worth a scene.
        try:
            from agentic_robotics.adapters.toyblocks.toyblocks_run_manifest import (
                write_run_manifest,
            )

            logger.info(f"Wrote run manifest to {write_run_manifest(self._current, config)}.")
        except Exception:
            logger.exception("Could not write the run manifest; continuing without one.")

        return self._current

    def dispose_current(self) -> None:
        """Tear the loaded scene down. Must run on the simulator's thread."""
        if self._current is None:
            return
        try:
            self._current.dispose()
        except Exception:
            logger.exception("Disposing the previous scene failed; building the next one anyway.")
        self._current = None


ISAAC_MODES = frozenset({"episode", "human", "page", "menu", "generate-env"})
"""Modes that import ToyBlocks, so a light interpreter has to hand over before they start."""

REEXEC_MARKER = "AGENTIC_ROBOTICS_REEXECED"
"""Set on the way into a re-exec, so a heavy interpreter that still cannot import ToyBlocks
fails with the resolver's message rather than spawning itself forever."""


def reexec_under_isaac_python_if_needed(args: argparse.Namespace) -> None:
    """Hand this command to an interpreter with ToyBlocks in it, if this one lacks it.

    This repo can be started from either of two interpreters -- a light one holding `mcp` and the
    model client, and one holding Isaac -- and which of them is on someone's PATH differs by
    machine. Rather than making that the user's problem, a mode that needs Isaac re-runs itself
    under an interpreter that has it, saying which one and why.

    :param args: Parsed command line, for the mode and an explicit `--python`
    """
    import os

    if args.mode not in ISAAC_MODES or os.environ.get(REEXEC_MARKER):
        return
    if importlib.util.find_spec("toyblocks") is not None:
        return

    python = boot.resolve_isaac_python(args.python)
    logger.info(f"{sys.executable} cannot import toyblocks; re-running under {python}.")
    # `src` leads the path so the heavy interpreter finds this package even where it is not
    # installed
    pythonpath = os.pathsep.join(
        filter(None, (str(boot.AGENTIC_ROBOTICS_ROOT / "src"), os.environ.get("PYTHONPATH")))
    )
    os.execve(
        str(python),
        [str(python), "-m", "agentic_robotics.adapters.toyblocks.cli", *sys.argv[1:]],
        {**os.environ, REEXEC_MARKER: "1", "PYTHONPATH": pythonpath},
    )


def require_free_ports(*ports: int) -> None:
    """Refuse to boot if a port is taken.

    Viser hops to the next free port when its default is occupied, which leaves the page's iframe
    pointed at a dead 8080; and the web server would otherwise die with EADDRINUSE minutes later,
    after Isaac has finished booting. Fail in the first second instead.

    :param ports: Ports that must be free
    :raises SystemExit: If any is already bound
    """
    import socket

    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as exc:
                raise SystemExit(
                    f"Port {port} is already in use (a previous run still shutting down?). "
                    f"Wait for it or kill it, then retry."
                ) from exc


POLICY_BACKED_MODES = frozenset({"episode", "human", "page", "menu", "leap", "pool"})
"""Modes that build a skill server, so a remote policy has to be up before they start."""


def require_policy_server(
    mode: str, settings: ExperimentSettings, port_override: int | None = None
) -> None:
    """Refuse to boot a ManiFlow-backed mode when nothing is serving the policy.

    `ManiFlowMCPServer` opens its websocket in `_build_skill_backend`, which runs after
    Isaac has booted and the scene has been built, so a forgotten policy server costs the whole
    boot before anything reports it. Same reason `require_free_ports` exists: fail in the first
    second instead.

    A TCP connect is as far as this goes. Reading the server's metadata would settle more -- which
    checkpoint, which language mode -- and needs the websocket client, which lives in the heavy
    venv this check deliberately runs ahead of.

    :param mode: Mode about to run
    :raises SystemExit: If the configured backend is ManiFlow and its port answers nothing
    """
    if mode not in POLICY_BACKED_MODES:
        return

    import socket

    if settings.skill_backend != SkillBackend.MANIFLOW:
        return

    maniflow = MANIFLOW_CONFIGS[settings.maniflow]
    host, port = maniflow.policy_host, port_override or maniflow.policy_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2.0)
        if sock.connect_ex((host, port)) == 0:
            return

    raise SystemExit(
        f"The skill backend is '{settings.skill_backend}' and nothing is listening on "
        f"{host}:{port}. Serve {maniflow.checkpoint_dir} first (`agentic-robotics serve`), or "
        f"pass --backend {SkillBackend.ORACLE} or set skill_backend in the experiment file."
    )


NO_PACKAGE_REASON = "no package or saved copy here"
"""Why a variant is greyed out when nothing built or saved exists for it."""


def block_mismatch_reason(task_slug: str, config: ToyBlocksConfig) -> str | None:
    """One line on why a variant's blocks do not fit the served checkpoint, or None when they do.

    Reads the built package, or the saved copy a load would restore. Nothing to say under the
    oracle backend, which plans against the scene's own geometry.

    :param task_slug: Slug of a built task package
    :param config: The run's configuration, for the backend and the served checkpoint
    :return: The reason, or None when the blocks fit or there is nothing to check
    """
    if config.skill_backend != SkillBackend.MANIFLOW:
        return None

    from agentic_robotics.adapters.toyblocks.block_asset_check import BlockAssetsUnknown
    from agentic_robotics.adapters.toyblocks.block_asset_check import (
        block_mismatch_reason as mismatch_reason,
    )

    package_dir = boot.task_package_dir(task_slug)
    try:
        return mismatch_reason(package_dir / "env.py", Path(config.maniflow.checkpoint_dir))
    except BlockAssetsUnknown as exc:
        return f"cannot check its blocks: {exc}"


def require_blocks_match(
    task_slug: str, settings: ExperimentSettings, declared_library: str | None = None
) -> None:
    """Refuse one built task package whose blocks the served ManiFlow checkpoint never trained on.

    A policy given blocks of another size or library runs every skill and moves nothing, which
    is the failure `block_asset_check` exists for. Nothing to check under the oracle backend,
    which plans against the scene's own geometry.

    A task that declares a library the checkpoint never trained on (the Search & Spell
    unpatterned tasks, `eval_blocks_36_0`) is deliberately outside the checkpoint's blocks:
    then the package is checked against the declaration instead, and the run says so.

    :param task_slug: Slug of a built task package
    :param settings: The experiment, for the backend and the served checkpoint
    :param declared_library: The glyph library the task's config names, if any
    :raises RuntimeError: If the blocks differ, or if either side cannot be read
    """
    if settings.skill_backend != SkillBackend.MANIFLOW:
        return

    from agentic_robotics.adapters.toyblocks.block_asset_check import (
        BlockAssetMismatch,
        BlockAssetsUnknown,
        assert_blocks_from_library,
        assert_blocks_match,
        trained_libraries,
    )

    env_py = boot.task_package_dir(task_slug) / "env.py"
    checkpoint_dir = Path(MANIFLOW_CONFIGS[settings.maniflow].checkpoint_dir)
    try:
        if declared_library is not None and declared_library not in trained_libraries(
            checkpoint_dir
        ):
            assert_blocks_from_library(env_py, declared_library)
            logger.warning(
                f"{task_slug} spawns {declared_library}, which {checkpoint_dir.name} never "
                f"trained on: the task declares it, so this run is deliberately outside the "
                f"checkpoint's blocks."
            )
            return
        assert_blocks_match(env_py, checkpoint_dir)
    except (BlockAssetMismatch, BlockAssetsUnknown) as exc:
        raise RuntimeError(f"{exc}\n    Pass --skip-block-check to run anyway.") from exc


def require_matching_blocks(args: argparse.Namespace, settings: ExperimentSettings) -> None:
    """Refuse a ManiFlow run whose task spawns blocks the checkpoint never trained on.

    Same slot and same reason as `require_policy_server`: the mismatch it catches costs a full
    Isaac boot and a whole episode to notice otherwise, and it does not announce itself when it
    happens -- the skills run, the arm moves, and the blocks stay where they were.

    A task with several PDDL variants is checked variant by variant, since a run draws one at
    random and they are generated independently. Unbuilt variants are left alone: `prepare_scene`
    already refuses those, with a message that says what to run.

    Failing to *read* either side fails the run too, rather than passing on the assumption that
    unreadable means fine. Three of the ten most recent checkpoints carry no `config.yaml` at
    all, so that path is reachable -- which is what `--skip-block-check` is for.

    :param args: Parsed command line, for the mode, the task and `--skip-block-check`
    :raises SystemExit: If the blocks differ, or if either side cannot be read
    """
    if args.mode not in POLICY_BACKED_MODES or args.skip_block_check:
        return

    for variant in task_variants.all_variants(chosen_task_config(args, settings)):
        if not boot.task_package_is_built(variant.task_slug):
            continue
        try:
            require_blocks_match(
                variant.task_slug, settings, args.usd_library or variant.glyph_library
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc


def settings_from_args(args: argparse.Namespace) -> ExperimentSettings:
    """The experiment named by `--experiment`, with this invocation's overrides applied.

    `--backend`, or the backend `--skills` implements, replaces the skill backend; `--model` and
    `--maniflow` replace the model and the checkpoint. Each is validated against its table.

    :raises SystemExit: If `--skills` names a module whose backend is unknown and `--backend` is
        absent, or an override names something that does not exist
    """
    settings = experiment.load_experiment(args.experiment)
    backend = getattr(args, "backend", None)
    skills = getattr(args, "skills", None)
    if backend is None and skills is not None:
        backend = boot.SKILL_MODULE_BACKENDS.get(skills)
        if backend is None:
            raise SystemExit(
                f"--skills {skills} is not in runner_bootstrap.SKILL_MODULE_BACKENDS; pass "
                "--backend to say which backend it implements."
            )
    settings = experiment.with_overrides(
        settings,
        model=getattr(args, "model", None),
        skill_backend=backend,
        maniflow=getattr(args, "maniflow", None),
    )
    logger.info(
        f"Experiment {settings.name}: backend={settings.skill_backend} model={settings.model} "
        f"maniflow={settings.maniflow}"
    )
    return settings


def codegen_seed(task: TaskConfig | ResolvedTask, args: argparse.Namespace) -> int:
    """The Stage-1 seed: `--seed` when given, else one derived from the task's slug."""
    return args.seed if args.seed is not None else boot.seed_for_task(task.task_slug)


def prepare_scene(
    task: TaskConfig | ResolvedTask,
    args: argparse.Namespace,
    settings: ExperimentSettings,
    dir_prefix: str | None = None,
) -> ToyBlocksConfig:
    """Build the task package if asked, then build this run's configuration.

    Codegen writes the package the adapter's gym registration reads, so it comes first; the
    configuration names the task, so it decides which environment gets built.

    :param task: The `TaskConfig` being run
    :param args: Parsed command line, for `--generate-env`, `--python` and the Stage-1 numbers
    :param settings: The experiment the run belongs to
    :param dir_prefix: Prefix of the run directory; the task's slug when omitted
    :return: The run's configuration
    """
    boot.ensure_task_package(
        task.task_slug,
        task.pddl_file,
        generate_env=args.generate_env,
        reset_states=args.reset_states,
        envs=args.envs,
        seed=codegen_seed(task, args),
        codegen_python=boot.resolve_isaac_python(args.python),
        usd_library=args.usd_library or glyph_library_for(task),
    )
    output_dir = boot.create_run_dir(
        dir_prefix or task.task_slug.replace("_", "-"),
        Path(args.runs_dir) if args.runs_dir else None,
    )
    config = experiment.build_config(
        settings, task_config=task, output_dir=output_dir, policy_port=args.policy_port
    )
    logger.info(
        f"Run: task={task.task_slug} env={task.env_name} backend={config.skill_backend} "
        f"model={config.model_config.model_id} camera={config.camera_name} viewer={config.viewer}"
    )
    return config


def run_episode_mode(args: argparse.Namespace, settings: ExperimentSettings) -> None:
    """One episode, headless: no page, no human, the model against the prompt and the budget.

    With `--baseline`, the episode is a LEAP evaluation: `leap.evaluate` wraps the skill server
    so the tool set is checked against `EvaluationAgent` and the baseline's descriptions are
    served, and the run directory is prefixed with the baseline.

    With `--plan-mode`, the episode is a Rao release evaluation: `release/rao_modes.py` runs the
    closed or online loop on the skill server's own tool descriptions, and the run directory is
    prefixed with the mode.
    """
    if args.plan_mode is not None:
        if args.baseline is not None:
            raise SystemExit("--plan-mode and --baseline are two evaluations; pass one.")
        from agentic_robotics.adapters.toyblocks.release.rao_modes import rao_agentic_thread

        task = chosen_task(args, settings)
        config = prepare_scene(
            task,
            args,
            settings,
            dir_prefix=f"rao-{args.plan_mode}-{task.task_slug.replace('_', '-')}",
        )
        boot.run_episode(
            boot.import_skill_module(config, args.skills, quiet_isaac=not args.verbose_isaac),
            config,
            agent_thread=rao_agentic_thread,
            agent_thread_kwargs={"mode": args.plan_mode, "instance": args.instance},
            preflight=not args.no_preflight,
        )
        return

    if args.baseline is None:
        if args.instance is not None:
            raise SystemExit(
                "--instance needs --baseline or --plan-mode: the plain episode starts from the "
                "package's own reset-state draw."
            )
        task = chosen_task(args, settings)
        config = prepare_scene(task, args, settings)
        boot.run_episode(
            boot.import_skill_module(config, args.skills, quiet_isaac=not args.verbose_isaac),
            config,
            preflight=not args.no_preflight,
        )
        return

    from agentic_robotics.adapters.toyblocks.leap.evaluate import (
        evaluation_agentic_thread,
        load_description_set,
    )

    task = chosen_task(args, settings)
    descriptions = load_description_set(args.baseline, args.docstrings, str(settings.skill_backend))
    config = prepare_scene(
        task, args, settings, dir_prefix=f"eval-{args.baseline}-{task.task_slug.replace('_', '-')}"
    )
    boot.run_episode(
        boot.import_skill_module(config, args.skills, quiet_isaac=not args.verbose_isaac),
        config,
        agent_thread=evaluation_agentic_thread,
        agent_thread_kwargs={
            "baseline": args.baseline,
            "descriptions": descriptions,
            "descriptions_path": args.docstrings,
            "instance": args.instance,
            "allow_declaration_mismatch": args.allow_declaration_mismatch,
            "model_host": args.model_host,
            "claude_code": {"effort": args.effort, "max_turns": args.max_turns},
        },
        preflight=not args.no_preflight,
    )


def run_human_mode(args: argparse.Namespace, settings: ExperimentSettings) -> None:
    """One episode with a human in the model's chair, reached through the same boot as `episode`.

    The only difference from `episode` is which agent thread runs: `human_model_client` supplies a
    `ModelClient` whose responses come from a person clicking on the observation, so the loop, the
    transcript, the budget and the manifest are the model's and only the chair changes.
    """
    task = chosen_task(args, settings)
    require_free_ports(args.port)
    config = prepare_scene(task, args, settings)

    from agentic_robotics.adapters.toyblocks.human_model_client import human_agentic_thread

    boot.run_episode(
        boot.import_skill_module(config, args.skills, quiet_isaac=not args.verbose_isaac),
        config,
        agent_thread=human_agentic_thread,
        agent_thread_kwargs={"host": args.host, "port": args.port},
        preflight=not args.no_preflight,
    )


def run_page_mode(args: argparse.Namespace, settings: ExperimentSettings) -> None:
    """One episode with the interjection page served around it."""
    task = chosen_task(args, settings)
    require_free_ports(args.port, VISER_PORT)
    config = prepare_scene(task, args, settings)

    from agentic_robotics.adapters.toyblocks.web_ui.server import agentic_web_thread

    boot.run_episode(
        boot.import_skill_module(config, args.skills, quiet_isaac=not args.verbose_isaac),
        config,
        agent_thread=agentic_web_thread,
        agent_thread_kwargs={"host": args.host, "port": args.port},
        preflight=not args.no_preflight,
    )


def run_leap_mode(args: argparse.Namespace, settings: ExperimentSettings) -> None:
    """The LEAP experimentation phase: the naive agent learning its skills under a dollar cap.

    The same boot as `episode` -- variant draw, policy-server and block checks, codegen, config
    patch, preflight, manifest -- with the simulator-state tools mixed into the skill server and
    `leap.run_leap`'s loop on the agent thread. `--task` may name a variant slug rather than a
    task key, which is how a run and its pool simulators are pinned to one scene. Artifacts are
    what `leap.run_leap` describes.
    """
    task = chosen_task(args, settings)
    config = prepare_scene(
        task, args, settings, dir_prefix=f"leap-{args.condition}-{task.task_slug.replace('_', '-')}"
    )

    from agentic_robotics.adapters.toyblocks.leap.loop import ContextPolicy
    from agentic_robotics.adapters.toyblocks.leap.prices import fill_prompt_markers
    from agentic_robotics.adapters.toyblocks.leap.run_leap import (
        leap_agentic_thread,
        prompt_file_for,
    )
    from agentic_robotics.adapters.toyblocks.leap.skill_server import build_server_class

    prompt_file = args.prompt_file or prompt_file_for(args.condition)
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    # The developer's prompt carries [[MARKER]]s: prices come from OpenRouter's listing for the
    # run's model, the context-clear gate from the policy the loop enforces; a marker that cannot be
    # filled refuses the run.
    gate_percent = round(100 * ContextPolicy().self_wipe_fraction)
    prompt = fill_prompt_markers(
        prompt,
        config.model_config.model_id,
        {"MIN_CLEARABLE_CONTEXT_PERCENT": str(gate_percent)},
    )
    logger.info(f"Prompt file: {prompt_file}")

    boot.run_episode(
        boot.import_skill_module(config, args.skills, quiet_isaac=not args.verbose_isaac),
        config,
        agent_thread=leap_agentic_thread,
        agent_thread_kwargs={"args": args, "prompt": prompt},
        preflight=not args.no_preflight,
        server_factory=build_server_class,
    )


def run_pool_mode(args: argparse.Namespace, settings: ExperimentSettings) -> None:
    """One pre-warmed simulator for subagent trials, serving its tools over streamable HTTP.

    The same boot as `leap`, serving the skill server at ``http://127.0.0.1:PORT/mcp`` (or on
    every interface with --lan) instead of running an agent, until the process is stopped. No
    viewer: nobody watches a pool simulator, each would take a port, and the viewer's reset hook
    breaks on a state write before any step, which is how every trial starts. Start one per
    simulator the pool should hold, each pinned to the same variant slug the `leap` run uses,
    then pass their URLs to `leap` as `--pool-url`.
    """
    task = chosen_task(args, settings)
    require_free_ports(args.pool_port)
    config = prepare_scene(
        task, args, settings, dir_prefix=f"pool-{task.task_slug.replace('_', '-')}-{args.pool_port}"
    )
    config = dataclasses.replace(config, viewer=False)

    from agentic_robotics.adapters.toyblocks.leap.pool_server import serve_thread
    from agentic_robotics.adapters.toyblocks.leap.skill_server import (
        build_frame_counted_server_class,
        build_server_class,
    )

    boot.run_episode(
        boot.import_skill_module(config, args.skills, quiet_isaac=not args.verbose_isaac),
        config,
        agent_thread=serve_thread,
        agent_thread_kwargs={"host": args.host, "port": args.pool_port},
        preflight=not args.no_preflight,
        server_factory=build_frame_counted_server_class
        if args.frame_count_tool
        else build_server_class,
    )


def run_menu_mode(args: argparse.Namespace, settings: ExperimentSettings) -> None:
    """Boot Isaac with no scene and let the page choose what to load, one task after another.

    Four demos in one sitting is four menu choices rather than four process restarts, which is why
    this exists rather than running `page` repeatedly. Nothing happens until someone types: the
    first message is the task, and when that episode ends the page waits for the next one.
    """
    require_free_ports(args.port, VISER_PORT)

    tasks = load_tasks()

    # The adapter needs *some* built task in its config to import -- importing one task package
    # registers every other -- but it builds no environment until the page asks for one. The run
    # directory this creates holds nothing. Any built variant serves; the page draws again per load.
    boot_variant = next(
        (
            variant
            for task in tasks.values()
            for variant in task_variants.all_variants(task)
            if boot.task_package_is_built(variant.task_slug)
        ),
        None,
    )
    if boot_variant is None:
        raise SystemExit(
            "No task has a package built under data/tasks/. "
            "Build one with `agentic-robotics generate-env`."
        )
    config = experiment.build_config(
        settings,
        task_config=boot_variant,
        output_dir=boot.create_run_dir(
            "menu-boot-", Path(args.runs_dir) if args.runs_dir else None
        ),
        policy_port=args.policy_port,
    )

    from agentic_robotics.adapters.toyblocks.web_ui.server import agentic_web_thread

    skills = boot.import_skill_module(  # Boots Isaac; no scene yet
        config, args.skills, quiet_isaac=not args.verbose_isaac
    )

    from agentic_robotics.adapters.toyblocks.toyblocks_runner import SimulatorDispatcher

    dispatcher = SimulatorDispatcher()
    loader = OracleSceneLoader(
        dispatcher,
        skills,
        config,
        require_preflight=not args.no_preflight,
        require_block_check=not args.skip_block_check,
    )

    def serve(finished: threading.Event) -> None:
        agentic_web_thread(
            None,
            finished,
            host=args.host,
            port=args.port,
            task_from_page=True,
            task_loader=loader,
        )

    boot.run_dispatcher(dispatcher, serve, on_finish=loader.dispose_current)


def run_mock_mode(args: argparse.Namespace, settings: ExperimentSettings) -> None:
    """The page with no simulator behind it: the loop, the channel, and a stub tool surface.

    Everything except the scene. It still calls the model once someone types, so it costs money and
    not GPU.
    """
    import asyncio

    from agentic_robotics.adapters.toyblocks.web_ui.server import (
        build_mock_server,
        run_web_session,
    )

    require_free_ports(args.port)
    asyncio.run(run_web_session(build_mock_server(), host=args.host, port=args.port))


def run_portal_mode(args: argparse.Namespace, settings: ExperimentSettings) -> None:
    """Serve the run portal: every finished run, rendered from its transcript and frames.

    No simulator and no model -- it reads the runs directory off disk, so it is safe to leave
    running beside an episode and refresh to watch one land. With `--static` it writes the same
    pages as files instead, self-contained so a copied runs tree can be browsed anywhere.

    :param args: Parsed arguments; `--runs-dir`, `--static`, `--host` and `--port` are read
    """
    from agentic_robotics.adapters.toyblocks.portal import (
        DEFAULT_RUNS_ROOT,
        PORTAL_DIR_NAME,
        build_static,
        serve,
    )

    runs_root = Path(args.runs_dir) if args.runs_dir else DEFAULT_RUNS_ROOT
    if not runs_root.is_dir():
        raise SystemExit(f"No runs directory at {runs_root}.")

    if args.static is not None:
        out_dir = Path(args.static) if args.static else runs_root / PORTAL_DIR_NAME
        written = build_static(runs_root, out_dir)
        logger.info(f"Wrote {len(written)} portal pages; open {out_dir / 'index.html'}.")
        return

    port = args.port if args.port != CHAT_PORT else PORTAL_PORT
    require_free_ports(port)
    serve(runs_root, args.host, port)


def run_generate_env_mode(args: argparse.Namespace, settings: ExperimentSettings) -> None:
    """Stage-1 codegen for every PDDL variant of a task: one env package, with reset states, each.

    A task owns a directory of variant problems, and a run draws one at random, so all of them are
    built here rather than only the one a draw would pick. `--reset-states` (default 32) applies to
    each variant.
    """
    task = chosen_task_config(args, settings)
    variants = task_variants.all_variants(task)
    if not variants:
        raise SystemExit(
            f"Task '{task.task_slug}' has no PDDL variant to build under "
            f"{task_variants.PROBLEMS_DIR / task.task_slug}/ or "
            f"{task_variants.PROBLEMS_DIR / f'{task.task_slug}.pddl'}."
        )
    library = args.usd_library or glyph_library_for(task)
    for variant in variants:
        boot.ensure_task_package(
            variant.task_slug,
            variant.pddl_file,
            generate_env=True,
            reset_states=args.reset_states,
            envs=args.envs,
            seed=codegen_seed(variant, args),
            codegen_python=boot.resolve_isaac_python(args.python),
            force=args.force,
            usd_library=library,
        )
        print(f"  {variant.task_slug}: {boot.task_package_dir(variant.task_slug)}")
    print(f"Task '{task.task_slug}': {len(variants)} variant package(s) built against {library}.")


def run_serve_mode(args: argparse.Namespace, settings: ExperimentSettings) -> None:
    """Serve the ManiFlow policy the experiment names, with every flag derived from it.

    No path is typed by hand: `MANIFLOW_CONFIGS` holds one `ManiFlowConfig` per checkpoint, naming
    the directory and the step file together, because a hand-typed step file drifts from the
    checkpoint it points into, and `steps_200000` asked of a directory whose newest weights are
    `steps_170000` reads as a serving bug rather than a stale string. This checks the files it
    resolves to before the GPU is touched.

    The process is replaced rather than spawned: the server is what you started, so it owns the
    terminal, the logs and the signal that stops it.

    :param args: Parsed arguments; only `--python` is read, to name the interpreter
    :raises SystemExit: If the port is taken or the checkpoint is missing a file the server needs
    """
    import os

    maniflow = MANIFLOW_CONFIGS[settings.maniflow]
    if args.policy_port is not None:
        maniflow = dataclasses.replace(maniflow, policy_port=args.policy_port)

    checkpoint = maniflow.safetensors_path
    train_config = (
        boot.toyblocks_root() / "maniflow-tb/examples/Franka/train_maniflow_multimodal.yaml"
    )
    policy_config = maniflow.checkpoint_dir / "config.yaml"
    stats = maniflow.checkpoint_dir / "stats.json"

    for path, what in (
        (checkpoint, "config.maniflow.safetensors_path"),
        (policy_config, "the checkpoint's config.yaml"),
        (stats, "the checkpoint's stats.json"),
        (train_config, "the maniflow-tb train config"),
    ):
        if not path.is_file():
            raise SystemExit(f"No file at {path}, named by {what}.")

    require_free_ports(maniflow.policy_port)

    python = boot.resolve_isaac_python(args.python)
    hvla_path = boot.toyblocks_root() / "maniflow-tb"
    env = dict(os.environ)
    env["H_VLA_PATH"] = str(hvla_path)
    env["PYTHONPATH"] = ":".join(filter(None, (env.get("PYTHONPATH"), str(hvla_path))))

    argv = [
        str(python),
        "-m",
        "toyblocks.serving.serve_maniflow_policy",
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(policy_config),
        "--train_config",
        str(train_config),
        "--stats_path",
        str(stats),
        "--port",
        str(maniflow.policy_port),
        "--seed",
        str(settings.rng_seed),
        "--deterministic_cuda",
    ]
    logger.info(f"Serving {checkpoint.name} on port {maniflow.policy_port} under {python}.")
    os.execve(str(python), argv, env)


def run_tasks_mode(args: argparse.Namespace, settings: ExperimentSettings) -> None:
    """List the tasks, how many PDDL variants each owns, and how many are ready on this machine."""
    tasks = load_tasks()
    width = max(len(key) for key in tasks)
    for key, task in sorted(tasks.items()):
        variants = task_variants.all_variants(task)
        ready = sum(1 for v in variants if variant_available(v))
        state = "no pddl" if not variants else f"{ready}/{len(variants)} ready"
        prompt = "prompt" if task.prompt else "NO PROMPT"
        print(f"{key:<{width}}  {state:<14}  {prompt:<9}  {task_variants.PROBLEMS_DIR / key}")


MODES = {
    "episode": run_episode_mode,
    "human": run_human_mode,
    "page": run_page_mode,
    "menu": run_menu_mode,
    "mock": run_mock_mode,
    "portal": run_portal_mode,
    "generate-env": run_generate_env_mode,
    "serve": run_serve_mode,
    "tasks": run_tasks_mode,
    "leap": run_leap_mode,
    "pool": run_pool_mode,
}


def parse_args() -> argparse.Namespace:
    """Parse the command line for every mode at once, since they share most of it."""
    parser = argparse.ArgumentParser(
        prog="agentic-robotics",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=sorted(MODES), help="What to run")
    parser.add_argument(
        "--task",
        default=None,
        help=("Task key, or one variant's slug (default: the experiment's task)"),
    )
    parser.add_argument("--reset-states", type=int, default=32, help="Reset states Stage-1 samples")
    parser.add_argument("--envs", type=int, default=8, help="Parallel envs Stage-1 samples with")
    parser.add_argument(
        "--usd-library",
        default=None,
        help=(
            "Glyph asset library Stage-1 builds letter and digit blocks from "
            f"(default: {DEFAULT_GLYPH_LIBRARY}). Applies to codegen only -- a package that is "
            "already built or restored keeps the library it was built against"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed passed to Stage-1 codegen (default: derived from the task slug, so each "
        "variant package samples its own reset states)",
    )
    parser.add_argument(
        "--skills",
        default=None,
        help=(
            "Skill module to drive (default: the one implementing config.skill_backend, "
            "the configured backend's own module)"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("oracle", "maniflow"),
        default=None,
        help=(
            "Skill backend to run, overriding config.skill_backend (default: derived from "
            "--skills through runner_bootstrap.SKILL_MODULE_BACKENDS, else config.py's)"
        ),
    )
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="Start an episode even if some block cannot be named by any pixel",
    )
    parser.add_argument(
        "--physx-scale",
        type=int,
        default=1,
        help="Divide the PhysX GPU buffer capacities by this; 16 takes an agent process from "
        "8.4 GB to 6.1 GB on the Rao scenes so four fit on the workstation (2026-10-14). "
        "1 leaves them as ToyBlocks sets them",
    )
    parser.add_argument(
        "--verbose-isaac",
        action="store_true",
        help="Let Kit print its own warnings; by default it is held to errors only",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="generate-env: regenerate even if the package is already built or saved",
    )
    parser.add_argument(
        "--generate-env",
        action="store_true",
        help="Run Stage-1 codegen if the task package is missing, rather than printing its command",
    )
    parser.add_argument(
        "--skip-block-check",
        action="store_true",
        help="Run a ManiFlow backend even when the task's blocks do not match the checkpoint's",
    )
    parser.add_argument(
        "--python",
        default=None,
        help=(
            "Interpreter Stage-1 codegen runs under. Defaults to this one where it can import "
            f"toyblocks, else ${boot.CODEGEN_PYTHON_ENV_VAR}, else a known heavy venv."
        ),
    )
    parser.add_argument(
        "--experiment",
        default=experiment.DEFAULT_EXPERIMENT,
        help=(
            "Experiment whose settings this run uses: a name under the adapter's experiments/ "
            "directory or a path to a TOML file (default: default)"
        ),
    )
    parser.add_argument(
        "--runs-dir",
        default=None,
        help="Directory to write this run under, instead of $AGENTIC_ROBOTICS_RUNS_DIR or runs/",
    )
    parser.add_argument("--port", type=int, default=CHAT_PORT, help="Port the page is served on")
    parser.add_argument(
        "--static",
        nargs="?",
        const="",
        default=None,
        metavar="DIR",
        help=(
            "portal: write the pages as files to DIR (default: <runs dir>/portal) instead of "
            "serving"
        ),
    )
    leap = parser.add_argument_group("leap and pool")
    leap.add_argument(
        "--condition",
        choices=("naive", "harness"),
        default="naive",
        help="leap: naive (spawn_subagents) or harness (plan_experiment and run_experiment)",
    )
    leap.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="leap: the agent's prompt (default: the condition's)",
    )
    leap.add_argument(
        "--budget-usd", type=float, default=100.0, help="leap: dollar cap on model calls"
    )
    leap.add_argument(
        "--ignore-budget",
        action="store_true",
        help="leap: start even if OpenRouter credit cannot cover --budget-usd",
    )
    leap.add_argument(
        "--context-limit",
        type=int,
        default=None,
        help="leap: prompt tokens at which context is wiped (default: MAX_CONTEXT_SIZE_TOKENS)",
    )
    leap.add_argument(
        "--max-tool-calls",
        type=int,
        default=10_000,
        help="leap: ceiling on tool calls; the cap is the stop",
    )
    leap.add_argument(
        "--max-idle-turns",
        type=int,
        default=3,
        help="leap: end the phase, forfeiting the budget, after this many consecutive turns "
        "without a tool call",
    )
    leap.add_argument(
        "--pool-url",
        action="append",
        default=[],
        dest="pool_urls",
        help="leap: URL of a `pool` simulator, once per simulator; enables spawn_subagents",
    )
    leap.add_argument("--pool-port", type=int, default=8101, help="pool: port to serve on")
    leap.add_argument(
        "--frame-count-tool",
        action="store_true",
        help="pool: also serve get_frame_count, the video frame counter, for probes that cut a "
        "clip per call (le_skill_trials.py); not for a pool a LEAP agent will use, whose "
        "declaration check refuses the extra tool",
    )
    leap.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="leap: a finished leap run directory to continue from, in a fresh context, under "
        "this run's --budget-usd (the earlier spend counts against it)",
    )
    leap.add_argument(
        "--example-tasks",
        type=Path,
        default=None,
        help="leap: the hand-holding setting; copy this directory's example tasks "
        "(examples/agent/handholding_tasks.py) into the workspace's logs/example_tasks/ in place "
        "of the two example initial states",
    )
    leap.add_argument(
        "--example-states",
        type=Path,
        default=None,
        help="leap: copy this directory's *.json initial states into the workspace's "
        "logs/initial_states/ instead of the two LEAP example states under data/example_states/ "
        "(the release phase seeds data/example_states_release/, one cluttered scene)",
    )
    evaluation = parser.add_argument_group("episode, as a LEAP evaluation")
    evaluation.add_argument(
        "--baseline",
        choices=("control", "oracle", "naive", "harness"),
        default=None,
        help="episode: check the served tools against EvaluationAgent and serve this "
        "baseline's descriptions: control, the declared ones; naive or harness, the set in "
        "--docstrings; oracle, the developer's set (not yet written)",
    )
    evaluation.add_argument(
        "--instance",
        type=int,
        default=None,
        help="episode: reset-state index of the task instance to run (default: the package's "
        "random draw); the suite uses 0 and 1 for every baseline so comparisons are paired",
    )
    evaluation.add_argument(
        "--allow-declaration-mismatch",
        action="store_true",
        help="episode: if the served tools differ from EvaluationAgent's declaration, record the "
        "difference and run anyway (for a skill surface the declarations do not yet cover)",
    )
    evaluation.add_argument(
        "--docstrings",
        type=Path,
        default=None,
        help="episode: a docstring snapshot JSON (a leap run's final one) whose descriptions "
        "the naive or harness baseline is evaluated with",
    )
    evaluation.add_argument(
        "--model-host",
        choices=("openrouter", "claude-code"),
        default="openrouter",
        help="episode: what runs the model loop: openrouter, the repo's loop over the OpenAI "
        "Responses API; claude-code, the Claude Agent SDK's bundled CLI, with the skills served "
        "to it over HTTP (needs CLAUDE_CODE_OAUTH_TOKEN and the claudecode extra)",
    )
    evaluation.add_argument(
        "--effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default=None,
        help="episode with --model-host claude-code: the CLI's effort level (default: the "
        "model's own)",
    )
    evaluation.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="episode with --model-host claude-code: ceiling on tool-use round trips, the "
        "CLI's own stop (default: none; the motor cutoff and the call ceiling end the episode)",
    )
    rao = parser.add_argument_group("episode, as a Rao release evaluation")
    rao.add_argument(
        "--plan-mode",
        choices=("closed", "online"),
        default=None,
        help="episode: run a Rao instance closed-loop (plan-act) or online (a committed "
        "symbolic plan, each step's pixel chosen on the current image); see release/rao_modes.py",
    )
    leap.add_argument(
        "--model",
        default=None,
        help="leap and episode: model the agent and its subagents run on, by its "
        "OPENAI_API_MODEL_CONFIGS name (default: the experiment's model)",
    )
    parser.add_argument(
        "--maniflow",
        default=None,
        help=("serve and the pool: the MANIFLOW_CONFIGS key to use instead of the experiment's"),
    )
    parser.add_argument(
        "--policy-port",
        type=int,
        default=None,
        help="serve binds this port and every other mode connects to it, instead of the "
        "checkpoint entry's (for running several policy servers on one node)",
    )
    parser.add_argument(
        "--lan",
        dest="host",
        action="store_const",
        const="0.0.0.0",
        default="127.0.0.1",
        help=(
            "Serve the page on every interface instead of loopback. There is no authentication: "
            "whoever reaches the port can move the arm. Never implied, always typed."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Dispatch to one mode."""
    from agentic_robotics.core.logging import configure_logging

    configure_logging()  # Before the re-exec, so its "re-running under" line is printed
    args = parse_args()
    settings = settings_from_args(args)
    boot.PHYSX_SCALE = max(1, args.physx_scale)
    require_policy_server(args.mode, settings, getattr(args, "policy_port", None))
    require_matching_blocks(args, settings)
    reexec_under_isaac_python_if_needed(args)
    boot.resolve_openrouter_key()
    MODES[args.mode](args, settings)


if __name__ == "__main__":
    main()
