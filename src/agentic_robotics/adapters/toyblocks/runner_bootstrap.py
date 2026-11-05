"""Shared bootstrap for the runners that drive :mod:`inline_oracle`.

A runner is a task, a prompt, and an ordering. The ordering is identical for every task and is the
part that is easy to get wrong, so it lives here: build the run's configuration, then import the
skill module (which boots Isaac), then hand the tool surface to the agentic loop.

Everything here takes the task as an argument. Nothing in this module knows about a particular
task such as ``r_on_red``; the caller supplies that, and the prompt, which is the one thing that
should never be shared between tasks.

``cli.py`` is the one caller.
"""

from __future__ import annotations

import argparse
import functools
import importlib
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import threading
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from isaaclab.sim import SimulationCfg, SimulationContext

    from agentic_robotics.adapters.toyblocks.config import (
        SkillBackend,
        ToyBlocksConfig,
    )
    from agentic_robotics.adapters.toyblocks.toyblocks_mcp_server import ToyBlocksMCPServer
    from agentic_robotics.adapters.toyblocks.toyblocks_runner import SimulatorDispatcher

logger = logging.getLogger("agentic_robotics.examples.runner_bootstrap")

AGENTIC_ROBOTICS_ROOT = Path(__file__).resolve().parents[4]

NESTED_TOYBLOCKS_ROOT = AGENTIC_ROBOTICS_ROOT.parent
"""Where ToyBlocks sits when this repo is its `agentic_robotics/` submodule (the nested layout)."""


@functools.cache
def toyblocks_root() -> Path:
    """Find the ToyBlocks checkout without assuming how the two repos are arranged.

    Resolved on first call, not at import, so that modules needing only this repo's own code --
    the Search & Spell scorer and its known-answer check, the saved-task helpers -- import on a
    machine without ToyBlocks. The installed package is asked first, so a workstation holding the
    two as siblings works. That answer is unavailable from an interpreter without ToyBlocks
    installed -- the light venv, which the page's mock mode runs in -- so the nested layout (this
    repo cloned under a ToyBlocks checkout) is tried next, and only a machine where neither holds
    is an error. The answer is cached, so the lookup runs once per process.

    :return: The checkout root, the directory holding `src/toyblocks`
    :raises SystemExit: If neither route finds one
    """
    spec = importlib.util.find_spec("toyblocks")
    if spec is not None and spec.origin is not None:
        return Path(spec.origin).resolve().parents[2]
    if (NESTED_TOYBLOCKS_ROOT / "src/toyblocks").is_dir():
        return NESTED_TOYBLOCKS_ROOT
    raise SystemExit(
        "Cannot find the ToyBlocks checkout. It is neither importable from this interpreter "
        f"({sys.executable}) nor at {NESTED_TOYBLOCKS_ROOT}. Run from an interpreter with "
        "toyblocks installed, or clone this repo under a ToyBlocks checkout."
    )


def domain_path() -> Path:
    """The blocks PDDL domain directory inside the ToyBlocks checkout."""
    return toyblocks_root() / "src/toyblocks/tamp/blocks"


TASKS_DIR = AGENTIC_ROBOTICS_ROOT / "data/tasks"
"""Task instances -- slug, PDDL, prompt -- as data rather than as constants in a runner, and the
generated task packages beside them.

A generated package is version-controlled here and imports as ``agentic_robotics.tasks.<slug>``
(see :mod:`agentic_robotics.tasks`), so ToyBlocks's tree is never written to. The packages are
small -- the four demo scenes are 360K together -- and `reset_states.pkl` is the set of scenes an
episode actually draws from, so keeping it is what makes two sessions comparable rather than
merely similar.
"""

TASK_PACKAGE_ROOT = "agentic_robotics.tasks"
"""Package the generated task packages import under."""


def task_package_module(task_name: str) -> str:
    """Module name a task's generated package imports as."""
    return f"{TASK_PACKAGE_ROOT}.{task_name}"


FALLBACK_RUNS_DIR = AGENTIC_ROBOTICS_ROOT / "runs"
"""Where runs go when the configured runs root does not exist on this machine."""

CODEGEN_PYTHON_ENV_VAR = "AGENTIC_ROBOTICS_PYTHON"
"""Environment variable naming the interpreter Stage-1 codegen runs under, if not this one."""

HEAVY_PYTHON_CANDIDATES = (Path("python"),)
"""Interpreters tried for codegen when this one cannot import ToyBlocks, in order."""


def resolve_isaac_python(explicit: str | Path | None = None) -> Path:
    """Pick an interpreter that can import ToyBlocks, and say so when there is none.

    Codegen boots Isaac, so it needs an interpreter with ToyBlocks in it. The runner's own
    interpreter is that one whenever the runner itself boots Isaac, which is the workstation case;
    in a cluster job a runner may be started from the light venv, and the heavy one has to be named.
    An explicit answer always wins, because neither guess is right on every machine.

    :param explicit: An interpreter given on the command line, which is used unchecked
    :return: Path to the interpreter
    :raises SystemExit: If no candidate can import ToyBlocks
    """
    if explicit is not None:
        return Path(explicit)

    from_env = os.environ.get(CODEGEN_PYTHON_ENV_VAR)
    if from_env:
        return Path(from_env)

    if importlib.util.find_spec("toyblocks") is not None:
        return Path(sys.executable)

    for candidate in HEAVY_PYTHON_CANDIDATES:
        if not candidate.is_file():
            continue
        probe = subprocess.run(
            [str(candidate), "-c", "import toyblocks"], capture_output=True, check=False
        )
        if probe.returncode == 0:
            return candidate

    tried = ", ".join(str(c) for c in (Path(sys.executable), *HEAVY_PYTHON_CANDIDATES))
    raise SystemExit(
        f"No interpreter found that can import toyblocks (tried {tried}). "
        f"Pass --python, or set {CODEGEN_PYTHON_ENV_VAR}."
    )


def require_eula_acceptance() -> None:
    """Refuse to boot Isaac where its licence prompt would hang or kill the process.

    Isaac reads a Yes/No off stdin when ``OMNI_KIT_ACCEPT_EULA`` is unset. A subprocess with no
    terminal answers EOF and dies, and codegen therefore fails with nothing that says why. Accepting
    a licence on someone's behalf is not this program's decision, so it says what to set instead.

    :raises SystemExit: If the variable is unset
    """
    if not os.environ.get("OMNI_KIT_ACCEPT_EULA"):
        raise SystemExit(
            "OMNI_KIT_ACCEPT_EULA is unset. Isaac stops at its licence prompt, and a subprocess "
            "with no terminal dies there. Export OMNI_KIT_ACCEPT_EULA=YES to accept it."
        )


def resolve_runs_dir(config_dir: Path, explicit: Path | None = None) -> Path:
    """Decide where this run's directory is created under.

    The configured runs root (`$AGENTIC_ROBOTICS_RUNS_DIR`, or a shared directory a job mounts)
    is the answer where it exists; where it does not, runs land in this repo rather than nowhere.

    :param config_dir: The configured runs root
    :param explicit: A directory given on the command line
    :return: An existing directory to create the run's own directory under
    """
    if explicit is not None:
        explicit.mkdir(parents=True, exist_ok=True)
        return explicit
    if config_dir.is_dir():
        return config_dir
    logger.info(f"{config_dir} is not on this machine; runs go under {FALLBACK_RUNS_DIR}.")
    FALLBACK_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return FALLBACK_RUNS_DIR


SKILL_MODULES = {"oracle": "inline_oracle", "maniflow": "maniflow_skills"}
"""Module holding the skill surface for each `SkillBackend`. Importing one boots Isaac Sim."""

SKILLS_PACKAGE = "agentic_robotics.adapters.toyblocks.skills"
"""Where a bare skill-module name such as `inline_oracle` resolves; a dotted name is imported as
is."""

SKILL_MODULE_BACKENDS = {
    "inline_oracle": "oracle",
    "maniflow_skills": "maniflow",
}
"""The `SkillBackend` each known skill module implements, for `agentic-robotics --skills` to derive
`--backend` from. A module absent here needs `--backend` spelled out."""


def default_skill_module(backend: SkillBackend) -> str:
    """The skill module implementing a backend.

    Derived rather than pinned: a constant here and a backend in the experiment settings would be
    two defaults for one decision, so changing the backend would leave the module behind, and a
    run would die at server construction reporting that the two disagree.

    :param backend: The configured skill backend
    :return: Module name for :func:`import_skill_module`
    """
    return SKILL_MODULES[backend]


def task_package_dir(task_name: str) -> Path:
    """Directory a task's generated env package lives in."""
    return TASKS_DIR / task_name


def task_package_is_built(task_name: str) -> bool:
    """Whether the generated task package for a task exists *and* carries its reset states.

    Both files, not just the module. Stage-1 writes the package first and saves
    `reset_states.pkl` at the end, so a codegen still running -- or one that died during the
    reset-state phase -- leaves a directory that imports fine and randomizes nothing: the env
    falls through to its `reset_states is None` branch, every episode gets the codegen spawn
    layout, and a session of ten runs is ten copies of one scene. Asking only for `__init__.py`
    would make that indistinguishable from a finished package, to this function and to every
    caller that keeps or regenerates one on its word.
    """
    package_dir = task_package_dir(task_name)
    return (package_dir / "__init__.py").is_file() and (package_dir / "reset_states.pkl").is_file()


def seed_for_task(task_name: str) -> int:
    """A codegen seed derived from the task slug, so every variant package samples differently.

    Variants of one family differ only in the PDDL, so a shared seed draws the same positions for
    the same object set: with the default seed 0 for every package, only 31 of the 64 LEAP
    variant packages come out distinct. CRC-32 of the slug is
    stable across runs and machines, which is what makes a rebuilt package reproducible.
    """
    return zlib.crc32(task_name.encode("utf-8"))


def stage_1_command(
    pddl_path: Path,
    *,
    reset_states: int,
    envs: int,
    seed: int,
    codegen_python: Path | None = None,
    usd_library: str | None = None,
) -> list[str]:
    """Build the Stage-1 codegen command that generates a task's env package.

    ``--domain_path`` is passed **relative to** :func:`toyblocks_root`, which is the cwd
    :func:`ensure_task_package` runs this under. That is not cosmetic: Stage 1 turns it into a
    module name with ``resolve_module_from_path``, which walks the path parts for the first one
    already in ``sys.modules``. The absolute form of a checkout named ``toyblocks`` has that
    name as a path part, which *is* the imported ``toyblocks`` package, so it resolves to the
    unimportable ``toyblocks.src.toyblocks.tamp.blocks``, every predicate class silently fails
    to load, and the sampled reset states come out as the codegen spawn layout.

    ``--pddl`` is absolute instead, and has to be: a problem file living in this repo has no path
    relative to the ToyBlocks checkout on a machine where the two are siblings. Stage 1 only opens
    it and takes the task slug from its basename, so nothing there depends on its shape.

    :param pddl_path: PDDL problem to generate an environment for
    :param reset_states: Reset states to sample; must divide by ``envs``
    :param envs: Parallel environments used for reset sampling
    :param seed: Seed passed to Stage-1 codegen
    :param codegen_python: Interpreter to run it under; resolved from this machine if omitted
    :param usd_library: Glyph asset library the scene's letter and digit blocks spawn from;
        omitted leaves Stage-1 on its own default, which is ``train_blocks``
    :return: The command, as an argument list
    """
    command = [
        str(codegen_python or resolve_isaac_python()),
        "scripts/generate_task_from_pddl.py",
        "--domain_path",
        str(domain_path().relative_to(toyblocks_root())),
        "--pddl",
        str(pddl_path),
        "--num_reset_states",
        str(reset_states),
        "--num-envs",
        str(envs),
        "--seed",
        str(seed),
        "--output_dir",
        str(TASKS_DIR),
        "--headless",
    ]
    # Which library a package was built against is what `block_asset_check` compares to the
    # checkpoint's, and a package carries no record of the flag beyond the USD paths in its
    # `env.py` -- so passing the wrong one here is not visible until a ManiFlow run grasps air.
    if usd_library:
        command += ["--usd_library", usd_library]
    return command


def ensure_task_package(
    task_name: str,
    pddl_path: Path,
    *,
    generate_env: bool,
    reset_states: int,
    envs: int,
    seed: int,
    codegen_python: Path | None = None,
    force: bool = False,
    usd_library: str | None = None,
) -> None:
    """Ensure a task's generated package exists, generating it if asked to.

    :param task_name: Slug the package is generated under
    :param pddl_path: PDDL problem to generate it from
    :param generate_env: Whether to run Stage-1 codegen rather than exiting with its command
    :param codegen_python: Interpreter codegen runs under; resolved from this machine if omitted
    :param force: Regenerate even when a built package exists. Without it, a package whose reset
        states were sampled under different constraints -- a changed predicate, a widened window
        -- is silently kept, because "the package exists" is the only question asked.
    :param usd_library: Glyph asset library to build against; see :func:`stage_1_command`. It
        reaches only the codegen branch, since an already-built package was built against
        whatever library it was built against and this cannot change that.
    :raises SystemExit: If the package is missing and ``--generate-env`` was not passed
    """
    package_dir = task_package_dir(task_name)

    if force:
        if package_dir.exists():
            shutil.rmtree(package_dir)
            logger.info(f"Removed {package_dir}; --force regenerates it.")
    elif task_package_is_built(task_name):
        logger.info(f"Task package present at {package_dir}.")
        return

    command = stage_1_command(
        pddl_path,
        reset_states=reset_states,
        envs=envs,
        seed=seed,
        codegen_python=codegen_python,
        usd_library=usd_library,
    )

    if not generate_env:
        printable = " \\\n    ".join(command)
        raise SystemExit(
            f"Task package missing at {package_dir}.\n"
            f"Re-run with --generate-env, or build it yourself (from {toyblocks_root()}):"
            f"\n\n    {printable}\n"
        )

    require_eula_acceptance()
    logger.info(f"Generating the task package (this boots Isaac): {' '.join(command)}")
    subprocess.run(command, check=True, cwd=toyblocks_root())
    # `agentic_robotics.tasks` may already be imported, and its finder caches the directory's
    # contents, so a package generated afterwards would be invisible to `import_module`.
    importlib.invalidate_caches()

    if not task_package_is_built(task_name):
        raise RuntimeError(f"Stage-1 codegen finished but {package_dir} is still not built.")

    # Codegen prints `Validation failed for <block> in N envs` and carries on, having written an
    # unvalidated pose. Read the states it actually produced rather than trusting that it exited
    # zero. Imported here rather than at module scope to keep this module's import order alone.
    from agentic_robotics.adapters.toyblocks.reset_state_check import check_package

    report = check_package(package_dir)
    logger.info(report.summary())
    if not report.ok:
        logger.warning(
            f"'{task_name}' carries states the arm cannot fully act in. Blocks beyond "
            f"{0.55:.2f} m are unreachable and overlapping blocks are shoved apart while "
            f"settling, so neither shows up as an error at run time. Regenerate with --force "
            f"after loosening the scene (fewer blocks, or a wider sampling window), or pin "
            f"`instance_ids` past the bad states."
        )


def load_leap_tasks() -> dict[str, Any]:
    """The LEAP family `TaskConfig`s from `prompts/evaluation.leap_tasks()`, keyed by family slug.

    `config.TOYBLOCKS_TASKS` does not include these; `agentic-robotics` merges them into its task
    table so a LEAP variant slug (`stack_outside_isolated_rotate_00`) resolves like any other
    task's variant.
    """
    from agentic_robotics.adapters.toyblocks.leap.prompts.evaluation import leap_tasks

    return dict(leap_tasks())


def create_run_dir(dir_prefix: str, runs_dir: Path | None = None) -> Path:
    """Create this run's directory under the runs directory and return it.

    :param dir_prefix: Prefix of the directory's name; a timestamp follows
    :param runs_dir: Directory to create it under; the configured default, or this machine's
        fallback, when omitted
    """
    from agentic_robotics.adapters.toyblocks.config import DEFAULT_RUNS_ROOT
    from agentic_robotics.core.filesystem import create_unique_dir

    output_root_dir = resolve_runs_dir(DEFAULT_RUNS_ROOT, runs_dir)
    output_dir = create_unique_dir(parent_dir=output_root_dir, dir_prefix=dir_prefix)
    logger.info(f"Run output directory: {output_dir}")
    return output_dir


def parse_args(description: str) -> argparse.Namespace:
    """Parse a runner's command-line arguments.

    Only `--generate-env` is a flag, because it selects an action rather than a value. Every
    configured value lives in `config.py` or in the constants at the top of a runner: a flag
    default such as `--model-id` would silently outrank `config.py`, and a run would record the
    model the flag named rather than the one the config asked for.
    """
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--generate-env",
        action="store_true",
        help=(
            "Run Stage-1 codegen as a subprocess if the autogen package is missing, instead of "
            "exiting with the command printed."
        ),
    )
    return parser.parse_args()


def prepare_import_path() -> None:
    """Make `agentic_robotics` importable when running out of a venv that lacks it."""
    if importlib.util.find_spec("agentic_robotics") is None:
        sys.path.insert(0, str(AGENTIC_ROBOTICS_ROOT / "src"))


OPENROUTER_KEY_FILE = Path.home() / ".openrouter_key"
"""Where the OpenRouter key is read from when `OPENROUTER_API_KEY` is unset.

A cluster job gets the key as a secret in its environment and needs none of this. A
workstation has no such mechanism, and the alternative to a file is exporting the key from a
shell profile -- which is the same secret on the same disk, in a file that is also sourced by
every unrelated shell. The narrower thing is a file that only this code reads.
"""


def resolve_openrouter_key() -> None:
    """Populate `OPENROUTER_API_KEY` from `~/.openrouter_key` if the environment lacks it.

    Resolved *into the environment* rather than handed to the client, so `core`'s contract stays
    "read one environment variable" and every entry point inherits the fallback without knowing
    it exists -- including the subprocess a re-exec creates, which is handed this environment.

    The environment always wins. A job's secret is the live value and a file in `$HOME` is
    whatever was last written there, so a stale file must never be able to shadow a good secret.

    Silent when there is no file and no variable: `OpenAIModelClient` asserts on a missing key
    and names the environment variable, which is the thing to fix in either case.
    """
    if os.environ.get("OPENROUTER_API_KEY"):
        return

    if not OPENROUTER_KEY_FILE.is_file():
        return

    # Refuse rather than warn. A warning about file permissions is read once and then scrolls
    # past every session, and the failure it warns about is one this code caused by reading a
    # key anyone on the machine can also read.
    mode = OPENROUTER_KEY_FILE.stat().st_mode
    if mode & 0o077:
        raise SystemExit(
            f"{OPENROUTER_KEY_FILE} is readable by other users (mode {mode & 0o777:03o}). "
            f"Run `chmod 600 {OPENROUTER_KEY_FILE}`, or export OPENROUTER_API_KEY instead."
        )

    key = OPENROUTER_KEY_FILE.read_text().strip()
    if not key:
        logger.warning(f"{OPENROUTER_KEY_FILE} is empty; leaving OPENROUTER_API_KEY unset.")
        return

    os.environ["OPENROUTER_API_KEY"] = key
    logger.info(f"Read OPENROUTER_API_KEY from {OPENROUTER_KEY_FILE} ({len(key)} chars).")


ISAAC_LOG_SETTINGS = (
    "--/log/level=error",
    "--/log/outputStreamLevel=error",
)
"""Kit log settings applied unless a caller asks for Isaac's own output.

Kit prints something over a hundred `[Warning]` lines per boot -- GLFW failing to open a display
on a headless machine, ECC being enabled, a deprecated extension, and then one
`FabricManager::initializePointInstancer mismatched prototypes` per debug-marker prim on every
scene load. None of it is actionable here and all of it buries this repo's own logging, which is
the only output that says what the agent did.

Passed by appending to `sys.argv`, which is how Kit takes settings and how `AppLauncher` itself
passes them -- it appends `--/plugins/carb.tasking.plugin/threadCount=...` the same way.
`launch_app` reads argv with `parse_known_args`, so these survive its parser untouched.
"""


def quiet_isaac_logs() -> None:
    """Ask Kit for errors only, before anything launches it.

    Must run before the skill-module import: these are read when the app starts, and a warning
    printed during startup is already gone by the time any setting could be changed.
    """
    for setting in ISAAC_LOG_SETTINGS:
        if setting not in sys.argv:
            sys.argv.append(setting)
    logger.info("Isaac logs set to errors only; pass --verbose-isaac to see its warnings.")


def _quiet_isaac_logs_post_launch() -> None:
    """Re-apply the log settings through carb, for warnings raised after startup.

    The per-scene `initializePointInstancer` warnings are the bulk of the noise and they come
    long after launch, so they are reachable this way even if the argv route missed. Guarded:
    carb is only importable once Isaac has launched, and a wrong setting name here is inert
    rather than fatal -- worth attempting precisely because it costs nothing when it fails.
    """
    try:
        import carb

        settings = carb.settings.get_settings()
        settings.set("/log/level", "error")
        settings.set("/log/outputStreamLevel", "error")
    except Exception as error:  # Never worth failing a run over log verbosity
        logger.debug(f"Could not quiet Isaac logs through carb: {type(error).__name__}: {error}")


PHYSX_SCALE = 1
"""Divisor applied to the PhysX GPU buffer capacities of every environment this process builds;
`agentic-robotics --physx-scale` sets it. ToyBlocks sizes the buffers for many-env datagen
(`gpu_max_rigid_patch_count` 2**22 in `base_env_cfg.py`, Isaac Lab's defaults for the rest),
and they are most of an agent process's GPU memory: 8.4 GB per process at 1, 6.1 GB at 16 on
the one-env Blocksworld scenes, so four processes
fit on the 32 GB workstation instead of three. Left at 1, nothing changes."""

PHYSX_CAPACITY_FIELDS = (
    "gpu_max_rigid_contact_count",
    "gpu_max_rigid_patch_count",
    "gpu_found_lost_pairs_capacity",
    "gpu_found_lost_aggregate_pairs_capacity",
    "gpu_total_aggregate_pairs_capacity",
    "gpu_collision_stack_size",
    "gpu_heap_capacity",
    "gpu_temp_buffer_capacity",
)


def shrink_physx_capacities(scale: int) -> None:
    """Divide the PhysX GPU buffer capacities by `scale` on every `SimulationCfg` an environment
    is built with from now on.

    The env's `SimulationCfg` is built when ToyBlocks's config classes are imported, before any
    caller could reach its defaults, so the division is applied where the simulation is
    constructed: a wrapper on `SimulationContext.__init__`, which receives the cfg. Must run
    after Isaac has launched (the import of `isaaclab.sim` needs it) and before the environment
    is built, which is where `import_skill_module` calls it.
    """
    from isaaclab.sim import simulation_context as sc

    original = sc.SimulationContext.__init__

    def patched(self: SimulationContext, cfg: SimulationCfg | None = None) -> None:
        physx = getattr(cfg, "physx", None)
        if physx is not None:
            for name in PHYSX_CAPACITY_FIELDS:
                setattr(physx, name, max(1024, int(getattr(physx, name)) // scale))
            logger.info(
                f"PhysX GPU buffer capacities divided by {scale} "
                f"(rigid patches {physx.gpu_max_rigid_patch_count:,}, "
                f"contacts {physx.gpu_max_rigid_contact_count:,})."
            )
        original(self, cfg)

    sc.SimulationContext.__init__ = patched


def import_skill_module(
    config: ToyBlocksConfig, module_name: str | None = None, *, quiet_isaac: bool = True
) -> ModuleType:
    """Import the skill module, which boots Isaac Sim.

    Late, and inside a function, on purpose: importing it imports the ToyBlocks adapter, which
    launches Isaac. A bare module name resolves inside :data:`SKILLS_PACKAGE`.

    :param config: The run's configuration; its backend picks the module when none is named
    :param module_name: Skill module to import; the configured backend's own unless a caller names
        one
    :param quiet_isaac: Whether to hold Kit to errors only, so this repo's logging is readable
    :return: The imported skill module
    """
    require_eula_acceptance()
    if quiet_isaac:
        quiet_isaac_logs()

    name = module_name or default_skill_module(config.skill_backend)
    skills = importlib.import_module(name if "." in name else f"{SKILLS_PACKAGE}.{name}")

    if quiet_isaac:
        _quiet_isaac_logs_post_launch()
    if PHYSX_SCALE > 1:
        shrink_physx_capacities(PHYSX_SCALE)

    return skills


SIM_THREAD_SERVERS = {"oracle": "SimThreadOracleServer", "maniflow": "SimThreadManiFlowServer"}
"""Module-level symbol each skill module is expected to expose, per `SkillBackend`."""


def sim_thread_server_class(skills: ModuleType, config: ToyBlocksConfig) -> type:
    """The skill module's simulator-thread MCP server for the backend `config` names.

    Which skills an episode runs is `config.skill_backend`, not a symbol name written into the
    shared path. A module that does not implement the configured backend says so here rather than
    booting the other one.

    :param skills: The imported skill module, from :func:`import_skill_module`
    :param config: The adapter's `ToyBlocksConfig`
    :return: The server class to construct
    :raises SystemExit: If the module exposes no server for the configured backend
    """
    name = SIM_THREAD_SERVERS[config.skill_backend]
    server = getattr(skills, name, None)
    if server is None:
        raise SystemExit(
            f"config.skill_backend is '{config.skill_backend}' and {skills.__name__} exposes no "
            f"module-level '{name}'. Point --skills at a module that does, or set skill_backend "
            f"in config.py to a backend this one implements."
        )
    return server


def run_dispatcher(
    dispatcher: SimulatorDispatcher,
    thread_target: Callable[[threading.Event], None],
    *,
    on_finish: Callable[[], None] | None = None,
) -> None:
    """Drive the simulator's thread for a second thread that builds and disposes scenes itself.

    :func:`run_episode` builds one server up front; this builds none, because the page decides
    what to load and when. The second thread gets the dispatcher's `finished` event and hands
    scene construction, resets and disposal back to this thread through the dispatcher.

    :param dispatcher: The `SimulatorDispatcher` skill bodies are handed to
    :param thread_target: Runs on the second thread, taking the `finished` event to set on exit
    :param on_finish: Run on this thread after the second one ends, e.g. to dispose the last scene
    """
    finished = threading.Event()
    thread = threading.Thread(target=thread_target, args=(finished,), daemon=True)
    thread.start()

    try:
        dispatcher.execute(finished)
        thread.join()
    finally:
        if on_finish is not None:
            on_finish()


def run_episode(
    skills: ModuleType,
    config: ToyBlocksConfig,
    *,
    agent_thread: Callable[..., None] | None = None,
    agent_thread_kwargs: dict | None = None,
    preflight: bool = True,
    server_factory: Callable[[ModuleType, Any], type] | None = None,
    dispatcher: SimulatorDispatcher | None = None,
) -> None:
    """Expose the skill module's tool surface over MCP and run the agentic loop.

    Isaac must be stepped from the thread that created it, so the simulator stays on this thread
    and the agentic loop runs on a second one.

    :param skills: The imported skill module, from :func:`import_skill_module`
    :param config: The run's configuration, which the server is built from and which is written
        to the run directory as ``config.json`` before the agent's first turn
    :param agent_thread: What runs on the second thread, taking ``(server, finished)``; defaults
        to `toyblocks_runner._agentic_thread`, which runs the episode and exits. `web_ui`'s
        `agentic_web_thread` is the other one, which serves a page around the same episode.
    :param agent_thread_kwargs: Extra keyword arguments for ``agent_thread``
    :param preflight: Whether to refuse a scene whose blocks no pixel can name
    :param server_factory: Resolves the server class to build from the skill module and the
        config; defaults to :func:`sim_thread_server_class`. The LEAP modes pass one that mixes
        the simulator-state tools into the same class.
    :param dispatcher: The dispatcher the server's skills are handed to; defaults to a fresh
        `SimulatorDispatcher`.
    """
    from agentic_robotics.adapters.toyblocks.experiment import write_resolved_config
    from agentic_robotics.adapters.toyblocks.toyblocks_run_manifest import write_run_manifest
    from agentic_robotics.adapters.toyblocks.toyblocks_runner import (
        SimulatorDispatcher,
        _agentic_thread,
    )

    logger.info(f"Wrote the resolved configuration to {write_resolved_config(config)}.")
    dispatcher = dispatcher if dispatcher is not None else SimulatorDispatcher()
    server_class = (server_factory or sim_thread_server_class)(skills, config)
    mcp_server = server_class(dispatcher, config)

    # On this thread, which is the simulator's, and before the agent's first turn: the check
    # reads the live scene, and an episode nobody can act in is not worth the model calls.
    if preflight:
        from agentic_robotics.adapters.toyblocks.scene_preflight import preflight_scene

        preflight_scene(mcp_server)

    # Written after the server is built -- the earliest moment the tool surface and the rendered
    # resolution exist -- and before the agent's first turn, so a run that dies mid-episode still
    # leaves a manifest saying what produced it. Telemetry is never worth an episode.
    try:
        manifest_path = write_run_manifest(mcp_server, config)
        logger.info(f"Wrote run manifest to {manifest_path}.")
    except Exception:
        logger.exception("Could not write the run manifest; continuing without one.")

    logger.info(f"Tools handed to the model: {mcp_server.registered_tools}")
    logger.info(f"Prompt: {config.task_config.prompt}")

    finished = threading.Event()
    thread = threading.Thread(
        target=agent_thread or _agentic_thread,
        args=(mcp_server, finished),
        kwargs=agent_thread_kwargs or {},
        daemon=True,
    )
    thread.start()

    status = 0
    try:
        dispatcher.execute(finished)
        thread.join()
    except BaseException:
        # The teardown below ends the process, so the exception would never reach the top level:
        # log it here and carry its status out.
        logger.exception("The episode failed.")
        status = 1
    finally:
        logger.info(f"Run artifacts in {config.output_dir}")
        shutdown_or_exit(mcp_server, status=status)


def shutdown_or_exit(
    mcp_server: ToyBlocksMCPServer, timeout_s: float = 60.0, status: int = 0
) -> None:
    """Shut the server down from the simulator's thread, then end the process either way.

    `shutdown` exports the video and closes the gym environment, and closing Isaac can hang
    forever: the first LEAP pilot's process spun at full CPU for an hour after its summary was
    written, holding the GPU, and Stage-1 codegen exits directly for the same reason. The
    shutdown gets `timeout_s` on a helper thread, which is ample for the video export; if it is
    still running after that, the process is ended outright, since everything it would still do
    is the hang.

    A shutdown that *finishes* is not enough either: the 2026-10-02 pilots closed their environments
    within the timeout and then sat for twenty minutes at interpreter exit, held by Kit's own
    non-daemon threads, until they were killed by hand. So the process ends here in both cases.
    Every caller is the end of an `agentic-robotics` mode, and the artifacts are on disk before this
    is reached.

    The timeout cannot be relied on either: the 2026-10-02 delegation check's parent wrote its
    video and then sat in `env.close()` with the interpreter lock held, so this thread never
    got to time the helper out (main thread and helper both in a futex wait, no log line after
    the video's). What the shutdown has to do is close the video writer, which is pure Python
    and quick; closing the Isaac environment is what hangs, and the process ends anyway. So the
    video writer is closed here (`ToyBlocksMCPServer.shutdown_video`), on a helper thread under
    the timeout, and the environment is never closed.
    """

    def close_video() -> None:
        mcp_server.shutdown_video()
        mcp_server.has_shutdown = True

    worker = threading.Thread(target=close_video, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        logger.warning(
            f"The video export did not finish within {timeout_s:.0f} s; exiting the process."
        )
    else:
        logger.info("Video exported; exiting the process without closing Isaac.")
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
