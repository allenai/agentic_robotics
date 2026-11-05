"""The LEAP experimentation phase, run as `agentic-robotics leap`:

    agentic-robotics leap --task ai2_easy --condition naive --budget-usd 5 \\
        --pool-url http://127.0.0.1:8101/mcp

`agentic-robotics` owns the boot (variant draw, policy-server and block checks, codegen, config
patch, preflight, manifest); this module supplies what runs on the agent thread: the composed server
(the simulator's skills, perception and state tools; the file, docstring, context and subagent or
experiment tools locally) and `LeapAgenticLoop` under a dollar cap. The prompt comes from
`--prompt-file`, which defaults to the developer's `experimentation.md` under the ABC package's
`prompts/` directory (`prompt_file_for`). Artifacts land in the run directory: `transcript.jsonl`
and `images/`, `workspace/notes/`, `workspace/logs/docstrings/`, `phase_summary.json`, and
`video.mp4`.

`--resume RUN_DIR` continues a finished phase in a fresh context: the earlier run's workspace is
copied in, its final docstring snapshot is applied before the initial snapshot, the ledger starts
at what it spent, and the agent's first system message says so. The developer's plan is to launch at
$5, resume to $10, then to $50; each stop overshoots by about one turn.

Before the first model call, the tool set the agent will see is checked against the baseline's
declaration (`NaiveAgent` or `HarnessAgent` from the ABCs): names, descriptions and input schemas,
or the run refuses to start. Both baselines declare a subagent tool, so both need at least one
`agentic-robotics pool` simulator, pinned to the same variant slug, passed as `--pool-url`.

The boot is `cli.py`'s alone, so this phase gets every check an episode gets; a copy of the boot
without the block check let two pilots run on the wrong glyph library (2026-09-29).
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import argparse
    import threading

    from agentic_robotics.adapters.toyblocks.leap.cost import CostLedger
    from agentic_robotics.adapters.toyblocks.leap.workspace import Workspace
    from agentic_robotics.adapters.toyblocks.toyblocks_mcp_server import ToyBlocksMCPServer

logger = logging.getLogger("agentic_robotics.examples.leap.run_leap")

CONDITIONS = ("naive", "harness")
"""The experimenting baselines, by the older name `agentic-robotics --condition` still uses."""


def prompt_file_for(condition: str) -> Path:
    """The developer's experimentation prompt, one file for both baselines: what differs
    between Naive and Harness is the tool set, and the Harness-specific sentence lives in
    `run_experiment`'s docstring."""
    from agentic_robotics.adapters.toyblocks.leap.prompts_dir import PROMPTS_DIR

    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; one of {CONDITIONS}.")
    return PROMPTS_DIR / "experimentation.md"


async def run_experimentation_phase(
    server: ToyBlocksMCPServer, args: argparse.Namespace, prompt: str
) -> None:
    """Connect an agent to the composed server and run the phase; writes `phase_summary.json`."""
    from mcp import Client as MCPClient

    from agentic_robotics.adapters.toyblocks.leap import messages
    from agentic_robotics.adapters.toyblocks.leap.composed_server import (
        ComposedServer,
        register_docstring_tool,
    )
    from agentic_robotics.adapters.toyblocks.leap.cost import CostLedger
    from agentic_robotics.adapters.toyblocks.leap.declarations import (
        assert_serves,
        declared_descriptions,
    )
    from agentic_robotics.adapters.toyblocks.leap.harness import (
        ExperimentHarness,
        register_harness_tools,
    )
    from agentic_robotics.adapters.toyblocks.leap.leap_agents import (
        HarnessAgent,
        NaiveAgent,
        NoOpMotorSkillsAgent,
    )
    from agentic_robotics.adapters.toyblocks.leap.loop import (
        ContextControl,
        ContextPolicy,
        LeapAgenticLoop,
        register_context_tool,
    )
    from agentic_robotics.adapters.toyblocks.leap.motor_skills_agent import MotorSkillsAgent
    from agentic_robotics.adapters.toyblocks.leap.openrouter_budget import require_budget
    from agentic_robotics.adapters.toyblocks.leap.subagents import (
        DEFAULT_PREAMBLE_FILE,
        SimulatorPool,
        SubagentSpawner,
        register_spawn_tool,
    )
    from agentic_robotics.adapters.toyblocks.leap.timing import tool_call_seconds
    from agentic_robotics.adapters.toyblocks.leap.workspace import Workspace, register_file_tools
    from agentic_robotics.core.foundation_models import (
        OpenAIModelClient,
        Transcript,
        create_openai_message,
    )

    baseline = {"naive": NaiveAgent, "harness": HarnessAgent}[args.condition]
    if not args.pool_urls:
        raise SystemExit(
            f"The {args.condition} baseline needs at least one --pool-url: its subagents run on "
            "pool simulators."
        )

    config = server.config
    # The baselines in `leap_agents.py` are composed from the oracle motor-skill ABC. The
    # ManiFlow skills carry a `steps` parameter and their own docstrings, declared in
    # `leap/maniflow_skills/maniflow_motor_skills_agent.py`, so under the ManiFlow backend the
    # same composition is rebuilt on that ABC. The
    # composition itself stays single-sourced in src: only the motor-skill base is swapped.
    motor_skills: type = MotorSkillsAgent
    if str(config.skill_backend) == "maniflow":
        from agentic_robotics.adapters.toyblocks.leap.maniflow_skills.maniflow_motor_skills_agent import (  # noqa: E501
            MotorSkillsAgent as ManiFlowMotorSkillsAgent,
        )

        motor_skills = ManiFlowMotorSkillsAgent
        no_op = type(
            NoOpMotorSkillsAgent.__name__,
            (ManiFlowMotorSkillsAgent,),
            {"get_refusal_text": NoOpMotorSkillsAgent.__dict__["get_refusal_text"]},
        )
        swap = {MotorSkillsAgent: ManiFlowMotorSkillsAgent, NoOpMotorSkillsAgent: no_op}
        baseline = type(
            baseline.__name__,
            tuple(swap.get(base, base) for base in baseline.__bases__),
            {"__doc__": baseline.__doc__},
        )
        logger.info(f"{baseline.__name__} recomposed on the ManiFlow motor-skill declaration.")
    workspace = Workspace(config.output_dir / "workspace")
    if getattr(args, "example_tasks", None):
        seed_example_states(workspace, source=args.example_tasks, target_name="example_tasks")
    elif getattr(args, "example_states", None):
        seed_example_states(workspace, source=args.example_states)
    else:
        seed_example_states(workspace)
    ledger = CostLedger(args.budget_usd)
    require_budget(args.budget_usd, ignore=getattr(args, "ignore_budget", False))
    policy = (
        ContextPolicy(limit_tokens=args.context_limit) if args.context_limit else ContextPolicy()
    )
    control = ContextControl(self_wipe_fraction=policy.self_wipe_fraction)

    resumed = resume_from(args.resume, workspace, ledger) if args.resume else None

    composed = ComposedServer(
        "leap",
        simulator=server.mcp,
        snapshot_dir=workspace.logs_dir / "docstrings",
        # The declared docstrings are the phase's starting point whichever skill module serves
        # the tools; the skill server's own text only matches them under the ManiFlow skills.
        initial_descriptions=resumed["descriptions"]
        if resumed
        else declared_descriptions(baseline),
    )
    file_tools = register_file_tools(composed.local, workspace)
    register_docstring_tool(composed.local, composed)
    register_context_tool(composed.local, control)

    pool = SimulatorPool(args.pool_urls)
    logger.info(f"Pool simulators serve: {await pool.check()}")
    spawner = SubagentSpawner(
        pool=pool,
        parent=composed,
        workspace=workspace,
        ledger=ledger,
        model_config=config.model_config,
        preamble=DEFAULT_PREAMBLE_FILE.read_text(encoding="utf-8").strip(),
    )
    harness: ExperimentHarness | None = None
    if args.condition == "harness":
        harness = ExperimentHarness(spawner=spawner, workspace=workspace)
        register_harness_tools(composed.local, harness)
        # The skills stay listed with their docstrings; every call is refused.
        composed.refuse_calls(
            sorted(motor_skills.__abstractmethods__), NoOpMotorSkillsAgent.get_refusal_text
        )
    else:
        register_spawn_tool(composed.local, spawner)

    model_client = OpenAIModelClient(config=config.model_config)
    transcript = Transcript(out_dir=config.output_dir)
    summary: dict[str, Any] = {"condition": args.condition, "finished": False}
    started = time.time()
    if resumed:
        summary.update(resumed_from=str(args.resume), prior_cost_usd=resumed["prior_cost_usd"])

    async with composed:
        # The experimentation phase serves the declared (Control) descriptions, or, when resumed,
        # the earlier run's final set; the initial snapshot the composed server just wrote is the
        # record of what passed.
        assert_serves(
            composed.visible_tools(),
            baseline,
            descriptions=resumed["descriptions"] if resumed else None,
        )
        logger.info(
            f"Tools handed to the model, checked against {baseline.__name__}: "
            f"{composed.visible_tool_names}"
        )
        async with MCPClient(composed.server) as mcp_client:
            agent = LeapAgenticLoop(
                composed,
                mcp_client,
                model_client,
                transcript,
                ledger=ledger,
                workspace=workspace,
                file_tool_names=file_tools,
                control=control,
                context_policy=policy,
                metadata_callback=lambda: {"frame_count": server.frame_count},
                max_idle_turns=args.max_idle_turns,
            )
            try:
                await agent.enter_loop(
                    task=prompt,
                    history=[create_openai_message(prompt, role="user")],
                    max_tool_calls=args.max_tool_calls,
                    opening_note=messages.resumed_phase_note(
                        resumed["prior_cost_usd"], ledger.status(), workspace.listing()
                    )
                    if resumed
                    else None,
                )
                summary["finished"] = True
            except Exception:
                logger.exception("Experimentation phase failed; ending after cleanup.")
            finally:
                summary.update(
                    model=config.model_config.model_id,
                    cost_usd=ledger.total_usd,
                    tool_calls=agent.tool_calls_made,
                    context_wipes=agent.wipe_count,
                    reprompts=agent.reprompts,
                    ended_idle=agent.ended_idle,
                    keep_alives=agent.keep_alives,
                    keep_alive_cost_usd=round(agent.keep_alive_cost_usd, 4),
                    subagent_episodes=spawner.episodes_run,
                    experiments_run=harness.experiments_run if harness else 0,
                    final_docstrings=composed.descriptions(),
                    seconds=round(time.time() - started, 1),
                    model_calls=len(model_client.usage_history),
                    model_seconds=round(
                        sum(u.duration_s or 0.0 for u in model_client.usage_history), 3
                    ),
                )
                transcript_path = config.output_dir / "transcript.jsonl"
                if transcript_path.is_file():
                    summary["tool_seconds"] = tool_call_seconds(transcript_path)
                (config.output_dir / "phase_summary.json").write_text(
                    json.dumps(summary, indent=1), encoding="utf-8"
                )


EXAMPLE_STATES_DIR = Path(__file__).resolve().parents[5] / "data" / "example_states"
"""The example initial states every experimenting agent starts with: one
simple three-block scene and one cluttered `ai2_stacked` scene, both held out from the
evaluation, written by `runs/example_states_20261005/make_states.py` into the `ai2_stacked_09`
package so every simulator of the phase can take either."""


def seed_example_states(
    workspace: Workspace, source: Path = EXAMPLE_STATES_DIR, target_name: str = "initial_states"
) -> list[str]:
    """Copy the example initial states into the workspace's read-only `logs/initial_states/`.

    The hand-holding setting (`--example-tasks`) passes its own directory and `example_tasks`.

    :param workspace: This run's workspace
    :param source: Directory whose `*.json` files are copied
    :param target_name: Subdirectory of `logs/` they are copied into
    :return: The workspace-relative paths copied, in name order; empty when there are none
    """
    if not source.is_dir():
        logger.warning(f"No example states under {source}; the workspace starts without them.")
        return []
    target = workspace.logs_dir / target_name
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(source.glob("*.json")):
        shutil.copyfile(path, target / path.name)
        copied.append(workspace.relative(target / path.name))
    logger.info(f"Example initial states in the workspace: {copied}")
    return copied


def final_snapshot(docstrings_dir: Path) -> dict[str, str]:
    """The description set of the highest-numbered snapshot under `docstrings_dir`.

    :raises FileNotFoundError: If there is no snapshot
    """
    snapshots = sorted(docstrings_dir.glob("[0-9]*.json"))
    if not snapshots:
        raise FileNotFoundError(f"No docstring snapshot under {docstrings_dir}.")
    return json.loads(snapshots[-1].read_text(encoding="utf-8"))["tools"]


def resume_from(run_dir: Path, workspace: Workspace, ledger: CostLedger) -> dict[str, Any]:
    """Carry a finished phase's workspace, final descriptions and spend into this run.

    :param run_dir: The earlier run's directory (holding `phase_summary.json` and `workspace/`)
    :param workspace: This run's workspace, which receives a copy of the earlier one
    :param ledger: This run's ledger, charged with what the earlier run spent
    :return: The earlier run's final description set and its cost
    """
    summary = json.loads((run_dir / "phase_summary.json").read_text(encoding="utf-8"))
    shutil.copytree(run_dir / "workspace", workspace.root, dirs_exist_ok=True)
    descriptions = final_snapshot(workspace.logs_dir / "docstrings")
    prior_cost = float(summary["cost_usd"])
    if prior_cost > 0:
        ledger.add(prior_cost)
    logger.info(
        f"Resumed from {run_dir}: ${prior_cost:.2f} already spent, "
        f"{len(descriptions)} tool descriptions restored, workspace copied."
    )
    return {"descriptions": descriptions, "prior_cost_usd": prior_cost}


def leap_agentic_thread(
    server: ToyBlocksMCPServer, finished: threading.Event, **kwargs: Any
) -> None:
    """Run the phase on this thread, which is not the simulator's; release it when done."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_experimentation_phase(server, **kwargs))
    finally:
        loop.close()
        finished.set()
