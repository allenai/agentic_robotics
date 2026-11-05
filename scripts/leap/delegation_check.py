"""Exercise the delegation tools against a pool simulator, with no parent agent.

Each delegation tool must have run successfully at least three times before either tool suite
counts as done. The tools are
`spawn_subagents` (naive condition) and `plan_experiment`, `run_experiment`,
`run_experiment` (harness condition). This driver stands in for the parent model: it
registers the same tools on a composed server, exactly as `run_leap.py` does, and calls them
through an MCP client with the JSON arguments a model would send. So the pydantic schemas, the
validation errors, the pool check-out and queueing, the per-trial model choice, the inherited
descriptions, the logs and the results files are all the real ones. Only the trials call a model:
`--trial-model` (default haiku-4.5, the cheapest configured) stands in as the parent's model, and
one trial names sonnet-5.5 to exercise the per-trial choice.

No parent simulator is booted. The spawner reads the parent's composed server only for its
descriptions, so the parent here is composed over the pool URL too, and the trials' starting
scene is what the pool simulator's `get_sim_state` returns at the start.

    python scripts/leap/delegation_check.py --pool-url http://127.0.0.1:8101/mcp \\
        --out-dir <runs dir>/delegation_check

Writes `summary.json` and `summary.md` under the output directory; the workspace under it holds
every trial's log and the experiments' plan and results files, as a real run's would.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mcp import Client as MCPClient

from agentic_robotics.adapters.toyblocks.leap.composed_server import (
    ComposedServer,
    register_docstring_tool,
)
from agentic_robotics.adapters.toyblocks.leap.cost import CostLedger
from agentic_robotics.adapters.toyblocks.leap.harness import (
    ExperimentHarness,
    register_harness_tools,
)
from agentic_robotics.adapters.toyblocks.leap.subagents import (
    DEFAULT_PREAMBLE_FILE,
    SimulatorPool,
    SubagentSpawner,
    register_spawn_tool,
)
from agentic_robotics.adapters.toyblocks.leap.workspace import (
    Workspace,
    register_file_tools,
)
from agentic_robotics.adapters.toyblocks.runner_bootstrap import (
    resolve_openrouter_key,
)
from agentic_robotics.core.foundation_models import OPENAI_API_MODEL_CONFIGS
from agentic_robotics.core.states import SimState

logger = logging.getLogger("agentic_robotics.examples.leap.delegation_check")

REQUIRED_SUCCESSES = 3
"""Successful calls each delegation tool needs before its suite counts as exercised."""

DELEGATION_TOOLS = ("spawn_subagents", "plan_experiment", "run_experiment")

STATE_FILE = "notes/start_state.json"
"""Workspace path the starting scene is written to, for the path form of `initial_state`."""

DESCRIPTION_MARKER = "CHECK-MARKER-7f3a"
"""Put into get_sim_state's description before the first batch; a trial is asked to quote it."""


@dataclass
class Call:
    """One tool call the driver made and what came back."""

    tool: str
    label: str
    expect_error: bool
    is_error: bool
    seconds: float
    text: str
    """The result text, cut to a few hundred characters."""
    ok: bool = field(init=False)
    """Whether the outcome matched the expectation."""

    def __post_init__(self) -> None:
        self.ok = self.is_error == self.expect_error


def _text_of(result: Any) -> str:
    return "\n".join(getattr(block, "text", "") for block in result.content)


class Driver:
    """Calls the delegation tools through the composed server and keeps the tally."""

    def __init__(self, client: MCPClient, out_dir: Path) -> None:
        self._client = client
        self._out_dir = out_dir
        self.calls: list[Call] = []
        self.trial_results: list[dict[str, Any]] = []

    async def call(
        self, tool: str, args: dict[str, Any], *, label: str, expect_error: bool = False
    ) -> str:
        """Call a tool as the model would and record the outcome."""
        start = time.monotonic()
        result = await self._client.call_tool(tool, args)
        seconds = round(time.monotonic() - start, 1)
        text = _text_of(result)
        call = Call(tool, label, expect_error, result.is_error, seconds, text[:600])
        self.calls.append(call)
        mark = "ok" if call.ok else "UNEXPECTED"
        logger.info(
            f"[{mark}] {tool} ({label}) in {seconds} s, is_error={result.is_error}: {text[:200]!r}"
        )
        if tool in ("spawn_subagents", "run_experiment") and not result.is_error:
            payload = json.loads(text)
            rows = payload["trials"] if isinstance(payload, dict) else payload
            for row in rows:
                self.trial_results.append({"call": label, **row})
        return text

    def tally(self) -> dict[str, dict[str, int]]:
        """Per tool: successful calls, expected errors, and unexpected outcomes."""
        counts: dict[str, dict[str, int]] = {}
        for call in self.calls:
            row = counts.setdefault(
                call.tool, {"successes": 0, "expected_errors": 0, "unexpected": 0}
            )
            if not call.ok:
                row["unexpected"] += 1
            elif call.is_error:
                row["expected_errors"] += 1
            else:
                row["successes"] += 1
        return counts


def trial(
    prompt: str,
    tools: list[str],
    max_tool_calls: int,
    state: dict[str, Any] | str | None,
    **extra: Any,
) -> dict[str, Any]:
    """An episode as the model's JSON would spell it; `state` is inline, a workspace path, or absent."""
    spec: dict[str, Any] = {"prompt": prompt, "tools": tools, "max_tool_calls": max_tool_calls}
    if state is not None:
        spec["initial_state"] = state
    spec.update(extra)
    return spec


async def read_start_state(pool_url: str) -> SimState:
    """The pool simulator's current state, which every trial here starts from."""
    async with MCPClient(pool_url) as client:
        result = await client.call_tool("get_sim_state", {})
    if result.is_error:
        raise RuntimeError(f"get_sim_state on the pool failed: {_text_of(result)}")
    return SimState.model_validate_json(result.content[0].text)


async def main(args: argparse.Namespace) -> int:
    resolve_openrouter_key()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    workspace = Workspace(out_dir / "workspace")
    ledger = CostLedger(args.budget_usd)

    start_state = await read_start_state(args.pool_url)
    state = start_state.model_dump()
    workspace.resolve(STATE_FILE, writable=True).write_text(
        start_state.model_dump_json(indent=1), encoding="utf-8"
    )
    block_names = [obj.name for obj in start_state.objects]
    logger.info(f"Starting scene: blocks {block_names}.")

    pool = SimulatorPool([args.pool_url])
    served = await pool.check()
    logger.info(f"Pool serves: {served}")

    parent = ComposedServer(
        "delegation-check", simulator=args.pool_url, snapshot_dir=workspace.logs_dir / "docstrings"
    )
    register_file_tools(parent.local, workspace)
    register_docstring_tool(parent.local, parent)
    spawner = SubagentSpawner(
        pool=pool,
        parent=parent,
        workspace=workspace,
        ledger=ledger,
        model_config=OPENAI_API_MODEL_CONFIGS[args.trial_model],
        preamble=DEFAULT_PREAMBLE_FILE.read_text(encoding="utf-8").strip(),
    )
    register_spawn_tool(parent.local, spawner)
    harness = ExperimentHarness(spawner=spawner, workspace=workspace)
    register_harness_tools(parent.local, harness)
    perceive = ["get_sim_state", "get_image_observation"]
    report = "Your final message must state what you did and what you observed."

    async with parent, MCPClient(parent.server) as client:
        listing = await client.list_tools(cache_mode="bypass")
        names = sorted(t.name for t in listing.tools)
        logger.info(f"Parent sees: {names}")
        missing = [t for t in DELEGATION_TOOLS if t not in names]
        if missing:
            raise RuntimeError(f"Delegation tools not served: {missing}")
        driver = Driver(client, out_dir)
        c = driver.call

        # A description edit before the first batch; trial A is asked to quote it back, which
        # checks that trials inherit the parent's current descriptions.
        original = parent.description_of("get_sim_state")
        await c(
            "update_tool_docstring",
            {"tool_name": "get_sim_state", "description": f"{DESCRIPTION_MARKER}. {original}"},
            label="marker",
        )

        # --- spawn_subagents: three batches, then the validation errors ---
        await c(
            "spawn_subagents",
            {
                "episodes": [
                    trial(
                        "Call get_sim_state once. Then write a file notes/trial_a.txt whose content is the "
                        "number of blocks in the scene followed by their names, one per line. In your final "
                        "message, give the number of blocks and quote, verbatim, the first sentence of the "
                        f"description of your get_sim_state tool. {report}",
                        [*perceive, "write_text_file"],
                        4,
                        state,
                    )
                ]
            },
            label="batch A: one trial, default model, inline state, write_text_file",
        )
        await c(
            "spawn_subagents",
            {
                "episodes": [
                    trial(
                        f"Read the file {STATE_FILE} with read_file, then call get_sim_state once, and "
                        "report whether the block names in the file and in the simulator are the same "
                        f"set, listing both. {report}",
                        ["get_sim_state", "read_file"],
                        3,
                        STATE_FILE,
                    ),
                    trial(
                        "Call get_image_observation once and describe, in one or two sentences, the "
                        f"blocks you see on the table and the letters or symbols on them. {report}",
                        ["get_image_observation"],
                        2,
                        state,
                    ),
                ]
            },
            label="batch B: two trials on one simulator (queueing), one from a state file",
        )
        await c(
            "spawn_subagents",
            {
                "episodes": [
                    trial(
                        "Call get_image_observation, then call pick once on any one block, giving the pixel "
                        "at the centre of that block in the image and steps=300. Then call get_sim_state and "
                        "report the gripper's finger_joint value and whether any block's z position rose "
                        f"by more than 2 cm compared with before the pick. {report}",
                        [*perceive, "pick"],
                        5,
                        state,
                    )
                ]
            },
            label="batch C: one trial running a skill (pick, steps=300)",
        )
        bad = dict(prompt="x", max_tool_calls=2, initial_state=state)
        await c(
            "spawn_subagents",
            {"episodes": [{**bad, "tools": ["update_tool_docstring"]}]},
            label="unknown tool",
            expect_error=True,
        )
        await c(
            "spawn_subagents",
            {"episodes": [{"prompt": "x", "tools": ["get_sim_state"], "max_tool_calls": 2}]},
            label="no state",
            expect_error=True,
        )
        await c(
            "spawn_subagents",
            {"episodes": [{**bad, "tools": ["get_sim_state"], "max_tool_calls": 101}]},
            label="ceiling above the maximum",
            expect_error=True,
        )

        # --- plan_experiment: three plans, then the validation errors ---
        plans = [
            {
                "hypothesis": "pick with steps=300 closes the gripper on a block from the home pose.",
                "success_criteria": "After pick, finger_joint reads between 0.3 and 0.6 and one block's z rose by more than 2 cm.",
                "trials": [
                    trial(
                        "Call get_image_observation, call pick once on any block (pixel at its centre, "
                        f"steps=300), then call get_sim_state. {report}",
                        [*perceive, "pick"],
                        5,
                        state,
                        label="pick-300",
                    )
                ],
            },
            {
                "hypothesis": "rotate cw with the default steps turns a block by roughly a quarter turn.",
                "success_criteria": "The block's quaternion changes and no block leaves the table.",
                "trials": [
                    trial(
                        "Call get_sim_state, call rotate once on any block (pixel at its centre, "
                        f"direction cw, default steps), then call get_sim_state again. {report}",
                        [*perceive, "rotate"],
                        5,
                        state,
                        label="rotate-cw",
                    )
                ],
            },
            {
                "hypothesis": "get_sim_state reports every block exactly once.",
                "success_criteria": "The block list has no repeated names.",
                "trials": [
                    trial(
                        f"Call open_gripper once, then get_sim_state once, and list the block names. {report}",
                        ["open_gripper", "get_sim_state"],
                        3,
                        state,
                        label="names-a",
                    ),
                    trial(
                        f"Call get_sim_state once and list the block names. Do not call open_gripper. {report}",
                        ["open_gripper", "get_sim_state"],
                        2,
                        STATE_FILE,
                        label="names-b",
                    ),
                ],
            },
        ]
        for i, plan in enumerate(plans):
            await c("plan_experiment", {"plan": plan}, label=f"plan {i}")
        dup = [
            trial("x", ["pick"], 2, state, label="same"),
            trial("x", ["pick"], 2, state, label="same"),
        ]
        await c(
            "plan_experiment",
            {"plan": {**plans[0], "trials": dup}},
            label="duplicate labels",
            expect_error=True,
        )

        # --- run_experiment ---
        for i in range(3):
            await c("run_experiment", {"id": i}, label=f"run {i}")
        await c("run_experiment", {"id": 0}, label="rerun", expect_error=True)
        await c("run_experiment", {"id": 99}, label="unknown experiment", expect_error=True)

    # --- artifacts and summary ---
    expected_files = [
        *(workspace.logs_dir / "subagents" / f"batch_{i:03d}" / "results.json" for i in range(3)),
        *(
            workspace.logs_dir / "experiments" / f"{i:03d}" / name
            for i in range(3)
            for name in ("plan.json", "results.json")
        ),
    ]
    files = {workspace.relative(p): p.is_file() for p in expected_files}
    logs = sorted(workspace.relative(p) for p in workspace.logs_dir.rglob("log.md"))
    trial_a_file = workspace.resolve("notes/trial_a.txt")
    marker_quoted = any(
        DESCRIPTION_MARKER in r.get("final_message", "")
        for r in driver.trial_results
        if r["call"].startswith("batch A")
    )

    tally = driver.tally()
    done = {t: tally.get(t, {}).get("successes", 0) >= REQUIRED_SUCCESSES for t in DELEGATION_TOOLS}
    unexpected = [asdict(c) for c in driver.calls if not c.ok]
    statuses = {}
    for r in driver.trial_results:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    summary = {
        "pool_url": args.pool_url,
        "trial_model": spawner.model_name,
        "cost_usd": round(ledger.total_usd, 4),
        "tally": tally,
        "three_successes": done,
        "trial_statuses": statuses,
        "trial_results": driver.trial_results,
        "unexpected": unexpected,
        "expected_files_present": files,
        "trial_logs": logs,
        "trial_a_wrote_file": trial_a_file.is_file(),
        "marker_quoted_by_trial_a": marker_quoted,
        "calls": [asdict(c) for c in driver.calls],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    lines = [
        "# Delegation check",
        "",
        f"Pool {args.pool_url}; trials on {spawner.model_name}; spent ${ledger.total_usd:.2f}.",
        "",
        "| tool | successes | expected errors | unexpected | >= 3 |",
        "|---|---|---|---|---|",
    ]
    for tool, row in tally.items():
        flag = "yes" if done.get(tool) else ("n/a" if tool not in DELEGATION_TOOLS else "NO")
        lines.append(
            f"| {tool} | {row['successes']} | {row['expected_errors']} | {row['unexpected']} | {flag} |"
        )
    lines += [
        "",
        f"Trial statuses: {statuses}",
        "",
        "| call | trial | model | status | calls | $ | final message |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in driver.trial_results:
        msg = r.get("final_message", "").replace("\n", " ")[:160]
        lines.append(
            f"| {r['call'][:28]} | {r.get('label', r['trial'])} | {r['model']} | {r['status']} | {r['tool_calls_used']} | {r['cost_usd']} | {msg} |"
        )
    lines += [
        "",
        f"Files present: {files}",
        f"Trial logs: {len(logs)}",
        f"Trial A wrote notes/trial_a.txt: {trial_a_file.is_file()}; quoted the description marker: {marker_quoted}",
    ]
    if unexpected:
        lines += ["", "## Unexpected outcomes"] + [
            f"- {u['tool']} ({u['label']}): is_error={u['is_error']}: {u['text'][:300]}"
            for u in unexpected
        ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    all_done = all(done.values()) and not unexpected and all(files.values())
    return 0 if all_done else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pool-url", required=True, help="An `agentic-robotics pool` simulator's MCP URL"
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="Where the workspace and summary go"
    )
    parser.add_argument(
        "--trial-model",
        default="haiku-4.5",
        choices=sorted(OPENAI_API_MODEL_CONFIGS),
        help="Stands in as the parent's model, so trials run on it unless they name another",
    )
    parser.add_argument("--budget-usd", type=float, default=5.0, help="Ledger cap over every trial")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s %(levelname)s %(name)s] %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    sys.exit(asyncio.run(main(parse_args())))
