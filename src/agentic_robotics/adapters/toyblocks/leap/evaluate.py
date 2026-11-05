"""The LEAP evaluation phase, run as `agentic-robotics episode --baseline <name>`:

    agentic-robotics episode --task <variant slug> --baseline control
    agentic-robotics episode --task <variant slug> --baseline naive \\
        --docstrings <leap run>/workspace/logs/docstrings/0007.json

`agentic-robotics` owns the boot; this module supplies the agent thread. It is `toyblocks_runner`'s
episode (the repo's `AgenticLoop` against the task prompt and the task's tool-call cutoff) with a
`ComposedServer` between the agent and the skill server, which is what lets the served
descriptions be replaced and the served set be checked. With `--instance k` the scene is reset
to the package's reset state k on the simulator's thread before the first model call (and the
visibility preflight re-run on it), so every baseline and model sees the same task instances
(instances 0 and 1 of every variant). When the loop ends the final scene is
scored with the package's goal classifier, and `evaluation_summary.json` records success and
each conjunct, tool calls by category (motor, perception, other), model calls, dollars and
wall-clock seconds: the E2 metrics. Before the first model call the tools the
agent will see must be exactly `EvaluationAgent`'s (the motor skills and the perception tools),
carrying the baseline's description set: Control, the declared docstrings; Naive and Harness, the
final docstring snapshot of that run's experimentation phase; Oracle, the developer's
`ORACLE_DOCSTRINGS` for the motor skills with the declared perception docstrings. The initial
snapshot written under the run's `docstrings/` is the record of what passed.

The model host is the repo's loop over OpenRouter unless `--model-host claude-code` is passed, in
which case the composed server is served over HTTP and the episode runs inside the Claude Code CLI
through the Claude Agent SDK (`core.claude_code_loop`); `make_claude_code_evaluation_loop_class`
is that host's counterpart of `make_evaluation_loop_class`, with the same counters, cutoff rule,
per-call snapshot and batched-image rule, and the summary carries the same fields plus
`cost_source: "sdk_estimate"`, since the SDK's dollar figure is a client-side estimate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import Counter
from inspect import cleandoc
from typing import TYPE_CHECKING, Any, TypeVar

from agentic_robotics.adapters.toyblocks.config import SEARCH_SPELL_DECLARATIONS

if TYPE_CHECKING:
    import threading
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from openai.types.responses.response_input_item_param import FunctionCallOutput

    from agentic_robotics.adapters.toyblocks.toyblocks_mcp_server import ToyBlocksMCPServer
    from agentic_robotics.core.agentic_loop import ToolCall
    from agentic_robotics.core.claude_code_loop import ClaudeCodeLoop

logger = logging.getLogger("agentic_robotics.examples.leap.evaluate")

T = TypeVar("T")

LEARNED_BASELINES = ("naive", "harness")
"""Baselines evaluated with the description set their experimentation phase ended on."""


def oracle_description_set(backend: str) -> dict[str, str]:
    """The developer's handwritten docstrings for the motor skills of `backend`, plus the declared
    perception docstrings: `oracle_agent.ORACLE_DOCSTRINGS` for the scripted skills,
    `maniflow_skills.maniflow_oracle_agent.DOCSTRINGS` for the ManiFlow skills.

    :raises SystemExit: If a motor skill is missing from the set or still reads "TODO"
    """
    from agentic_robotics.adapters.toyblocks.leap.declarations import (
        declared_tools,
        recompose_on_backend,
    )
    from agentic_robotics.adapters.toyblocks.leap.leap_agents import EvaluationAgent

    if str(backend) == "maniflow":
        from agentic_robotics.adapters.toyblocks.leap.maniflow_skills.maniflow_motor_skills_agent import (  # noqa: E501
            MotorSkillsAgent,
        )
        from agentic_robotics.adapters.toyblocks.leap.maniflow_skills.maniflow_oracle_agent import (
            DOCSTRINGS as handwritten,
        )
    else:
        from agentic_robotics.adapters.toyblocks.leap.motor_skills_agent import MotorSkillsAgent
        from agentic_robotics.adapters.toyblocks.leap.oracle_agent import (
            ORACLE_DOCSTRINGS as handwritten,
        )

    agent_cls = recompose_on_backend(EvaluationAgent, backend)
    declared = {name: text for name, (text, _) in declared_tools(agent_cls).items()}
    skills = sorted(MotorSkillsAgent.__abstractmethods__)
    missing = [name for name in skills if name not in handwritten]
    unwritten = [name for name in skills if handwritten.get(name, "").strip() in ("", "TODO")]
    if missing or unwritten:
        raise SystemExit(
            f"The handwritten docstring set for the {backend} skills is not complete: "
            f"missing {missing}, unwritten {unwritten}."
        )
    # Dedented like the Control docstrings are at registration, so the two sets differ only in
    # what they say, not in indentation or trailing whitespace.
    return {**declared, **{name: cleandoc(handwritten[name]) for name in skills}}


def load_description_set(
    baseline: str, docstrings_path: Path | None, backend: str
) -> dict[str, str] | None:
    """The description set the baseline is evaluated with, or None for the declared docstrings.

    :param backend: `config.skill_backend`, which picks the handwritten set for the oracle baseline

    :raises SystemExit: If a learned baseline has no snapshot, or the Oracle set is asked for
    """
    if baseline == "control":
        if docstrings_path is not None:
            raise SystemExit("--docstrings does not apply to the control baseline.")
        return None
    if baseline == "oracle":
        if docstrings_path is not None:
            raise SystemExit("--docstrings does not apply to the oracle baseline.")
        return oracle_description_set(backend)
    if docstrings_path is None:
        raise SystemExit(f"The {baseline} baseline needs --docstrings: its final snapshot JSON.")
    payload = json.loads(docstrings_path.read_text(encoding="utf-8"))
    return dict(payload["tools"])


async def on_simulator_thread(server: ToyBlocksMCPServer, work: Callable[[], T]) -> T:
    """Run `work()` on the simulator's thread through the server's dispatcher and await it."""

    async def job() -> T:
        return work()

    return await asyncio.wrap_future(server._dispatcher.submit(job()))


def reset_to_instance(server: ToyBlocksMCPServer, instance: int) -> None:
    """Reset env 0 to reset state `instance`, settle, and re-run the visibility preflight."""
    import torch

    from agentic_robotics.adapters.toyblocks.scene_preflight import preflight_scene

    env = server._toyblocks_env
    base = env.base_env
    available = int(getattr(base, "num_reset_states", 0) or 0)
    if not 0 <= instance < available:
        raise SystemExit(
            f"Instance {instance} is out of range: the package has {available} reset states."
        )
    base._reset_idx(
        torch.tensor([0], device=base.device),
        instance_ids=torch.tensor([instance], device=base.device),
    )
    env._sync_simulator()
    preflight_scene(server)
    server.record_current_frame()


SCORING_SETTLE_SYNCS = 5
"""Simulator syncs before scoring, as `render_task.py --settle-syncs` uses on solved states."""

GRIPPER_OPEN_BELOW = 0.25
"""`finger_joint` reading below which the goal classifier's gate counts the gripper as open."""

SETTLE_DISPLACEMENT_M = 0.002
"""A goal block that moves no more than this across the settle syncs counts as at rest, whatever
its reported velocity. The classifier's own gate reads PhysX velocities, and the top cube of a
six-cube Blocksworld tower reports a steady 0.04-0.05 m/s while sitting still (settle_probe,
2026-10-14, against the gate's 0.03), which would score a complete tower as a failure. Blocks that
do not move are therefore at rest, whatever velocities they report."""


def classify_scene(server: ToyBlocksMCPServer) -> dict[str, Any]:
    """The package's goal classifier on the scene as it is: success and every conjunct."""
    base = server._toyblocks_env.base_env
    base._compute_intermediate_values()
    result = base.evaluate()
    stats = {key: float(value[0]) for key, value in result.get("stats", {}).items()}
    return {"success": bool(result["success"][0]), "conjuncts": stats}


def open_gripper_for_scoring(server: ToyBlocksMCPServer) -> None:
    """Open the gripper the way the serving skill module does.

    The ManiFlow server ramps the gripper command through `ToyBlocksEnv.step`; the scripted
    server opens it through its motion layer, which settles and forgets the held
    block. Calling only the
    ManiFlow path would make a scripted-skills episode that ends with the gripper closed raise
    during scoring and be recorded with no success value.
    """
    if hasattr(server, "_command_gripper"):
        server._command_gripper(close=False)
        return
    server._motion.open_gripper()


def score_scene(server: ToyBlocksMCPServer, output_dir: Path | None = None) -> dict[str, Any]:
    """Score the final scene the way the task means it: settled, with the robot letting go.

    The autogen classifier gates success on every goal block being at rest and `finger_joint`
    reading open (`scene_settled_and_gripper_open`), a guard against success firing mid-skill
    during data generation. `stack` and `place_on_table` leave the gripper partly closed after
    releasing (finger 0.49-0.64 on 2026-10-04's pick-start probe), so a scene whose every
    conjunct holds would score as a failure (as in the Control suite,
    `stack_feasible_isolated_flip_01` instance 0). Here the scene is synced to rest, the gripper
    is opened if it reads closed (arm held still, the same command `open_gripper` sends), synced
    again, and classified;
    anything still held drops, which is the scene's honest end state. The raw classification is
    kept beside it, and the final `SimState` is written to `final_state.json` for re-scoring.
    `success` is the classifier corrected for letter case (`leap.spelling`: AI2, Ai2, aI2 or ai2
    all count), or for a Search & Spell instance the specification's rule
    (`leap.spelling.search_spell_report`, whose report is kept under `search_spell`; the
    classifier's own reading, the designated blocks stacked in order, stays `success_strict`),
    and for the velocity gate (`SETTLE_DISPLACEMENT_M`: a scene whose conjuncts all hold, gripper
    open, and whose blocks do not move across the settle syncs is a success,
    `settle_gate_overridden`); `success_strict` keeps the
    uncorrected classifier's reading only so that suites scored before the corrections can be
    compared.
    """
    from agentic_robotics.adapters.toyblocks.leap.sim_state import read_sim_state
    from agentic_robotics.adapters.toyblocks.leap.spelling import (
        search_spell_report_for,
        spelling_success,
    )

    task_slug = getattr(getattr(server.config, "task_config", None), "task_slug", None)
    env = server._toyblocks_env
    robot = env.base_env.scene.articulations["robot"]
    raw = classify_scene(server)
    finger_before = float(robot.data.joint_pos[0, robot.joint_names.index("finger_joint")])
    for _ in range(SCORING_SETTLE_SYNCS):
        env._sync_simulator()
    opened = finger_before >= GRIPPER_OPEN_BELOW
    if opened:
        open_gripper_for_scoring(server)
        for _ in range(SCORING_SETTLE_SYNCS):
            env._sync_simulator()
    # Settled by position: the blocks where they are now against where they are after another
    # round of syncs. A block that does not move is at rest, whatever velocity PhysX reports.
    before = {obj.name: obj.pose.position for obj in read_sim_state(env).objects}
    for _ in range(SCORING_SETTLE_SYNCS):
        env._sync_simulator()
    final = classify_scene(server)
    state = read_sim_state(env)
    after = {obj.name: obj.pose.position for obj in state.objects}
    displacement = max(
        (
            math.dist((before[n].x, before[n].y, before[n].z), (after[n].x, after[n].y, after[n].z))
            for n in after
            if n in before
        ),
        default=0.0,
    )
    finger_after = float(robot.data.joint_pos[0, robot.joint_names.index("finger_joint")])
    conjuncts_hold = all(v == 1 for k, v in final["conjuncts"].items() if k.startswith("g0_conj_"))
    gate_overridden = (
        not final["success"]
        and conjuncts_hold
        and finger_after < GRIPPER_OPEN_BELOW
        and displacement <= SETTLE_DISPLACEMENT_M
    )
    if output_dir is not None:
        (output_dir / "final_state.json").write_text(
            state.model_dump_json(indent=1), encoding="utf-8"
        )
    corrected = spelling_success(state.model_dump(), final["conjuncts"], task_slug)
    success = final["success"] or gate_overridden
    search_spell = search_spell_report_for(state.model_dump(), task_slug)
    return {
        **final,
        "success": success if corrected is None else corrected,
        "success_strict": final["success"],
        "raw": raw,
        **({"search_spell": search_spell} if search_spell is not None else {}),
        "finger_before_scoring": round(finger_before, 4),
        "gripper_opened_for_scoring": opened,
        "settle_displacement_m": round(displacement, 5),
        "settle_gate_overridden": gate_overridden,
    }


def motor_tool_names() -> frozenset[str]:
    """The motor skill tool names, from the ABC."""
    from agentic_robotics.adapters.toyblocks.leap.motor_skills_agent import MotorSkillsAgent

    return frozenset(MotorSkillsAgent.__abstractmethods__)


UNBUDGETED_MOTOR_TOOLS: frozenset[str] = frozenset()
"""Motor skills free under the motor cutoff. Empty: every motor skill, `return_home` included,
counts, so the budget measures every arm motion the agent chooses to spend."""


def budgeted_tool_names() -> frozenset[str]:
    """The motor skill tool names that count toward the cutoff."""
    return motor_tool_names() - UNBUDGETED_MOTOR_TOOLS


def search_spell_episode_caps(task_slug: str | None) -> tuple[int, int, int] | None:
    """(motor cutoff, ceiling on all calls, letter blocks) for a Search & Spell instance, from its
    family and `config`'s per-letter-block rate and factor; None for any other task."""
    from agentic_robotics.adapters.toyblocks.release.search_spell_scoring import family_for_slug

    family = family_for_slug(task_slug) if task_slug else None
    if family is None:
        return None
    from agentic_robotics.adapters.toyblocks import config as cfg

    letters = sum(1 for info in family.blocks.values() if info["kind"] == "letter")
    motor = cfg.SEARCH_SPELL_MOTOR_CALLS_PER_LETTER_BLOCK * letters
    return motor, round(cfg.SEARCH_SPELL_TOTAL_CALLS_FACTOR * motor), letters


def final_assistant_message(transcript_path: Path) -> str | None:
    """The text of the last assistant message in a transcript, or None."""
    last = None
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("role") == "assistant" and record.get("type") == "message":
            last = record
    if last is None:
        return None
    parts = [block.get("text", "") for block in last.get("content", []) if isinstance(block, dict)]
    return "\n".join(part for part in parts if part) or None


BATCHED_IMAGE_NOTE = (
    "Image omitted: this motor skill was followed by another in the same turn, so the next "
    "skill's image shows the scene after both."
)
"""Replaces the image of a motor skill's result when the agent issued another motor skill right
after it in the same turn (`batch_images` below). Agents batch `return_home` after the skill it
follows in most turns, so the earlier skill's image is superseded a few seconds later; such images
are about a third of all tool results at roughly 300 tokens each, and dropping them cuts an
episode's context growth by about a fifth without changing what the agent can see at decision
time."""


def make_evaluation_loop_class(
    total_ceiling: int | None = None,
    after_motor_call: Callable[[int, str], Awaitable[None]] | None = None,
    batch_images: bool = False,
) -> type:
    """`AgenticLoop` with a cutoff on motor skill calls (heavy imports deferred to call time).

    :param total_ceiling: The ceiling on calls of any kind; the ABC package's
        `MAX_EPISODE_TOOL_CALLS` when None
    :param after_motor_call: Awaited with (motor calls made, the call's tool name) after every
        motor skill call, for per-skill state recording
    :param batch_images: Whether a motor skill's result keeps its image when the agent issued
        another motor skill right after it in the same turn (`BATCHED_IMAGE_NOTE` replaces it)

    The evaluation cutoff is `LEAP_EVAL_BUDGET_FACTOR` calls per motor skill in the optimal plan,
    counting only motor skill calls, so looking is free and the cutoff is not what ends an
    episode that was going to finish. `MAX_EPISODE_TOOL_CALLS` from the ABC package bounds calls
    of any kind, a safety ceiling against perception loops. The counting and the two messages go
    through the base loop's hooks; see `EvaluationLoop`.
    """
    from agentic_robotics.adapters.toyblocks.leap.messages import (
        evaluation_final_update,
        evaluation_status_line,
    )
    from agentic_robotics.adapters.toyblocks.leap.recursive_agent import MAX_EPISODE_TOOL_CALLS
    from agentic_robotics.core.agentic_loop import AgenticLoop

    counted = motor_tool_names()
    budgeted = budgeted_tool_names()
    ceiling = MAX_EPISODE_TOOL_CALLS if total_ceiling is None else total_ceiling

    class EvaluationLoop(AgenticLoop):
        """The base loop with the motor-call cutoff, set through the base loop's hooks:
        `max_tool_calls` is the motor cutoff (`tool_call_cost` charges only the budgeted motor
        skills), the safety ceiling on calls of any kind is `should_continue`, and the two
        messages are `status_text` and `get_final_update`. `motor_calls_made` counts the budgeted
        skills; `skill_calls_made` every motor skill, `return_home` included, and indexes the
        per-call state snapshots."""

        motor_calls_made: int = 0
        skill_calls_made: int = 0
        total_calls_made: int = 0
        hit_cutoff: bool = False

        def status_text(self, tool_calls_left: int) -> str:
            return evaluation_status_line(tool_calls_left, ceiling - self.total_calls_made)

        def tool_call_cost(self, call: ToolCall) -> int:
            return 1 if call["name"] in budgeted else 0

        def should_continue(self) -> bool:
            if self.total_calls_made >= ceiling:
                self.hit_cutoff = True
                return False
            return True

        async def request_more_tool_calls(self) -> int:
            self.hit_cutoff = True  # The motor cutoff is spent
            return 0

        async def process_tool_calls(self, calls: list[ToolCall]) -> list[FunctionCallOutput]:
            outputs = await super().process_tool_calls(calls)
            if not batch_images:
                return outputs
            # Before the base loop appends them to the transcript, so the dropped images never
            # enter the context (`AgenticLoop.run_tool_calls` appends what this returns).
            for output, call, following in zip(outputs, calls, calls[1:], strict=False):
                if call["name"] in counted and following["name"] in counted:
                    blocks = output.get("output")
                    if isinstance(blocks, list) and any(
                        b.get("type") == "input_image" for b in blocks
                    ):
                        output["output"] = [b for b in blocks if b.get("type") != "input_image"] + [
                            {"type": "input_text", "text": BATCHED_IMAGE_NOTE}
                        ]
            return outputs

        async def process_tool_call(self, call: ToolCall) -> FunctionCallOutput:
            output = await super().process_tool_call(call)
            # Per call, not per turn: the agent batches several motor skills in one turn in
            # about half its turns, and a snapshot taken after the whole batch would record the
            # batch's final state under every call in it (an `open_gripper` followed by a flip in
            # the same turn reads closed).
            if call["name"] in counted:
                self.skill_calls_made += 1
                if call["name"] in budgeted:
                    self.motor_calls_made += 1
                if after_motor_call is not None:
                    await after_motor_call(self.skill_calls_made, call["name"])
            return output

        async def run_tool_calls(self, calls: list[ToolCall]) -> list[FunctionCallOutput]:
            outputs = await super().run_tool_calls(calls)
            self.total_calls_made += len(outputs)
            return outputs

        def get_final_update(self, task: str) -> str:
            return evaluation_final_update(task)

    return EvaluationLoop


def make_claude_code_evaluation_loop_class(
    motor_cutoff: int,
    total_ceiling: int | None = None,
    after_motor_call: Callable[[int, str], Awaitable[None]] | None = None,
    batch_images: bool = False,
) -> type:
    """`ClaudeCodeLoop` with the motor-call cutoff of `make_evaluation_loop_class`, enforced
    through the SDK's hooks (heavy imports deferred to call time).

    :param motor_cutoff: Motor skill calls after which the next model step is the final answer
    :param total_ceiling: The ceiling on calls of any kind; the ABC package's
        `MAX_EPISODE_TOOL_CALLS` when None
    :param after_motor_call: Awaited with (motor calls made, the call's tool name) after every
        motor skill call, for per-skill state recording
    :param batch_images: Whether a motor skill's result keeps its image when the agent issued
        another motor skill right after it in the same step (`BATCHED_IMAGE_NOTE` replaces it)

    The rule is the base loop's, at the same boundary: the budget is checked when a model step
    starts, so a step that overruns it runs every call it issued, and the first call of a step
    that starts over budget is denied with `evaluation_final_update`, the message the base loop
    gives on its final turn. The model answers that denial with its final message, which ends
    the run; any further call is denied the same way, and `max_turns` is the ceiling on that.
    """
    from agentic_robotics.adapters.toyblocks.leap.messages import (
        evaluation_final_update,
        evaluation_status_line,
    )
    from agentic_robotics.adapters.toyblocks.leap.recursive_agent import MAX_EPISODE_TOOL_CALLS
    from agentic_robotics.core.claude_code_loop import ClaudeCodeLoop, ToolCallRecord, strip_images

    counted = motor_tool_names()
    budgeted = budgeted_tool_names()
    ceiling = MAX_EPISODE_TOOL_CALLS if total_ceiling is None else total_ceiling

    class ClaudeCodeEvaluationLoop(ClaudeCodeLoop):
        """The Claude Code loop with the motor-call cutoff: `deny_reason` is the cutoff and the
        ceiling, `after_tool_call` the counting and the snapshot, `edit_tool_output` the
        batched-image rule, `status_text` the status line. Counts as `EvaluationLoop`'s."""

        motor_calls_made: int = 0
        skill_calls_made: int = 0
        total_calls_made: int = 0
        hit_cutoff: bool = False

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._over_budget_at_start: dict[str, bool] = {}

        def _over_budget(self) -> bool:
            return self.motor_calls_made >= motor_cutoff or self.total_calls_made >= ceiling

        def on_assistant_message(self, message_id: str, calls: list[ToolCallRecord]) -> None:
            over = self._over_budget()
            self._over_budget_at_start[message_id] = over
            if over:
                self.hit_cutoff = True

        def status_text(self) -> str | None:
            # Clamped at zero: the base loop never shows a negative count, since it leaves the
            # tool-call loop before the status line once the budget is spent.
            return evaluation_status_line(
                max(motor_cutoff - self.motor_calls_made, 0),
                max(ceiling - self.total_calls_made, 0),
            )

        def deny_reason(self, call: ToolCallRecord) -> str | None:
            if call.message_id is None:
                over = self._over_budget()  # No step known for this call: judge it now
            else:
                over = self._over_budget_at_start.get(call.message_id, False)
            if over:
                self.hit_cutoff = True
                return evaluation_final_update(self._task)
            return None

        async def after_tool_call(self, call: ToolCallRecord, response: Any) -> None:  # noqa: ANN401
            self.total_calls_made += 1
            if call.name in counted:
                self.skill_calls_made += 1
                if call.name in budgeted:
                    self.motor_calls_made += 1
                if after_motor_call is not None:
                    await after_motor_call(self.skill_calls_made, call.name)

        def edit_tool_output(self, call: ToolCallRecord, response: Any) -> Any | None:  # noqa: ANN401
            if not batch_images or call.name not in counted:
                return None
            batch = self.batch_of(call)
            following = batch[call.index + 1] if call.index + 1 < len(batch) else None
            if following is None or following.name not in counted:
                return None
            return strip_images(response, BATCHED_IMAGE_NOTE)

    return ClaudeCodeEvaluationLoop


def claude_code_model_id(openrouter_model_id: str) -> str:
    """The Messages-API model id behind an OpenRouter id (`anthropic/claude-opus-5.5` to
    `claude-opus-5-5`), the only kind Claude Code can host.

    OpenRouter writes the version with a dot and Anthropic with a dash; the CLI rejects the
    dotted form as an unknown model.

    :raises SystemExit: For a model OpenRouter routes to another vendor
    """
    vendor, _, model = openrouter_model_id.partition("/")
    if vendor != "anthropic" or not model:
        raise SystemExit(
            f"Claude Code hosts Anthropic models only; {openrouter_model_id!r} is not one."
        )
    return model.replace(".", "-")


def claude_agent_sdk_version() -> str:
    """The installed `claude-agent-sdk` version, which pins the bundled CLI."""
    from importlib.metadata import version

    return version("claude-agent-sdk")


def claude_code_summary(agent: ClaudeCodeLoop) -> dict[str, Any]:
    """The summary fields a Claude Code episode fills in place of the model client's usage.

    `cost_usd` is the SDK's `total_cost_usd`, an estimate from its bundled price table rather
    than a provider's bill, so `cost_source` says so; `model_seconds` is the SDK's API time.
    """
    result = agent.result
    fields: dict[str, Any] = {
        "model_calls": len(agent.usage_history),
        "cost_usd": round(result.total_cost_usd, 4) if result and result.total_cost_usd else None,
        "cost_source": "sdk_estimate",
        "model_seconds": round(result.duration_api_ms / 1000, 3) if result else None,
        "denials": agent.denials,
        "hook_ordering_fallbacks": agent.ordering_fallbacks,
    }
    if result is not None:
        fields["result"] = {
            "subtype": result.subtype,
            "is_error": result.is_error,
            "num_turns": result.num_turns,
            "stop_reason": result.stop_reason,
            "usage": result.usage,
            "model_usage": result.model_usage,
            "errors": result.errors,
            "session_id": result.session_id,
        }
    if agent.init_data is not None:
        fields["init"] = {
            key: agent.init_data.get(key)
            for key in ("model", "tools", "mcp_servers", "permissionMode", "claude_code_version")
        }
    return fields


def tool_call_counts(transcript_path: Path) -> dict[str, Any]:
    """Tool calls in the transcript, by category and by name."""
    from agentic_robotics.adapters.toyblocks.leap.motor_skills_agent import MotorSkillsAgent
    from agentic_robotics.adapters.toyblocks.leap.perception_tools_agent import (
        PerceptionToolsAgent,
    )

    motor = set(MotorSkillsAgent.__abstractmethods__)
    perception = set(PerceptionToolsAgent.__abstractmethods__)
    names: Counter[str] = Counter()
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("type") == "function_call":
            names[item["name"]] += 1
    return {
        "total": sum(names.values()),
        "motor": sum(n for name, n in names.items() if name in motor),
        "perception": sum(n for name, n in names.items() if name in perception),
        "other": sum(n for name, n in names.items() if name not in motor | perception),
        "by_name": dict(names),
    }


async def run_evaluation_episode(
    server: ToyBlocksMCPServer,
    *,
    baseline: str,
    descriptions: dict[str, str] | None,
    descriptions_path: Path | None,
    instance: int | None = None,
    allow_declaration_mismatch: bool = False,
    model_host: str = "openrouter",
    claude_code: dict[str, Any] | None = None,
) -> None:
    """One evaluation episode behind the composed server, checked against `EvaluationAgent`.

    :param allow_declaration_mismatch: Log how the served tools differ from the declaration and
        run anyway, instead of refusing; for a skill surface the declarations do not yet cover
    :param model_host: `openrouter` (the repo's loop) or `claude-code` (the Claude Agent SDK)
    :param claude_code: For the `claude-code` host, `effort` and `max_turns` (both may be None)
    """
    from contextlib import AsyncExitStack

    from mcp import Client as MCPClient

    from agentic_robotics.adapters.toyblocks.leap.composed_server import ComposedServer
    from agentic_robotics.adapters.toyblocks.leap.declarations import (
        assert_serves,
        declared_descriptions,
        recompose_on_backend,
    )
    from agentic_robotics.adapters.toyblocks.leap.leap_agents import EvaluationAgent
    from agentic_robotics.adapters.toyblocks.leap.timing import tool_call_seconds
    from agentic_robotics.core.foundation_models import (
        OpenAIModelClient,
        Transcript,
        create_openai_message,
    )

    config = server.config
    # Under the ManiFlow backend the served skills carry `steps` and their own docstrings, so the
    # declaration the episode is checked against is EvaluationAgent rebuilt on that ABC.
    EvaluationAgent = recompose_on_backend(EvaluationAgent, str(config.skill_backend))
    composed = ComposedServer(
        f"leap-eval-{baseline}",
        simulator=server.mcp,
        snapshot_dir=config.output_dir / "docstrings",
        # Control is evaluated with the declared docstrings, installed rather than assumed to be
        # what the skill server says: they are only the same under the ManiFlow skills.
        initial_descriptions=descriptions or declared_descriptions(EvaluationAgent),
    )
    if model_host not in ("openrouter", "claude-code"):
        raise SystemExit(f"Unknown model host {model_host!r}.")
    model_client = (
        OpenAIModelClient(config=config.model_config) if model_host == "openrouter" else None
    )
    transcript = Transcript(out_dir=config.output_dir)
    prompt = config.task_config.prompt
    summary: dict[str, Any] = {
        "baseline": baseline,
        "task": config.task_config.task_slug,
        "instance": instance,
        "docstrings": str(descriptions_path) if descriptions_path else None,
        "model": config.model_config.model_id,
        "model_host": model_host,
        "max_tool_calls": config.task_config.max_tool_calls,
        "finished": False,
        "success": None,
    }
    if model_host == "claude-code":
        from agentic_robotics.core.claude_code_loop import ClaudeCodeConfig

        options = claude_code or {}
        claude_code_config = ClaudeCodeConfig(
            model=claude_code_model_id(config.model_config.model_id),
            mcp_server_key="toyblocks",
            effort=options.get("effort"),
            max_turns=options.get("max_turns"),
            cwd=config.output_dir,
        )
        summary["model"] = claude_code_config.model
        summary["claude_code"] = {
            "effort": claude_code_config.effort,
            "max_turns": claude_code_config.max_turns,
            "sdk_version": claude_agent_sdk_version(),
        }
    started = time.time()

    # Search & Spell: the motor cutoff and the all-calls ceiling come from the instance's family
    # (11 motor calls per letter block, 2.5x that for calls of any kind), and the state after
    # every motor call is recorded with the target letters it exposes, for the exhaustiveness
    # reading. Other tasks keep the task config's cutoff and the ABC package's ceiling.
    motor_cutoff = config.task_config.max_tool_calls
    total_ceiling: int | None = None
    caps = search_spell_episode_caps(config.task_config.task_slug)
    exposure: list[dict[str, Any]] = []
    after_motor_call = None
    if caps is not None:
        motor_cutoff, total_ceiling, summary["letter_blocks"] = caps
        summary["total_ceiling"] = total_ceiling
        from agentic_robotics.adapters.toyblocks.leap.sim_state import read_sim_state
        from agentic_robotics.adapters.toyblocks.release.search_spell_scoring import (
            exposed_target_letters,
            family_for_slug,
        )

        family = family_for_slug(config.task_config.task_slug)
        states_path = config.output_dir / "states.jsonl"

        async def after_motor_call(motor_calls: int, tool_name: str) -> None:
            state = await on_simulator_thread(
                server, lambda: read_sim_state(server._toyblocks_env).model_dump()
            )
            exposed = exposed_target_letters(state, family)
            exposure.append({"motor_call": motor_calls, "skill": tool_name, "exposed": exposed})
            with states_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"motor_call": motor_calls, "skill": tool_name, "state": state})
                    + "\n"
                )

    async with composed:
        try:
            assert_serves(composed.visible_tools(), EvaluationAgent, descriptions=descriptions)
            logger.info(
                f"Tools handed to the model, checked against EvaluationAgent with the {baseline} "
                f"descriptions: {composed.visible_tool_names}"
            )
        except RuntimeError as exc:
            if not allow_declaration_mismatch:
                raise
            # A skill surface other than the declared one (the scripted skills, for one): say
            # exactly how it differs and run anyway, so the delta is on record.
            summary["declaration_mismatch"] = str(exc)
            (config.output_dir / "declaration_mismatch.txt").write_text(str(exc), encoding="utf-8")
            logger.warning(f"Running despite a declaration mismatch:\n{exc}")
        if instance is not None:
            await on_simulator_thread(server, lambda: reset_to_instance(server, instance))
            logger.info(f"Scene reset to reset state {instance}.")
        async with AsyncExitStack() as stack:
            metadata_callback = lambda: {"frame_count": server.frame_count}  # noqa: E731
            if model_host == "claude-code":
                from agentic_robotics.adapters.toyblocks.leap.composed_server import (
                    serve_over_http,
                )

                mcp_url = await stack.enter_async_context(serve_over_http(composed))
                agent = make_claude_code_evaluation_loop_class(
                    motor_cutoff, total_ceiling, after_motor_call, batch_images=caps is not None
                )(
                    transcript,
                    claude_code_config,
                    mcp_url,
                    composed.visible_tool_names,
                    metadata_callback=metadata_callback,
                )
            else:
                mcp_client = await stack.enter_async_context(MCPClient(composed.server))
                agent = make_evaluation_loop_class(
                    total_ceiling, after_motor_call, batch_images=caps is not None
                )(mcp_client, model_client, transcript, metadata_callback=metadata_callback)
            try:
                if model_host == "claude-code":
                    await agent.run(task=prompt, prompt=prompt)
                else:
                    await agent.enter_loop(
                        task=prompt,
                        history=[create_openai_message(prompt, role="user")],
                        max_tool_calls=motor_cutoff,
                    )
                summary["finished"] = True
            except Exception as exc:
                summary["error"] = f"{type(exc).__name__}: {exc}"[:1000]
                logger.exception("Evaluation episode failed; ending after cleanup.")
            finally:
                summary["motor_cutoff"] = motor_cutoff
                summary["motor_calls_made"] = agent.motor_calls_made
                summary["motor_skill_calls"] = getattr(agent, "skill_calls_made", None)
                summary["unbudgeted_motor_tools"] = sorted(UNBUDGETED_MOTOR_TOOLS)
                summary["total_calls_made"] = agent.total_calls_made
                summary["hit_cutoff"] = agent.hit_cutoff
                if caps is not None:
                    summary["exposure"] = exposure
                    summary["target_letters_exposed_ever"] = sorted(
                        {item for record in exposure for item in record["exposed"]}
                    )
                try:
                    summary.update(
                        await on_simulator_thread(
                            server, lambda: score_scene(server, config.output_dir)
                        )
                    )
                except Exception:
                    logger.exception("Scoring the final scene failed.")
                if not summary["finished"]:
                    # An episode the loop did not finish (a model-API error such as the 2026-10-07
                    # org budget 403, or a crash) measured nothing: keep the scene's verdict for
                    # the record but leave the episode unscored, so the suite does not count it.
                    summary["scene_success"] = summary.get("success")
                    summary["success"] = None
                transcript_path = config.output_dir / "transcript.jsonl"
                if transcript_path.is_file():
                    summary["tool_calls"] = tool_call_counts(transcript_path)
                    summary["tool_seconds"] = tool_call_seconds(transcript_path)
                    if caps is not None:
                        from agentic_robotics.adapters.toyblocks.release.search_spell_scoring import (  # noqa: E501
                            declared_outcome,
                        )

                        summary["declared"] = declared_outcome(
                            final_assistant_message(transcript_path),
                            SEARCH_SPELL_DECLARATIONS,
                        )
                if model_host == "claude-code":
                    summary.update(claude_code_summary(agent))
                else:
                    usage = model_client.usage_history
                    summary["model_calls"] = len(usage)
                    summary["cost_usd"] = round(sum(u.cost_usd or 0.0 for u in usage), 4)
                    summary["model_seconds"] = round(sum(u.duration_s or 0.0 for u in usage), 3)
                summary["seconds"] = round(time.time() - started, 1)
                (config.output_dir / "evaluation_summary.json").write_text(
                    json.dumps(summary, indent=1), encoding="utf-8"
                )
                logger.info(
                    f"Evaluation episode: success={summary['success']}, "
                    f"{summary.get('tool_calls', {}).get('total')} tool calls, "
                    f"${summary['cost_usd']}."
                )


def evaluation_agentic_thread(
    server: ToyBlocksMCPServer, finished: threading.Event, **kwargs: Any
) -> None:
    """Run the episode on this thread, which is not the simulator's; release it when done."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_evaluation_episode(server, **kwargs))
    finally:
        loop.close()
        finished.set()
