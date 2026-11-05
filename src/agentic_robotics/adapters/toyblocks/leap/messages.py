"""Every message the LEAP code puts in front of the model, apart from the prompt files and the
tool docstrings.

The developer writes all model-facing text. The code imports from here and never inlines a
string the model will read; edit the wording here and nowhere else. Text the developer has not
rewritten reads as it did in the 2026-09-29 pilots.

Not here, and where to find them:

- The system prompts: the developer's `experimentation.md` and `subagent_preamble.md` under the ABC
  package's `prompts/` directory (see `leap/prompts_dir.py`).
- The tool docstrings: the ABCs under `src/agentic_robotics/adapters/toyblocks/leap/`.
- Errors raised inside tools (a bad path, an unknown tool): beside the tool that raises them.
- The core loop's per-turn "tool calls remaining" line and its final-answer request:
  `src/agentic_robotics/core/agentic_loop.py`. Evaluation episodes (`agentic-robotics episode
  --baseline`) use `evaluation_status_line` and `evaluation_final_update` below instead, because
  their cutoff counts motor skill calls only.
"""

from __future__ import annotations


def status_line(
    tool_calls_left: int, prompt_tokens: int, limit_tokens: int, budget_status: str
) -> str:
    """The first line of every experimentation-phase turn's system message.

    :param tool_calls_left: Tool calls left under the phase's ceiling
    :param prompt_tokens: Size of the previous model call's prompt, in tokens
    :param limit_tokens: Context size at which the history is cleared
    :param budget_status: The ledger's "spent of cap" text
    """
    remaining = max(limit_tokens - prompt_tokens, 0)
    return (
        f"Tool calls remaining: {tool_calls_left}. Context used: {prompt_tokens:,} of "
        f"{limit_tokens:,} tokens ({remaining:,} remaining). Budget spent: {budget_status}."
    )


def context_warning(threshold: float, limit_tokens: int, may_clear_early: bool) -> str:
    """Appended to the status line the first time the context crosses a warning threshold.

    :param threshold: The fraction of the limit just crossed (0.5, 0.9 or 0.95)
    :param limit_tokens: Context size at which the history is cleared
    :param may_clear_early: Whether clear_context is available at this threshold
    """
    early = " You may call clear_context to clear it early." if may_clear_early else ""
    return (
        f"Warning: your context is over {round(100 * threshold)}% full. When it "
        f"reaches {limit_tokens:,} tokens it will be cleared, keeping only your task "
        "instructions and your workspace files; write anything you want to keep to "
        f"a file before then.{early}"
    )


def clear_context_refused(percent_used: int, min_percent: int) -> str:
    """The error a clear_context call gets below the threshold at which clearing is allowed."""
    return (
        f"Your context is {percent_used}% full; clear_context is available only at "
        f"{min_percent}% or more."
    )


def context_limit_reason(limit_tokens: int) -> str:
    """Why the context was cleared, when the limit did it; fills `context_cleared_note`."""
    return f"it reached the limit of {limit_tokens:,} tokens"


CLEAR_CONTEXT_REASON = "you asked for it with clear_context"
"""Why the context was cleared, when the agent did it; fills `context_cleared_note`."""


def context_cleared_note(reason: str, workspace_listing: str) -> str:
    """The system message that opens the fresh history after a context clear."""
    return (
        f"[Context cleared] Your conversation history was cleared because {reason}. Nothing "
        "from before this message is visible to you any more, apart from what you wrote to "
        f"files. Your workspace holds:\n{workspace_listing}"
    )


def context_limit_file_turn_note(limit_tokens: int, workspace_listing: str) -> str:
    """The system message for the one file-tools-only turn the agent gets at the context limit
    when it never cleared the context itself; the history is cleared after this turn."""
    return (
        f"Your context has reached its limit of {limit_tokens:,} tokens and will be cleared "
        "after this turn. Only the file tools are available now: write anything you want to "
        f"keep to your workspace. Your workspace holds:\n{workspace_listing}"
    )


def resumed_phase_note(prior_cost_usd: float, budget_status: str, workspace_listing: str) -> str:
    """The system message that opens a resumed experimentation phase: a fresh context over the
    earlier run's workspace and its final tool descriptions, with the budget raised."""
    return (
        f"[Phase resumed] An earlier session of this experimentation phase spent "
        f"${prior_cost_usd:.2f} and ended when its budget ran out. Its workspace files and its "
        "final tool descriptions are in place; its conversation is not. The budget has been "
        f"raised: {budget_status} spent so far. Your workspace holds:\n{workspace_listing}"
    )


def no_tool_call_note(budget_status: str, calls_before_exit: int) -> str:
    """Appended when a turn made no tool call while budget remains; the phase continues."""
    return (
        "[No tool call] The experimentation phase only ends once your budget is spent. "
        f"Budget spent: {budget_status}. After {calls_before_exit} turns in a row without "
        "a tool call, your remaining budget will be forfeited and the phase will end."
    )


def final_file_turn_note(budget_status: str) -> str:
    """The system message for the one file-tools-only turn after the budget cap."""
    return (
        f"Your budget is exhausted ({budget_status}). This is your final turn, and "
        "only the file tools are available: write anything you want to keep to your "
        "workspace now. After this turn you will be asked for your final answer."
    )


def subagent_status_line(tool_calls_left: int) -> str:
    """The per-turn system message a subagent episode gets."""
    return f"Tool calls remaining: {tool_calls_left}."


def harness_trial_preamble(label: str, hypothesis: str, success_criteria: str) -> str:
    """Put in front of a trial's prompt when the harness runs it as part of an experiment."""
    return (
        f"This is trial '{label}' of an experiment.\n"
        f"Hypothesis under test: {hypothesis}\n"
        f"Success criteria: {success_criteria}\n\n"
        "Carry out the trial below, then end with a report that states whether the criteria "
        "were met, what you observed that decides it, and anything that did not go as the "
        "trial expected.\n\n"
    )


def evaluation_status_line(motor_calls_left: int, total_calls_left: int) -> str:
    """The system message before every evaluation-episode turn.

    Only motor skill calls count toward the cutoff.
    Perception and other tools are free up to a safety ceiling on all calls.

    :param motor_calls_left: Motor skill calls left under the cutoff
    :param total_calls_left: Calls of any kind left under the safety ceiling
    """
    return (
        f"You have {motor_calls_left} motor skill calls remaining. Perception tool calls "
        f"do not count toward that limit ({total_calls_left} calls of any kind remaining)."
    )


def evaluation_final_update(task: str) -> str:
    """Asks for the final answer once an evaluation episode's cutoff is reached."""
    return (
        f"Recall your current task:\n\n{task}\n\n"
        "You have no further motor skill calls available. Respond with your final answer."
    )


def final_update(task: str) -> str:
    """Asks for the final answer once an experimentation-phase or subagent loop has left its
    tool-call loop (the core loop's own wording, kept here so no LEAP text comes from
    `src/agentic_robotics/core`)."""
    return (
        f"Recall your current task:\n\n{task}\n\n"
        "You have no further tool calls available. Respond with your final answer."
    )
