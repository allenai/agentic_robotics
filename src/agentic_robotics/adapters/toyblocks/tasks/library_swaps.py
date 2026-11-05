"""Existing scenes rebuilt with another glyph asset library, as `TaskConfig`s for
`agentic-robotics` to resolve.

A ManiFlow checkpoint only sees the blocks its training data rendered, and `block_asset_check`
refuses a scene whose blocks come from another library. Regenerating a scene with
`--usd-library` samples new reset states, so a scene whose start must stay fixed is copied instead:
its env package (`data/tasks/<slug>/` and the autogen tree) is the original's with only the block
table pointed at the other library, keeping the original's `reset_states.pkl`, and its PDDL is
`data/problems/<slug>.pddl`. Each task takes its prompt and call budget from the task it copies, so
the agent sees the same episode. `cli.py`'s `load_tasks` merges this table.
"""

from __future__ import annotations

from typing import Any

from agentic_robotics.adapters.toyblocks.config import TOYBLOCKS_TASKS, TaskConfig

LIBRARY_SWAPS: dict[str, tuple[str, str]] = {
    "ai2_medium_07_lo": ("ai2_medium", "train_blocks_large"),
}
"""Each swapped scene's slug, mapped to the task whose prompt it reuses and the library its blocks
come from."""


def library_swap_tasks() -> dict[str, Any]:
    """The swapped scenes as `TaskConfig`s, keyed by slug."""
    tasks = {}
    for slug, (source_key, library) in LIBRARY_SWAPS.items():
        source = TOYBLOCKS_TASKS[source_key]
        tasks[slug] = TaskConfig(
            name=f"{source.name}: {slug} with {library} blocks",
            task_slug=slug,
            prompt=source.prompt,
            max_tool_calls=source.max_tool_calls,
            glyph_library=library,
        )
    return tasks
