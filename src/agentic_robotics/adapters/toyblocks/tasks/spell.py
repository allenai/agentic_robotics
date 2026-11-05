"""Registration of the experimentation phase's example scene.

`spell_example` (DUCK, one problem, four letters) is the scene the release experimentation phase
runs in (`agentic-robotics leap --example-states data/example_states_release`); its package stays
registered here while that phase and its continuation run. The Search & Spell evaluation tasks
themselves are `TOYBLOCKS_TASKS` entries in `config.py`. The word tasks NULL, MENU, YES, CHECK
and EMU under `data/problems/spell_<word>/` are not registered; whether their files go is the
developer's call.
"""

from __future__ import annotations

from typing import Any

from agentic_robotics.adapters.toyblocks.config import TaskConfig

EXAMPLE_TASK = "spell_example"


def spell_tasks() -> dict[str, Any]:
    """The experimentation phase's example scene, keyed `spell_example`."""
    return {
        EXAMPLE_TASK: TaskConfig(
            name="Search & Spell example scene (DUCK), for the experimentation phase",
            task_slug=EXAMPLE_TASK,
            prompt="",
            max_tool_calls=80,
        )
    }
