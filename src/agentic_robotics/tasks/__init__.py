"""Generated task packages, imported from this repo's ``data/tasks/`` directory.

Each task package (``<slug>/__init__.py``, ``env.py``, ``env_cfg.py``, ``reset_states.pkl``) is
written there by ToyBlocks's task generator and registers its gym environment with entry points
relative to its own module name, so it imports as ``agentic_robotics.tasks.<slug>`` from wherever
this package's search path points. Keeping the packages here, rather than inside ToyBlocks's
source tree, means ToyBlocks is never written to and an ordinary (non-editable) install of it
works.
"""

from __future__ import annotations

from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parents[3] / "data/tasks"
"""The directory the task packages live in; also where the generator writes new ones."""

__path__.append(str(TASKS_DIR))
