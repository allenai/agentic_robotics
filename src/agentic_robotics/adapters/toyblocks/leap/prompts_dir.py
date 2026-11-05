"""Where the developer's LEAP prompt files live: the `prompts/` directory of the ABC package.

Resolved from the package's path rather than imported as a package, so the directory needs no
`__init__.py` and stays plain text files the developer edits.
"""

from pathlib import Path

from agentic_robotics.adapters.toyblocks.leap import constants

PROMPTS_DIR = Path(constants.__file__).resolve().with_name("prompts")
"""the developer's `experimentation.md` and `subagent_preamble.md`."""
