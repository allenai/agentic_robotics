"""The experiment drivers: what builds the task instances, runs an evaluation suite episode by
episode, and tabulates the results. Each module is a command (``python -m
agentic_robotics.adapters.toyblocks.drivers.<name>``, or the console script ``pyproject.toml``
names) with its own arguments; the episodes themselves are ``agentic-robotics episode`` runs."""
