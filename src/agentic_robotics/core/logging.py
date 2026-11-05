"""Utility definitions to configure logging."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Attach one handler to the `agentic_robotics` namespace, isolated from the root logger."""
    package_logger = logging.getLogger("agentic_robotics")
    if package_logger.handlers:
        return  # Already configured; don't attach a second handler

    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s %(levelname)s %(name)s] %(message)s")
    handler.setFormatter(formatter)
    package_logger.addHandler(handler)
    package_logger.setLevel(level)
    package_logger.propagate = False
